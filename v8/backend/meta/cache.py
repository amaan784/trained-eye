"""Content-addressed cache for Meta API calls.

The key is a SHA-256 over the model, the endpoint and the normalized request. Media
(raw bytes or base64 data URLs) is replaced by its own SHA-256 before hashing, so the
same image always produces the same key no matter how it was encoded. A hit means no
network call and no spend.

Layout: <root>/<key[:2]>/<key>.json, plus <key>.<name> for binary outputs such as
generated images. Binary outputs are stored exactly as the API returned them, so
watermarks and provenance metadata are kept.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

# Request fields that change how a response is delivered, not what it says.
TRANSPORT_FIELDS = frozenset({"stream", "stream_options", "timeout", "user"})


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _media_digest(value: str) -> str | None:
    """SHA-256 of the decoded bytes if `value` is a base64 data URL."""
    if not value.startswith("data:"):
        return None
    header, sep, payload = value.partition(",")
    if not sep or not header.endswith(";base64"):
        return None
    try:
        return sha256_hex(base64.b64decode(payload, validate=True))
    except ValueError:
        return None


def normalize(obj: Any, _top: bool = True) -> Any:
    """Return a JSON-safe copy with media hashed and None dropped.

    Keys are sorted when hashed, except under a JSON schema's "properties": strict
    structured output generates fields in that order, so it is kept. Transport fields
    are removed from the top level only, so a schema property that happens to be
    called "stream" still counts.
    """
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return {"sha256": sha256_hex(bytes(obj))}
    if isinstance(obj, str):
        digest = _media_digest(obj)
        return {"sha256": digest} if digest else obj
    if isinstance(obj, dict):
        return {
            str(k): (
                [[str(name), normalize(spec, False)] for name, spec in v.items()]
                if k == "properties" and isinstance(v, dict)
                else normalize(v, False)
            )
            for k, v in obj.items()
            if v is not None and not (_top and k in TRANSPORT_FIELDS)
        }
    if isinstance(obj, (list, tuple)):
        return [normalize(v, False) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def cache_key(model: str, endpoint: str, request: Any) -> str:
    canonical = json.dumps(
        {"model": model, "endpoint": endpoint, "request": normalize(request)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256_hex(canonical.encode("utf-8"))


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class Cache:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _path(self, key: str, suffix: str = "json") -> Path:
        return self.root / key[:2] / f"{key}.{suffix}"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(
        self, key: str, record: dict, blobs: dict[str, bytes] | None = None
    ) -> None:
        """Store a record and optional binary outputs. Blobs are written first."""
        names = []
        for name, data in (blobs or {}).items():
            if not name or "/" in name or name == "json":
                raise ValueError(f"bad blob name {name!r}")
            _atomic_write(self._path(key, name), data)
            names.append(name)
        stored = {"key": key, "created_at": time.time(), **record, "blobs": names}
        _atomic_write(self._path(key), json.dumps(stored, indent=1).encode("utf-8"))

    def blob(self, key: str, name: str) -> bytes:
        return self._path(key, name).read_bytes()

    def __contains__(self, key: str) -> bool:
        return self._path(key).is_file()
