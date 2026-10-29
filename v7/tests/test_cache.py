import base64

import pytest

from backend.meta.cache import Cache, cache_key, normalize, sha256_hex

PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"


def data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def test_key_is_stable_and_order_independent():
    a = cache_key("m", "/responses", {"x": 1, "y": [1, 2], "z": None})
    b = cache_key("m", "/responses", {"y": [1, 2], "x": 1})
    assert a == b and len(a) == 64


def test_key_depends_on_model_endpoint_and_request():
    base = cache_key("m", "/responses", {"x": 1})
    assert cache_key("m2", "/responses", {"x": 1}) != base
    assert cache_key("m", "/images/edits", {"x": 1}) != base
    assert cache_key("m", "/responses", {"x": 2}) != base


def test_media_hashed_the_same_as_bytes_or_data_url():
    as_bytes = cache_key("m", "/e", {"image": PNG})
    as_url = cache_key("m", "/e", {"image": data_url(PNG)})
    as_jpeg_label = cache_key("m", "/e", {"image": data_url(PNG, "image/jpeg")})
    assert as_bytes == as_url == as_jpeg_label
    assert normalize({"image": PNG}) == {"image": {"sha256": sha256_hex(PNG)}}
    assert cache_key("m", "/e", {"image": PNG + b"!"}) != as_bytes


def test_transport_fields_ignored_only_at_top_level():
    assert cache_key("m", "/e", {"x": 1, "stream": True}) == cache_key("m", "/e", {"x": 1})
    schema_a = {"schema": {"properties": {"stream": {"type": "string"}}}}
    schema_b = {"schema": {"properties": {}}}
    assert cache_key("m", "/e", schema_a) != cache_key("m", "/e", schema_b)


def test_schema_property_order_counts():
    a = {"schema": {"properties": {"reason": {}, "label": {}}}}
    b = {"schema": {"properties": {"label": {}, "reason": {}}}}
    assert cache_key("m", "/e", a) != cache_key("m", "/e", b)


def test_plain_strings_are_not_treated_as_media():
    assert normalize("data:text/plain,hello") == "data:text/plain,hello"
    assert normalize("data:image/png;base64,***") == "data:image/png;base64,***"


def test_put_get_and_blobs(tmp_path):
    cache = Cache(tmp_path)
    key = cache_key("m", "/e", {"x": 1})
    assert cache.get(key) is None and key not in cache
    cache.put(key, {"model": "m", "response": {"ok": True}}, blobs={"0.png": PNG})
    assert key in cache
    record = cache.get(key)
    assert record["response"] == {"ok": True}
    assert record["blobs"] == ["0.png"]
    assert cache.blob(key, "0.png") == PNG
    assert (tmp_path / key[:2] / f"{key}.json").is_file()
    assert not list(tmp_path.rglob(".tmp-*"))


def test_bad_blob_names(tmp_path):
    cache = Cache(tmp_path)
    for name in ("", "a/b", "json"):
        with pytest.raises(ValueError):
            cache.put("ab" * 32, {}, blobs={name: b"x"})
