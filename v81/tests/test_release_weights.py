"""te fetch-weights (backend/release/weights.py) against a fake Modal volume: nothing is
downloaded for real."""

import enum
import sys
import types

import pytest

from backend.release import weights
from backend.release.weights import fetch_weights


class FileEntryType(enum.IntEnum):  # as modal.volume.FileEntryType
    UNSPECIFIED = 0
    FILE = 1
    DIRECTORY = 2


class FakeVolume:
    """listdir and read_file of modal.Volume over {path: bytes or None (a folder)}."""

    def __init__(self, files: dict, short: str = ""):
        self.files, self.short, self.read = files, short, []

    def listdir(self, path):
        prefix = path.strip("/") + "/"
        if not any(p.startswith(prefix) for p in self.files):
            raise FileNotFoundError(f"No such file or directory: {path}")
        return [types.SimpleNamespace(
            path=p, type=FileEntryType.DIRECTORY if data is None else FileEntryType.FILE,
            size=0 if data is None else len(data), mtime=0)
            for p, data in self.files.items()
            if p.startswith(prefix) and "/" not in p[len(prefix):]]

    def read_file(self, path):
        self.read.append(path.rsplit("/", 1)[-1])
        data = self.files[path]
        if path.endswith(self.short):
            data = data[:-1] if self.short else data
        yield from (data[i:i + 4] for i in range(0, len(data), 4))


RUN = "runs/final/out/"
ADAPTER = {RUN + "adapter_config.json": b'{"r": 16}', RUN + "adapter_model.safetensors": b"w" * 30,
           RUN + "chat_template.jinja": b"{{ x }}", RUN + "README.md": b"# trainer card",
           RUN + "training_args.bin": b"pickle", RUN + "checkpoint-10": None,
           RUN + "prepared": None}


def test_only_the_top_level_weight_and_config_files_are_fetched(tmp_path):
    volume, out = FakeVolume(ADAPTER), tmp_path / "data/weights/final/adapter"
    report = fetch_weights("final", out, "adapter", volume=volume)
    assert sorted(p.name for p in out.iterdir()) == [
        "adapter_config.json", "adapter_model.safetensors", "chat_template.jinja"]
    assert (out / "adapter_model.safetensors").read_bytes() == b"w" * 30
    assert report["skipped"] == ["README.md", "checkpoint-10", "prepared", "training_args.bin"]
    assert report["remote"] == "/runs/final/out" and report["bytes"] == 9 + 30 + 7
    assert sorted(volume.read) == ["adapter_config.json", "adapter_model.safetensors",
                                   "chat_template.jinja"]  # no pickle or checkpoint was read
    with pytest.raises(ValueError, match="already has files"):
        fetch_weights("final", out, "adapter", volume=volume)


def test_a_short_download_or_a_folder_without_weights_leaves_nothing(tmp_path):
    out = tmp_path / "w"
    with pytest.raises(RuntimeError, match="downloaded 29 bytes"):
        fetch_weights("final", out, volume=FakeVolume(ADAPTER, short=".safetensors"))
    assert sorted(p.name for p in tmp_path.iterdir()) == []  # no half-made folder
    no_weights = {k: v for k, v in ADAPTER.items() if not k.endswith(".safetensors")}
    with pytest.raises(ValueError, match="no adapter weights"):
        fetch_weights("final", out, volume=FakeVolume(no_weights))
    with pytest.raises(RuntimeError, match="te export first"):
        fetch_weights("final", out, "merged", volume=FakeVolume(ADAPTER))
    merged = {"runs/final/merged/config.json": b"{}",
              "runs/final/merged/model-00001-of-00002.safetensors": b"a",
              "runs/final/merged/model-00002-of-00002.safetensors": b"b",
              "runs/final/merged/model.safetensors.index.json": b"{}"}
    report = fetch_weights("final", out, "merged", volume=FakeVolume(merged))
    assert len(report["files"]) == 4 and report["skipped"] == []
    for bad in ({"run": "Bad Name"}, {"run": "final", "kind": "gguf"}):
        with pytest.raises(ValueError):
            fetch_weights(out_dir=tmp_path / "x", volume=FakeVolume(ADAPTER), **bad)


def test_without_modal_it_says_how_to_install(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)
    with pytest.raises(RuntimeError, match=r'pip install -e "\.\[train\]"'):
        fetch_weights("final", tmp_path / "w")
    opened = []
    fake = types.SimpleNamespace(Volume=types.SimpleNamespace(
        from_name=lambda name: opened.append(name) or FakeVolume(ADAPTER)))
    monkeypatch.setitem(sys.modules, "modal", fake)
    fetch_weights("final", tmp_path / "w")
    assert opened == [weights.VOLUME] == ["trained-eye"]
