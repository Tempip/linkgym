"""linkgym.datasets with a mocked download: no test touches the network."""

import hashlib
import io
import re
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

from linkgym import datasets
from linkgym.datasets import Dataset, DatasetError, fetch

REPO = Path(__file__).resolve().parents[1]
CONTENT = b"linkgym test trace\n" * 1000
URL = "https://zenodo.org/records/123/files/fake.h5?download=1"


class Response(io.BytesIO):
    def __init__(self, data: bytes, length: int | None):
        super().__init__(data)
        self.headers = {} if length is None else {"Content-Length": str(length)}


@pytest.fixture
def cache(monkeypatch, tmp_path):
    """A fake published dataset and an empty cache directory."""
    directory = tmp_path / "cache"
    monkeypatch.setenv("LINKGYM_DATA_DIR", str(directory))
    fake = Dataset("fake.h5", hashlib.sha256(CONTENT).hexdigest(), len(CONTENT), "123", "test")
    monkeypatch.setitem(datasets.DATASETS, "fake", fake)
    return directory


def serve(monkeypatch, data=CONTENT, length="auto", error=None) -> list[str]:
    """Replace urlopen; return the list of requested URLs."""
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        if error is not None:
            raise error
        return Response(data, len(data) if length == "auto" else length)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return calls


def test_fetch_downloads_verifies_and_caches(monkeypatch, cache, capsys):
    calls = serve(monkeypatch)
    path = fetch("fake")
    assert path == cache / "fake.h5"
    assert path.read_bytes() == CONTENT
    assert calls == [URL]
    assert sorted(p.name for p in cache.iterdir()) == ["fake.h5"]  # no .part file left
    assert "100%" in capsys.readouterr().err


def test_cached_file_is_verified_not_downloaded(monkeypatch, cache):
    serve(monkeypatch)
    fetch("fake")
    calls = serve(monkeypatch, error=AssertionError("downloaded again"))
    assert fetch("fake", progress=False) == cache / "fake.h5"
    assert calls == []


def test_explicit_cache_dir(monkeypatch, cache, tmp_path):
    serve(monkeypatch)
    path = fetch("fake", cache_dir=tmp_path / "elsewhere", progress=False)
    assert path == tmp_path / "elsewhere" / "fake.h5"
    assert not cache.exists()


def test_default_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("LINKGYM_DATA_DIR", str(tmp_path / "data"))
    assert datasets.default_cache_dir() == tmp_path / "data"
    monkeypatch.delenv("LINKGYM_DATA_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert datasets.default_cache_dir() == tmp_path / "xdg" / "linkgym"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert datasets.default_cache_dir() == Path.home() / ".cache" / "linkgym"


def test_hash_mismatch_writes_nothing(monkeypatch, cache):
    serve(monkeypatch, data=CONTENT[:-1] + b"x")
    with pytest.raises(DatasetError, match="SHA-256"):
        fetch("fake", progress=False)
    assert list(cache.iterdir()) == []


@pytest.mark.parametrize(
    "data, length, message",
    [
        (CONTENT[:-10], None, "stopped after"),
        (CONTENT[:-10], len(CONTENT) - 10, "expected"),
        (CONTENT + b"x", None, "more than"),
    ],
    ids=["truncated", "wrong_length", "too_long"],
)
def test_size_mismatch_writes_nothing(monkeypatch, cache, data, length, message):
    serve(monkeypatch, data=data, length=length)
    with pytest.raises(DatasetError, match=message):
        fetch("fake", progress=False)
    assert list(cache.iterdir()) == []


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("no route to host"),
        urllib.error.HTTPError(URL, 404, "NOT FOUND", {}, None),
        TimeoutError("timed out"),
    ],
    ids=["url_error", "http_404", "timeout"],
)
def test_network_errors_are_dataset_errors(monkeypatch, cache, error):
    serve(monkeypatch, error=error)
    with pytest.raises(DatasetError, match=re.escape(URL)) as info:
        fetch("fake", progress=False)
    assert "download it yourself" in str(info.value)
    assert not (cache / "fake.h5").exists()


def test_corrupt_cache_raises_and_force_downloads_again(monkeypatch, cache):
    cache.mkdir()
    (cache / "fake.h5").write_bytes(b"not the dataset")
    calls = serve(monkeypatch)
    with pytest.raises(DatasetError, match=r"force=True"):
        fetch("fake", progress=False)
    assert calls == []
    assert fetch("fake", force=True, progress=False).read_bytes() == CONTENT
    assert calls == [URL]


def test_unknown_name_lists_the_datasets():
    with pytest.raises(DatasetError, match="munich-v1, munich-v1-test-alt"):
        fetch("munich-v2")


def test_unpublished_record(monkeypatch, cache):
    monkeypatch.setitem(datasets.DATASETS, "draft", Dataset("d.h5", "0" * 64, 1, None, "draft"))
    calls = serve(monkeypatch)
    with pytest.raises(DatasetError, match="not published yet"):
        fetch("draft")
    assert calls == []


def test_command_lists_and_fetches(monkeypatch, cache, capsys):
    serve(monkeypatch)
    assert datasets.main([]) == 0
    assert "munich-v1:" in capsys.readouterr().out
    assert datasets.main(["fake"]) == 0
    assert capsys.readouterr().out.strip() == str(cache / "fake.h5")
    assert datasets.main(["munich-v2"]) == 1
    assert "unknown dataset" in capsys.readouterr().err


def test_registry_matches_the_protocol_hashes():
    # The v0.2 experiment checked these files by hash; the published ones must be the same
    protocol = (REPO / "examples" / "evaluate_v02.py").read_text(encoding="utf-8")
    for name, dataset in datasets.DATASETS.items():
        assert re.fullmatch(r"[0-9a-f]{64}", dataset.sha256)
        assert f'"data/{dataset.filename}",\n        "{dataset.sha256}"' in protocol, name


@pytest.mark.parametrize("name", sorted(datasets.DATASETS))
def test_local_copy_matches_the_registry(name):
    dataset = datasets.DATASETS[name]
    path = REPO / "data" / dataset.filename
    if not path.exists():
        pytest.skip(f"{path} not present (generated locally, not committed)")
    assert path.stat().st_size == dataset.size
    assert datasets._sha256(path) == dataset.sha256


def test_import_does_not_load_torch():
    code = "import sys, linkgym.datasets; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
