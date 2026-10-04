"""Published trace datasets, downloaded from Zenodo and verified by SHA-256.

    from linkgym.datasets import fetch

    path = fetch("munich-v1")  # downloaded on first use, then read from the cache

Every name refers to one file of one Zenodo *version* record, never to the concept record
(which resolves to the latest version), so the file behind a name never changes; its SHA-256
and size are fixed here and checked on download and on every later call. Files are cached
in ``$LINKGYM_DATA_DIR`` if set, else in ``$XDG_CACHE_HOME/linkgym`` or ``~/.cache/linkgym``.

Also a command: ``python -m linkgym.datasets [NAME ...]`` lists the datasets, or fetches
the named ones and prints their paths. Standard library only.

Stability: stable (docs/api_stability.md).
"""

from __future__ import annotations

import hashlib
import http.client
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DATASETS",
    "Dataset",
    "DatasetError",
    "ZENODO_RECORD",
    "default_cache_dir",
    "fetch",
]

# The Zenodo version record holding both Munich v1 files (DOI 10.5281/zenodo.23135098)
ZENODO_RECORD: str | None = "23135098"
URL = "https://zenodo.org/records/{record}/files/{filename}?download=1"
TIMEOUT = 60  # seconds without data before a download fails
CHUNK = 1 << 20


class DatasetError(RuntimeError):
    """A dataset could not be fetched, or a file failed verification."""


@dataclass(frozen=True)
class Dataset:
    """One file of a Zenodo version record.

    :param filename: File name in the record and in the cache
    :param sha256: SHA-256 of the file (hex)
    :param size: File size [bytes]
    :param record: Zenodo version record ID; ``None`` if not published yet
    :param description: One line on the content
    """

    filename: str
    sha256: str
    size: int
    record: str | None
    description: str


DATASETS: dict[str, Dataset] = {
    "munich-v1": Dataset(
        filename="munich-v1.h5",
        sha256="4b3bbb246c40f4513a685d069a9e913649d58a88e4dbdc908b3fa4e074f3cd9c",
        size=67_002_864,
        record=ZENODO_RECORD,
        description="Munich ray-traced traces: 76 trajectories of 4000 slots on 15 streets "
        "(train, val and test splits)",
    ),
    "munich-v1-test-alt": Dataset(
        filename="munich-v1-test-alt.h5",
        sha256="0969ea3638aa34c2c7547d66d8de9746bc3109f59a324844c07dadb9b359c347",
        size=20_283_984,
        record=ZENODO_RECORD,
        description="the 23 test trajectories of munich-v1 traced again with 8 million rays "
        "(alternative realization)",
    ),
}


def default_cache_dir() -> Path:
    """``$LINKGYM_DATA_DIR`` if set, else ``$XDG_CACHE_HOME/linkgym`` or ``~/.cache/linkgym``."""
    if os.environ.get("LINKGYM_DATA_DIR"):
        return Path(os.environ["LINKGYM_DATA_DIR"]).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base).expanduser() / "linkgym"


def fetch(
    name: str,
    *,
    cache_dir: str | os.PathLike | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Return the local path of dataset ``name``, downloading it on first use.

    A cached file is checked against the dataset's SHA-256 on every call. A download goes
    to a ``.part`` file next to the target and is moved into place only after its size and
    SHA-256 match, so an interrupted or corrupt download never leaves a file under the
    dataset's name.

    :param name: A key of :data:`DATASETS`, e.g. ``"munich-v1"``
    :param cache_dir: Directory for the file; default :func:`default_cache_dir`
    :param force: Download again even if the file is cached
    :param progress: Print download progress to stderr
    :raises DatasetError: Unknown name, record not published, download failure, or a size
        or SHA-256 mismatch (of the download or of the cached file)
    """
    try:
        dataset = DATASETS[name]
    except KeyError:
        raise DatasetError(
            f"unknown dataset {name!r}; available: {', '.join(sorted(DATASETS))}"
        ) from None
    directory = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    path = directory / dataset.filename
    if path.exists() and not force:
        digest = _sha256(path)
        if digest != dataset.sha256:
            raise DatasetError(
                f"{path} has SHA-256 {digest}, expected {dataset.sha256}: the file is corrupt "
                f"or not this dataset. Delete it, or call fetch({name!r}, force=True) to "
                "download it again."
            )
        return path
    if dataset.record is None:
        raise DatasetError(
            f"{name} is not published yet (no Zenodo record in linkgym.datasets). Generate it "
            "with linkgym-traces (docs/channels.md, Munich dataset) and pass its path as "
            "trace_path."
        )
    directory.mkdir(parents=True, exist_ok=True)
    _download(URL.format(record=dataset.record, filename=dataset.filename), path, dataset, progress)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, path: Path, dataset: Dataset, progress: bool) -> None:
    part = path.with_name(path.name + ".part")
    digest = hashlib.sha256()
    received = 0
    if progress:
        print(f"Downloading {url} ({dataset.size / 1e6:.1f} MB)", file=sys.stderr, flush=True)
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response, part.open("wb") as f:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != dataset.size:
                raise DatasetError(f"{url} has {length} bytes, expected {dataset.size}")
            shown = 0
            while block := response.read(CHUNK):
                received += len(block)
                if received > dataset.size:
                    raise DatasetError(f"{url} sent more than the expected {dataset.size} bytes")
                f.write(block)
                digest.update(block)
                if progress and received * 10 // dataset.size > shown:
                    shown = received * 10 // dataset.size
                    print(f"\r  {10 * shown}%", end="", file=sys.stderr, flush=True)
        if progress:
            print(file=sys.stderr)
        if received != dataset.size:
            raise DatasetError(
                f"the download of {url} stopped after {received} of {dataset.size} bytes"
            )
        if digest.hexdigest() != dataset.sha256:
            raise DatasetError(
                f"{url} has SHA-256 {digest.hexdigest()}, expected {dataset.sha256}; nothing "
                f"was written to {path}"
            )
        os.replace(part, path)
    except (OSError, http.client.HTTPException) as error:  # URLError, HTTPError, timeouts
        raise DatasetError(
            f"could not download {url}: {error}. You can download it yourself, check that its "
            f"SHA-256 is {dataset.sha256} and put it at {path}."
        ) from error
    finally:
        part.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    names = sys.argv[1:] if argv is None else argv
    if not names:
        for name, dataset in DATASETS.items():
            status = "" if dataset.record else " (not published yet)"
            print(f"{name}: {dataset.description}, {dataset.size / 1e6:.1f} MB{status}")
        return 0
    try:
        for name in names:
            print(fetch(name))
    except DatasetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
