"""Build the Zenodo package of the Munich dataset v1, in a folder outside the repository.

    python examples/rt/package_zenodo.py [--data data] [--out ../zenodo/munich-v1]

Copies into the output folder: the two trace files, checked against the sizes and SHA-256
in linkgym.datasets; the generator configuration (examples/rt/munich.json); the route
figure (docs/assets/munich_routes.png); the dataset README (examples/rt/zenodo/README.md,
with the record's version DOI filled in from linkgym.datasets.ZENODO_RECORD); the ODbL 1.0
text as LICENSE and the attribution NOTICE. Then writes SHA256SUMS over all of them, in the
format of ``sha256sum -c``. Until the Zenodo record is set, the DOI stays a placeholder and
the script says so.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from linkgym.datasets import DATASETS, ZENODO_RECORD

REPO = Path(__file__).resolve().parents[2]
SOURCES = REPO / "examples" / "rt" / "zenodo"
PLACEHOLDER_DOI = "10.5281/zenodo.XXXXXXX"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=REPO / "data", help="trace files")
    parser.add_argument("--out", type=Path, default=REPO.parent / "zenodo" / "munich-v1")
    args = parser.parse_args()
    out = args.out.resolve()
    if out == REPO or REPO in out.parents:
        raise SystemExit(f"{out} is inside the repository; build the package outside it")
    out.mkdir(parents=True, exist_ok=True)

    files = []
    for name in ("munich-v1", "munich-v1-test-alt"):
        dataset = DATASETS[name]
        source = args.data / dataset.filename
        if source.stat().st_size != dataset.size or sha256(source) != dataset.sha256:
            raise SystemExit(f"{source} is not the published {name} (size or SHA-256 differs)")
        target = out / dataset.filename
        if not target.exists() or sha256(target) != dataset.sha256:
            shutil.copyfile(source, target)
        files.append(target)
    for source, name in (
        (REPO / "examples" / "rt" / "munich.json", "munich.json"),
        (REPO / "docs" / "assets" / "munich_routes.png", "munich_routes.png"),
        (SOURCES / "ODbL-1.0.txt", "LICENSE"),
        (SOURCES / "NOTICE", "NOTICE"),
    ):
        shutil.copyfile(source, out / name)
        files.append(out / name)

    doi = PLACEHOLDER_DOI if ZENODO_RECORD is None else f"10.5281/zenodo.{ZENODO_RECORD}"
    record_url = (
        "record URL to be added"
        if ZENODO_RECORD is None
        else f"https://zenodo.org/records/{ZENODO_RECORD}"
    )
    readme = (SOURCES / "README.md").read_text(encoding="utf-8")
    readme = readme.replace("@DOI@", doi).replace("@RECORD_URL@", record_url)
    (out / "README.md").write_bytes(readme.encode("utf-8"))
    files.append(out / "README.md")

    lines = [f"{sha256(path)}  {path.name}\n" for path in sorted(files, key=lambda p: p.name)]
    (out / "SHA256SUMS").write_bytes("".join(lines).encode("utf-8"))
    for path in sorted(out.iterdir()):
        print(f"{path.stat().st_size:>12,}  {path.name}")
    if ZENODO_RECORD is None:
        print(f"Not final: no Zenodo record in linkgym.datasets, the README has {doi}")


if __name__ == "__main__":
    main()
