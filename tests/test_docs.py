"""The README and docs code and the quickstart notebook run as documented."""

import re
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def python_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r"```python\r?\n(.*?)```", text, flags=re.DOTALL)


def readme_python_blocks() -> list[str]:
    return python_blocks(REPO / "README.md")


def test_readme_has_two_python_snippets():
    assert len(readme_python_blocks()) == 2


@pytest.mark.parametrize("index", [0, 1], ids=["evaluate_olla", "train_ppo"])
def test_readme_snippet_runs_verbatim(index):
    if index == 1:
        pytest.importorskip("stable_baselines3")
    code = readme_python_blocks()[index]
    exec(compile(code, f"README.md python block {index + 1}", "exec"), {"__name__": "__readme__"})


def test_bring_your_own_channel_snippet_runs_verbatim(tmp_path, monkeypatch):
    blocks = python_blocks(REPO / "docs" / "channels.md")
    assert len(blocks) == 1
    monkeypatch.chdir(tmp_path)  # the snippet writes my_channel.h5
    exec(compile(blocks[0], "docs/channels.md python block", "exec"), {"__name__": "__docs__"})
    assert (tmp_path / "my_channel.h5").exists()


@pytest.mark.slow
# pyzmq warns about the Windows default event loop; it falls back to a selector thread
@pytest.mark.filterwarnings("ignore:Proactor event loop:RuntimeWarning")
def test_quickstart_notebook_runs():
    # Needs the `docs` extra
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    notebook = nbformat.read(REPO / "examples" / "quickstart.ipynb", as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO / "examples")}},
    )
    start = time.perf_counter()
    client.execute()
    assert time.perf_counter() - start < 120
