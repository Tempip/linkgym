"""The README and docs code and the quickstart notebook run as documented."""

import re
import shutil
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def python_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r"```python\r?\n(.*?)```", text, flags=re.DOTALL)


def readme_python_blocks() -> list[str]:
    return python_blocks(REPO / "README.md")


def test_readme_has_four_python_snippets():
    assert len(readme_python_blocks()) == 4


@pytest.mark.parametrize("index", [0, 1], ids=["evaluate_olla", "train_ppo"])
def test_readme_snippet_runs_verbatim(index):
    if index == 1:
        pytest.importorskip("stable_baselines3")
    code = readme_python_blocks()[index]
    exec(compile(code, f"README.md python block {index + 1}", "exec"), {"__name__": "__readme__"})


def test_readme_dataset_snippet_runs_on_the_sample(monkeypatch):
    # No network in tests: fetch returns the sample trace instead of downloading munich-v1
    sample = REPO / "tests" / "data" / "munich_sample.h5"
    monkeypatch.setattr("linkgym.datasets.fetch", lambda name, **kwargs: sample)
    code = readme_python_blocks()[2]
    namespace = {"__name__": "__readme__"}
    exec(compile(code, "README.md python block 3", "exec"), namespace)
    assert namespace["env"].unwrapped.channel_info["split"] == "train"
    namespace["env"].close()


def test_readme_own_channel_snippet_runs_verbatim(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the snippet writes my_channel.h5
    namespace = {"__name__": "__readme__"}
    exec(compile(readme_python_blocks()[3], "README.md python block 4", "exec"), namespace)
    namespace["env"].reset(seed=0)
    namespace["env"].close()


def test_bring_your_own_channel_snippet_runs_verbatim(tmp_path, monkeypatch):
    blocks = python_blocks(REPO / "docs" / "channels.md")
    assert len(blocks) == 1
    monkeypatch.chdir(tmp_path)  # the snippet writes my_channel.h5
    exec(compile(blocks[0], "docs/channels.md python block", "exec"), {"__name__": "__docs__"})
    assert (tmp_path / "my_channel.h5").exists()


def test_rt_tutorial_runs_verbatim(tmp_path, monkeypatch):
    # The non-GPU steps: the configuration check, then inspect, train and evaluate on the
    # sample trace, all Python blocks in order in one namespace
    pytest.importorskip("stable_baselines3")
    text = (REPO / "docs" / "tutorial_rt.md").read_text(encoding="utf-8")
    configs = re.findall(r"```json\r?\n(.*?)```", text, flags=re.DOTALL)
    assert len(configs) == 1
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tutorial.json").write_text(configs[0], encoding="utf-8")
    (tmp_path / "tests" / "data").mkdir(parents=True)
    shutil.copy(REPO / "tests" / "data" / "munich_sample.h5", tmp_path / "tests" / "data")
    namespace = {"__name__": "__docs__"}
    blocks = python_blocks(REPO / "docs" / "tutorial_rt.md")
    assert len(blocks) == 6
    for i, block in enumerate(blocks):
        exec(compile(block, f"docs/tutorial_rt.md python block {i + 1}", "exec"), namespace)
    assert [r.split for r in namespace["config"].routes] == ["train", "val", "test"]
    assert namespace["diff"]["num_clusters"] == 1


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
