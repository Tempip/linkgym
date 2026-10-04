# Contributing

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12.

```bash
git clone https://github.com/Tempip/linkgym.git
cd linkgym
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only torch
uv pip install -e ".[dev,train,docs]"  # docs: runs the notebook test
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pre-commit install
```

With the hook installed, every commit runs ruff (lint and format). The commands below
assume the virtual environment is active.

## Checks

```bash
ruff check .
ruff format --check .
pytest -q -m "not slow"   # what CI runs
pytest -q                 # everything, including slow tests (PPO training script, notebook)
```

The full run, including the slow tests, takes about four minutes on a desktop CPU.

The Sionna RT generator has GPU tests (marker `rt`), skipped without sionna-rt and a CUDA
GPU. Run them in a separate environment with the `rt` extra (see
[docs/channels.md](docs/channels.md#installation)):

```bash
pytest -q -m rt
```

## Commits

Small commits with [Conventional Commits](https://www.conventionalcommits.org/) prefixes, as
in the existing history: `feat:`, `fix:`, `docs:`, `test:`, `chore:`. Add user-visible
changes to `CHANGELOG.md` under `[Unreleased]`.

Results in `docs/results/` are produced by the scripts in `examples/`; if a change affects
them, regenerate them with those scripts rather than editing the files by hand.

Releases follow [RELEASING.md](RELEASING.md). What a release may change in the public
API is set out in [docs/api_stability.md](docs/api_stability.md): a change to the public
names of a module also updates `tests/test_public_api.py`.

## Issues

Bug reports and questions: https://github.com/Tempip/linkgym/issues. For a bug, include the
linkgym, sionna-no-rt, torch and gymnasium versions and a minimal script that reproduces it.
