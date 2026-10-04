# API stability

Since 0.3.0.

This page is for projects that depend on linkgym and pin its version. It says which parts
of linkgym are stable, which are experimental, and what each kind of release may change.

## Status of each part

| part | status | notes |
|---|---|---|
| `linkgym` (`ENV_ID`, `ScenarioConfig`, `evaluate`, `__version__`) | stable | |
| The environment `linkgym/LinkAdaptation-v0` | stable | through `gymnasium.make`: actions, observations, reward and info keys as in [environment.md](environment.md) |
| `linkgym.phy` | stable | [phy.md](phy.md) |
| `linkgym.channels` | stable | channel interface, trace format version 1, trace sources and helpers ([channels.md](channels.md)) |
| `linkgym.evaluation` | stable | |
| `linkgym.datasets` | stable | the dataset names and their files never change |
| `linkgym.baselines` | stable | |
| The `linkgym-traces` command | stable | subcommands and options |
| Trace format version 1 | stable | readable by every future linkgym version |
| `linkgym.beams` | experimental | [beams.md](beams.md) |
| The Python API of `linkgym.rt` | experimental | use the `linkgym-traces` command |
| `linkgym.sim`, `linkgym.env`, `linkgym.config` | internal | use `linkgym.phy`, `gymnasium.make` and `linkgym.ScenarioConfig` |

Within a stable module, the public API is the names in its `__all__`. Names that start with
an underscore, and names not in `__all__`, are internal, even if they can be imported. Each
module's docstring states its status.

## What a release may change

1. **Patch releases** (0.3.0 → 0.3.1) only fix bugs. They change no stable API. They do not
   change the outputs of the environment, the simulator or `linkgym.phy`, or any published
   result; golden tests check this.
2. **Minor releases** (0.3 → 0.4) may add APIs. A stable API is removed or changed
   incompatibly only after at least one minor release in which using it emits a
   `DeprecationWarning`, with a "Deprecated" entry in the CHANGELOG.
3. **Experimental APIs** may change in any minor release, with a CHANGELOG entry. They
   become stable after at least one minor release without changes.
4. **Data formats** have their own version numbers, independent of linkgym's. Trace format
   version 1 stays readable; a new format gets a new version number.
5. **Published results** stay reproducible with the versions pinned in their protocol.
   Pinned dependencies, such as Sionna, change only in minor releases, and only after the
   golden tests pass with the new version.
6. **Public names are snapshotted.** `tests/test_public_api.py` records the public names
   of `linkgym`, `linkgym.phy`, `linkgym.beams`, `linkgym.channels`, `linkgym.evaluation`
   and `linkgym.datasets`. Changing them means updating that test, this page and the
   CHANGELOG.

## Pinning linkgym in your project

- Depend on a compatible release, for example `linkgym ~= 0.3.0`, which accepts 0.3.x
  patches but not 0.4.
- Keep a lockfile (`uv lock`, or `pip freeze` into a requirements file), so that Sionna,
  PyTorch and the other dependencies are pinned exactly too.
- Datasets fetched with `linkgym.datasets.fetch` are pinned by name: each name always
  points to the same files, checked by SHA-256.
