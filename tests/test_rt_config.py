"""Trace generator configuration, route geometry and CLI; none of these need Sionna RT."""

import copy
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from linkgym import channels
from linkgym.rt import config as rt_config
from linkgym.rt.config import (
    ConfigError,
    Route,
    category_from_los_share,
    load_config,
    parse_config,
    split_overlap,
)

REPO = Path(__file__).resolve().parents[1]
MUNICH = REPO / "examples" / "rt" / "munich.json"

MINIMAL = {
    "scene": "munich",
    "transmitter": {"position": [0.0, 0.0, 20.0]},
    "num_slots": 1000,
    "solver": {},
    "routes": [
        {"name": "a", "split": "train", "waypoints": [[0, 0], [20, 0]]},
        {"name": "b", "split": "test", "waypoints": [[0, 50], [0, 80], [30, 80]]},
    ],
}


def with_changes(path, value):
    """MINIMAL with the entry at ``path`` (keys and list indices) set or removed (None)."""
    data = copy.deepcopy(MINIMAL)
    target = data
    for key in path[:-1]:
        target = target[key]
    if value is None:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return data


def test_munich_config():
    config = load_config(MUNICH)
    assert config.scene == "munich"
    assert config.tx_position == (116.5, 80.5, 23.4)
    assert (config.speed_mps, config.num_slots, config.anchor_spacing_slots) == (15.0, 4000, 10)
    solver = config.solver
    assert solver.max_depth == 10 and solver.diffraction and solver.diffraction_lit_region
    assert not solver.refraction  # thin slabs are not valid for solid OSM blocks
    assert (solver.samples_per_src, solver.max_num_paths_per_src) == (4_000_000, 10_000_000)
    assert config.min_mean_gain_db == -150.0
    assert config.attribution_text == rt_config.OSM_ATTRIBUTION
    splits = [r.split for r in config.routes]
    assert len(config.routes) == 15
    assert (splits.count("train"), splits.count("val"), splits.count("test")) == (8, 3, 4)
    assert len({r.group for r in config.routes}) == 15
    total = sum(r.length for r in config.routes)
    assert 3250 < total < 3350
    # Split by street: no route within 10 m of a route of another split
    assert max(split_overlap(config).values()) == 0.0


def test_defaults_and_derived_quantities():
    config = parse_config(MINIMAL)
    assert config.slot_duration_s == 0.5e-3
    assert config.slot_length_m == pytest.approx(7.5e-3)
    assert config.anchor_spacing_m == pytest.approx(0.075)
    assert config.solver == rt_config.SolverSettings()
    assert config.tx_antenna == config.rx_antenna == rt_config.Antenna("iso", "V")
    # 20 m at 7.5 mm per slot: 2667 slot positions, two trajectories of 1000 slots
    a, b = config.routes
    assert config.num_trajectories(a) == 2
    assert config.slot_distances(a).shape == (2, 1000)
    assert config.num_trajectories(b) == 8  # 60 m: 8001 slot positions
    assert parse_config(with_changes(["subcarrier_spacing_hz"], 15e3)).slot_duration_s == 1e-3


def test_route_points_follow_the_polyline():
    route = Route("r", "train", 0, ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0)))
    assert route.length == 15.0
    xy, direction = route.points(np.array([0.0, 4.0, 10.0, 12.5, 15.0]))
    np.testing.assert_allclose(xy, [[0, 0], [4, 0], [10, 0], [10, 2.5], [10, 5]])
    # At a waypoint, the direction is that of the next segment
    np.testing.assert_allclose(direction, [[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])


def test_trajectory_count_keeps_whole_trajectories():
    config = parse_config(MINIMAL)
    exact = Route("exact", "train", 0, ((0.0, 0.0), (999 * 7.5e-3, 0.0)))  # 1000 slots
    assert config.num_trajectories(exact) == 1
    short = Route("short", "train", 0, ((0.0, 0.0), (998 * 7.5e-3, 0.0)))
    assert config.num_trajectories(short) == 0


@pytest.mark.parametrize(
    "path, value, match",
    [
        (["scene"], None, "missing 'scene'"),
        (["unknown"], 1, r"unknown key\(s\) \['unknown'\]"),
        (["transmitter", "position"], [0, 0], "'position' must be a list of three numbers"),
        (["transmitter", "look_at"], [1, 1, 1], "either 'orientation' or 'look_at'"),
        (["transmitter", "antenna"], {"pattern": "horn"}, "'pattern' must be one of"),
        (["receiver"], {"antenna": {"polarization": "VH"}}, "'polarization' must be one of"),
        (["subcarrier_spacing_hz"], 20e3, "5G NR numerologies"),
        (["prb_sampling"], "edge", "'prb_sampling' must be one of"),
        (["speed_mps"], 0, "'speed_mps' must be > 0"),
        (["num_slots"], 1005, "must be a multiple of anchor_spacing_slots"),
        (["solver", "max_depth"], -1, "'max_depth' must be an integer >= 0"),
        (["solver", "diffraction"], "yes", "'diffraction' must be true or false"),
        (["solver", "deterministic"], False, r"unknown key\(s\) \['deterministic'\]"),
        (["routes"], [], "non-empty list"),
        (["routes", 0, "waypoints"], [[0, 0]], "at least two"),
        (["routes", 0, "waypoints"], [[0, 0], [0, 0], [5, 0]], "consecutive waypoints"),
        (["routes", 0, "waypoints"], [[0, 0], [5, 0]], "shorter than one trajectory"),
        (["routes", 0, "category"], "indoor", "'category' must be one of"),
        (["routes", 1, "name"], "a", r"duplicate names \['a'\]"),
    ],
)
def test_config_validation(path, value, match):
    if path == ["transmitter", "look_at"]:
        data = with_changes(["transmitter", "orientation"], [0, 0, 0])
        data["transmitter"]["look_at"] = value
    else:
        data = with_changes(path, value)
    with pytest.raises(ConfigError, match=match):
        parse_config(data)


def test_load_config_names_the_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(with_changes(["speed_mps"], -1)))
    with pytest.raises(ConfigError, match=re.escape(str(path))):
        load_config(path)
    path.write_text("{not json")
    with pytest.raises(ConfigError, match="cannot read the configuration"):
        load_config(path)


def test_to_dict_round_trip():
    config = load_config(MUNICH)
    again = parse_config(json.loads(json.dumps(config.to_dict())), base_dir=config.base_dir)
    assert again == config


def test_select_routes():
    config = parse_config(MINIMAL)
    assert [r.name for r in config.select(["b"]).routes] == ["b"]
    assert config.select(None) is config
    with pytest.raises(ConfigError, match="unknown route"):
        config.select(["c"])
    assert [r.name for r in config.select_splits(["test"]).routes] == ["b"]
    with pytest.raises(ConfigError, match="unknown split"):
        config.select_splits(["val"])


def test_solver_overrides():
    config = parse_config(MINIMAL)
    changed = config.with_solver({"samples_per_src": 8_000_000, "refraction": False})
    assert changed.solver.samples_per_src == 8_000_000 and not changed.solver.refraction
    assert changed.solver.max_depth == config.solver.max_depth
    assert changed.to_dict()["solver"]["samples_per_src"] == 8_000_000
    with pytest.raises(ConfigError, match="unknown key"):
        config.with_solver({"deterministic": False})
    with pytest.raises(ConfigError, match="must be true or false"):
        config.with_solver({"refraction": 0})


def test_min_mean_gain_db():
    assert parse_config(MINIMAL).min_mean_gain_db is None
    config = parse_config(with_changes(["min_mean_gain_db"], -150))
    assert config.min_mean_gain_db == -150.0
    with pytest.raises(ConfigError, match="'min_mean_gain_db' must be a number"):
        parse_config(with_changes(["min_mean_gain_db"], "low"))


def test_cli_solver_override_parsing():
    from linkgym.rt.cli import _solver_overrides

    assert _solver_overrides(["samples_per_src=8000000", "refraction=false"]) == {
        "samples_per_src": 8000000,
        "refraction": False,
    }
    with pytest.raises(ConfigError, match="KEY=VALUE"):
        _solver_overrides(["samples_per_src"])


def test_store_paths_needs_prb_centres(tmp_path):
    from linkgym.rt.generate import generate

    config = parse_config(with_changes(["prb_sampling"], "mean12"))
    with pytest.raises(ConfigError, match="prb_sampling 'center'"):
        generate(config, tmp_path / "t.h5", store_paths=True)
    assert not (tmp_path / "t.h5").exists()


def test_split_overlap():
    data = copy.deepcopy(MINIMAL)
    data["routes"].append({"name": "c", "split": "val", "waypoints": [[5, 5], [25, 5]]})
    shares = split_overlap(parse_config(data))
    # a and c run 5 m apart over their whole length; b is far from both
    assert shares["a"] == 1.0 and shares["c"] == 1.0 and shares["b"] == 0.0


@pytest.mark.parametrize(
    "share, category",
    [(0.0, "nlos"), (0.05, "nlos"), (0.06, "transition"), (0.74, "transition"), (0.75, "los")],
)
def test_category_from_los_share(share, category):
    assert category_from_los_share(share) == category


def test_constants_match_the_trace_format():
    assert rt_config.PRB_SAMPLINGS == channels.PRB_SAMPLINGS
    assert rt_config.CATEGORIES == channels.CATEGORIES


def rt_modules_after(code):
    """Sionna RT modules loaded after running ``code`` in a fresh interpreter."""
    report = "print([m for m in ('sionna.rt', 'mitsuba', 'drjit') if m in sys.modules])"
    code = f"import sys\n{code}\n{report}"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_cli_and_config_do_not_import_sionna_rt():
    code = (
        "from linkgym.rt.cli import main\ntry:\n    main(['--help'])\nexcept SystemExit:\n    pass"
    )
    assert rt_modules_after(code) == "[]"


def test_env_does_not_import_sionna_rt():
    try:
        importlib.metadata.version("sionna-rt")
        # sionna-rt replaces sionna/__init__.py with one that imports Sionna RT on any
        # `import sionna` (docs/channels.md); linkgym cannot prevent that there
        pytest.skip("sionna-rt is installed in this environment")
    except importlib.metadata.PackageNotFoundError:
        pass
    assert rt_modules_after("import gymnasium, linkgym\ngymnasium.make(linkgym.ENV_ID)") == "[]"


def test_only_the_generator_imports_sionna_rt():
    pattern = re.compile(r"^\s*(import|from)\s+(sionna\.rt|mitsuba|drjit)\b", re.MULTILINE)
    offenders = [
        p.relative_to(REPO).as_posix()
        for p in (REPO / "src" / "linkgym").rglob("*.py")
        if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == ["src/linkgym/rt/generate.py"]


def test_generate_without_sionna_rt_explains_the_extra(tmp_path):
    try:
        import sionna.rt  # noqa: F401

        pytest.skip("Sionna RT is installed")
    except ImportError:
        pass
    code = (
        f"from linkgym.rt.cli import main; import sys; sys.exit(main(['check', {str(MUNICH)!r}]))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 1
    assert 'pip install "linkgym[rt]"' in out.stderr and "own environment" in out.stderr
