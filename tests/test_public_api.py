"""Snapshot of the public names of linkgym's modules (docs/api_stability.md).

A failure here means a public name was added, removed or renamed. That is a change of the
public API: update this snapshot together with docs/api_stability.md and the CHANGELOG
(and, for a stable name being removed or changed, deprecate it for one minor release first).
"""

import importlib
import inspect

import pytest

PUBLIC = {
    "linkgym": ["ENV_ID", "ScenarioConfig", "__version__", "evaluate"],
    "linkgym.phy": [
        "MAX_MCS",
        "MCS_CATEGORY",
        "MCS_TABLE_INDEX",
        "MIN_MCS",
        "Transmission",
        "effective_sinr",
        "tb_size_per_mcs",
        "transmit",
    ],
    "linkgym.beams": [
        "Codebook",
        "beam_gain",
        "best_beam",
        "dft_codebook",
        "rsrp_dbm",
        "steering_vector",
    ],
    "linkgym.paths": [
        "CHANNEL_KIND",
        "PATH_TRACE_FORMAT_VERSION",
        "ArrayChannelEpisode",
        "ArrayChannelSource",
        "PathTraceInfo",
        "PathTraceSource",
        "PathWindow",
        "inspect_path_trace",
        "read_path_window",
        "reconstruct_cfr",
        "rotation_matrix",
        "write_path_trace",
    ],
    "linkgym.channels": [
        "CATEGORIES",
        "GAIN_UNIT",
        "PRB_SAMPLINGS",
        "TRACE_FORMAT",
        "TRACE_FORMAT_VERSION",
        "ChannelEpisode",
        "ChannelSource",
        "TraceChannelSource",
        "TraceData",
        "TraceFormatError",
        "TraceInfo",
        "inspect_trace",
        "link_budget_snr_db",
        "read_trace",
        "tx_power_for_median_snr",
        "write_trace",
    ],
    "linkgym.evaluation": [
        "METRICS",
        "cluster_bootstrap",
        "evaluate",
        "evaluate_episodes",
        "paired_bootstrap",
        "trace_episodes",
    ],
    "linkgym.datasets": [
        "DATASETS",
        "Dataset",
        "DatasetError",
        "ZENODO_RECORD",
        "default_cache_dir",
        "fetch",
    ],
}

# Public-looking functions and classes defined in a module that are deliberately not part
# of its public API
INTERNAL = {"linkgym.datasets": {"main"}}  # the `python -m linkgym.datasets` entry point


@pytest.mark.parametrize("name", sorted(PUBLIC))
def test_public_names(name):
    module = importlib.import_module(name)
    assert sorted(module.__all__) == sorted(PUBLIC[name]), (
        f"{name}.__all__ changed; update tests/test_public_api.py, docs/api_stability.md "
        "and the CHANGELOG"
    )
    missing = [n for n in module.__all__ if not hasattr(module, n)]
    assert not missing, f"{name}.__all__ lists missing names {missing}"


@pytest.mark.parametrize("name", sorted(PUBLIC))
def test_no_accidental_public_definitions(name):
    module = importlib.import_module(name)
    defined = {
        n
        for n, obj in vars(module).items()
        if not n.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and obj.__module__ == name
    }
    undeclared = defined - set(module.__all__) - INTERNAL.get(name, set())
    assert not undeclared, (
        f"{name} defines {sorted(undeclared)} without listing them in __all__: make them "
        "private (leading underscore), add them to __all__, or list them in INTERNAL"
    )
