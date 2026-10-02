"""Channel sources: trace format validation, selection, SNR modes, simulator protocol."""

import pickle
import warnings
from functools import partial

import gymnasium
import h5py
import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env as gymnasium_check_env
from gymnasium.utils.env_checker import data_equivalence

from linkgym import ENV_ID, ScenarioConfig
from linkgym.channels import (
    ChannelEpisode,
    TraceChannelSource,
    TraceFormatError,
    inspect_trace,
    link_budget_snr_db,
    read_trace,
    write_trace,
)
from linkgym.env import LinkAdaptationEnv
from linkgym.sim import MIN_MCS, LinkSimulator

SPLITS = ["train", "train", "val", "val", "test", "test"]
GROUPS = [0, 0, 1, 1, 2, 2]
NUM_SLOTS = 120
NUM_PRBS = 52
BANDWIDTH = NUM_PRBS * 12 * 30e3


def synthetic_gain(num_slots=NUM_SLOTS, seed=0):
    """Rayleigh-like gains with a path loss of 110-120 dB per trajectory."""
    rng = np.random.default_rng(seed)
    path_loss_db = rng.uniform(110.0, 120.0, size=(len(SPLITS), 1, 1))
    fading = rng.exponential(size=(len(SPLITS), num_slots, NUM_PRBS))
    return (fading * 10.0 ** (-path_loss_db / 10.0)).astype(np.float32)


def write_synthetic(path, num_slots=NUM_SLOTS, **kwargs):
    gain = synthetic_gain(num_slots)
    attrs = {
        "carrier_frequency_hz": 3.5e9,
        "subcarrier_spacing_hz": 30e3,
        "slot_duration_s": 0.5e-3,
        "group": np.array(GROUPS),
    } | kwargs
    write_trace(path, gain, SPLITS, **attrs)
    return gain


@pytest.fixture
def trace(tmp_path):
    path = tmp_path / "trace.h5"
    return path, write_synthetic(path)


def make_trace_env(path, **kwargs):
    kwargs = {"channel": "trace", "trace_path": path, "episode_length": 40} | kwargs
    return gymnasium.make(ENV_ID, **kwargs)


def make_source(path, num_slots=40, **kwargs):
    scenario = {
        "num_prbs": NUM_PRBS,
        "subcarrier_spacing": 30e3,
        "carrier_frequency": 3.5e9,
        "slot_duration": 0.5e-3,
    }
    return TraceChannelSource(path, num_slots=num_slots, **(scenario | kwargs))


def episode_sinr(env, seed):
    """Per-PRB SINR of every slot of an episode, from the privileged info."""
    _, info = env.reset(seed=seed)
    sinr = [info["privileged"]["sinr_prb"]]
    for _ in range(env.unwrapped.config.episode_length - 1):
        *_, info = env.step(0)
        sinr.append(info["privileged"]["sinr_prb"])
    return np.stack(sinr), info


def selected(env, key, seeds=range(20)):
    """Values of channel_info[key] over episodes reset with ``seeds``."""
    values = set()
    for seed in seeds:
        env.reset(seed=seed)
        values.add(env.unwrapped.channel_info[key])
    return values


# Trace format


def test_write_read_round_trip(tmp_path):
    path = tmp_path / "trace.h5"
    gain = synthetic_gain()
    rx_position = np.zeros((len(SPLITS), NUM_SLOTS, 3))
    write_trace(
        path,
        gain,
        SPLITS,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30e3,
        slot_duration_s=0.5e-3,
        prb_sampling="mean12",
        group=np.array(GROUPS),
        rx_position=rx_position,
        num_paths=np.full((len(SPLITS), 12), 7),
        los=np.ones((len(SPLITS), 12)),
        attrs={"scene": "synthetic"},
    )
    data = read_trace(path)
    np.testing.assert_array_equal(data.gain, gain)
    with h5py.File(path, "r") as f:
        assert f["num_paths"].dtype == np.int16 and f["los"].dtype == np.int8
        assert f["gain"].chunks is None and f["gain"].compression is None  # contiguous
    assert data.split.tolist() == SPLITS
    assert data.group.tolist() == GROUPS
    assert (data.num_trajectories, data.num_slots, data.num_prbs) == gain.shape
    assert data.attrs["format"] == "linkgym-trace"
    assert data.attrs["format_version"] == 1
    assert data.attrs["gain_unit"] == "linear_power_gain"
    assert data.attrs["prb_sampling"] == "mean12"
    assert data.attrs["slot_duration_s"] == 0.5e-3
    assert data.attrs["scene"] == "synthetic"


def set_attr(key, value):
    def apply(f):
        f.attrs[key] = value

    return apply


def del_attr(key):
    def apply(f):
        del f.attrs[key]

    return apply


def set_dataset(name, transform):
    def apply(f):
        data = transform(f[name][()])
        del f[name]
        f.create_dataset(name, data=data)

    return apply


def del_dataset(name):
    def apply(f):
        del f[name]

    return apply


def add_dataset(name, data):
    def apply(f):
        f.create_dataset(name, data=data)

    return apply


def both(*datasets):
    def apply(f):
        for name, shape in datasets:
            f.create_dataset(name, data=np.zeros(shape, np.int8))

    return apply


def set_split(labels):
    def apply(f):
        del f["split"]
        f.create_dataset("split", data=np.array(labels, dtype=object), dtype=h5py.string_dtype())

    return apply


def with_value(value):
    def transform(gain):
        gain = gain.copy()
        gain[1, 2, 3] = value
        return gain

    return transform


@pytest.mark.parametrize(
    "mutate, match",
    [
        (set_attr("format", "other"), "attribute 'format' must be 'linkgym-trace'"),
        (set_attr("format_version", 2), "unsupported format_version 2"),
        (del_attr("slot_duration_s"), r"missing attributes \['slot_duration_s'\]"),
        (set_attr("gain_unit", "dB"), "attribute 'gain_unit' must be 'linear_power_gain'"),
        (set_attr("prb_sampling", "edge"), "attribute 'prb_sampling' must be one of"),
        (set_attr("carrier_frequency_hz", -3.5e9), "'carrier_frequency_hz' must be a positive"),
        (set_attr("subcarrier_spacing_hz", "30 kHz"), "'subcarrier_spacing_hz' must be a positive"),
        (set_attr("slot_duration_s", np.inf), "'slot_duration_s' must be a positive"),
        (del_dataset("gain"), "missing dataset 'gain'"),
        (del_dataset("split"), "missing dataset 'split'"),
        (set_dataset("gain", lambda g: g.astype(np.float64)), "'gain' must be float32"),
        (set_dataset("gain", lambda g: g[:, :, 0]), "with 3 dimensions"),
        (set_dataset("gain", lambda g: g[:, :0]), "'gain' is empty"),
        (set_attr("num_prb", 51), "'gain' has 52 PRBs, attribute 'num_prb' says 51"),
        (set_dataset("gain", with_value(np.nan)), "NaN or infinite"),
        (set_dataset("gain", with_value(np.inf)), "NaN or infinite"),
        (set_dataset("gain", with_value(-1e-12)), "negative values"),
        (set_split(SPLITS[:-1]), "'split' must hold 6 strings"),
        (set_dataset("split", lambda s: np.arange(6)), "'split' must hold 6 strings"),
        (set_split([*SPLITS[:-1], ""]), "empty labels"),
        (add_dataset("group", np.zeros(5, np.int32)), r"'group' must have shape \(6,\)"),
        (add_dataset("rx_position", np.zeros((6, NUM_SLOTS, 2))), "'rx_position' must have"),
        (add_dataset("num_paths", np.zeros(6, np.int16)), "'num_paths' must have"),
        (add_dataset("los", np.zeros((6, 2, 2), np.int8)), "'los' must have"),
        (both(("num_paths", (6, 3)), ("los", (6, 4))), "must have the same shape"),
    ],
)
def test_read_trace_rejects_invalid_files(tmp_path, mutate, match):
    path = tmp_path / "trace.h5"
    write_trace(
        path,
        synthetic_gain(),
        SPLITS,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30e3,
        slot_duration_s=0.5e-3,
    )
    with h5py.File(path, "r+") as f:
        mutate(f)
    with pytest.raises(TraceFormatError, match=match) as error:
        read_trace(path)
    assert str(error.value).startswith(f"{path}: ")


def test_read_trace_rejects_non_hdf5(tmp_path):
    path = tmp_path / "trace.h5"
    path.write_text("not a trace")
    with pytest.raises(TraceFormatError, match="cannot open as HDF5"):
        read_trace(path)


def test_write_trace_validates(tmp_path):
    gain = synthetic_gain()
    gain[0, 0, 0] = np.nan
    with pytest.raises(TraceFormatError, match="NaN"):
        write_trace(
            tmp_path / "trace.h5",
            gain,
            SPLITS,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30e3,
            slot_duration_s=0.5e-3,
        )


@pytest.mark.parametrize(
    "trace_kwargs, env_kwargs, match",
    [
        ({}, {"num_prbs": 51}, "trace has 52 PRBs, the scenario has 51"),
        ({}, {"carrier_frequency": 28e9}, "trace carrier_frequency_hz is 3500000000.0"),
        (
            {"subcarrier_spacing_hz": 15e3, "slot_duration_s": 1e-3},
            {},
            "trace subcarrier_spacing_hz is 15000.0, the scenario has 30000.0",
        ),
        ({"slot_duration_s": 1e-3}, {}, "trace slot_duration_s is 0.001, the scenario has 0.0005"),
        ({}, {"episode_length": 121}, "trajectories have 120 slots, episodes need 121"),
        (
            {},
            {"trace_splits": ["holdout"]},
            r"split\(s\) \['holdout'\] not in the trace; available: \['test', 'train', 'val'\]",
        ),
    ],
)
def test_trace_must_match_scenario(tmp_path, trace_kwargs, env_kwargs, match):
    path = tmp_path / "trace.h5"
    write_synthetic(path, **trace_kwargs)
    with pytest.raises(TraceFormatError, match=match) as error:
        make_trace_env(path, **env_kwargs)
    assert str(path) in str(error.value)


# Scenario fields


def test_default_scenario_is_tdl():
    config = ScenarioConfig()
    assert (config.channel, config.trace_path, config.trace_splits) == ("tdl", None, ("train",))
    assert (config.snr_mode, config.tx_power_dbm, config.noise_figure_db) == (
        "normalized",
        None,
        7.0,
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"channel": "rt"}, "channel must be one of"),
        ({"channel": "trace"}, "channel='trace' requires trace_path"),
        ({"snr_mode": "absolute"}, "snr_mode must be one of"),
        ({"snr_mode": "link_budget", "tx_power_dbm": 30.0}, "requires channel='trace'"),
        (
            {"channel": "trace", "trace_path": "t.h5", "snr_mode": "link_budget"},
            "requires tx_power_dbm",
        ),
        (
            {
                "channel": "trace",
                "trace_path": "t.h5",
                "snr_mode": "link_budget",
                "tx_power_dbm": 30.0,
                "snr_db": 10.0,
            },
            "snr_db is not used",
        ),
        ({"trace_splits": []}, "trace_splits must be non-empty"),
        ({"trace_splits": ["train", ""]}, "trace_splits must be non-empty"),
    ],
)
def test_scenario_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ScenarioConfig(**kwargs)


def test_scenario_normalizes_trace_fields(tmp_path):
    config = ScenarioConfig(channel="trace", trace_path=tmp_path / "t.h5", trace_splits="test")
    assert config.trace_path == str(tmp_path / "t.h5")
    assert config.trace_splits == ("test",)
    assert ScenarioConfig(trace_splits=["val", "test"]).trace_splits == ("val", "test")


# Selection


def test_selection_is_deterministic_and_slices_the_trajectory(trace):
    path, gain = trace
    source = make_source(path)
    first, second = source.generate(40, 8, seed=5), source.generate(40, 8, seed=5)
    assert torch.equal(first.gain, second.gain)
    assert first.info == second.info
    assert source.generate(40, 8, seed=6).info != first.info

    assert first.gain.dtype == torch.float32
    assert first.gain.shape == (8, 40, NUM_PRBS)
    assert first.reference_snr_db is None
    for link, (t, o) in enumerate(zip(first.info["trajectory"], first.info["offset"], strict=True)):
        assert 0 <= o <= NUM_SLOTS - 40
        expected = gain[t, o : o + 40] / gain[t].astype(np.float64).mean()
        np.testing.assert_allclose(first.gain[link].numpy(), expected, rtol=1e-6)


def test_env_selection_derives_from_the_env_seed(trace):
    path, _ = trace
    env_a, env_b = make_trace_env(path), make_trace_env(path)
    env_a.action_space.seed(1)
    actions = [env_a.action_space.sample() for _ in range(40)]

    first = [env_a.reset(seed=3)]
    first += [env_a.step(a) for a in actions]
    info_a = env_a.unwrapped.channel_info
    second = [env_b.reset(seed=3)]
    second += [env_b.step(a) for a in actions]
    for a, b in zip(first, second, strict=True):
        assert data_equivalence(a, b, exact=True)
    assert env_b.unwrapped.channel_info == info_a

    env_b.reset()  # continues the random stream of seed 3
    assert env_b.unwrapped.channel_info != info_a
    env_b.reset(seed=3)
    assert env_b.unwrapped.channel_info == info_a


def test_split_selection(trace):
    path, _ = trace
    episode = make_source(path, splits=("val",)).generate(40, 200, seed=0)
    assert set(episode.info["split"]) == {"val"}
    assert set(episode.info["trajectory"]) == {2, 3}
    assert set(episode.info["group"]) == {1}

    episode = make_source(path, splits=("train", "test")).generate(40, 200, seed=0)
    assert set(episode.info["trajectory"]) == {0, 1, 4, 5}
    assert set(episode.info["split"]) == {"train", "test"}

    assert selected(make_trace_env(path), "split") == {"train"}  # default
    assert selected(make_trace_env(path, trace_splits="test"), "trajectory") == {4, 5}


# SNR modes


@pytest.mark.parametrize("snr_db", [12.0, None])
def test_normalized_mean_snr_is_the_scenario_snr(tmp_path, snr_db):
    """With episodes as long as the trajectories, the window is the whole trajectory."""
    path = tmp_path / "trace.h5"
    write_synthetic(path, num_slots=40)
    env = make_trace_env(path, snr_db=snr_db)
    for seed in range(5):
        sinr, info = episode_sinr(env, seed)
        channel = env.unwrapped.channel_info
        assert channel["offset"] == 0
        if snr_db is not None:
            assert info["snr_db"] == snr_db
        assert channel["realized_snr_db"] == pytest.approx(info["snr_db"], abs=1e-4)
        mean_db = 10.0 * np.log10(sinr.astype(np.float64).mean())
        assert mean_db == pytest.approx(info["snr_db"], abs=1e-4)


def test_normalized_window_is_scaled_by_the_trajectory_mean(trace):
    path, gain = trace
    env = make_trace_env(path, snr_db=12.0)
    sinr, _ = episode_sinr(env, 7)
    channel = env.unwrapped.channel_info
    t, o = channel["trajectory"], channel["offset"]
    expected = 10.0**1.2 * gain[t, o : o + 40] / gain[t].astype(np.float64).mean()
    np.testing.assert_allclose(sinr, expected, rtol=1e-5)
    mean_db = 10.0 * np.log10(sinr.astype(np.float64).mean())
    assert channel["realized_snr_db"] == pytest.approx(mean_db, abs=1e-4)


def test_link_budget_snr_db_arithmetic():
    # k T0 at 290 K: -173.975 dBm/Hz
    assert link_budget_snr_db(0.0, 0.0, 1.0) == pytest.approx(173.9752, abs=1e-4)
    # 30 dBm over 52 PRBs at 30 kHz (18.72 MHz), noise figure 7 dB: noise -94.252 dBm
    assert link_budget_snr_db(30.0, 7.0, 18.72e6) == pytest.approx(124.2521, abs=1e-4)
    assert link_budget_snr_db(40.0, 7.0, 18.72e6) == pytest.approx(134.2521, abs=1e-4)
    assert link_budget_snr_db(30.0, 7.0, 2 * 18.72e6) == pytest.approx(121.2418, abs=1e-4)
    assert link_budget_snr_db(30.0, 7.0, 1e6, temperature_k=580.0) == pytest.approx(
        30.0 + 173.9752 - 60.0 - 7.0 - 3.0103, abs=1e-4
    )


def test_link_budget_env_sinr_is_reference_snr_times_gain(trace):
    path, gain = trace
    env = make_trace_env(path, snr_mode="link_budget", tx_power_dbm=30.0)
    sinr, info = episode_sinr(env, 0)
    channel = env.unwrapped.channel_info
    t, o = channel["trajectory"], channel["offset"]
    reference = link_budget_snr_db(30.0, 7.0, BANDWIDTH)
    np.testing.assert_allclose(sinr, 10.0 ** (reference / 10.0) * gain[t, o : o + 40], rtol=1e-6)
    # info["snr_db"] is the realized mean SNR of the episode, not a draw
    assert info["snr_db"] == channel["realized_snr_db"]
    mean_db = 10.0 * np.log10(sinr.astype(np.float64).mean())
    assert info["snr_db"] == pytest.approx(mean_db, abs=1e-4)

    env = make_trace_env(path, snr_mode="link_budget", tx_power_dbm=30.0, noise_figure_db=10.0)
    _, info_nf10 = episode_sinr(env, 0)
    assert env.unwrapped.channel_info["trajectory"] == t
    assert info_nf10["snr_db"] == pytest.approx(info["snr_db"] - 3.0, abs=1e-4)


# Environment


def test_common_random_numbers_across_policies(trace):
    """Same seed, different actions: same trajectory, window, SNR and ACK uniforms."""
    path, _ = trace
    env_a, env_b = make_trace_env(path), make_trace_env(path)
    env_b.action_space.seed(0)
    _, info_a = env_a.reset(seed=11)
    _, info_b = env_b.reset(seed=11)
    assert env_a.unwrapped.channel_info == env_b.unwrapped.channel_info
    assert info_a["snr_db"] == info_b["snr_db"]
    for _ in range(40):
        np.testing.assert_array_equal(
            info_a["privileged"]["sinr_prb"], info_b["privileged"]["sinr_prb"]
        )
        _, _, _, _, info_a = env_a.step(5 - MIN_MCS)
        _, _, _, _, info_b = env_b.step(env_b.action_space.sample())
        u_a = env_a.unwrapped.last_result.ack_uniform
        u_b = env_b.unwrapped.last_result.ack_uniform
        assert u_a.tolist() == u_b.tolist()


TRACE_CONFIGS = [{}, {"snr_mode": "link_budget", "tx_power_dbm": 30.0}]


@pytest.mark.parametrize("kwargs", TRACE_CONFIGS, ids=["normalized", "link_budget"])
def test_gymnasium_check_env_with_trace(trace, kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gymnasium_check_env(make_trace_env(trace[0], **kwargs).unwrapped)


@pytest.mark.parametrize("kwargs", TRACE_CONFIGS, ids=["normalized", "link_budget"])
def test_sb3_check_env_with_trace(trace, kwargs):
    env_checker = pytest.importorskip("stable_baselines3.common.env_checker")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env_checker.check_env(make_trace_env(trace[0], **kwargs).unwrapped, warn=True)


def test_tdl_channel_info_has_the_realized_snr():
    env = gymnasium.make(ENV_ID, episode_length=40)
    sinr, _ = episode_sinr(env, 0)
    channel = env.unwrapped.channel_info
    assert list(channel) == ["realized_snr_db"]
    mean_db = 10.0 * np.log10(sinr.astype(np.float64).mean())
    assert channel["realized_snr_db"] == pytest.approx(mean_db, abs=1e-4)


# Simulator protocol


class ConstantSource:
    """Unit gain everywhere; a minimal ChannelSource."""

    num_prbs = NUM_PRBS

    def __init__(self, reference_snr_db=None, shape=None, dtype=torch.float32):
        self.reference_snr_db = reference_snr_db
        self.shape = shape
        self.dtype = dtype

    def generate(self, num_slots, batch_size, seed):
        shape = self.shape or (batch_size, num_slots, self.num_prbs)
        return ChannelEpisode(
            gain=torch.ones(shape, dtype=self.dtype),
            reference_snr_db=self.reference_snr_db,
            info={"seed": [seed] * batch_size},
        )


@pytest.mark.parametrize(
    "reference, expected_db",
    [(None, [15.0, 15.0]), (3.0, [3.0, 3.0]), (np.array([0.0, 10.0]), [0.0, 10.0])],
    ids=["scenario", "float", "per_link"],
)
def test_simulator_applies_the_reference_snr(reference, expected_db):
    sim = LinkSimulator(5, batch_size=2, snr_db=15.0, channel_source=ConstantSource(reference))
    for link, snr_db in enumerate(expected_db):
        torch.testing.assert_close(sim.sinr[link], torch.full((5, NUM_PRBS), 10.0 ** (snr_db / 10)))
    assert sim.channel_info["realized_snr_db"] == pytest.approx(expected_db, abs=1e-5)
    assert len(sim.channel_info["seed"]) == 2


def test_simulator_checks_the_source():
    with pytest.raises(ValueError, match="channel_source has 52 PRBs, the simulator 51"):
        LinkSimulator(5, num_prbs=51, channel_source=ConstantSource())
    with pytest.raises(ValueError, match=r"with shape \(1, 4, 52\), expected torch.float32"):
        LinkSimulator(5, channel_source=ConstantSource(shape=(1, 4, NUM_PRBS)))
    with pytest.raises(ValueError, match="channel gain is torch.float64"):
        LinkSimulator(5, channel_source=ConstantSource(dtype=torch.float64))


# Lazy loading


def record_gain_reads(monkeypatch):
    """Record the keys of every read of the 'gain' dataset."""
    reads = []
    original = h5py.Dataset.__getitem__

    def recording(self, key):
        if self.name == "/gain":
            reads.append(key)
        return original(self, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recording)
    return reads


def test_validation_streams_one_trajectory_at_a_time(trace, monkeypatch):
    path, gain = trace
    reads = record_gain_reads(monkeypatch)
    info = inspect_trace(path)
    assert reads == list(range(len(SPLITS)))
    np.testing.assert_allclose(info.mean_gain, gain.astype(np.float64).mean(axis=(1, 2)))
    assert info.shape == gain.shape

    source = make_source(path)
    held = [v for v in (*vars(source).values(), *vars(source.info).values())]
    assert all(v.size < gain.size / 10 for v in held if isinstance(v, np.ndarray))


def test_generate_reads_only_the_episode_windows(trace, monkeypatch):
    path, gain = trace
    source = make_source(path)
    reads = record_gain_reads(monkeypatch)
    episode = source.generate(40, 3, seed=1)
    windows = zip(episode.info["trajectory"], episode.info["offset"], strict=True)
    assert reads == [(t, slice(o, o + 40)) for t, o in windows]
    source.close()


def test_source_pickles_and_opens_one_handle_per_process(trace):
    path, _ = trace
    source = make_source(path)
    first = source.generate(40, 4, seed=2)  # opens the handle
    copy = pickle.loads(pickle.dumps(source))
    assert copy._file is None
    assert torch.equal(copy.generate(40, 4, seed=2).gain, first.gain)

    handle = source._file
    source._file_pid = -1  # as seen from a forked child
    assert torch.equal(source.generate(40, 4, seed=2).gain, first.gain)
    assert source._file is not handle
    for s in (source, copy):
        s.close()
    handle.close()


def test_env_close_releases_the_trace_file(tmp_path):
    path = tmp_path / "trace.h5"
    write_synthetic(path)
    env = make_trace_env(path)
    env.reset(seed=0)
    env.step(0)
    env.close()
    path.unlink()  # fails on Windows while a handle is open


def trace_env_fns(path, n):
    return [
        partial(LinkAdaptationEnv, channel="trace", trace_path=str(path), episode_length=40)
    ] * n


def test_async_vector_env_matches_sync(trace):
    path, _ = trace
    sync = gymnasium.vector.SyncVectorEnv(trace_env_fns(path, 3))
    spawned = gymnasium.vector.AsyncVectorEnv(trace_env_fns(path, 3), context="spawn")
    try:
        obs_sync, _ = sync.reset(seed=[1, 2, 3])
        obs_spawned, _ = spawned.reset(seed=[1, 2, 3])
        np.testing.assert_array_equal(obs_sync, obs_spawned)
        for a in range(5):
            actions = np.array([a, a + 5, a + 10])
            out_sync, out_spawned = sync.step(actions), spawned.step(actions)
            for x, y in zip(out_sync[:4], out_spawned[:4], strict=True):
                np.testing.assert_array_equal(x, y)
    finally:
        sync.close()
        spawned.close()


def test_sb3_subproc_vec_env_matches_dummy(trace):
    vec_env = pytest.importorskip("stable_baselines3.common.vec_env")
    path, _ = trace
    dummy = vec_env.DummyVecEnv(trace_env_fns(path, 2))
    subproc = vec_env.SubprocVecEnv(trace_env_fns(path, 2), start_method="spawn")
    try:
        for env in (dummy, subproc):
            env.seed(5)
        np.testing.assert_array_equal(dummy.reset(), subproc.reset())
        for a in range(5):
            actions = np.array([a, a + 7])
            out_dummy, out_subproc = dummy.step(actions), subproc.step(actions)
            # DummyVecEnv keeps rewards in a float32 buffer, SubprocVecEnv returns float64
            for x, y in zip(out_dummy[:3], out_subproc[:3], strict=True):
                np.testing.assert_array_equal(np.float32(x), np.float32(y))
    finally:
        dummy.close()
        subproc.close()
