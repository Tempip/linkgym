"""Channel sources: the per-slot, per-PRB channel power gain of an episode.

:class:`ChannelSource` is the extension point for bringing your own channels. Built-in
sources: :class:`linkgym.sim.TDLChannelGain` (3GPP TDL fading, the default) and
:class:`TraceChannelSource` (gains read from a trace file, see :func:`write_trace` and
docs/channels.md).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from linkgym.config import SNR_MODES

TRACE_FORMAT = "linkgym-trace"
TRACE_FORMAT_VERSION = 1
GAIN_UNIT = "linear_power_gain"
PRB_SAMPLINGS = ("center", "mean12")
BOLTZMANN = 1.380649e-23  # J/K
REFERENCE_TEMPERATURE = 290.0  # K


@dataclass(frozen=True)
class ChannelEpisode:
    """Channel of one episode for ``batch_size`` links.

    :param gain: [batch_size, num_slots, num_prbs] float32 tensor, always the linear power
        gain |h|^2 of each PRB in each slot (non-negative, finite)
    :param reference_snr_db: SNR [dB] at unit gain. If `None`, the simulator uses the
        scenario SNR (drawn or fixed); the gain should then have unit mean so that the
        scenario SNR is the mean SNR. If set (a float, or a [batch_size] array for
        per-link values), the simulator uses it instead, e.g. from a link budget.
    :param info: Metadata of the episode: each value a JSON-serializable list with one
        entry per link, e.g. the trajectory index
    """

    gain: torch.Tensor
    reference_snr_db: float | np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)


class ChannelSource(Protocol):
    """Source of per-slot, per-PRB channel gains; the extension point for custom channels.

    This interface is stable. An implementation provides:

    - ``num_prbs``: the number of PRBs of the gains it returns;
    - ``generate(num_slots, batch_size, seed)``: a :class:`ChannelEpisode` whose ``gain``
      has shape [batch_size, num_slots, num_prbs]. The result must depend only on the
      arguments: same arguments, same gains. Use ``seed`` for all randomness and do not
      read or modify global random state.

    The simulator computes ``sinr = 10 ** (snr_db / 10) * gain``, where ``snr_db`` is
    ``reference_snr_db`` if the episode sets it and the scenario SNR otherwise. Pass a
    source to :class:`linkgym.sim.LinkSimulator` with ``channel_source=...``.
    """

    num_prbs: int

    def generate(self, num_slots: int, batch_size: int, seed: int) -> ChannelEpisode: ...


class TraceFormatError(ValueError):
    """A trace file does not follow the format or does not match the scenario."""


@dataclass(frozen=True)
class TraceData:
    """Validated contents of a trace file (format version 1)."""

    path: str
    gain: np.ndarray  # float32 [T, N, num_prb]
    split: np.ndarray  # str [T]
    group: np.ndarray | None  # int [T]
    attrs: dict[str, Any]

    @property
    def num_trajectories(self) -> int:
        return self.gain.shape[0]

    @property
    def num_slots(self) -> int:
        return self.gain.shape[1]

    @property
    def num_prbs(self) -> int:
        return self.gain.shape[2]


def link_budget_snr_db(
    tx_power_dbm: float,
    noise_figure_db: float,
    bandwidth_hz: float,
    temperature_k: float = REFERENCE_TEMPERATURE,
) -> float:
    """SNR [dB] at unit gain: transmit power over k T B times the noise figure.

    The transmit power is assumed spread uniformly over ``bandwidth_hz``, so the SNR of a
    PRB with gain g is this value plus 10 log10(g).
    """
    noise_dbm = 10.0 * math.log10(BOLTZMANN * temperature_k * bandwidth_hz) + 30.0
    return tx_power_dbm - noise_dbm - noise_figure_db


def write_trace(
    path: str | os.PathLike,
    gain: np.ndarray,
    split: list[str] | np.ndarray,
    *,
    carrier_frequency_hz: float,
    subcarrier_spacing_hz: float,
    slot_duration_s: float,
    prb_sampling: str = "center",
    group: np.ndarray | None = None,
    rx_position: np.ndarray | None = None,
    rx_velocity: np.ndarray | None = None,
    num_paths: np.ndarray | None = None,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Write a trace file (format version 1); the result is validated by reading it back.

    :param gain: [T, N, num_prb] linear power gain |h|^2 per trajectory, slot and PRB
    :param split: [T] split label per trajectory, e.g. "train", "val", "test"
    :param prb_sampling: "center" (one frequency point per PRB) or "mean12" (mean over the
        12 subcarriers of the PRB)
    :param attrs: Optional extra root attributes (str, int, float), e.g. scene or versions
    """
    import h5py

    gain = np.asarray(gain, dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = TRACE_FORMAT
        f.attrs["format_version"] = TRACE_FORMAT_VERSION
        f.attrs["gain_unit"] = GAIN_UNIT
        f.attrs["carrier_frequency_hz"] = float(carrier_frequency_hz)
        f.attrs["subcarrier_spacing_hz"] = float(subcarrier_spacing_hz)
        f.attrs["slot_duration_s"] = float(slot_duration_s)
        f.attrs["num_prb"] = int(gain.shape[-1]) if gain.ndim == 3 else -1
        f.attrs["prb_sampling"] = prb_sampling
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
        f.create_dataset("gain", data=gain)
        f.create_dataset("split", data=np.asarray(split, dtype=object), dtype=h5py.string_dtype())
        optional = {
            "group": (group, np.int32),
            "rx_position": (rx_position, np.float32),
            "rx_velocity": (rx_velocity, np.float32),
            "num_paths": (num_paths, np.int16),
        }
        for name, (data, dtype) in optional.items():
            if data is not None:
                f.create_dataset(name, data=np.asarray(data, dtype=dtype))
    read_trace(path)


def read_trace(path: str | os.PathLike) -> TraceData:
    """Read and validate a trace file; raise :class:`TraceFormatError` on any problem."""
    import h5py

    path = os.fspath(path)

    def fail(message: str) -> TraceFormatError:
        return TraceFormatError(f"{path}: {message}")

    try:
        f = h5py.File(path, "r")
    except OSError as e:
        raise fail(f"cannot open as HDF5 ({e})") from e
    with f:
        attrs = {key: _plain(value) for key, value in f.attrs.items()}
        if attrs.get("format") != TRACE_FORMAT:
            raise fail(f"attribute 'format' must be {TRACE_FORMAT!r}, got {attrs.get('format')!r}")
        if attrs.get("format_version") != TRACE_FORMAT_VERSION:
            raise fail(
                f"unsupported format_version {attrs.get('format_version')!r}; "
                f"this linkgym reads version {TRACE_FORMAT_VERSION}"
            )
        required = (
            "gain_unit",
            "carrier_frequency_hz",
            "subcarrier_spacing_hz",
            "slot_duration_s",
            "num_prb",
            "prb_sampling",
        )
        missing = [key for key in required if key not in attrs]
        if missing:
            raise fail(f"missing attributes {missing}")
        if attrs["gain_unit"] != GAIN_UNIT:
            raise fail(f"attribute 'gain_unit' must be {GAIN_UNIT!r}, got {attrs['gain_unit']!r}")
        if attrs["prb_sampling"] not in PRB_SAMPLINGS:
            raise fail(f"attribute 'prb_sampling' must be one of {PRB_SAMPLINGS}")
        for key in ("carrier_frequency_hz", "subcarrier_spacing_hz", "slot_duration_s"):
            value = attrs[key]
            if not isinstance(value, float | int) or not math.isfinite(value) or value <= 0:
                raise fail(f"attribute {key!r} must be a positive number, got {value!r}")

        for name in ("gain", "split"):
            if name not in f:
                raise fail(f"missing dataset {name!r}")
        gain_ds = f["gain"]
        if gain_ds.dtype != np.float32 or gain_ds.ndim != 3:
            raise fail(
                f"dataset 'gain' must be float32 with 3 dimensions [T, N, num_prb], "
                f"got {gain_ds.dtype} with shape {gain_ds.shape}"
            )
        num_traj, num_slots, num_prb = gain_ds.shape
        if num_traj < 1 or num_slots < 1:
            raise fail(f"dataset 'gain' is empty, shape {gain_ds.shape}")
        if num_prb != attrs["num_prb"]:
            raise fail(
                f"dataset 'gain' has {num_prb} PRBs, attribute 'num_prb' says {attrs['num_prb']}"
            )
        gain = gain_ds[()]
        if not np.all(np.isfinite(gain)):
            raise fail("dataset 'gain' contains NaN or infinite values")
        if np.any(gain < 0):
            raise fail("dataset 'gain' contains negative values")

        split_ds = f["split"]
        if split_ds.shape != (num_traj,) or split_ds.dtype.kind not in "OSU":
            raise fail(f"dataset 'split' must hold {num_traj} strings, got shape {split_ds.shape}")
        split = np.array(split_ds.asstr()[()], dtype=object)
        if any(not label for label in split):
            raise fail("dataset 'split' contains empty labels")

        expected = {
            "group": (num_traj,),
            "rx_position": (num_traj, num_slots, 3),
            "rx_velocity": (num_traj, 3),
        }
        for name, shape in expected.items():
            if name in f and f[name].shape != shape:
                raise fail(f"dataset {name!r} must have shape {shape}, got {f[name].shape}")
        if "num_paths" in f and (f["num_paths"].ndim != 2 or f["num_paths"].shape[0] != num_traj):
            raise fail(f"dataset 'num_paths' must have shape ({num_traj}, A)")
        group = f["group"][()] if "group" in f else None

    return TraceData(path=path, gain=gain, split=split, group=group, attrs=attrs)


def _plain(value: Any) -> Any:
    """HDF5 attribute value as a plain Python value."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


class TraceChannelSource:
    """Channel gains read from a trace file, one window of a trajectory per link.

    For each link of an episode, a trajectory is drawn uniformly (with replacement) among
    those whose split label is in ``splits``, and a start slot uniformly among the slots
    that leave ``num_slots`` slots in the trajectory. The draws use only ``seed``.

    SNR modes:

    - ``"normalized"``: each trajectory is divided by its mean gain over all its slots and
      PRBs; the simulator then applies the scenario SNR.
    - ``"link_budget"``: absolute gains; the reference SNR is
      :func:`link_budget_snr_db` over the allocated bandwidth.

    :param path: Trace file (format version 1)
    :param num_prbs: Number of PRBs of the scenario; must match the file
    :param subcarrier_spacing: Subcarrier spacing [Hz]; must match the file
    :param carrier_frequency: Carrier frequency [Hz]; must match the file
    :param slot_duration: Slot duration [s]; must match the file
    :param num_slots: Longest episode to serve; trajectories must be at least this long
    :param splits: Split labels to sample from
    :param snr_mode: "normalized" or "link_budget"
    :param tx_power_dbm: Transmit power [dBm], required for "link_budget"
    :param noise_figure_db: Receiver noise figure [dB], for "link_budget"
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        num_prbs: int,
        subcarrier_spacing: float,
        carrier_frequency: float,
        slot_duration: float,
        num_slots: int,
        splits: tuple[str, ...] = ("train",),
        snr_mode: str = "normalized",
        tx_power_dbm: float | None = None,
        noise_figure_db: float = 7.0,
    ) -> None:
        trace = read_trace(path)
        self.trace = trace

        def fail(message: str) -> TraceFormatError:
            return TraceFormatError(f"{trace.path}: {message}")

        if trace.num_prbs != num_prbs:
            raise fail(f"trace has {trace.num_prbs} PRBs, the scenario has {num_prbs}")
        checks = {
            "carrier_frequency_hz": carrier_frequency,
            "subcarrier_spacing_hz": subcarrier_spacing,
            "slot_duration_s": slot_duration,
        }
        for key, value in checks.items():
            if not math.isclose(trace.attrs[key], value, rel_tol=1e-9):
                raise fail(f"trace {key} is {trace.attrs[key]}, the scenario has {value}")
        if trace.num_slots < num_slots:
            raise fail(f"trajectories have {trace.num_slots} slots, episodes need {num_slots}")
        if not splits:
            raise ValueError("splits must not be empty")
        available = sorted(set(trace.split))
        unknown = [s for s in splits if s not in available]
        if unknown:
            raise fail(f"split(s) {unknown} not in the trace; available: {available}")
        if snr_mode not in SNR_MODES:
            raise ValueError(f"snr_mode must be one of {SNR_MODES}, got {snr_mode!r}")

        self.num_prbs = num_prbs
        self.splits = tuple(splits)
        self.snr_mode = snr_mode
        self._eligible = np.flatnonzero(np.isin(trace.split, self.splits))
        self._mean_gain = trace.gain.astype(np.float64).mean(axis=(1, 2))
        if snr_mode == "normalized":
            zero = [int(i) for i in self._eligible if self._mean_gain[i] <= 0]
            if zero:
                raise fail(f"trajectories {zero} have zero mean gain; cannot normalize them")
            self.reference_snr_db = None
        else:
            if tx_power_dbm is None:
                raise ValueError("snr_mode='link_budget' requires tx_power_dbm")
            bandwidth = num_prbs * 12 * subcarrier_spacing
            self.reference_snr_db = link_budget_snr_db(tx_power_dbm, noise_figure_db, bandwidth)

    def generate(self, num_slots: int, batch_size: int, seed: int) -> ChannelEpisode:
        trace = self.trace
        if num_slots > trace.num_slots:
            raise ValueError(f"episodes of {num_slots} slots exceed the {trace.num_slots} slots")
        rng = np.random.default_rng(seed)
        trajectories = rng.choice(self._eligible, size=batch_size)
        offsets = rng.integers(0, trace.num_slots - num_slots + 1, size=batch_size)
        gain = np.stack(
            [trace.gain[t, o : o + num_slots] for t, o in zip(trajectories, offsets, strict=True)]
        )
        if self.snr_mode == "normalized":
            gain = (gain / self._mean_gain[trajectories][:, None, None]).astype(np.float32)
        info = {
            "trajectory": trajectories.tolist(),
            "offset": offsets.tolist(),
            "split": [str(trace.split[t]) for t in trajectories],
        }
        if trace.group is not None:
            info["group"] = [int(trace.group[t]) for t in trajectories]
        return ChannelEpisode(
            gain=torch.from_numpy(np.ascontiguousarray(gain)),
            reference_snr_db=self.reference_snr_db,
            info=info,
        )
