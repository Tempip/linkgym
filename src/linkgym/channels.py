"""Channel sources: the per-slot, per-PRB channel power gain of an episode.

:class:`ChannelSource` is the extension point for bringing your own channels. Built-in
sources: :class:`linkgym.sim.TDLChannelGain` (3GPP TDL fading, the default) and
:class:`TraceChannelSource` (gains read from a trace file, see :func:`write_trace` and
docs/channels.md).
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from linkgym.config import SNR_MODES

TRACE_FORMAT = "linkgym-trace"
TRACE_FORMAT_VERSION = 1
GAIN_UNIT = "linear_power_gain"
PRB_SAMPLINGS = ("center", "mean12")
CATEGORIES = ("los", "nlos", "transition")
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
    source to :class:`linkgym.sim.LinkSimulator` with ``channel_source=...``. A source may
    also have a ``close()`` method releasing its resources; the simulator calls it in
    :meth:`~linkgym.sim.LinkSimulator.close`.
    """

    num_prbs: int

    def generate(self, num_slots: int, batch_size: int, seed: int) -> ChannelEpisode: ...


class TraceFormatError(ValueError):
    """A trace file does not follow the format or does not match the scenario."""


@dataclass(frozen=True)
class TraceInfo:
    """Validated header and statistics of a trace file (format version 1), without the gains.

    :param mean_gain: [T] float64 mean gain of each trajectory over all its slots and PRBs
    """

    path: str
    shape: tuple[int, int, int]  # [T, N, num_prb] of the gain dataset
    split: np.ndarray  # str [T]
    group: np.ndarray | None  # int [T]
    category: np.ndarray | None  # str [T]
    attrs: dict[str, Any]
    mean_gain: np.ndarray

    @property
    def num_trajectories(self) -> int:
        return self.shape[0]

    @property
    def num_slots(self) -> int:
        return self.shape[1]

    @property
    def num_prbs(self) -> int:
        return self.shape[2]


@dataclass(frozen=True)
class TraceData:
    """Validated contents of a trace file (format version 1), gains included."""

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


def tx_power_for_median_snr(
    trace: str | os.PathLike,
    splits: Sequence[str],
    target_snr_db: float,
    noise_figure_db: float = 7.0,
) -> float:
    """Transmit power [dBm] that puts the median per-slot wideband SNR at ``target_snr_db``.

    For ``snr_mode="link_budget"``: one transmit power for all trajectories places the
    typical SNR inside the range of the BLER tables (-5 to 20 dB for PDSCH table 1) while
    keeping the real power differences between and along trajectories, unlike
    ``"normalized"``, which scales each trajectory separately.

    The median is taken over every slot of every trajectory whose split is in ``splits``,
    of the slot's wideband SNR: the reference SNR of :func:`link_budget_snr_db` over the
    trace's bandwidth (``num_prb`` x 12 x subcarrier spacing, as in
    :class:`TraceChannelSource`) plus 10 log10 of the slot's gain averaged over its PRBs.
    Compute it on the train split only, so that val and test data never influence the
    scenario definition.

    :param trace: Trace file (format version 1)
    :param splits: Split labels whose trajectories set the median, e.g. ``["train"]``
    :param target_snr_db: Median per-slot wideband SNR to reach [dB]
    :param noise_figure_db: Receiver noise figure [dB], as given to the scenario
    """
    info = inspect_trace(trace)
    if not splits:
        raise ValueError("splits must not be empty")
    available = sorted(set(info.split))
    unknown = [s for s in splits if s not in available]
    if unknown:
        raise ValueError(f"split(s) {unknown} not in {info.path}; available: {available}")
    eligible = np.flatnonzero(np.isin(info.split, list(splits)))
    with _open_trace(info.path) as f:
        dataset = f["gain"]
        wideband = np.concatenate([dataset[t].mean(axis=1, dtype=np.float64) for t in eligible])
    with np.errstate(divide="ignore"):
        median_gain_db = float(np.median(10.0 * np.log10(wideband)))
    if not math.isfinite(median_gain_db):
        raise ValueError(f"more than half of the slots of splits {list(splits)} have zero gain")
    bandwidth = info.num_prbs * 12 * info.attrs["subcarrier_spacing_hz"]
    return target_snr_db - median_gain_db - link_budget_snr_db(0.0, noise_figure_db, bandwidth)


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
    num_paths: np.ndarray | None = None,
    los: np.ndarray | None = None,
    category: list[str] | np.ndarray | None = None,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Write a trace file (format version 1); the result is validated by reading it back.

    ``gain`` is stored contiguous and uncompressed, so that a window of slots of one
    trajectory is a single contiguous read.

    :param gain: [T, N, num_prb] linear power gain |h|^2 per trajectory, slot and PRB
    :param split: [T] split label per trajectory, e.g. "train", "val", "test"
    :param prb_sampling: "center" (one frequency point per PRB) or "mean12" (mean over the
        12 subcarriers of the PRB)
    :param group: [T] group of each trajectory, e.g. the route it was cut from
    :param rx_position: [T, N, 3] receiver position [m]
    :param num_paths: [T, A] number of propagation paths at A points of each trajectory
    :param los: [T, A] 1 if a line-of-sight path exists at these points, else 0
    :param category: [T] route category of each trajectory: "los", "nlos" or "transition"
    :param attrs: Optional extra root attributes (str, int, float or arrays of them)
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
        if category is not None:
            data = np.asarray(category, dtype=object)
            f.create_dataset("category", data=data, dtype=h5py.string_dtype())
        optional = {
            "group": (group, np.int32),
            "rx_position": (rx_position, np.float32),
            "num_paths": (num_paths, np.int16),
            "los": (los, np.int8),
        }
        for name, (data, dtype) in optional.items():
            if data is not None:
                f.create_dataset(name, data=np.asarray(data, dtype=dtype))
    inspect_trace(path)


def inspect_trace(path: str | os.PathLike) -> TraceInfo:
    """Validate a trace file without loading all its gains into memory.

    The gains are read one trajectory at a time to check their values and to compute the
    mean gain of each trajectory. Raise :class:`TraceFormatError` on any problem.
    """
    path = os.fspath(path)
    with _open_trace(path) as f:
        return _validate(f, path)


def read_trace(path: str | os.PathLike) -> TraceData:
    """Read and validate a trace file, gains included; raise :class:`TraceFormatError`."""
    path = os.fspath(path)
    with _open_trace(path) as f:
        info = _validate(f, path)
        gain = f["gain"][()]
    return TraceData(path=path, gain=gain, split=info.split, group=info.group, attrs=info.attrs)


def _open_trace(path: str):
    """Open a trace file read-only."""
    import h5py

    try:
        return h5py.File(path, "r")
    except OSError as e:
        raise TraceFormatError(f"{path}: cannot open as HDF5 ({e})") from e


def _validate(f, path: str) -> TraceInfo:
    """Check an open trace file; the gains are streamed one trajectory at a time."""

    def fail(message: str) -> TraceFormatError:
        return TraceFormatError(f"{path}: {message}")

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
    mean_gain = np.empty(num_traj)
    for t in range(num_traj):
        block = gain_ds[t]
        if not np.all(np.isfinite(block)):
            raise fail(f"dataset 'gain' contains NaN or infinite values (trajectory {t})")
        if np.any(block < 0):
            raise fail(f"dataset 'gain' contains negative values (trajectory {t})")
        mean_gain[t] = block.mean(dtype=np.float64)

    split_ds = f["split"]
    if split_ds.shape != (num_traj,) or split_ds.dtype.kind not in "OSU":
        raise fail(f"dataset 'split' must hold {num_traj} strings, got shape {split_ds.shape}")
    split = np.array(split_ds.asstr()[()], dtype=object)
    if any(not label for label in split):
        raise fail("dataset 'split' contains empty labels")

    expected = {"group": (num_traj,), "rx_position": (num_traj, num_slots, 3)}
    for name, shape in expected.items():
        if name in f and f[name].shape != shape:
            raise fail(f"dataset {name!r} must have shape {shape}, got {f[name].shape}")
    for name in ("num_paths", "los"):
        if name in f and (f[name].ndim != 2 or f[name].shape[0] != num_traj):
            raise fail(f"dataset {name!r} must have shape ({num_traj}, A), got {f[name].shape}")
    if "num_paths" in f and "los" in f and f["num_paths"].shape != f["los"].shape:
        raise fail("datasets 'num_paths' and 'los' must have the same shape")
    group = f["group"][()] if "group" in f else None
    category = None
    if "category" in f:
        ds = f["category"]
        if ds.shape != (num_traj,) or ds.dtype.kind not in "OSU":
            raise fail(f"dataset 'category' must hold {num_traj} strings, got shape {ds.shape}")
        category = np.array(ds.asstr()[()], dtype=object)
        unknown = sorted(set(category) - set(CATEGORIES))
        if unknown:
            raise fail(f"dataset 'category' has values {unknown}; allowed: {list(CATEGORIES)}")

    return TraceInfo(
        path=path,
        shape=(num_traj, num_slots, num_prb),
        split=split,
        group=group,
        category=category,
        attrs=attrs,
        mean_gain=mean_gain,
    )


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

    The file is validated once, in ``__init__``, without keeping its gains in memory; each
    episode then reads only its windows. Each process opens its own read-only handle on
    first use (again after a fork), so the source can be pickled and used in subprocess
    vector environments. :meth:`close` releases the handle.

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
        info = inspect_trace(path)
        self.info = info

        def fail(message: str) -> TraceFormatError:
            return TraceFormatError(f"{info.path}: {message}")

        if info.num_prbs != num_prbs:
            raise fail(f"trace has {info.num_prbs} PRBs, the scenario has {num_prbs}")
        checks = {
            "carrier_frequency_hz": carrier_frequency,
            "subcarrier_spacing_hz": subcarrier_spacing,
            "slot_duration_s": slot_duration,
        }
        for key, value in checks.items():
            if not math.isclose(info.attrs[key], value, rel_tol=1e-9):
                raise fail(f"trace {key} is {info.attrs[key]}, the scenario has {value}")
        if info.num_slots < num_slots:
            raise fail(f"trajectories have {info.num_slots} slots, episodes need {num_slots}")
        if not splits:
            raise ValueError("splits must not be empty")
        available = sorted(set(info.split))
        unknown = [s for s in splits if s not in available]
        if unknown:
            raise fail(f"split(s) {unknown} not in the trace; available: {available}")
        if snr_mode not in SNR_MODES:
            raise ValueError(f"snr_mode must be one of {SNR_MODES}, got {snr_mode!r}")

        self.num_prbs = num_prbs
        self.splits = tuple(splits)
        self.snr_mode = snr_mode
        self._eligible = np.flatnonzero(np.isin(info.split, self.splits))
        if snr_mode == "normalized":
            zero = [int(i) for i in self._eligible if info.mean_gain[i] <= 0]
            if zero:
                raise fail(f"trajectories {zero} have zero mean gain; cannot normalize them")
            self.reference_snr_db = None
        else:
            if tx_power_dbm is None:
                raise ValueError("snr_mode='link_budget' requires tx_power_dbm")
            bandwidth = num_prbs * 12 * subcarrier_spacing
            self.reference_snr_db = link_budget_snr_db(tx_power_dbm, noise_figure_db, bandwidth)
        self._file = None
        self._file_pid = None

    def generate(
        self,
        num_slots: int,
        batch_size: int,
        seed: int,
        *,
        trajectories: list[int] | np.ndarray | None = None,
        offsets: list[int] | np.ndarray | None = None,
    ) -> ChannelEpisode:
        """Gains of ``batch_size`` windows; random unless pinned.

        :param trajectories: Pin the trajectory of each link ([batch_size] indices into the
            file, with a split in ``splits``); requires ``offsets``
        :param offsets: Pin the start slot of each link's window
        """
        info = self.info
        if num_slots > info.num_slots:
            raise ValueError(f"episodes of {num_slots} slots exceed the {info.num_slots} slots")
        rng = np.random.default_rng(seed)
        if (trajectories is None) != (offsets is None):
            raise ValueError("pin both trajectories and offsets, or neither")
        if trajectories is None:
            trajectories = rng.choice(self._eligible, size=batch_size)
            offsets = rng.integers(0, info.num_slots - num_slots + 1, size=batch_size)
        else:
            trajectories, offsets = self._check_pin(trajectories, offsets, num_slots, batch_size)
        dataset = self._gain_dataset()
        gain = np.stack(
            [dataset[t, o : o + num_slots] for t, o in zip(trajectories, offsets, strict=True)]
        )
        if self.snr_mode == "normalized":
            gain = (gain / info.mean_gain[trajectories][:, None, None]).astype(np.float32)
        episode_info = {
            "trajectory": trajectories.tolist(),
            "offset": offsets.tolist(),
            "split": [str(info.split[t]) for t in trajectories],
        }
        if info.group is not None:
            episode_info["group"] = [int(info.group[t]) for t in trajectories]
        if info.category is not None:
            episode_info["category"] = [str(info.category[t]) for t in trajectories]
        return ChannelEpisode(
            gain=torch.from_numpy(np.ascontiguousarray(gain)),
            reference_snr_db=self.reference_snr_db,
            info=episode_info,
        )

    def _check_pin(self, trajectories, offsets, num_slots: int, batch_size: int):
        trajectories = np.asarray(trajectories)
        offsets = np.asarray(offsets)
        if trajectories.shape != (batch_size,) or offsets.shape != (batch_size,):
            raise ValueError(f"pin one trajectory and one offset per link ({batch_size})")
        if trajectories.dtype.kind not in "iu" or offsets.dtype.kind not in "iu":
            raise ValueError("trajectories and offsets must be integers")
        not_eligible = [int(t) for t in trajectories if t not in self._eligible]
        if not_eligible:
            raise ValueError(
                f"trajectories {not_eligible} are not in the splits {list(self.splits)} "
                f"of {self.info.path}"
            )
        last = self.info.num_slots - num_slots
        bad = [int(o) for o in offsets if not 0 <= o <= last]
        if bad:
            raise ValueError(f"offsets {bad} outside [0, {last}] for episodes of {num_slots} slots")
        return trajectories, offsets

    def close(self) -> None:
        """Close the file handle of this process, if one is open."""
        if self._file is not None and self._file_pid == os.getpid():
            self._file.close()
        self._file = None
        self._file_pid = None

    def _gain_dataset(self):
        """The gain dataset, through a read-only handle owned by the current process."""
        if self._file is None or self._file_pid != os.getpid():
            self._file = _open_trace(self.info.path)
            self._file_pid = os.getpid()
        return self._file["gain"]

    def __getstate__(self) -> dict[str, Any]:
        # File handles belong to one process: a copy opens its own on first use
        return self.__dict__ | {"_file": None, "_file_pid": None}
