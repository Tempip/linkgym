"""Trace format 2: ray-traced paths at anchors, and multi-antenna reconstruction.

Stability: experimental (docs/api_stability.md); docs/paths.md has the details.

A format-2 trace (``channel_kind = "paths"``) stores, for every anchor (one path solve
every ``anchor_spacing_slots`` slots) of every trajectory, the propagation paths that Sionna
RT found for a single-antenna transmitter traced as a synthetic array: complex coefficient
``a``, delay ``tau``, Doppler shift, and the unit direction vectors of departure (``k_tx``)
and arrival (``k_rx``). :func:`reconstruct_cfr` rebuilds from them the channel of every
antenna of any transmit array (rows, columns, spacing, orientation) the way Sionna RT's
synthetic arrays do: the path gets the phase exp(+j 2 pi r . k_tx) at the antenna at r
(positions in wavelengths). The element pattern and polarization are those of the trace.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from linkgym.beams import Codebook, _element_positions, beam_gain
from linkgym.channels import (
    CATEGORIES,
    TRACE_FORMAT,
    TraceFormatError,
    _plain,
    link_budget_snr_db,
)
from linkgym.config import SNR_MODES

__all__ = [
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
]

PATH_TRACE_FORMAT_VERSION = 2
CHANNEL_KIND = "paths"
DEFAULT_ANTENNA = {"pattern": "iso", "polarization": "V"}
_CHUNK = 1 << 20  # paths validated per read
_RESERVED = {
    "format",
    "format_version",
    "channel_kind",
    "carrier_frequency_hz",
    "subcarrier_spacing_hz",
    "slot_duration_s",
    "num_prb",
    "prb_sampling",
    "num_slots",
    "anchor_spacing_slots",
    "synthetic_array",
    "tx_antenna",
    "rx_antenna",
    "tx_orientation",
}


@dataclass(frozen=True)
class PathTraceInfo:
    """Validated metadata of a format-2 trace (the paths themselves stay on disk).

    :param mean_gain: [T] mean single-antenna (1x1) power gain of each trajectory over all
        its slots and PRBs, as format 1 computes it from its gains
    """

    path: str
    num_trajectories: int
    num_slots: int
    num_prbs: int
    anchor_spacing: int
    num_paths: int
    split: np.ndarray
    group: np.ndarray | None
    category: np.ndarray | None
    mean_gain: np.ndarray
    attrs: dict[str, Any]

    @property
    def num_anchors(self) -> int:
        return self.num_slots // self.anchor_spacing

    @property
    def shape(self) -> tuple[int, int, int]:
        """(trajectories, slots, PRBs), as format 1's ``TraceInfo.shape``."""
        return (self.num_trajectories, self.num_slots, self.num_prbs)


@dataclass(frozen=True)
class PathWindow:
    """The paths of the anchors that cover a window of slots of one trajectory.

    :param a: [P] complex64 path coefficients
    :param tau: [P] float32 delays [s]
    :param doppler: [P] float32 Doppler shifts [Hz]
    :param k_tx: [P, 3] float32 unit directions of departure (scene frame)
    :param counts: [J] number of paths of each anchor, in order
    :param steps: [J, 2] first slot and number of slots of the window within each anchor
    :param carrier_frequency: Carrier frequency [Hz]
    :param frequencies: [F] PRB centre frequencies relative to the carrier [Hz]
    :param slot_duration: Slot duration [s]
    """

    a: np.ndarray
    tau: np.ndarray
    doppler: np.ndarray
    k_tx: np.ndarray
    counts: np.ndarray
    steps: np.ndarray
    carrier_frequency: float
    frequencies: np.ndarray
    slot_duration: float

    @property
    def num_slots(self) -> int:
        return int(self.steps[:, 1].sum())


def rotation_matrix(orientation: Sequence[float]) -> torch.Tensor:
    """Rotation matrix [3, 3] (float64) of Sionna RT's orientation (alpha, beta, gamma).

    The angles [rad] rotate about the z, y and x axes (TR 38.901, (7.1-4)), as
    ``sionna.rt.utils.rotation_matrix`` (geometry.py:89-123), which Sionna RT applies to
    the antenna positions of a device (antenna_array.py:108).
    """
    a, b, c = (float(x) for x in orientation)
    sa, ca, sb, cb, sc, cc = (
        math.sin(a),
        math.cos(a),
        math.sin(b),
        math.cos(b),
        math.sin(c),
        math.cos(c),
    )
    return torch.tensor(
        [
            [ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
            [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
            [-sb, cb * sc, cb * cc],
        ],
        dtype=torch.float64,
    )


def reconstruct_cfr(
    window: PathWindow,
    *,
    num_rows: int = 1,
    num_cols: int = 1,
    spacing: tuple[float, float] = (0.5, 0.5),
    orientation: Sequence[float] = (0.0, 0.0, 0.0),
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Channel frequency response of every antenna of a transmit array, [S, F, N].

    For slot s of anchor j and PRB centre frequency f, antenna n at position r_n
    (wavelengths, rotated by ``orientation``) sees

        h = sum_p a_p exp(-j 2 pi (f_c + f) tau_p) exp(j 2 pi f_D,p t) exp(j 2 pi r_n . k_tx,p)

    with t the time since the anchor's path solve: the model of linkgym's generator and of
    Sionna RT's ``Paths.cfr`` with a synthetic array. Antennas are in ``rt.PlanarArray``
    order (column by column), as in :mod:`linkgym.beams`; a 1x1 array gives the channel
    whose power is format 1's gain.

    :param window: Paths of a window, from :func:`read_path_window` or a source
    :param num_rows: Rows of the array
    :param num_cols: Columns of the array
    :param spacing: (vertical, horizontal) element spacing [wavelengths]
    :param orientation: Orientation (alpha, beta, gamma) [rad] of the array, as in Sionna
        RT. Positions follow it exactly; the element response stays that of the trace
        (exact for rotations about z with the default isotropic, vertically polarized
        elements)
    :param dtype: Output dtype; phases are always computed in float64
    """
    y, z = _element_positions(num_rows, num_cols, spacing)
    positions = torch.stack([torch.zeros_like(y), y, z], dim=1) @ rotation_matrix(orientation).T
    a = torch.from_numpy(np.asarray(window.a, dtype=np.complex128))
    tau = torch.from_numpy(np.asarray(window.tau, dtype=np.float64))
    doppler = torch.from_numpy(np.asarray(window.doppler, dtype=np.float64))
    k_tx = torch.from_numpy(np.asarray(window.k_tx, dtype=np.float64))
    frequencies = torch.from_numpy(np.asarray(window.frequencies, dtype=np.float64))
    a = a * torch.exp(-2j * math.pi * window.carrier_frequency * tau)
    array_phase = torch.exp(2j * math.pi * (positions @ k_tx.T))  # [N, P]
    blocks, start = [], 0
    for count, (first, num) in zip(window.counts.tolist(), window.steps.tolist(), strict=True):
        part = slice(start, start + count)
        start += count
        if count == 0:
            blocks.append(torch.zeros((num, len(frequencies), positions.shape[0]), dtype=dtype))
            continue
        t = (first + torch.arange(num, dtype=torch.float64)) * window.slot_duration
        evolution = a[part] * torch.exp(2j * math.pi * torch.outer(t, doppler[part]))  # [S, P]
        delay = torch.exp(-2j * math.pi * torch.outer(frequencies, tau[part]))  # [F, P]
        blocks.append(
            torch.einsum(
                "sp,fp,np->sfn",
                evolution.to(dtype),
                delay.to(dtype),
                array_phase[:, part].to(dtype),
            )
        )
    return torch.cat(blocks)


def write_path_trace(
    path: str | os.PathLike,
    *,
    split: Sequence[str] | np.ndarray,
    path_offset: np.ndarray,
    a: np.ndarray,
    tau: np.ndarray,
    doppler: np.ndarray,
    k_tx: np.ndarray,
    k_rx: np.ndarray,
    carrier_frequency_hz: float,
    subcarrier_spacing_hz: float,
    slot_duration_s: float,
    num_prb: int,
    num_slots: int,
    anchor_spacing_slots: int,
    tx_antenna: dict[str, str] | None = None,
    rx_antenna: dict[str, str] | None = None,
    tx_orientation: Sequence[float] = (0.0, 0.0, 0.0),
    mean_gain: np.ndarray | None = None,
    group: np.ndarray | None = None,
    category: Sequence[str] | np.ndarray | None = None,
    rx_position: np.ndarray | None = None,
    los: np.ndarray | None = None,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Write a format-2 trace (``channel_kind = "paths"``); it is validated by reading it back.

    The paths of trajectory t, anchor j are ``path_offset[t, j]`` to
    ``path_offset[t, j + 1]`` of the flat path arrays: trajectory-major, anchors in order.
    Keep the paths of an anchor in a fixed order (the generator sorts them) so that files,
    and the sums over paths, are reproducible.

    :param split: [T] split label per trajectory
    :param path_offset: [T, A + 1] int64 offsets into the path arrays, A = num_slots /
        anchor_spacing_slots; ``path_offset[t, A] == path_offset[t + 1, 0]``
    :param a: [P] complex path coefficients (passband, element patterns included)
    :param tau: [P] delays [s]
    :param doppler: [P] Doppler shifts [Hz]
    :param k_tx: [P, 3] unit directions of departure; ``k_rx``: of arrival (scene frame)
    :param tx_antenna: Element pattern and polarization the paths were traced with, e.g.
        ``{"pattern": "iso", "polarization": "V"}`` (the default); ``rx_antenna`` likewise
    :param tx_orientation: Orientation (alpha, beta, gamma) [rad] of the traced transmitter
    :param mean_gain: [T] mean 1x1 gain of each trajectory over its slots and PRBs; computed
        from the paths if `None`
    :param group: [T] group of each trajectory (e.g. its route); ``category``: [T] route
        category; ``rx_position``: [T, num_slots, 3] receiver positions [m];
        ``los``: [T, A] line-of-sight flag per anchor
    :param attrs: Extra root attributes (provenance); the format's own keys are reserved
    """
    import h5py

    path_offset = np.asarray(path_offset, dtype=np.int64)
    if path_offset.ndim != 2:
        raise ValueError(f"path_offset must be [T, A + 1], got shape {path_offset.shape}")
    clash = sorted(set(attrs or {}) & _RESERVED)
    if clash:
        raise ValueError(f"attrs {clash} are reserved by the format")
    # The values as stored, so that the mean gain is computed from what readers will see
    arrays = {
        "path_offset": path_offset,
        "a": np.asarray(a, dtype=np.complex64),
        "tau": np.asarray(tau, dtype=np.float32),
        "doppler": np.asarray(doppler, dtype=np.float32),
        "k_tx": np.asarray(k_tx, dtype=np.float32),
        "k_rx": np.asarray(k_rx, dtype=np.float32),
    }
    if mean_gain is None:
        grid = PathTraceInfo(
            path=os.fspath(path),
            num_trajectories=path_offset.shape[0],
            num_slots=int(num_slots),
            num_prbs=int(num_prb),
            anchor_spacing=int(anchor_spacing_slots),
            num_paths=int(path_offset[-1, -1]),
            split=np.asarray(split, dtype=object),
            group=None,
            category=None,
            mean_gain=np.zeros(path_offset.shape[0]),
            attrs={
                "carrier_frequency_hz": float(carrier_frequency_hz),
                "subcarrier_spacing_hz": float(subcarrier_spacing_hz),
                "slot_duration_s": float(slot_duration_s),
            },
        )
        mean_gain = _mean_gains(arrays, grid)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = TRACE_FORMAT
        f.attrs["format_version"] = PATH_TRACE_FORMAT_VERSION
        f.attrs["channel_kind"] = CHANNEL_KIND
        f.attrs["carrier_frequency_hz"] = float(carrier_frequency_hz)
        f.attrs["subcarrier_spacing_hz"] = float(subcarrier_spacing_hz)
        f.attrs["slot_duration_s"] = float(slot_duration_s)
        f.attrs["num_prb"] = int(num_prb)
        f.attrs["prb_sampling"] = "center"
        f.attrs["num_slots"] = int(num_slots)
        f.attrs["anchor_spacing_slots"] = int(anchor_spacing_slots)
        f.attrs["synthetic_array"] = True
        f.attrs["tx_antenna"] = json.dumps(tx_antenna or DEFAULT_ANTENNA, sort_keys=True)
        f.attrs["rx_antenna"] = json.dumps(rx_antenna or DEFAULT_ANTENNA, sort_keys=True)
        f.attrs["tx_orientation"] = np.asarray(tx_orientation, dtype=np.float64)
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
        f.create_dataset("split", data=np.asarray(split, dtype=object), dtype=h5py.string_dtype())
        f.create_dataset("num_paths", data=np.diff(path_offset, axis=1).astype(np.int32))
        for name, data in arrays.items():
            f.create_dataset(name, data=data)
        if category is not None:
            data = np.asarray(category, dtype=object)
            f.create_dataset("category", data=data, dtype=h5py.string_dtype())
        optional = {
            "group": (group, np.int32),
            "rx_position": (rx_position, np.float32),
            "los": (los, np.int8),
        }
        for name, (data, dtype) in optional.items():
            if data is not None:
                f.create_dataset(name, data=np.asarray(data, dtype=dtype))
        f.create_dataset("mean_gain", data=np.asarray(mean_gain, dtype=np.float64))
    inspect_path_trace(path)


def inspect_path_trace(path: str | os.PathLike, *, deep: bool = False) -> PathTraceInfo:
    """Validate a format-2 trace without loading its paths into memory.

    The path arrays are checked in chunks. With ``deep=True``, the mean gain of every
    trajectory is also recomputed from its paths and compared with the stored one (this
    reconstructs every anchor). Raise :class:`~linkgym.channels.TraceFormatError` on any
    problem.
    """
    path = os.fspath(path)
    with _open(path) as f:
        info = _validate(f, path)
        if deep:
            recomputed = _mean_gains(f, info)
            if not np.allclose(recomputed, info.mean_gain, rtol=1e-6, atol=0):
                bad = np.flatnonzero(~np.isclose(recomputed, info.mean_gain, rtol=1e-6, atol=0))
                raise TraceFormatError(
                    f"{path}: dataset 'mean_gain' does not match the paths of trajectories "
                    f"{bad.tolist()[:10]}"
                )
        return info


def read_path_window(
    path: str | os.PathLike, trajectory: int, offset: int, num_slots: int
) -> PathWindow:
    """The paths covering slots ``offset`` to ``offset + num_slots`` of one trajectory."""
    path = os.fspath(path)
    with _open(path) as f:
        info = _validate(f, path)
        return _window(f, info, trajectory, offset, num_slots)


@dataclass(frozen=True)
class ArrayChannelEpisode:
    """Channel of every transmit antenna in one episode, for ``batch_size`` links.

    :param channel: [batch_size, num_slots, num_prbs, num_antennas] complex64 channel of
        each antenna; antennas in ``rt.PlanarArray`` order, as in :mod:`linkgym.beams`
    :param reference_snr_db: SNR [dB] at unit gain, as in
        :class:`~linkgym.channels.ChannelEpisode`: `None` if the channel is normalized (the
        scenario SNR applies), else a link budget
    :param info: Metadata of the episode, one entry per link, as in
        :class:`~linkgym.channels.ChannelEpisode`
    :param beam_gain: [batch_size, num_slots, num_prbs, num_beams] float32 |w^H h|^2 of
        every beam of the source's codebook, or `None` without a codebook
    """

    channel: torch.Tensor
    reference_snr_db: float | np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)
    beam_gain: torch.Tensor | None = None


class ArrayChannelSource(Protocol):
    """Source of per-antenna channels: :class:`~linkgym.channels.ChannelSource` for arrays.

    Experimental. An implementation provides ``num_prbs``, ``num_antennas`` and
    ``generate(num_slots, batch_size, seed)``, which returns an :class:`ArrayChannelEpisode`
    whose ``channel`` has shape [batch_size, num_slots, num_prbs, num_antennas] and depends
    only on the arguments (``seed`` for all randomness, no global random state).
    """

    num_prbs: int
    num_antennas: int

    def generate(self, num_slots: int, batch_size: int, seed: int) -> ArrayChannelEpisode: ...


class PathTraceSource:
    """Per-antenna channels rebuilt from a path trace (format 2), one window per link.

    The :class:`ArrayChannelSource` counterpart of
    :class:`~linkgym.channels.TraceChannelSource`, with the same checks of the file against
    the scenario, the same random draws of trajectories and windows for a given seed (on a
    file with the same splits and number of slots), the same pins and the same episode
    info. Each link is one transmitter-receiver link of a trajectory. Windows may start
    between anchors. The file is validated once; each process opens its own read-only
    handle on first use, so the source can be pickled; :meth:`close` releases it.

    The channel of each antenna comes from :func:`reconstruct_cfr` for the array given
    here; the element pattern, polarization and carrier frequency are those of the trace
    (docs/paths.md).

    SNR modes:

    - ``"normalized"``: the channel of each trajectory is divided by the square root of its
      ``mean_gain``, the mean single-antenna gain. A single antenna then has unit mean
      power, as in format 1, and a beam up to ``num_antennas`` times more.
    - ``"link_budget"``: absolute channels; the reference SNR is
      :func:`~linkgym.channels.link_budget_snr_db` over the allocated bandwidth, for the
      whole transmit power (unit-norm beams keep it).

    :param path: Path trace (format 2)
    :param num_prbs: Number of PRBs of the scenario; must match the file
    :param subcarrier_spacing: Subcarrier spacing [Hz]; must match the file
    :param carrier_frequency: Carrier frequency [Hz]; must match the file
    :param slot_duration: Slot duration [s]; must match the file
    :param num_slots: Longest episode to serve; trajectories must be at least this long
    :param splits: Split labels to sample from
    :param snr_mode: "normalized" or "link_budget"
    :param tx_power_dbm: Transmit power [dBm], required for "link_budget"
    :param noise_figure_db: Receiver noise figure [dB], for "link_budget"
    :param num_rows: Rows of the transmit array
    :param num_cols: Columns of the transmit array
    :param spacing: (vertical, horizontal) element spacing [wavelengths]
    :param orientation: Orientation (alpha, beta, gamma) [rad] of the array; the traced
        transmitter's (attribute ``tx_orientation``) if `None`
    :param codebook: Optional :class:`~linkgym.beams.Codebook` for this array; each
        episode then also has the gain of every beam
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
        num_rows: int = 1,
        num_cols: int = 1,
        spacing: tuple[float, float] = (0.5, 0.5),
        orientation: Sequence[float] | None = None,
        codebook: Codebook | None = None,
    ) -> None:
        info = inspect_path_trace(path)
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
        if num_rows < 1 or num_cols < 1:
            raise ValueError(
                f"the array needs at least one row and column, got {num_rows}x{num_cols}"
            )
        spacing = (float(spacing[0]), float(spacing[1]))
        if codebook is not None and (
            (codebook.num_rows, codebook.num_cols) != (num_rows, num_cols)
            or tuple(codebook.spacing) != spacing
        ):
            raise ValueError(
                f"the codebook is for a {codebook.num_rows}x{codebook.num_cols} array with "
                f"spacing {tuple(codebook.spacing)}, the source has {num_rows}x{num_cols} "
                f"with spacing {spacing}"
            )

        self.num_prbs = num_prbs
        self.num_antennas = num_rows * num_cols
        self.splits = tuple(splits)
        self.snr_mode = snr_mode
        self.array = {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "spacing": spacing,
            "orientation": tuple(
                float(x)
                for x in (info.attrs["tx_orientation"] if orientation is None else orientation)
            ),
        }
        self.codebook = codebook
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
    ) -> ArrayChannelEpisode:
        """Channels of ``batch_size`` windows; random unless pinned.

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
        f = self._handle()
        channels = []
        for t, o in zip(trajectories.tolist(), offsets.tolist(), strict=True):
            h = reconstruct_cfr(_window(f, info, t, o, num_slots), **self.array)
            if self.snr_mode == "normalized":
                h = h * float(1.0 / math.sqrt(info.mean_gain[t]))
            channels.append(h)
        channel = torch.stack(channels)
        episode_info = {
            "trajectory": trajectories.tolist(),
            "offset": offsets.tolist(),
            "split": [str(info.split[t]) for t in trajectories],
        }
        if info.group is not None:
            episode_info["group"] = [int(info.group[t]) for t in trajectories]
        if info.category is not None:
            episode_info["category"] = [str(info.category[t]) for t in trajectories]
        return ArrayChannelEpisode(
            channel=channel,
            reference_snr_db=self.reference_snr_db,
            info=episode_info,
            beam_gain=None if self.codebook is None else beam_gain(channel, self.codebook),
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

    def _handle(self):
        """A read-only handle on the file, owned by the current process."""
        if self._file is None or self._file_pid != os.getpid():
            self._file = _open(self.info.path)
            self._file_pid = os.getpid()
        return self._file

    def __getstate__(self) -> dict[str, Any]:
        # File handles belong to one process: a copy opens its own on first use
        return self.__dict__ | {"_file": None, "_file_pid": None}


def _is_path_trace(path: str | os.PathLike) -> bool:
    """Whether ``path`` is an HDF5 file that declares itself a path trace (not validated)."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            return _plain(f.attrs.get("channel_kind")) == CHANNEL_KIND
    except OSError:
        return False


def _open(path: str):
    import h5py

    try:
        return h5py.File(path, "r")
    except OSError as e:
        raise TraceFormatError(f"{path}: cannot open as HDF5 ({e})") from e


def _frequencies(info: PathTraceInfo) -> np.ndarray:
    """PRB centre frequencies relative to the carrier [Hz], as the generator computes them."""
    scs = info.attrs["subcarrier_spacing_hz"]
    subcarriers = (np.arange(12 * info.num_prbs) - 6 * info.num_prbs) * scs
    return subcarriers.reshape(info.num_prbs, 12).mean(axis=1)


def _window(f, info: PathTraceInfo, trajectory: int, offset: int, num_slots: int) -> PathWindow:
    if not 0 <= trajectory < info.num_trajectories:
        raise ValueError(f"trajectory {trajectory} outside [0, {info.num_trajectories})")
    if num_slots < 1 or not 0 <= offset <= info.num_slots - num_slots:
        raise ValueError(
            f"window of {num_slots} slots at offset {offset} does not fit in {info.num_slots} slots"
        )
    spacing = info.anchor_spacing
    first_anchor, last_anchor = offset // spacing, (offset + num_slots - 1) // spacing
    row = f["path_offset"][trajectory, first_anchor : last_anchor + 2]
    begin, end = int(row[0]), int(row[-1])
    anchors = np.arange(first_anchor, last_anchor + 1)
    start = np.maximum(offset, anchors * spacing)
    stop = np.minimum(offset + num_slots, (anchors + 1) * spacing)
    return PathWindow(
        a=f["a"][begin:end],
        tau=f["tau"][begin:end],
        doppler=f["doppler"][begin:end],
        k_tx=f["k_tx"][begin:end],
        counts=np.diff(row),
        steps=np.stack([start - anchors * spacing, stop - start], axis=1),
        carrier_frequency=float(info.attrs["carrier_frequency_hz"]),
        frequencies=_frequencies(info),
        slot_duration=float(info.attrs["slot_duration_s"]),
    )


def _mean_gains(f, info: PathTraceInfo) -> np.ndarray:
    """Mean 1x1 gain of every trajectory, computed as format 1 does from float32 gains."""
    gains = np.empty(info.num_trajectories)
    for t in range(info.num_trajectories):
        window = _window(f, info, t, 0, info.num_slots)
        h = reconstruct_cfr(window, dtype=torch.complex128)[..., 0]
        gains[t] = (h.abs().square().numpy().astype(np.float32)).mean(dtype=np.float64)
    return gains


def _validate(f, path: str) -> PathTraceInfo:
    def fail(message: str) -> TraceFormatError:
        return TraceFormatError(f"{path}: {message}")

    attrs = {key: _plain(value) for key, value in f.attrs.items()}
    if attrs.get("format") != TRACE_FORMAT:
        raise fail(f"attribute 'format' must be {TRACE_FORMAT!r}, got {attrs.get('format')!r}")
    if attrs.get("format_version") != PATH_TRACE_FORMAT_VERSION:
        raise fail(
            f"format_version {attrs.get('format_version')!r} is not a paths trace "
            f"(version {PATH_TRACE_FORMAT_VERSION}); read format 1 with linkgym.channels"
        )
    if attrs.get("channel_kind") != CHANNEL_KIND:
        raise fail(f"attribute 'channel_kind' must be {CHANNEL_KIND!r}")
    missing = sorted(_RESERVED - {"format", "format_version", "channel_kind"} - set(attrs))
    if missing:
        raise fail(f"missing attributes {missing}")
    for key in ("carrier_frequency_hz", "subcarrier_spacing_hz", "slot_duration_s"):
        value = attrs[key]
        if not isinstance(value, float | int) or not math.isfinite(value) or value <= 0:
            raise fail(f"attribute {key!r} must be a positive number, got {value!r}")
    for key in ("num_prb", "num_slots", "anchor_spacing_slots"):
        if not isinstance(attrs[key], int) or attrs[key] < 1:
            raise fail(f"attribute {key!r} must be a positive integer, got {attrs[key]!r}")
    if attrs["prb_sampling"] != "center":
        raise fail("attribute 'prb_sampling' must be 'center' in a paths trace")
    if attrs["synthetic_array"] is not True:
        raise fail("attribute 'synthetic_array' must be true")
    for key in ("tx_antenna", "rx_antenna"):
        try:
            antenna = json.loads(attrs[key])
        except (TypeError, json.JSONDecodeError):
            antenna = None
        if not isinstance(antenna, dict) or not {"pattern", "polarization"} <= set(antenna):
            raise fail(f"attribute {key!r} must be JSON with 'pattern' and 'polarization'")
    if np.shape(attrs["tx_orientation"]) != (3,):
        raise fail("attribute 'tx_orientation' must hold 3 angles")
    num_slots, spacing = attrs["num_slots"], attrs["anchor_spacing_slots"]
    if num_slots % spacing:
        raise fail(f"num_slots {num_slots} is not a multiple of anchor_spacing_slots {spacing}")
    num_anchors = num_slots // spacing

    for name in ("split", "path_offset", "a", "tau", "doppler", "k_tx", "k_rx", "mean_gain"):
        if name not in f:
            raise fail(f"missing dataset {name!r}")
    split_ds = f["split"]
    num_traj = split_ds.shape[0] if split_ds.ndim == 1 else -1
    if num_traj < 1 or split_ds.dtype.kind not in "OSU":
        raise fail(f"dataset 'split' must hold one string per trajectory, got {split_ds.shape}")
    split = np.array(split_ds.asstr()[()], dtype=object)
    if any(not label for label in split):
        raise fail("dataset 'split' contains empty labels")

    offsets = f["path_offset"][()]
    if offsets.dtype != np.int64 or offsets.shape != (num_traj, num_anchors + 1):
        raise fail(
            f"dataset 'path_offset' must be int64 [{num_traj}, {num_anchors + 1}], got "
            f"{offsets.dtype} {offsets.shape}"
        )
    if offsets[0, 0] != 0 or np.any(np.diff(offsets, axis=1) < 0):
        raise fail("dataset 'path_offset' must start at 0 and never decrease")
    if num_traj > 1 and np.any(offsets[1:, 0] != offsets[:-1, -1]):
        raise fail("dataset 'path_offset': each trajectory must start where the previous ends")
    num_paths = int(offsets[-1, -1])
    expected = {
        "a": ((num_paths,), "c"),
        "tau": ((num_paths,), "f"),
        "doppler": ((num_paths,), "f"),
        "k_tx": ((num_paths, 3), "f"),
        "k_rx": ((num_paths, 3), "f"),
    }
    for name, (shape, kind) in expected.items():
        ds = f[name]
        if (
            ds.shape != shape
            or ds.dtype.kind != kind
            or ds.dtype.itemsize != (8 if kind == "c" else 4)
        ):
            raise fail(
                f"dataset {name!r} must be {'complex64' if kind == 'c' else 'float32'} "
                f"{shape}, got {ds.dtype} {ds.shape}"
            )
    for begin in range(0, num_paths, _CHUNK):
        end = min(begin + _CHUNK, num_paths)
        a, tau, doppler = f["a"][begin:end], f["tau"][begin:end], f["doppler"][begin:end]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(doppler))):
            raise fail(f"paths {begin}-{end}: 'a' or 'doppler' has NaN or infinite values")
        if not np.all(np.isfinite(tau)) or np.any(tau < 0):
            raise fail(f"paths {begin}-{end}: 'tau' must be finite and non-negative")
        for name in ("k_tx", "k_rx"):
            norm = np.linalg.norm(f[name][begin:end].astype(np.float64), axis=1)
            if np.any(np.abs(norm - 1) > 1e-4):
                raise fail(f"paths {begin}-{end}: {name!r} must hold unit vectors")

    if "num_paths" in f and not np.array_equal(f["num_paths"][()], np.diff(offsets, axis=1)):
        raise fail("dataset 'num_paths' does not match 'path_offset'")
    shapes = {
        "group": (num_traj,),
        "rx_position": (num_traj, num_slots, 3),
        "los": (num_traj, num_anchors),
    }
    for name, shape in shapes.items():
        if name in f and f[name].shape != shape:
            raise fail(f"dataset {name!r} must have shape {shape}, got {f[name].shape}")
    category = None
    if "category" in f:
        ds = f["category"]
        if ds.shape != (num_traj,) or ds.dtype.kind not in "OSU":
            raise fail(f"dataset 'category' must hold {num_traj} strings, got shape {ds.shape}")
        category = np.array(ds.asstr()[()], dtype=object)
        unknown = sorted(set(category) - set(CATEGORIES))
        if unknown:
            raise fail(f"dataset 'category' has values {unknown}; allowed: {list(CATEGORIES)}")
    mean_gain = f["mean_gain"][()]
    if mean_gain.shape != (num_traj,) or mean_gain.dtype != np.float64:
        raise fail(f"dataset 'mean_gain' must be float64 [{num_traj}], got {mean_gain.shape}")
    if not np.all(np.isfinite(mean_gain)) or np.any(mean_gain < 0):
        raise fail("dataset 'mean_gain' must be finite and non-negative")

    return PathTraceInfo(
        path=path,
        num_trajectories=num_traj,
        num_slots=num_slots,
        num_prbs=attrs["num_prb"],
        anchor_spacing=spacing,
        num_paths=num_paths,
        split=split,
        group=f["group"][()] if "group" in f else None,
        category=category,
        mean_gain=mean_gain,
        attrs=attrs,
    )
