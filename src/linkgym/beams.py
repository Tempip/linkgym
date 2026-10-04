"""Antenna-array utilities for beam management: DFT codebooks, steering vectors, beam gain, RSRP.

Stability: experimental (see docs/api_stability.md); docs/beams.md has the details.

Conventions, chosen to match Sionna RT's ``rt.PlanarArray`` and its synthetic arrays:

- The array has ``num_rows`` x ``num_cols`` elements in the y-z plane of its local frame,
  boresight along +x. Element k is in row ``k % num_rows`` (row 0 at the top) and column
  ``k // num_rows``: numbered column by column from the top left (antenna_array.py:135-138).
  Its position in wavelengths is y = d_h (column - (num_cols - 1) / 2),
  z = -d_v (row - (num_rows - 1) / 2) (antenna_array.py:212-215).
- A plane wave leaving or arriving along the unit vector k (local frame) reaches element n
  with the phase exp(+j 2 pi r_n . k): its steering vector, the phase Sionna RT's synthetic
  arrays apply (paths.py:1006-1023).
- The gain of beam weights w on a channel h (antenna axis last) is |w^H h|^2.

Front/back ambiguity: the elements lie in the y-z plane, so the phase depends only on k_y
and k_z. A direction and its mirror image behind the array, (-k_x, k_y, k_z), have the same
steering vector: with isotropic elements every beam also points backwards, and a beam gain
cannot tell a user in front of the array from one behind it. Real arrays break the tie with
directional elements (e.g. ``"tr38901"``) or a ground plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from linkgym.phy import _sionna_import_error

try:
    from sionna.phy.mimo import grid_of_beams_dft, grid_of_beams_dft_ula
except ImportError as _error:
    _hint = _sionna_import_error(_error)
    if _hint is None:
        raise
    raise _hint from _error

__all__ = ["Codebook", "beam_gain", "best_beam", "dft_codebook", "rsrp_dbm", "steering_vector"]


@dataclass(frozen=True)
class Codebook:
    """A set of beams for a planar array.

    :param weights: [B, N] complex64 beam weights, one unit-norm row per beam
    :param u: [B] direction cosine along the array's y axis (k_y) each beam points to
    :param v: [B] direction cosine along the array's z axis (k_z) each beam points to
    :param num_rows: Rows of the array
    :param num_cols: Columns of the array
    :param spacing: (vertical, horizontal) element spacing [wavelengths]
    """

    weights: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    num_rows: int
    num_cols: int
    spacing: tuple[float, float]

    @property
    def num_beams(self) -> int:
        return self.weights.shape[0]

    @property
    def num_elements(self) -> int:
        return self.weights.shape[1]


def dft_codebook(
    num_rows: int,
    num_cols: int,
    *,
    oversampling: tuple[int, int] = (1, 1),
    spacing: tuple[float, float] = (0.5, 0.5),
) -> Codebook:
    """DFT grid of beams for a ``num_rows`` x ``num_cols`` array (ULA if one row).

    The weights are Sionna PHY's ``grid_of_beams_dft_ula`` (one row) or
    ``grid_of_beams_dft`` (precoding.py:265, 332), unchanged; their element order is
    the array's (column by column). Beam (m_v, m_h) has index m_v * num_cols * O_h + m_h
    and points to

        u = nu(m_h, num_cols * O_h) / d_h,    v = -nu(m_v, num_rows * O_v) / d_v

    with nu(m, M) = m / M wrapped into [-1/2, 1/2]. The minus sign of v is the vertical
    sign fix: rows are numbered from the top, so Sionna's increasing phase over the rows
    points downwards. With B = N (no oversampling) and half-wavelength spacing the beams
    are orthonormal; with oversampling (O_v, O_h) they form a tight frame and the beam gains
    of a channel add up to O_v O_h ||h||^2. With spacing above half a wavelength the beams
    have grating lobes, and ``u``, ``v`` give the one closest to broadside; below half a
    wavelength, some beams point to |u| > 1, outside the visible region.

    :param oversampling: (vertical, horizontal) oversampling factors; the vertical one
        must be 1 for a single row
    :param spacing: (vertical, horizontal) element spacing [wavelengths]
    """
    o_v, o_h = (int(o) for o in oversampling)
    d_v, d_h = (float(d) for d in spacing)
    if num_rows < 1 or num_cols < 1 or o_v < 1 or o_h < 1 or d_v <= 0 or d_h <= 0:
        raise ValueError("array sizes and oversampling must be >= 1, spacings > 0")
    if num_rows == 1 and o_v != 1:
        raise ValueError("a single-row array (ULA) has no vertical beams; use oversampling (1, O)")
    if num_rows == 1:
        weights = grid_of_beams_dft_ula(num_cols, o_h)
    else:
        weights = grid_of_beams_dft(num_rows, num_cols, o_v, o_h).reshape(-1, num_rows * num_cols)
    m_v, m_h = torch.meshgrid(
        torch.arange(num_rows * o_v), torch.arange(num_cols * o_h), indexing="ij"
    )
    u = _wrap(m_h.reshape(-1) / (num_cols * o_h)) / d_h
    v = -_wrap(m_v.reshape(-1) / (num_rows * o_v)) / d_v
    return Codebook(
        weights=weights.to(torch.complex64),
        u=u.to(torch.float64),
        v=v.to(torch.float64),
        num_rows=num_rows,
        num_cols=num_cols,
        spacing=(d_v, d_h),
    )


def steering_vector(
    num_rows: int,
    num_cols: int,
    directions,
    *,
    spacing: tuple[float, float] = (0.5, 0.5),
) -> torch.Tensor:
    """Steering vectors exp(+j 2 pi r_n . k) of the array, [..., N] complex64.

    :param directions: [..., 3] unit vectors k in the array's local frame (boresight +x).
        From Sionna's zenith angle theta and azimuth phi:
        k = (sin theta cos phi, sin theta sin phi, cos theta).
    :param spacing: (vertical, horizontal) element spacing [wavelengths]
    """
    k = torch.as_tensor(directions, dtype=torch.float64)
    if k.shape[-1:] != (3,):
        raise ValueError(f"directions must have shape [..., 3], got {tuple(k.shape)}")
    y, z = _element_positions(num_rows, num_cols, spacing)
    phase = 2 * math.pi * (k[..., 1:2] * y + k[..., 2:3] * z)
    return torch.polar(torch.ones_like(phase), phase).to(torch.complex64)


def beam_gain(h, weights) -> torch.Tensor:
    """Beam gain |w_b^H h|^2 of every beam, [..., B], real.

    :param h: [..., N] channel per antenna (complex), e.g. [slots, prbs, N]; the antenna
        axis is last
    :param weights: a :class:`Codebook` or its [B, N] weights
    """
    if isinstance(weights, Codebook):
        weights = weights.weights
    h = torch.as_tensor(h)
    weights = torch.as_tensor(weights)
    if h.shape[-1] != weights.shape[-1]:
        raise ValueError(
            f"h has {h.shape[-1]} antennas on its last axis, the weights {weights.shape[-1]}"
        )
    dtype = torch.promote_types(torch.promote_types(h.dtype, weights.dtype), torch.complex64)
    return (h.to(dtype) @ weights.to(dtype).conj().transpose(0, 1)).abs().square()


def rsrp_dbm(beam_gain, tx_power_dbm: float, *, prb_axis: int = -2) -> torch.Tensor:
    """RSRP [dBm] of every beam: power per resource element + 10 log10(mean_PRB G_b).

    RSRP is the average received power per resource element of the reference signal
    (TS 38.215). Here the transmit power is spread uniformly over all 12 subcarriers of all
    PRBs, with no reference-signal power boosting, and the reference signal of beam b sees
    the beam gain G_b of each PRB.

    :param beam_gain: [..., num_prbs, B] absolute linear power gain per PRB and beam
        (path gain and array gain, e.g. from a link-budget trace and :func:`beam_gain`)
    :param tx_power_dbm: Total transmit power [dBm] over the band
    :param prb_axis: Axis of the PRBs in ``beam_gain``
    :output rsrp: [..., B] (the PRB axis removed), -inf for a zero gain
    """
    g = torch.as_tensor(beam_gain, dtype=torch.float64)
    num_prbs = g.shape[prb_axis]
    per_re_dbm = tx_power_dbm - 10 * math.log10(12 * num_prbs)
    return per_re_dbm + 10 * torch.log10(g.mean(dim=prb_axis))


def best_beam(values, dim: int = -1) -> torch.Tensor:
    """Index of the strongest beam, e.g. of beam gains or RSRPs, along ``dim``."""
    return torch.as_tensor(values).argmax(dim=dim)


def _wrap(nu: torch.Tensor) -> torch.Tensor:
    """m / M wrapped into [-1/2, 1/2] as in Sionna's grid of beams: m / M up to M / 2."""
    return torch.where(nu <= 0.5, nu, nu - 1.0)


def _element_positions(num_rows: int, num_cols: int, spacing) -> tuple[torch.Tensor, torch.Tensor]:
    """(y, z) positions [wavelengths] of the elements, in rt.PlanarArray's order."""
    d_v, d_h = spacing
    k = torch.arange(num_rows * num_cols, dtype=torch.float64)
    row, col = k % num_rows, torch.div(k, num_rows, rounding_mode="floor")
    return d_h * (col - (num_cols - 1) / 2), -d_v * (row - (num_rows - 1) / 2)
