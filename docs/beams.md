# Antenna arrays and beams: `linkgym.beams`

Since 0.3.0.

Status: **experimental** (see [api_stability.md](api_stability.md)): the names and
conventions may still change in 0.5.

Utilities for beam management on a planar antenna array: a DFT codebook, steering vectors,
the gain of each beam on a channel, the RSRP of each beam and the best beam. They work on
any channel given per antenna (PyTorch tensors, antenna axis last), for example one
reconstructed from ray-traced paths.

| name | what it does |
|---|---|
| `dft_codebook(num_rows, num_cols, *, oversampling=(1, 1), spacing=(0.5, 0.5))` | `Codebook(weights [B, N], u [B], v [B], ...)`: DFT grid of beams and the direction each beam points to |
| `steering_vector(num_rows, num_cols, directions, *, spacing)` | exp(+j 2π r_n·k) for unit vectors k [..., 3]: the channel of a plane wave from k, [..., N] |
| `beam_gain(h, weights)` | \|w_bᴴ h\|² for every beam: [..., N] → [..., B] |
| `rsrp_dbm(beam_gain, tx_power_dbm, *, prb_axis=-2)` | RSRP of every beam [dBm]: [..., num_prbs, B] → [..., B] |
| `best_beam(values, dim=-1)` | index of the strongest beam |

## Conventions

They match Sionna RT, so a channel from Sionna RT, or reconstructed from its paths, can be
used directly. An `rt`-marked test checks them against `rt.PlanarArray` and the per-antenna
coefficients of a synthetic-array path solve.

- **Array.** `num_rows` x `num_cols` elements in the y-z plane of the array's local frame,
  boresight along +x. Element k is in row `k % num_rows` (row 0 at the top) and column
  `k // num_rows`: numbered column by column from the top left, as `rt.PlanarArray`. A ULA
  is one row (`num_rows=1`), along y.
- **Positions** in wavelengths: y = d_h (column − (num_cols − 1)/2), z = −d_v (row −
  (num_rows − 1)/2), with the spacings `spacing = (d_v, d_h)` (default half a wavelength).
- **Phase.** A plane wave leaving or arriving along the unit vector k reaches element n with
  the phase exp(+j 2π r_n·k): that is its steering vector, the phase Sionna RT's synthetic
  arrays apply to each antenna. From Sionna's zenith angle θ and azimuth φ,
  k = (sin θ cos φ, sin θ sin φ, cos θ), in the array's local frame.
- **Beam gain** of weights w on a channel h: \|wᴴh\|². A beam pointing exactly at a plane
  wave has the full array gain N (+15 dB for 32 elements).

## The DFT codebook

The weights are Sionna PHY's `grid_of_beams_dft_ula` (one row) or `grid_of_beams_dft`,
unchanged. Beam (m_v, m_h) has index m_v · num_cols · O_h + m_h and points to the direction
cosines

    u = ν(m_h, num_cols · O_h) / d_h  (along y),    v = −ν(m_v, num_rows · O_v) / d_v  (along z)

with ν(m, M) = m/M wrapped into [−1/2, 1/2]. The minus sign on v is needed because rows are
numbered from the top: Sionna's increasing phase over the rows points downwards. A beam
whose (u, v) has u² + v² > 1 points outside the visible region; no plane wave comes from
there (8 of the 32 beams of a 4 x 8 UPA).

- **Without oversampling** and with half-wavelength spacing, the N beams are orthonormal: the
  gains of all beams add up to ‖h‖².
- **With oversampling** (O_v, O_h), there are O_v·O_h·N beams on a finer grid. They are no
  longer orthogonal, and the gains add up to O_v·O_h·‖h‖².
- **Spacing above half a wavelength** gives each beam grating lobes, and (u, v) is the one
  closest to broadside.

**Front/back ambiguity.** The elements lie in the y-z plane, so the phase depends only on
k_y and k_z. A direction and its mirror image behind the array, (−k_x, k_y, k_z), have the
same steering vector. With isotropic elements, every beam also points backwards, and a beam
gain cannot tell a user in front of the array from one behind it. Real arrays break the tie
with directional elements (e.g. Sionna RT's `"tr38901"` pattern) or a ground plane; when you
place an array in a scene, orient it towards the area it serves.

## RSRP

RSRP (TS 38.215) is the average received power per resource element of a reference signal,
for example the SSB of each beam. `rsrp_dbm` assumes the total transmit power is spread
uniformly over all 12 subcarriers of all PRBs, with no reference-signal power boosting:

    RSRP_b [dBm] = tx_power_dbm − 10 log10(12 · num_prbs) + 10 log10(mean over PRBs of G_b)

`G_b` must be the **absolute** linear power gain (path gain and array gain), as in a trace
read with `snr_mode="link_budget"`. With `"normalized"` traces, each trajectory is scaled
separately, and RSRPs of different users are not comparable.

## Example

```python
import torch

from linkgym import beams

codebook = beams.dft_codebook(4, 8)  # UPA with 4 rows and 8 columns: 32 beams
k = torch.tensor([0.8, 0.36, 0.48], dtype=torch.float64)  # unit vector, array's frame
h = beams.steering_vector(4, 8, k)  # channel per antenna of a plane wave from k, [32]
gains = beams.beam_gain(h, codebook)  # [32]
best = int(beams.best_beam(gains))
print(best, float(codebook.u[best]), float(codebook.v[best]))  # the beam closest to k
```
