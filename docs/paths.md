# Path traces and antenna arrays: `linkgym.paths`

Since 0.4.0.

Status: **experimental** (see [api_stability.md](api_stability.md)): the names may still
change in 0.5. The file format has its own version number (2) and stays readable.

A format-1 trace ([channels.md](channels.md#trace-format-version-1)) stores the gain of one
fixed antenna configuration. A **path trace** (trace format 2, `channel_kind = "paths"`)
stores the ray-traced propagation paths themselves, at every anchor of every trajectory.
From them linkgym rebuilds, at load time, the channel of every antenna of any transmit
array: its size, spacing and orientation, and a codebook, are chosen without tracing again.

| name | what it does |
|---|---|
| `write_path_trace(path, *, split, path_offset, a, tau, doppler, k_tx, k_rx, ...)` | writes a path trace and validates it by reading it back |
| `inspect_path_trace(path, *, deep=False)` | validates a file and returns `PathTraceInfo` (shape, splits, groups, categories, `mean_gain`, attributes); `deep=True` also recomputes the mean gains |
| `read_path_window(path, trajectory, offset, num_slots)` | the paths that cover a window of slots, `PathWindow` |
| `reconstruct_cfr(window, *, num_rows=1, num_cols=1, spacing=(0.5, 0.5), orientation=(0, 0, 0), dtype=torch.complex64)` | the channel of every antenna, [slots, PRBs, antennas] |
| `rotation_matrix(orientation)` | Sionna RT's rotation (α, β, γ) as a [3, 3] float64 tensor |
| `ArrayChannelSource`, `ArrayChannelEpisode` | the interface of per-antenna channel sources: `ChannelSource` for arrays |
| `PathTraceSource(path, *, ..., num_rows, num_cols, spacing, orientation, codebook)` | per-antenna channels of random or pinned windows of a path trace |

## How the channel of an array is rebuilt

Sionna RT's synthetic arrays trace the paths once, from the centre of the array. Each
antenna then only adds a phase to each path: antenna n at position r_n (in wavelengths)
multiplies path p by exp(+j 2π r_n·k_tx,p), where k_tx,p is the path's unit direction of
departure. The amplitude, delay and Doppler shift of a path are the same at every antenna.
A single-antenna trace that keeps each path's coefficient, delay, Doppler shift and
direction therefore holds everything Sionna RT uses to build any synthetic array with the
same element.

For slot s of anchor j (t seconds after the anchor's path solve), PRB centre frequency f
(relative to the carrier f_c) and antenna n, `reconstruct_cfr` computes

    h[s, f, n] = Σ_p a_p · exp(−j 2π (f_c + f) τ_p) · exp(j 2π f_D,p t) · exp(j 2π r_n·k_tx,p)

summed over the paths of anchor j, with all phases in float64. This is the model of the
format-1 generator ([channels.md](channels.md#what-the-generator-does)) with the array
phase added:

- **Antennas** are in `rt.PlanarArray` order, column by column, with the positions and the
  phase convention of [beams.md](beams.md#conventions), so `linkgym.beams` codebooks and
  beam gains apply directly.
- **Orientation** (α, β, γ), in radians, rotates the antenna positions about z, y and x as
  Sionna RT does (TR 38.901, `rotation_matrix`). `PathTraceSource` defaults to the
  orientation the transmitter was traced with (attribute `tx_orientation`).
- **A 1x1 array** gives the channel whose power is format 1's gain. On the committed sample,
  the 1x1 reconstruction matches the gains of `munich_sample.h5` to a relative 1e-5.
- **Checked against Sionna RT.** An `rt`-marked test traces 4x8 and 1x32 arrays with
  Sionna RT itself, with the transmitter at three orientations (including pitch and roll),
  on a line-of-sight and a non-line-of-sight receiver. The reconstruction from the
  single-antenna paths matches Sionna RT's per-antenna coefficients to a few 1e-6 of the
  largest antenna response, the precision of Sionna's float32. Ignoring the orientation
  gives errors of order 1.

## What can and cannot change at load time

| can change at load time | fixed at generation (recorded in the attributes) |
|---|---|
| array size: rows and columns | element pattern and polarization, at both ends (`tx_antenna`, `rx_antenna`; isotropic, vertical by default) |
| element spacing, in wavelengths | carrier frequency: the coefficients, and the wavelength that scales the array |
| orientation of the array: positions are exact for any rotation | the element response under the orientation: it is the traced one. With isotropic, vertically polarized elements it is exact for any rotation about z (yaw); for pitch or roll it is an approximation, unless the transmitter was traced with that orientation |
| codebook (anything in `linkgym.beams`) | scene, transmitter position, receiver routes, solver settings |
| SNR mode and transmit power | the receiver array: one antenna (`k_rx` is stored for receive arrays later) |
| windows, splits and pins, as in format 1 | PRB grid, slot duration and anchor spacing, which must match the scenario |

Dual-polarized arrays and 2x2 polarization are not supported: every path has one complex
coefficient, for the traced polarization.

## Trace format, version 2

An HDF5 file with T trajectories of N slots, solved at A = N / `anchor_spacing_slots`
anchors each. Write it with `write_path_trace`; `inspect_path_trace` validates it.

Root attributes:

| attribute | type | value |
|---|---|---|
| `format` | str | `"linkgym-trace"` |
| `format_version` | int | `2` |
| `channel_kind` | str | `"paths"` |
| `carrier_frequency_hz`, `subcarrier_spacing_hz`, `slot_duration_s` | float | as in format 1 |
| `num_prb` | int | number of PRBs |
| `prb_sampling` | str | `"center"`: the channel is rebuilt at the centre of each PRB |
| `num_slots` | int | N, slots per trajectory |
| `anchor_spacing_slots` | int | slots between path solves; divides N |
| `synthetic_array` | bool | `true`: paths traced once per device, as Sionna RT's synthetic arrays |
| `tx_antenna`, `rx_antenna` | str | JSON with the `pattern` and `polarization` the paths were traced with |
| `tx_orientation` | float64 [3] | orientation (α, β, γ) [rad] of the traced transmitter |

Other attributes are kept in `PathTraceInfo.attrs`. The generator writes the same
provenance as for format 1 (scene and its SHA-256, solver, versions, routes, attribution,
configuration); see [channels.md](channels.md#attributes-written-by-the-generator).

Datasets, per trajectory:

| dataset | type, shape | required | content |
|---|---|---|---|
| `split` | str [T] | yes | split label of each trajectory |
| `path_offset` | int64 [T, A + 1] | yes | the paths of trajectory t, anchor j are `path_offset[t, j]` to `path_offset[t, j + 1]` of the path datasets; `path_offset[t, A] == path_offset[t + 1, 0]` |
| `num_paths` | int32 [T, A] | yes | paths of each anchor, `diff(path_offset)` |
| `mean_gain` | float64 [T] | yes | mean single-antenna gain over all slots and PRBs, computed as format 1 does from float32 gains: the scale of `normalized` mode |
| `group`, `category`, `rx_position`, `los` | as in format 1 | no | group (route), route category, receiver positions [T, N, 3], line of sight per anchor [T, A] |

Datasets, per path (P paths in all, trajectory by trajectory, anchors in order):

| dataset | type, shape | content |
|---|---|---|
| `a` | complex64 [P] | path coefficient, as Sionna RT's `Paths.a` (element patterns included, before the carrier phase) |
| `tau` | float32 [P] | delay [s] |
| `doppler` | float32 [P] | Doppler shift [Hz] |
| `k_tx` | float32 [P, 3] | unit direction of departure, scene frame |
| `k_rx` | float32 [P, 3] | unit direction of arrival, scene frame |

A path takes 40 bytes. All datasets are contiguous and uncompressed, so the paths of a
window are one read per dataset.

**Validation.** Shapes and dtypes; offsets that start at 0, never decrease, run on from one
trajectory to the next and match `num_paths`; finite values, delays ≥ 0, direction vectors
of unit norm (to 1e-4); `num_slots` a multiple of the anchor spacing; categories among
`los`, `nlos`, `transition`; mean gains finite and ≥ 0. `inspect_path_trace(path,
deep=True)` also rebuilds every trajectory and compares its mean gain with `mean_gain`.
Format-1 readers (`inspect_trace`, `read_trace`, `channel="trace"`) reject path traces with
their "unsupported format_version" error.

**Precision and reproducibility.** float32 is Sionna RT's own precision, so storing the
paths loses nothing. The paths of each anchor are stored in the generator's fixed order
(by delay, coefficient and Doppler shift), so sums are reproducible, and generating the
same configuration twice gives byte-identical files.

## Generating path traces

```bash
linkgym-traces generate config.json -o paths.h5 --store-paths
```

The generator runs exactly as for format 1 ([channels.md](channels.md#generating-traces-with-sionna-rt)):
the same path solves, the same dropped trajectories, the same attributes. Instead of the
gains, it stores the valid paths of every anchor of the kept trajectories. It takes the
same time and GPU memory; only the file is larger, 40 bytes per path. `prb_sampling` must
be `"center"`.

`tests/data/munich_paths_sample.h5` (0.7 MB, ODbL, like the Munich dataset) holds the paths
of the first 100 slots of the four trajectories of `munich_sample.h5`, solved again with the
dataset's configuration by `examples/rt/make_paths_sample.py`. That script checks that the
gains, path counts and line-of-sight flags equal those of the format-1 sample, bit for bit.

### The Munich paths dataset

Version v2 of the Munich dataset ([channels.md](channels.md#munich-dataset)), DOI
[10.5281/zenodo.23267742](https://doi.org/10.5281/zenodo.23267742), holds the paths of all
its trajectories, generated with `--store-paths`:

| name | content | size |
|---|---|---:|
| `munich-v2` | 76 trajectories of 4000 slots, 400 path solves each: 7,836,232 paths | 318 MB |
| `munich-v2-test-alt` | the 23 test trajectories of the second ray-tracing realization: 1,807,394 paths | 74 MB |

`linkgym.datasets.fetch("munich-v2")` downloads a file once and checks its SHA-256; pass the
path to `PathTraceSource` (below). The single-antenna gains rebuilt from these files are
identical to those of `munich-v1` and `munich-v1-test-alt` up to float32 rounding (3 of
15.8 M gains differ by one unit in the last place); the splits, routes, positions and mean
gains are identical.

### Dependency on Sionna RT internals

Sionna RT 2.1.0 keeps the direction vectors of the paths only in the private attributes
`Paths._k_tx` and `Paths._k_rx` (`sionna/rt/path_solvers/paths.py`, lines 925-926). They
are what its own synthetic arrays use (lines 957-975), where Sionna notes that rebuilding
them from the public angles is ill-conditioned near the poles. The generator reads them
there:

- linkgym pins `sionna-rt==2.1.0` exactly, in the `rt` extra;
- if the attributes are missing or have an unexpected shape, the generator stops with an
  error that names the installed version;
- the `rt` test above compares the reconstruction with Sionna RT's own synthetic-array
  output, so a change in Sionna RT shows up as a failing test before a release.

Reading path traces needs neither Sionna RT nor a GPU.

## Using a path trace: `PathTraceSource`

`PathTraceSource` is the `ArrayChannelSource` counterpart of `TraceChannelSource`. It has
the same checks of the file against the scenario, the same random draws of trajectories and
windows for a given seed (on a file with the same splits and number of slots), the same
pins (`trajectories`, `offsets`) and the same episode info. Each link is the single link of
one trajectory. `generate` returns an `ArrayChannelEpisode`:

- `channel`: [batch, slots, PRBs, antennas] complex64;
- `beam_gain`: [batch, slots, PRBs, beams] float32, |wᴴh|² for every beam of the codebook,
  if one was given (its array must be the source's);
- `reference_snr_db` and `info`, as in `ChannelEpisode`.

SNR modes:

- `"normalized"`: each trajectory's channel is divided by the square root of its
  `mean_gain`. A single antenna then has unit mean power, as in format 1, and a beam up to
  `num_antennas` times more (the array gain).
- `"link_budget"`: absolute channels, and the reference SNR of the link budget for the
  whole transmit power; a unit-norm beam keeps that power.

```python
from linkgym.beams import best_beam, dft_codebook
from linkgym.paths import PathTraceSource

codebook = dft_codebook(4, 8)  # 4 x 8 planar array, 32 beams
source = PathTraceSource(
    "tests/data/munich_paths_sample.h5",  # 4 trajectories of 100 slots
    num_prbs=52,
    subcarrier_spacing=30e3,
    carrier_frequency=3.5e9,
    slot_duration=0.5e-3,
    num_slots=50,
    splits=("train",),
    num_rows=4,
    num_cols=8,
    codebook=codebook,
)
episode = source.generate(num_slots=50, batch_size=2, seed=0)
print(episode.channel.shape)  # [2 links, 50 slots, 52 PRBs, 32 antennas]
gain = episode.beam_gain.mean(dim=(1, 2))  # mean gain of each beam, [2, 32]
print(episode.info["trajectory"], best_beam(gain).tolist())
```

`linkgym.evaluation.trace_episodes` also lists the episodes of a path trace; pass each
episode's `options["trajectory"]` and `options["offset"]` to `generate` as `trajectories`
and `offsets` to pin it.

The reconstruction runs on the CPU in PyTorch. On the densest trajectory of the Munich
sample (about 1200 paths per anchor), its 100 slots for a 4x8 array take about 14 ms with 8
threads; the cost grows with the number of slots, PRBs, antennas and paths.

## Limitations

- The environment (`channel="trace"`) does not read path traces yet; use
  `PathTraceSource` directly, for example to study beam selection. Caching beam gains for
  the environment and several users per trace are planned for 0.5.
- One receive antenna, one polarization, and the element pattern traced at generation (see
  the table above).
- Plane waves across the array: the array is small compared with the distances to the
  scatterers, as in Sionna RT's synthetic arrays.
- Between anchors, the paths, their amplitudes and their delays are those of the anchor;
  only the Doppler phase evolves (as in format 1).
