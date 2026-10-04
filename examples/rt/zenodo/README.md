# linkgym Munich ray-traced channel traces, v1

Ray-traced 5G NR channel traces along 15 streets around one rooftop base station in the
`munich` scene of Sionna RT: the linear power gain per slot and per PRB of a single-antenna
downlink, for UEs moving along the street centres. They were made for
[linkgym](https://github.com/Tempip/linkgym), a Gymnasium environment for link adaptation
(MCS selection), and are split by street into train, val and test, so that agents can be
trained on some streets and evaluated on others. A second ray-tracing realization of the
test streets, with another ray budget, lets results be checked against the ray tracer's
non-convergence.

- **Version DOI of this record:** [@DOI@](https://doi.org/@DOI@) (@RECORD_URL@)
- **Generator and documentation:** <https://github.com/Tempip/linkgym>, in particular
  [docs/channels.md](https://github.com/Tempip/linkgym/blob/main/docs/channels.md#munich-dataset)
- **License:** Open Database License (ODbL) 1.0; derived from OpenStreetMap data,
  (c) OpenStreetMap contributors (see License)

## Files

| file | content | size |
|---|---|---:|
| `munich-v1.h5` | 76 trajectories of 4000 slots x 52 PRBs: train 39, val 14, test 23 | 67,002,864 bytes |
| `munich-v1-test-alt.h5` | the 23 test trajectories traced again with 8 million rays and a 1e6 path buffer | 20,283,984 bytes |
| `munich.json` | the generator configuration of `munich-v1.h5` | |
| `munich_routes.png` | top view of the scene, the base station and the routes by split | |
| `README.md` | this file | |
| `LICENSE` | the full text of the Open Database License 1.0 | |
| `NOTICE` | license notice and OpenStreetMap attribution | |
| `SHA256SUMS` | SHA-256 of every other file (`sha256sum -c SHA256SUMS`) | |

## What it is

- **Scene:** Sionna RT 2.1's built-in `munich` scene, about 1.5 x 1.2 km around the
  Frauenkirche, built from OpenStreetMap data (buildings as extruded blocks with ITU radio
  materials). Its SHA-256 over the XML and every referenced file is stored in the traces
  (`scene_sha256`, `d0b76661...`).
- **Base station:** one transmitter at (116.5, 80.5, 23.4) m in scene coordinates, 4 m above
  a 19.4 m roof at its west edge, facing the Marienhof square; isotropic, vertically
  polarized. The site was chosen for line-of-sight coverage; the method is in
  docs/channels.md.
- **UE:** 1.5 m above ground, isotropic, vertically polarized, moving at 15 m/s along the
  street centres (at least 1.6 m from any surface). Some of the streets are pedestrian
  zones; the speed matches linkgym's default scenario, not traffic.
- **Grid:** carrier 3.5 GHz, subcarrier spacing 30 kHz, 52 PRBs (18.72 MHz), slots of
  0.5 ms; the gain of a PRB is |h|^2 at its centre frequency.
- **Trajectories:** each route is cut into trajectories of 4000 slots (2 s, 30 m). Paths are
  solved every 10 slots (7.5 cm); in between, the channel evolves with each path's Doppler
  shift.
- **Gain:** the linear power gain of the single-antenna channel, the absolute path gain
  including the (0 dBi) antenna gains, not normalized.

| split | streets | trajectories kept (configured) | LoS / transition / NLoS trajectories | LoS share |
|---|---:|---:|---|---:|
| train | 8 | 39 (58) | 8 / 13 / 18 | 0.29 |
| val | 3 | 14 (14) | 3 / 6 / 5 | 0.32 |
| test | 4 | 23 (29) | 5 / 10 / 8 | 0.29 |

No street is in two splits, and no route comes within 10 m of a route of another split.
The LoS share is the share of path solves with a line-of-sight path; a route is `nlos` up
to 0.05, `los` from 0.75 and `transition` in between. Per route:

| route | split | category | trajectories kept (dropped) | LoS share | mean path gain |
|---|---|---|---:|---:|---:|
| north-canyon | train | los | 8 (0) | 0.93 | -80.7 dB |
| east-avenue | train | nlos | 9 (0) | 0.00 | -109.4 dB |
| se-avenue | train | nlos | 3 (7) | 0.00 | -129.4 dB |
| canyon-cross | train | transition | 5 (0) | 0.12 | -88.3 dB |
| ne-north-south | train | nlos | 3 (1) | 0.00 | -145.4 dB |
| sw-avenue | train | nlos | 2 (4) | 0.00 | -121.8 dB |
| kaufinger | train | nlos | 1 (7) | 0.00 | -130.4 dB |
| rathaus-west | train | transition | 8 (0) | 0.41 | -86.6 dB |
| north-street | val | nlos | 5 (0) | 0.00 | -111.0 dB |
| nw-avenue | val | transition | 6 (0) | 0.25 | -93.6 dB |
| nw-diagonal | val | los | 3 (0) | 1.00 | -85.9 dB |
| long-diagonal | test | nlos | 7 (0) | 0.00 | -108.1 dB |
| west-street | test | transition | 10 (0) | 0.21 | -90.0 dB |
| south-street | test | los | 5 (0) | 0.90 | -82.3 dB |
| south-curve | test | nlos | 1 (6) | 0.00 | -126.7 dB |

The mean path gain is 10 log10 of the mean linear gain of the route's kept trajectories.

## File format

HDF5, linkgym trace format version 1 (full specification in
[docs/channels.md](https://github.com/Tempip/linkgym/blob/main/docs/channels.md#trace-format-version-1)).

| dataset | type, shape | content |
|---|---|---|
| `gain` | float32 [T, 4000, 52] | linear power gain per trajectory, slot and PRB |
| `split` | str [T] | `train`, `val` or `test` |
| `group` | int32 [T] | the route (street) of each trajectory |
| `category` | str [T] | route category: `los`, `transition` or `nlos` |
| `rx_position` | float32 [T, 4000, 3] | UE position [m] in scene coordinates |
| `num_paths` | int16 [T, 400] | number of paths at each path solve |
| `los` | int8 [T, 400] | 1 if a line-of-sight path exists at the path solve |

Root attributes describe the grid (`carrier_frequency_hz`, `subcarrier_spacing_hz`,
`slot_duration_s`, `num_prb`, `prb_sampling`) and the provenance: `scene`, `scene_sha256`,
`attribution`, the radio devices, `solver` (all path solver arguments), `routes` (per route:
split, category, waypoints, kept and dropped trajectories with the reason), `versions` and
the full `generator_config`.

## Loading

With linkgym 0.2.0 or later, which downloads the files of this record and checks their
SHA-256:

```python
import gymnasium as gym

from linkgym.datasets import fetch

path = fetch("munich-v1")  # or "munich-v1-test-alt"
env = gym.make(
    "linkgym/LinkAdaptation-v0", channel="trace", trace_path=str(path), trace_splits=["train"]
)
```

Without linkgym, with h5py:

```python
import h5py

with h5py.File("munich-v1.h5", "r") as f:
    split = f["split"][()].astype(str)  # [76]
    gain = f["gain"][0]  # [4000, 52] linear power gain of trajectory 0
```

## How it was generated

With the linkgym trace generator (`linkgym-traces generate`, Sionna RT), at commit
`7087409` of <https://github.com/Tempip/linkgym>; the generator is unchanged in the linkgym
0.2.0 release. The `versions` attribute of the files reads `linkgym 0.1.0` because the
generator was not released yet when they were made.

- **Software:** sionna-rt 2.1.0, mitsuba 3.9.1 (variant `cuda_ad_mono_polarized`), drjit
  1.5.0, numpy 2.5.3, h5py 3.16.0, Python 3.12.14; an NVIDIA RTX 3060 (12 GB) on Windows 11.
- **Path solver** (Sionna RT `PathSolver`): deterministic; `max_depth` 10; line of sight,
  specular reflection and diffraction including the lit region on; refraction, diffuse
  reflection and edge diffraction off; `samples_per_src` 4,000,000;
  `max_num_paths_per_src` 10,000,000; `seed` 42; synthetic array.
- **Dropped trajectories:** a trajectory with a path solve without any path, or with a mean
  gain below -150 dB, was dropped: 25 of the 101 configured trajectories, all on NLoS
  streets far from the base station. The reason and mean gain of each are in the `routes`
  attribute.
- **Commands:**

  ```bash
  linkgym-traces generate munich.json -o munich-v1.h5
  linkgym-traces generate munich.json -o munich-v1-test-alt.h5 --splits test --solver samples_per_src=8000000 max_num_paths_per_src=1000000
  ```

  74 and 27 minutes on the RTX 3060. The frequency response is summed in float64 with the
  paths in a fixed order, so generating the same configuration twice with the same versions
  gives byte-identical files. Other GPUs, drivers or library versions may give slightly
  different gains.

## Known limitations

- **One realization, not a converged prediction.** The ray tracer samples rays, and more
  rays find more weak paths; the channel does not converge within budgets that fit a 12 GB
  GPU. Against 16 million rays, on 11 check stretches, the EESM effective SINR differed by
  0.39 dB in the median and 1.9 dB at the 95th percentile, and the mean gain of a stretch by
  0.4 dB in the median and 9.1 dB at most. The strongest paths (line of sight, first
  reflections) are stable; the per-PRB fading detail, the frequency and time correlations
  and the tail of the effective SINR depend on the ray budget. The cause is diffraction in
  the lit region on finely meshed building edges. Treat the data as one deterministic
  realization of the scene, and check conclusions on `munich-v1-test-alt.h5`.
- **The two test realizations** differ in the mean gain of a trajectory by 0.12 dB in the
  median and 6.1 dB at most (on `long-diagonal`), and in the normalized effective SINR per
  slot by 2.6 dB at the 95th percentile of a typical trajectory (42 dB on the one kept
  `south-curve` trajectory, in its deep fades); NLoS streets differ most.
- **Refraction is off.** Sionna RT models surfaces as thin slabs; the OpenStreetMap
  buildings are solid blocks, so with refraction on, rays cross whole buildings through
  two 10 cm walls; on check stretches this changed NLoS and transition levels by -11 to
  +35 dB, mostly upwards, and made them depend on the ray budget.
- **Anchor spacing.** Between path solves, a path that appears or disappears (at a shadow
  boundary) is picked up at the next solve: a few slots around abrupt changes have errors of
  several dB; elsewhere the anchored channel follows a per-slot solve closely (NMSE about
  -18.5 dB on NLoS and transition stretches with diffraction).
- **Coverage.** Far NLoS streets have stretches with almost no signal. They are kept unless
  a trajectory's mean gain is below -150 dB, and deep fades inside kept trajectories stay
  as they are. Normalizing a trajectory by its mean linear gain (linkgym's default SNR mode)
  then puts its weak stretches far below the drawn SNR, into outage: 9% of the slots of the
  linkgym v0.2 test episodes.
- **Scope.** One scene, one base station, single isotropic antennas, no interference, no
  vegetation or vehicles; buildings are extruded footprints.

## Versions and DOIs

Zenodo gives every version of a dataset its own **version DOI**, and the dataset as a
whole a **concept DOI** that always resolves to the latest version.

- **This record is version v1**, DOI [@DOI@](https://doi.org/@DOI@). Its files never change.
  Cite this DOI for results, and download from this record: linkgym's
  `fetch("munich-v1")` uses this record's file URLs and checks each file's SHA-256 (the
  values in `SHA256SUMS`).
- **The concept DOI** is shown on the record page ("Cite all versions"). Use it only to
  refer to the dataset in general. A later version can have different files and give
  different results.
- **A new version** (other routes, solver settings or scene) will be a new record with its
  own DOI, and a new name in linkgym (for example `munich-v2`); `munich-v1` keeps pointing
  to this record.

## Citation

```bibtex
@dataset{rodrigues2026munich,
  author    = {Rodrigues Souza, Pedro José},
  title     = {linkgym Munich ray-traced channel traces, v1},
  year      = {2026},
  version   = {v1},
  publisher = {Zenodo},
  doi       = {@DOI@}
}
```

Please also cite linkgym (<https://github.com/Tempip/linkgym>, "Cite this repository") and
Sionna RT if you use them.

## License

This database is made available under the Open Database License (ODbL) 1.0: the full text
is in `LICENSE`, and at <https://opendatacommons.org/licenses/odbl/1-0/>. It derives from
the `munich` scene of Sionna RT, which was built from OpenStreetMap data, (c) OpenStreetMap
contributors, available under the ODbL (<https://www.openstreetmap.org/copyright>). See
`NOTICE` for the attribution to keep when you use or share the data or works produced from
it.

linkgym is not affiliated with or endorsed by NVIDIA. Sionna is a trademark of NVIDIA
Corporation.
