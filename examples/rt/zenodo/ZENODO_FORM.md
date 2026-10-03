# Zenodo form: linkgym Munich ray-traced channel traces, v1

Values for the zenodo.org upload form ("New upload"). Steps: [RELEASING.md](../../../RELEASING.md),
"Dataset release". Files: the package built by `examples/rt/package_zenodo.py`.

## Basic information

| field | value |
|---|---|
| Digital Object Identifier | "Do you already have a DOI for this upload?" -> **No** -> **Get a DOI now!** |
| Resource type | Dataset |
| Title | linkgym Munich ray-traced channel traces, v1 (Sionna RT, 3.5 GHz, 52 PRBs) |
| Publication date | the day you publish (YYYY-MM-DD) |
| Creators | Rodrigues, Pedro (type: Personal; role: none needed; add ORCID and affiliation if you have them) |
| Description | the text below |
| Licenses | Open Data Commons Open Database License v1.0 (search "Open Database License"; SPDX `ODbL-1.0`) |
| Copyright | Map data (c) OpenStreetMap contributors; traces (c) 2026 Pedro Rodrigues |

Description:

> Ray-traced 5G NR channel traces along 15 streets around one rooftop base station in the
> "munich" scene of Sionna RT, made for linkgym, a Gymnasium environment for link
> adaptation (MCS selection). Each trace holds the linear power gain per slot (0.5 ms) and
> per PRB (52 PRBs at 30 kHz, 3.5 GHz carrier) of a single-antenna downlink, for a UE
> moving at 15 m/s along the street centres: 76 trajectories of 4000 slots (2 s), split by
> street into train (8 streets, 39 trajectories), val (3 streets, 14) and test (4 streets,
> 23), with route categories line-of-sight, transition and NLoS. munich-v1-test-alt.h5
> holds the 23 test trajectories traced again with another ray budget, to check results
> against the ray tracer's non-convergence.
>
> Generated with the linkgym trace generator on Sionna RT 2.1.0 (deterministic path solver:
> maximum depth 10, line of sight, specular reflection and diffraction including the lit
> region, refraction off, 4 million rays). The files are HDF5 (linkgym trace format v1) and
> carry the full generator configuration and provenance as attributes. README.md documents
> the content, the generation and the known limitations: the dataset is one deterministic
> ray-tracing realization of the scene, not a converged prediction.
>
> Load it with linkgym 0.2.0 or later, linkgym.datasets.fetch("munich-v1"), which downloads
> the files of this version and checks their SHA-256; or read it with h5py.
>
> Derived from OpenStreetMap data, (c) OpenStreetMap contributors, available under the Open
> Database License.

## Recommended information

| field | value |
|---|---|
| Keywords and subjects | 5G NR; link adaptation; MCS selection; ray tracing; Sionna RT; wireless channel; channel traces; reinforcement learning; OpenStreetMap; Munich; HDF5 |
| Languages | English |
| Dates | 2026-10-02, type "Created" (generation of the files) |
| Version | v1 |
| Publisher | Zenodo (default) |

## Related works

| relation | identifier | scheme | resource type |
|---|---|---|---|
| Is supplement to | https://github.com/Tempip/linkgym | URL | Software |
| Is documented by | https://github.com/Tempip/linkgym/blob/main/docs/channels.md#munich-dataset | URL | Other |
| Is derived from | https://www.openstreetmap.org | URL | Dataset |
| References | https://github.com/NVlabs/sionna-rt | URL | Software |

Once linkgym 0.2.0 has its software DOI, a metadata edit (no new version needed) can add
"Is supplement to" that DOI.

## Leave empty or default

Contributors, Funding, Alternate identifiers, References, Software section, Communities.
Visibility: Public; files: Public; no embargo.
