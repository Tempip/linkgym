# Channels

The channel of an episode is a linear power gain |h|^2 per slot and per PRB. The
simulator turns it into the per-PRB SINR

```
sinr = 10 ** (snr_db / 10) * gain
```

where `snr_db` is the scenario SNR, or a reference SNR set by the channel source (link
budget, below). Two sources are built in, chosen with the `channel` scenario field:

- `channel="tdl"` (default): 3GPP TR 38.901 TDL fading generated with Sionna, as in
  v0.1.
- `channel="trace"`: gains read from an HDF5 trace file (`trace_path`), for example
  computed offline by ray tracing along UE trajectories, or measured.

## Channel source interface

`linkgym.channels.ChannelSource` is the extension point for other channel models. It is
a stable interface. A source has:

- `num_prbs`: the number of PRBs of the gains it returns;
- `generate(num_slots, batch_size, seed) -> ChannelEpisode`.

`ChannelEpisode` has three fields:

| field | content |
|---|---|
| `gain` | float32 tensor [batch_size, num_slots, num_prbs], always the linear power gain, finite and non-negative |
| `reference_snr_db` | `None`: the simulator uses the scenario SNR, and the gain should have unit mean. A float, or a [batch_size] array: the simulator uses that SNR instead. |
| `info` | metadata, one JSON-serializable list entry per link, e.g. the trajectory index |

`generate` must depend only on its arguments: use `seed` for all randomness and do not
read or modify global random state. A source can be passed to
`linkgym.sim.LinkSimulator(..., channel_source=source)`. The environment builds its
source from the scenario; to use your own channel in the environment, write it to a
trace file (see [Bring your own channel](#bring-your-own-channel)).

`linkgym.sim.TDLChannelGain` is the TDL source: the power delay profile is normalized,
so the gain has unit mean and the scenario SNR is the mean SNR. It samples the channel
at one frequency per PRB, spaced by the PRB bandwidth, once per slot.

## Trace format, version 1

A trace is an HDF5 file holding T trajectories of N consecutive slots each. Write it
with `linkgym.channels.write_trace` and read it with `read_trace`; both validate it.

Root attributes:

| attribute | type | value |
|---|---|---|
| `format` | str | `"linkgym-trace"` |
| `format_version` | int | `1` |
| `gain_unit` | str | `"linear_power_gain"` |
| `carrier_frequency_hz` | float | carrier frequency [Hz] |
| `subcarrier_spacing_hz` | float | subcarrier spacing [Hz] |
| `slot_duration_s` | float | time between consecutive slots [s] (0.5e-3 at 30 kHz) |
| `num_prb` | int | number of PRBs, equal to the last dimension of `gain` |
| `prb_sampling` | str | `"center"` (one frequency point per PRB) or `"mean12"` (mean over the 12 subcarriers of the PRB) |

Other attributes (for example the scene or the generator versions) are allowed and kept
in `TraceData.attrs`.

Datasets:

| dataset | type, shape | required | content |
|---|---|---|---|
| `gain` | float32 [T, N, num_prb] | yes | linear power gain of the single-layer channel per trajectory, slot and PRB, including antenna and beamforming gains |
| `split` | str [T] | yes | split label of each trajectory, e.g. `"train"`, `"val"`, `"test"` |
| `group` | int [T] | no | group of each trajectory, e.g. the street it runs along |
| `rx_position` | float32 [T, N, 3] | no | UE position [m] |
| `rx_velocity` | float32 [T, 3] | no | UE velocity [m/s] |
| `num_paths` | int [T, A] | no | number of propagation paths at A points of each trajectory |

The environment reads `gain`, `split` and `group`; the other optional datasets are for
analysis.

### Validation

A file is rejected with `TraceFormatError` (a `ValueError`, its message starts with the
file path) if:

- it is not an HDF5 file, `format` is not `"linkgym-trace"` or `format_version` is not 1;
- a required attribute is missing, `gain_unit` or `prb_sampling` has another value, or
  `carrier_frequency_hz`, `subcarrier_spacing_hz` or `slot_duration_s` is not a positive,
  finite number;
- `gain` or `split` is missing, `gain` is not float32 with three dimensions, is empty,
  has a PRB count other than `num_prb`, or holds NaN, infinite or negative values;
- `split` does not hold one non-empty string per trajectory;
- an optional dataset has the wrong shape.

With `channel="trace"`, the environment also requires the trace to match the scenario:
same `num_prbs`, `carrier_frequency` and `subcarrier_spacing`, and a slot duration equal
to that of the numerology (relative tolerance 1e-9); trajectories at least
`episode_length` slots long; and every label in `trace_splits` present in the file.
Gains are not resampled in time or frequency.

## Selection and splits

At each `reset()`, the environment draws a trajectory uniformly, with replacement, among
those whose split label is in `trace_splits` (default `("train",)`), and a start slot
uniformly among the `N - episode_length + 1` that fit. The episode is that window. Both
draws come from the channel seed, which derives from the `reset()` seed: the same seed
gives the same trajectory and window, and every policy evaluated with the same seeds
faces the same channels.

Splits are per trajectory, so the slots of one trajectory are never in two splits. The
trace generator assigns the labels; for example, all trajectories of one street go to the
same split, recorded in `group`. Seeds and splits are independent: train on
`trace_splits=["train"]`, select on `["val"]` and report on `["test"]`.

`env.unwrapped.channel_info` describes the current episode: `trajectory`, `offset` (start
slot), `split`, `group` (if the file has it) and `realized_snr_db`, the mean SNR of the
episode's slots (10 log10 of the mean linear per-PRB SINR). With the TDL channel it holds
only `realized_snr_db`.

## SNR modes

### `snr_mode="normalized"` (default)

Each trajectory is divided by its mean gain over all its slots and PRBs. The scenario
SNR (`snr_db`, or a draw from `snr_db_range`) is then the mean SNR over the whole
trajectory, and `info["snr_db"]` reports it. The variation within the trajectory, such
as the path loss changing along a street, is kept, so the mean SNR of a window differs
from the scenario SNR; `channel_info["realized_snr_db"]` holds it.

### `snr_mode="link_budget"`

The gains are used as they are, with the SNR at unit gain from a link budget:

```
reference_snr_db = tx_power_dbm - (10 log10(k T0 B) + 30) - noise_figure_db
```

with k Boltzmann's constant, T0 = 290 K and B = `num_prbs` x 12 x `subcarrier_spacing`
the allocated bandwidth, over which the transmit power is spread uniformly
(`linkgym.channels.link_budget_snr_db`). `tx_power_dbm` is required and
`noise_figure_db` defaults to 7 dB; `snr_db` must be `None` and `snr_db_range` is not
used. `info["snr_db"]` is the realized mean SNR of the episode.

For example, 30 dBm over 52 PRBs at 30 kHz (18.72 MHz) with a 7 dB noise figure gives a
reference SNR of 124.25 dB; a gain of -110 dB then means an SNR of 14.25 dB.

## Limitations

- **SINR outside the BLER data.** With `link_budget`, a UE close to the transmitter can
  see a SINR far above 20 dB, where the PDSCH table 1 BLER data ends: the BLER stays at
  its 20 dB value, so MCS 28 keeps its TBLER floor of 0.0215 and higher SINR brings
  nothing. The observation clips the wideband SINR to [-10, 40] dB. Below -5 dB, the
  BLER stays at its -5 dB value. See [benchmarks.md](benchmarks.md#known-limitations).
- **Single link.** No interference and no noise other than thermal noise; the gain is
  that of one effective single-layer channel.
- **Fixed grid.** The trace must have the scenario's PRB count, subcarrier spacing and
  slot duration; there is no interpolation.
- **Memory.** Each environment instance loads the whole `gain` dataset
  (4 x T x N x num_prb bytes); a vector of n environments loads it n times.
- **Overlapping windows.** Windows are drawn with replacement and may overlap; the number
  of distinct episodes is bounded by the size of the trace.

## Bring your own channel

Any channel available as a numpy array of linear power gains [T, N, num_prb] can be
used: write it with `write_trace` and pass the file to `gym.make`.

```python
import gymnasium as gym
import numpy as np

import linkgym
from linkgym.baselines import OLLAPolicy
from linkgym.channels import write_trace

# Linear power gain |h|^2 per trajectory, slot and PRB: here 6 trajectories of 2000 slots
# of Rayleigh block fading, constant over 10 slots
rng = np.random.default_rng(0)
gain = np.repeat(rng.exponential(size=(6, 200, 52)), 10, axis=1)
split = ["train", "train", "train", "train", "val", "test"]

write_trace(
    "my_channel.h5",
    gain,
    split,
    carrier_frequency_hz=3.5e9,  # must match the scenario
    subcarrier_spacing_hz=30e3,
    slot_duration_s=0.5e-3,
)

env = gym.make("linkgym/LinkAdaptation-v0", channel="trace", trace_path="my_channel.h5")
obs, info = env.reset(seed=0)
print(env.unwrapped.channel_info)

result = linkgym.evaluate(
    OLLAPolicy(bler_target=0.1),
    env_kwargs={"channel": "trace", "trace_path": "my_channel.h5", "trace_splits": ["test"]},
    seeds=[1000, 1001],
)
print(result["goodput_mbps"]["mean"], result["observed_tbler"]["mean"])
```
