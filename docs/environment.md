# LinkAdaptation-v0

`linkgym/LinkAdaptation-v0` is MCS selection for a single-cell, single-user 5G NR
downlink (PDSCH, MCS table 1) on top of Sionna SYS 2.1. The environment class is
`linkgym.env.LinkAdaptationEnv`.

```python
import gymnasium
import linkgym  # registers linkgym/LinkAdaptation-v0

env = gymnasium.make("linkgym/LinkAdaptation-v0", speed=3.0)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

The package never changes torch's thread count. On the CPU one thread is faster for this
environment (see [benchmarks.md](benchmarks.md)); call `torch.set_num_threads(1)` in your
own script.

## Episode

- An episode has `episode_length` slots (default 1000). The channel of all slots is
  generated (TDL) or read from a trace file in `reset()`; see [channels.md](channels.md).
- The step that transmits the last slot returns `truncated=True`. `terminated` is always
  `False`. The environment is registered without `max_episode_steps`; it truncates itself.
- Calling `step()` after the last slot raises `RuntimeError`.
- With `channel="trace"`, `reset(seed=s, options={"trajectory": i, "offset": o})` pins the
  episode to trajectory `i` of the trace file (its split must be in `trace_splits`),
  starting at slot `o`. The SNR, the channel seed and the ACK draws come from `s` exactly
  as without options; only the choice of trajectory and window is replaced. The same seed
  and options give the same episode for every policy. See
  [channels.md](channels.md#pinned-episodes).

## Timing

At step t the agent chooses the MCS of slot t. With feedback delay d (`feedback_delay`,
default 1), the newest report it can see is the one of slot t - d.

```
call          reset()      step(a_0)      step(a_1)      step(a_2)
returns       obs_0, i_0   obs_1, i_1     obs_2, i_2     obs_3, i_3
slot sent         -        slot 0 (a_0)   slot 1 (a_1)   slot 2 (a_2)
agent         a_0 = pi(obs_0)  a_1 = pi(obs_1)  a_2 = pi(obs_2)  ...
newest report in obs_t:
  d = 1           -        slot 0         slot 1         slot 2
  d = 2           -          -            slot 0         slot 1
```

| agent chooses slot | from | newest report, d = 1 | newest report, d = 2 |
|---:|---|---|---|
| 0 | `reset()` | none | none |
| 1 | `step(a_0)` | slot 0 | none |
| 2 | `step(a_1)` | slot 1 | slot 0 |
| t | `step(a_{t-1})` | slot t - 1 | slot t - 2 |

A report only exists once its slot has been transmitted, and the pre-generated SINR of
slot t never enters the observation used to choose slot t. `tests/test_env.py` checks
this for d = 1 and d = 3.

## Action

`Discrete(26)`: action a selects MCS a + 3, i.e. MCS 3 to 28. Sionna 2.1 ships no BLER
data for MCS 0-2 of PDSCH table 1.

## Observation

`Box(-1, 1, (3 * K,), float32)` with the last K reports (`num_reports`, default 4), most
recent first. Report i (i = 0 is the most recent) occupies `obs[3 * i : 3 * i + 3]`:

| index | content | scaling |
|---|---|---|
| `3 * i` | wideband SINR of the slot [dB]: 10 log10 of the mean linear per-PRB SINR, independent of the MCS | clipped to [-10, 40] dB, then `(sinr_db + 10) / 25 - 1` |
| `3 * i + 1` | HARQ feedback | ACK = +1, NACK = -1 |
| `3 * i + 2` | MCS used in the slot | `(mcs - 3) / 12.5 - 1` |

For the default K = 4: `obs[0:3]` is the newest report, `obs[3:6]` the one before, up to
`obs[9:12]`.

Reports that do not exist yet (the first d + K - 1 steps of an episode) are all zeros.
A SINR entry of 0 also encodes 15 dB; the HARQ entry, which is never 0 for a real
report, tells the two apart.

## Reward

Delivered bits of slot t divided by the TB size of MCS 28 for the configured allocation
(42016 bits for 52 PRB and 12 data symbols, computed with Sionna's `MCSDecoderNR` and
`TransportBlockNR`). The reward is in [0, 1]; there is no penalty term. Delivered bits
are the TB information bits on ACK and 0 on NACK.

## Info

| key | `reset` | `step` | content |
|---|:---:|:---:|---|
| `mcs` | | yes | MCS used in slot t |
| `ack` | | yes | `True` for ACK |
| `bits` | | yes | delivered TB information bits of slot t |
| `tbler` | | yes | TBLER of slot t; the ACK is drawn from it |
| `snr_db` | yes | yes | mean SNR of the episode [dB]: the scenario SNR, or with `snr_mode="link_budget"` the realized mean SNR of the episode's slots |
| `num_allocated_re` | yes | yes | data resource elements per slot (constant, 7488 by default) |
| `report` | `None` | yes | the raw report that entered the observation: `slot`, `sinr_wideband_db` (unclipped), `ack`, `mcs`; `None` while no report exists |
| `privileged` | yes | yes | `sinr_prb` (float32, linear per-PRB SINR) and `num_data_symbols` of the slot the next action decides; `None` after the last slot |

`privileged` is for the oracle baseline only. It holds the channel of the slot being
decided, which no real scheduler knows; agents must not use it.

`env.unwrapped.last_result` holds the simulator outcome of the last slot, including
`ack_uniform`, the uniform draw behind the ACK. It is meant for analysis and tests and is
not part of the observation or the info.

`env.unwrapped.channel_info` describes the channel of the current episode: the trace
trajectory, start slot and split, and the realized mean SNR of the episode's slots
(`realized_snr_db`). See [channels.md](channels.md#selection-and-splits).

## Scenario

`linkgym.ScenarioConfig` fields. Each one can be passed to `gymnasium.make` as a keyword,
or a whole `config=ScenarioConfig(...)` can be passed and overridden by keywords.

| field | default | meaning |
|---|---|---|
| `episode_length` | 1000 | slots per episode |
| `snr_db` | `None` | fixed mean SNR [dB]; `None` draws it per episode |
| `snr_db_range` | (5.0, 20.0) | range of the uniform per-episode SNR draw [dB] |
| `speed` | 15.0 | UE speed [m/s] (TDL) |
| `feedback_delay` | 1 | delay d of the reports [slots], >= 1 |
| `num_reports` | 4 | number K of reports in the observation |
| `tdl_model` | "A" | TR 38.901 TDL profile (TDL) |
| `delay_spread` | 100e-9 | RMS delay spread [s] (TDL) |
| `carrier_frequency` | 3.5e9 | carrier frequency [Hz] |
| `num_prbs` | 52 | allocated PRBs |
| `subcarrier_spacing` | 30e3 | subcarrier spacing [Hz] |
| `num_data_symbols` | 12 | OFDM symbols per slot carrying data |
| `channel` | "tdl" | "tdl" (TDL fading) or "trace" (gains read from `trace_path`) |
| `trace_path` | `None` | trace file, required for `channel="trace"` |
| `trace_splits` | ("train",) | split labels of the trace trajectories to draw from |
| `snr_mode` | "normalized" | "normalized" (scenario SNR) or "link_budget" (trace only) |
| `tx_power_dbm` | `None` | transmit power [dBm], required for `snr_mode="link_budget"` |
| `noise_figure_db` | 7.0 | receiver noise figure [dB] (link budget) |

The channel fields are described in [channels.md](channels.md).

## Randomness

All randomness derives from the seed passed to `reset()`. `reset(seed)` seeds
`self.np_random`, which draws the SNR (if not fixed, and not with
`snr_mode="link_budget"`) and then a simulator seed; the simulator derives independent
channel and ACK seeds from it, using its own `torch.Generator` for the ACK draws. With a
trace, the channel seed selects the trajectory and the start slot. Sionna's global
generator is left unchanged by the episode generation.

The channel and the ACK uniforms do not depend on the actions. With the same seed, every
policy faces the same SNR, the same channel and the same ACK draw in every slot, so
policies can be compared on common random numbers.

## Baselines and evaluation

`linkgym.baselines` has `FixedMCSPolicy`, `ILLAPolicy`, `OLLAPolicy`, `OraclePolicy` and
`SB3Policy` (an adapter for Stable-Baselines3 models). All have `reset()` and
`act(obs, info) -> action` and act through `env.step`. At step t, `act` receives what the
previous `step` (or `reset`) returned. ILLA and OLLA read only `info["report"]` and
`info["num_allocated_re"]`; OLLA gets the report's ACK as HARQ feedback and its wideband
SINR as effective SINR. The oracle runs ILLA on `info["privileged"]`.

`linkgym.evaluate(policy, env_kwargs, seeds, episodes_per_seed)` returns goodput
[Mbit/s], observed TBLER and mean MCS as mean and sample std across seeds. Results are in
[benchmarks.md](benchmarks.md).

## Known limitations

- With `feedback_delay` d > 1, the observation does not include the agent's own actions
  for the d - 1 slots whose feedback is still pending, although a real scheduler knows
  them. The problem is therefore partially observable for d > 1. The default d = 1 is
  unaffected. Planned for v0.2.
- The limitations of the underlying simulator (coarse BLER tables, SINR clamping, no
  HARQ retransmissions, no interference, no channel estimation error, flat mean SNR)
  are listed in [benchmarks.md](benchmarks.md#known-limitations).
