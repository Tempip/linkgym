# Benchmarks

Speed of the link simulator (`linkgym.sim`, M1) and of the Gymnasium environment
`LinkAdaptation-v0` (M2), and link adaptation baselines on both, built on Sionna SYS 2.1.0.
All numbers are from a single machine, CPU only.

## Machine

- CPU: AMD Ryzen 7 7700X 8-Core Processor (16 logical CPUs)
- OS: Windows-11-10.0.26200-SP0
- Python 3.12.14, torch 2.14.0+cpu, sionna-no-rt 2.1.0, gymnasium 1.3.0, numpy 2.5.3
- torch default intra-op threads: 8

## Simulator speed

Reproduce with `python examples/benchmark.py --sections sim`.

- `step`: `LinkSimulator.step` with fixed MCS 14. The episode's channel is generated before
  timing starts.
- `olla`: the full OLLA loop (`run_episode(sim, "olla")`), one OLLA decision plus one step per
  slot.
- One step advances all B links by one slot, so link-slots/s = B x steps/s. "x real time"
  compares link-slots/s with the 2000 slots/s of one real link at 30 kHz subcarrier spacing
  (0.5 ms slots).
- 200 slots per run, median of 5 runs after 1 warm-up run. A second run of the whole benchmark
  was within 3% of these numbers.

| workload | B | threads | ms/step | steps/s | link-slots/s | x real time |
|---|---:|---:|---:|---:|---:|---:|
| step | 1 | 1 | 0.852 | 1174 | 1174 | 0.59 |
| olla | 1 | 1 | 1.684 | 594 | 594 | 0.30 |
| step | 16 | 1 | 0.871 | 1148 | 18367 | 9.18 |
| olla | 16 | 1 | 1.786 | 560 | 8957 | 4.48 |
| step | 256 | 1 | 1.036 | 965 | 247162 | 123.58 |
| olla | 256 | 1 | 2.981 | 336 | 85889 | 42.94 |
| step | 1 | 8 | 1.031 | 970 | 970 | 0.48 |
| olla | 1 | 8 | 1.872 | 534 | 534 | 0.27 |
| step | 16 | 8 | 1.090 | 918 | 14683 | 7.34 |
| olla | 16 | 8 | 2.238 | 447 | 7149 | 3.57 |
| step | 256 | 8 | 1.351 | 740 | 189441 | 94.72 |
| olla | 256 | 8 | 3.456 | 289 | 74079 | 37.04 |

Observations:

- The time per step grows slowly with B: with one thread, going from B=1 to B=256 makes a
  plain step 22% slower (0.85 to 1.04 ms) and an OLLA step 77% slower (1.68 to 2.98 ms), so
  link-slots/s grow almost linearly with B.
- One thread is faster than the default 8 threads in every configuration.
- An OLLA step costs 2.0x (B=1) to 2.9x (B=256) a plain step. Each OLLA decision runs ILLA,
  which evaluates `PHYAbstraction` for all 29 MCS indices.

## Environment speed

Reproduce with `python examples/benchmark.py --sections env`.

`env.step` of `gymnasium.make("linkgym/LinkAdaptation-v0")` (default scenario, including
Gymnasium's `PassiveEnvChecker` and `OrderEnforcing` wrappers), random actions, one torch
thread. Median of 5 episodes of 1000 steps after 1 warm-up episode; `reset()` is not
timed. A second run gave the same numbers within 0.1%.

| workload | threads | ms/step | steps/s | x real time |
|---|---:|---:|---:|---:|
| env.step | 1 | 0.994 | 1006 | 0.50 |

One environment step costs about 0.14 ms more than a bare simulator step (0.85 ms, B=1, one
thread). Policies add their own cost on top: OLLA and ILLA call Sionna once per step.

## Environment baselines (M2)

Reproduce with `python examples/run_baselines.py` (143 s). Default `LinkAdaptation-v0`
scenario: 1000-slot episodes, mean SNR drawn per episode from U(5, 20) dB, 15 m/s,
feedback delay 1, TDL-A with 100 ns delay spread at 3.5 GHz. Seeds 0-4, 2 episodes per
seed; mean ± sample std (ddof=1) across seeds. All policies run through `env.step`; with
the same seed they face the same SNR, channel and ACK draws.

- `olla`, `illa`: Sionna OLLA and ILLA fed the report in `info["report"]`: the wideband
  SINR (mean linear per-PRB SINR) and the ACK of the previous slot.
- `oracle`: Sionna ILLA on the per-RE SINR of the slot being decided
  (`info["privileged"]`).

| policy | goodput Mbit/s | observed TBLER | mean MCS |
|---|---:|---:|---:|
| olla (target 0.05) | 34.92 ± 11.01 | 0.0534 ± 0.0007 | 15.12 ± 3.81 |
| olla (target 0.1) | 34.83 ± 10.50 | 0.1028 ± 0.0007 | 15.86 ± 3.74 |
| olla (target 0.2) | 32.62 ± 9.42 | 0.2022 ± 0.0004 | 16.78 ± 3.66 |
| illa (target 0.1) | 20.15 ± 2.45 | 0.5548 ± 0.0724 | 19.06 ± 3.87 |
| fixed (MCS 5) | 10.94 ± 0.07 | 0.0064 ± 0.0068 | 5.00 ± 0.00 |
| fixed (MCS 14) | 24.06 ± 5.17 | 0.2546 ± 0.1602 | 14.00 ± 0.00 |
| fixed (MCS 24) | 16.02 ± 14.40 | 0.7631 ± 0.2129 | 24.00 ± 0.00 |
| oracle (target 0.1) | 43.52 ± 11.84 | 0.0269 ± 0.0038 | 17.55 ± 3.58 |

Observations:

- OLLA reaches each target (0.053, 0.103, 0.202). Targets 0.05 and 0.1 give about the same
  goodput; 0.2 gives less.
- ILLA on the wideband report has a TBLER of 0.55. The wideband mean SINR is higher than the
  EESM effective SINR that decides the ACK, and ILLA has no correction for it or for the
  one-slot-old report at 15 m/s.
- The goodput std across seeds is large (about 10 Mbit/s for OLLA) because the mean SNR is
  drawn per episode and each seed has only 2 episodes.
- The oracle's goodput is 8.6 Mbit/s above the best OLLA mean.

## Simulator link adaptation baselines (M1)

Policies run directly on `linkgym.sim` with `run_episode`; ILLA and OLLA are fed the EESM
effective SINR of the transmitted MCS. Reproduce with `python examples/baseline.py --grid`
(2000 slots, seed 0).

Scenario: PDSCH, MCS table 1, 52 PRB at 30 kHz subcarrier spacing, 12 data symbols
(7488 REs per slot), SISO, TDL-A with 100 ns delay spread at 3.5 GHz. TBLER target 0.1 for
oracle, ILLA and OLLA. Fixed-MCS rows use the default scenario (15 dB, 3 m/s). All rows of a
scenario share the same seed, hence the same channel and the same ACK draws.

- `oracle`: Sionna ILLA on the current slot's per-RE SINR (no feedback delay), so its delay is
  0 in every scenario. It is the upper bound for the ILLA selection rule at this target, not
  for goodput in general.
- `illa`, `olla`: Sionna ILLA and OLLA, fed the ACK and effective SINR of the slot
  `delay` slots earlier.
- goodput counts TB information bits of ACKed slots; observed TBLER is the fraction of NACKed
  slots.

| policy | snr_db | speed | delay | goodput Mbit/s | observed TBLER | mean MCS |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 5 | 3 | 0 | 19.21 | 0.0280 | 9.22 |
| illa | 5 | 3 | 1 | 18.23 | 0.0695 | 9.11 |
| olla | 5 | 3 | 1 | 17.46 | 0.1000 | 9.16 |
| illa | 5 | 3 | 4 | 14.37 | 0.2410 | 9.08 |
| olla | 5 | 3 | 4 | 13.45 | 0.1010 | 7.31 |
| oracle | 5 | 15 | 0 | 20.27 | 0.0320 | 9.60 |
| illa | 5 | 15 | 1 | 13.87 | 0.2975 | 9.49 |
| olla | 5 | 15 | 1 | 14.19 | 0.1005 | 7.42 |
| illa | 5 | 15 | 4 | 10.06 | 0.4060 | 9.39 |
| olla | 5 | 15 | 4 | 9.46 | 0.1010 | 5.05 |
| oracle | 15 | 3 | 0 | 50.31 | 0.0310 | 20.22 |
| illa | 15 | 3 | 1 | 48.43 | 0.0570 | 20.07 |
| olla | 15 | 3 | 1 | 45.93 | 0.1000 | 20.12 |
| illa | 15 | 3 | 4 | 39.42 | 0.2155 | 20.04 |
| olla | 15 | 3 | 4 | 37.27 | 0.1005 | 17.51 |
| oracle | 15 | 15 | 0 | 51.57 | 0.0290 | 20.50 |
| illa | 15 | 15 | 1 | 36.44 | 0.2865 | 20.36 |
| olla | 15 | 15 | 1 | 38.96 | 0.1010 | 17.93 |
| illa | 15 | 15 | 4 | 28.66 | 0.3915 | 20.33 |
| olla | 15 | 15 | 4 | 28.72 | 0.1005 | 14.61 |
| fixed (MCS 5) | 15 | 3 | - | 11.01 | 0.0000 | 5.00 |
| fixed (MCS 14) | 15 | 3 | - | 32.08 | 0.0060 | 14.00 |
| fixed (MCS 24) | 15 | 3 | - | 14.10 | 0.7915 | 24.00 |

Observations:

- OLLA holds the 0.1 TBLER target in every scenario.
- ILLA's TBLER grows with speed and feedback delay (at 15 dB: 0.057 at 3 m/s with 1 slot of
  delay, 0.39 at 15 m/s with 4 slots).
- The oracle's TBLER is about 0.03, below the target: the TBLER curves are steep, so the
  highest MCS that meets the target often has a TBLER well below it.
- At 3 m/s with 1 slot of delay, ILLA has a higher goodput than OLLA; at 15 m/s with 1 slot of
  delay, OLLA has a higher goodput than ILLA.

## Known limitations

- **Coarse BLER tables.** Each (MCS, code block size) curve in Sionna's bundled tables has 15
  SINR points about 1.79 dB apart and there are 5 code block sizes (24 to 2000 bits). Between
  points, the waterfall shape comes from Sionna's log-linear interpolation, not from data.
- **SINR above 20 dB.** The PDSCH table 1 data ends at 20 dB; above it the BLER stays at its
  20 dB value (the interpolation grid is clamped to [-5, 30] dB). This leaves an error floor
  at the top MCS: in this setup MCS 28 has cb_bler 0.0043 and TBLER 0.0215 (5 code blocks) at
  every SINR from 20 dB up. MCS 27 has no floor in PDSCH table 1; in PUSCH table 1 (not used
  here) MCS 27 has a cb_bler floor of 0.0037.
- **MCS 0-2 missing.** There is no BLER data for MCS 0-2 in PDSCH table 1, so the simulator
  only accepts MCS 3-28.
- **Code block size.** Code blocks larger than 2000 bits use the 2000-bit curve. With 7488 REs
  every MCS from 3 to 28 has code blocks of 3768-8432 bits.
- **No HARQ retransmissions.** A NACKed transport block is lost; there is no soft combining.
- **No interference.** SINR is SNR x |h|^2 for a single link.
- **No channel estimation error.** SINR is computed from the true channel.
- **Flat mean SNR.** No path loss, shadowing or UE mobility beyond the TDL Doppler; the mean
  SNR is a fixed parameter.
- **One SINR sample per PRB.** The channel is sampled once per PRB and once per slot and held
  constant within them.
