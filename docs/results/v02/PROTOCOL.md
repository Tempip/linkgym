# v0.2 Munich experiment: protocol

Written and committed, together with `examples/train_ppo.py` and
`examples/evaluate_v02.py`, before any training on the Munich traces and before any
evaluation on their val or test split. Every training run records this commit (clean
working tree). Deviations, if any, are listed in the results README.

## Questions

- **Q1.** Does PPO trained on the generic TDL channel (the M3 gamma = 0 models, not
  retrained) work on the Munich test streets, compared with tuned OLLA?
- **Q2.** Does training on the site help? PPO gamma = 0 trained on the Munich train
  streets, evaluated on the held-out test streets, against tuned OLLA and against PPO-TDL.
- **Q3.** Do the conclusions hold per route category (line of sight, NLoS, transition) and
  on the alternative test realization?
- **Q4** (secondary, evaluation only). How does the Munich-trained PPO do on the TDL default
  scenario (the M3 test seeds)?

## Data

| file | SHA-256 | content |
|---|---|---|
| `data/munich-v1.h5` | `4b3bbb246c40f4513a685d069a9e913649d58a88e4dbdc908b3fa4e074f3cd9c` | train, val and test splits (4e6 rays, 1e7 path buffer) |
| `data/munich-v1-test-alt.h5` | `0969ea3638aa34c2c7547d66d8de9746bc3109f59a324844c07dadb9b359c347` | test split, alternative realization (8e6 rays, 1e6 path buffer) |

The files are generated locally from `examples/rt/munich.json` (see
[docs/channels.md](../../channels.md#munich-dataset)); the scripts check the hashes.
Trajectories are 4000 slots long; splits are by street.

| split | routes (category: kept trajectories) | trajectories |
|---|---|---:|
| train | north-canyon (los: 8), canyon-cross (transition: 5), rathaus-west (transition: 8), east-avenue (nlos: 9), se-avenue (nlos: 3), ne-north-south (nlos: 3), sw-avenue (nlos: 2), kaufinger (nlos: 1) | 39 |
| val | nw-diagonal (los: 3), nw-avenue (transition: 6), north-street (nlos: 5) | 14 |
| test | south-street (los: 5), west-street (transition: 10), long-diagonal (nlos: 7), south-curve (nlos: 1) | 23 |

## Scenario

`linkgym/LinkAdaptation-v0` with `channel="trace"` and `snr_mode="normalized"`; everything
else as in v0.1: mean SNR drawn per episode from U(5, 20) dB, feedback delay 1 slot,
K = 4 reports, 1000-slot episodes, 52 PRB at 30 kHz, PDSCH MCS table 1 (MCS 3-28). Q4 uses
the default TDL scenario.

## Episodes and seeds

| use | episodes | reset seeds |
|---|---|---|
| training (PPO-Munich seed s) | random windows of the 39 train trajectories | env i: s x 8 + i (as M3) |
| validation | every val trajectory x 4 non-overlapping windows (offsets 0, 1000, 2000, 3000): 56 pinned episodes | 5000-5055 |
| test, both realizations | every test trajectory x 4 windows: 92 pinned episodes | 6000-6091 |
| Q4 | TDL, M3 test seeds 1000-1009, 5 episodes each: 50 episodes | as M3 |

Episodes are enumerated by `linkgym.evaluation.trace_episodes` and run with
`env.reset(seed, options={"trajectory", "offset"})`. Episode k is the same route,
trajectory and window, with the same seed (hence the same SNR), in both test realizations,
and every policy runs the same episodes (common random numbers).

## Policies

- **Fixed MCS 14.**
- **ILLA 0.1**, no outer loop, fed the wideband SINR report.
- **Default OLLA**: Sionna's OLLA with TBLER target 0.1 and its default step
  `delta_up` = 1 dB, no selection.
- **Tuned OLLA**: the cell of the v0.1 grid TBLER target {0.05, 0.1, 0.2} x `delta_up`
  {0.1, 0.25, 0.5, 1.0} dB with the highest mean goodput on the 56 val episodes (ties: the
  first cell in that order).
- **Oracle 0.1**: ILLA on the per-RE SINR of the slot being decided.
- **PPO-TDL**: the three M3 models `runs/m3/gamma0_seed{0,1,2}/model.zip` (commit
  `ce7619f`), deterministic actions, unchanged.
- **PPO-Munich**: three runs, `examples/train_ppo.py --gamma 0 --seed s --channel trace
  --trace-path data/munich-v1.h5 --eval-freq 100000` for s = 0, 1, 2: the M3 hyperparameters
  (SB3 `PPO("MlpPolicy")`, `net_arch` 64x64, learning rate 3e-4, `n_steps` 2048,
  `batch_size` 64, `n_epochs` 10, `gae_lambda` 0.95, clip 0.2, `ent_coef` 0, `vf_coef` 0.5,
  `max_grad_norm` 0.5), 8 `SubprocVecEnv` workers, 1,015,808 steps (62 rollouts). Training
  episodes come from the train split only (the script refuses val or test). The final
  model of each run is used: no checkpoint selection, no hyperparameter changes.
- **PPO result**: per episode, the mean of the three training seeds' goodput (and TBLER,
  MCS) is "PPO-TDL" or "PPO-Munich"; every seed is also reported on its own.

## Selection

The val split is used only to choose the tuned OLLA cell, for the PPO-Munich learning
curves (all 56 val episodes at the start, every 100k steps and at the end of training)
and for a context table. Nothing else is selected on val, and nothing at all on test.

## Test

Evaluated once, after training and validation, on both realizations, all policies on the
same 92 episodes. `examples/evaluate_v02.py test` refuses to run again once its results
exist.

## Metrics

Per episode: goodput (TB information bits of ACKed slots per second), observed TBLER
(share of NACKed slots), mean MCS. Goodput is the primary metric.

## Statistics

- **Paired differences** against tuned OLLA (and PPO-Munich against PPO-TDL): the mean over
  episodes of the per-episode difference (every trajectory has 4 episodes, so episodes and
  trajectories weigh the same), in Mbit/s and in % of the reference mean.
- **95% CIs** by a cluster bootstrap that resamples trajectories with replacement, keeping
  their 4 episodes together (`linkgym.evaluation.cluster_bootstrap`), 10,000 resamples,
  seed 0, percentile interval. Per category, trajectories are resampled within the
  category.
- **Training-seed variance is not in these CIs**: they describe the test episodes for the
  given models. The per-seed results and their spread are reported next to them.
- **Primary comparisons** (overall, main realization): Q1: PPO-TDL - tuned OLLA; Q2:
  PPO-Munich - tuned OLLA and PPO-Munich - PPO-TDL. All other comparisons (per category,
  every other policy against tuned OLLA, per seed) are secondary. No correction for
  multiple comparisons.
- **Per-route table**: mean goodput of every policy on each test route and the paired
  difference against tuned OLLA with a cluster bootstrap over the route's trajectories (no
  interval for `south-curve`, which has one trajectory).
- **Q4**: as in M3, per-episode paired bootstrap (10,000 resamples, seed 0) against the M3
  tuned OLLA cell (TBLER target 0.1, `delta_up` 0.25 dB) on the 50 TDL test episodes.

## Decision rules

- **Q1, Q2** (main realization): a policy is *better* than its reference if the 95% CI of
  the paired goodput difference lies above zero, *worse* if it lies below zero, and shows
  *no detectable difference* otherwise.
- **Q3, robustness**: for each primary comparison, overall and per category, the
  conclusion *holds* if the mean difference has the same sign in both realizations and the
  95% CI excludes zero in both; it is *contradicted* if both CIs exclude zero with opposite
  signs; otherwise it is *inconclusive*.
- **Q4**: descriptive.

## Limitations stated in advance

- **Four test routes.** Each category is one or two streets (line of sight: one street with
  5 trajectories; transition: one street with 10; NLoS: two streets with 7 and 1). The
  bootstrap resamples trajectories within these streets: the results describe these
  streets and say little about routes or sites in general.
- **One site, one base station, one scene**, ray traced without convergence (see
  [docs/channels.md](../../channels.md#known-limitations-of-the-munich-traces)); the
  alternative realization is one other ray budget, not an independent channel.
- **Normalized SNR only**: the absolute level along a street is removed per trajectory.
- **Three training seeds** per PPO family.

## Compute plan

Projected on the M3 machine (Ryzen 7 7700X, CPU): about 20 min per PPO-Munich run
(including the learning curves), run one after the other, and about 20 min for all
evaluations with 8 worker processes.
