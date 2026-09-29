# linkgym

Gymnasium environments for 5G NR link adaptation (MCS selection), built on NVIDIA Sionna SYS.

**Status: pre-alpha, under construction.**

## Scope of v0.1

- Single cell.
- Action: MCS selection.
- Channel: precomputed or statistical.
- Comparison: PPO against Sionna's outer-loop link adaptation (OLLA) and fixed-MCS baselines.

## Related work

- [sionna-rl](https://github.com/tobiassugandi/sionna-rl): a reproducible study comparing PPO
  against Sionna's OLLA for single-link MCS selection.
- M. Tsampazi, N. N. Santhi, N. Perrotta, F. Dressler, T. Melodia, "ARIADNE: AI-RAN Informed
  Link Adaptation in Digital Twin Network Environments", European Wireless 2026.
  [arXiv:2605.29772](https://arxiv.org/abs/2605.29772). RL (PPO) link adaptation integrated
  with Sionna SYS over ray-traced channels, compared against OLLA and SALAD.

linkgym differs from these in that it aims to be a reusable, pip-installable package of
registered Gymnasium environments rather than a single study.

## License

MIT. See [LICENSE](LICENSE).

## Disclaimer

This project is not affiliated with or endorsed by NVIDIA. Sionna is a trademark of NVIDIA
Corporation.
