# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial project skeleton: packaging, license, CI, pre-commit and an import test.
- Project URLs in `pyproject.toml`, CI and license badges in the README, and `.gitattributes`
  enforcing LF line endings.
- `linkgym.sim`: single-cell, single-user downlink link simulator on Sionna SYS 2.1
  (TDL fading, EESM, PHYAbstraction) with seeded, batched links and oracle, ILLA, OLLA and
  fixed-MCS policies.
- `examples/baseline.py` and `examples/benchmark.py` for link adaptation baselines and CPU
  speed measurements; results in `docs/benchmarks.md`.
- Tests for reproducibility, instance isolation, Sionna RNG side effects and equivalence with
  `PHYAbstraction`'s ACK and delivered bits.

### Changed

- Pinned ruff to 0.16.9 in the `dev` extra to match the pre-commit hook version.
- Set the author name in `pyproject.toml` to Pedro Rodrigues.
