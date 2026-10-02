"""Gymnasium environment ``linkgym/LinkAdaptation-v0``: MCS selection on one 5G NR link."""

from __future__ import annotations

import dataclasses
from typing import Any

import gymnasium
import numpy as np
import torch
from gymnasium import spaces

from linkgym.channels import TraceChannelSource
from linkgym.config import ScenarioConfig
from linkgym.sim import (
    MAX_MCS,
    MIN_MCS,
    LinkResult,
    LinkSimulator,
    slot_duration,
    tb_size_per_mcs,
)

SINR_DB_MIN = -10.0
SINR_DB_MAX = 40.0
REPORT_SIZE = 3  # wideband SINR, HARQ, MCS


def scale_sinr_db(sinr_db: float) -> float:
    """Clip a SINR [dB] to [SINR_DB_MIN, SINR_DB_MAX] and map it linearly to [-1, 1]."""
    clipped = min(max(sinr_db, SINR_DB_MIN), SINR_DB_MAX)
    return 2.0 * (clipped - SINR_DB_MIN) / (SINR_DB_MAX - SINR_DB_MIN) - 1.0


def scale_mcs(mcs: int) -> float:
    """Map an MCS index linearly from [MIN_MCS, MAX_MCS] to [-1, 1]."""
    return 2.0 * (mcs - MIN_MCS) / (MAX_MCS - MIN_MCS) - 1.0


class LinkAdaptationEnv(gymnasium.Env):
    """MCS selection for a single-cell, single-user 5G NR downlink (PDSCH, MCS table 1).

    At step t the agent chooses the MCS of slot t. With feedback delay d
    (``feedback_delay``), the observation it sees holds the reports of slots <= t - d.
    An episode has ``episode_length`` slots; the channel is generated in :meth:`reset`,
    from the TDL model (``channel="tdl"``) or a trace file (``channel="trace"``, see
    docs/channels.md).
    The last step returns ``truncated=True``; ``terminated`` is always `False`.

    **Action**: ``Discrete(26)``; action a selects MCS a + 3 (MCS 3-28).

    **Observation**: ``Box(-1, 1, (3 * K,), float32)`` holding the last K
    (``num_reports``) reports, most recent first. Report i (i = 0 is the most recent)
    occupies ``obs[3 * i : 3 * i + 3]``:

    - ``obs[3 * i]``: wideband SINR of the slot, 10 log10 of its mean linear per-PRB
      SINR (independent of the MCS), clipped to [-10, 40] dB and mapped linearly to
      [-1, 1]
    - ``obs[3 * i + 1]``: HARQ feedback, +1 for ACK, -1 for NACK
    - ``obs[3 * i + 2]``: MCS used in the slot, mapped linearly from [3, 28] to [-1, 1]

    Reports that do not exist yet are all zeros. A SINR entry of 0 also means 15 dB;
    the HARQ entry, never 0 for a real report, tells the two cases apart.

    **Reward**: delivered bits of slot t divided by the TB size of MCS 28 for the
    configured allocation, in [0, 1].

    **Info** returned by :meth:`step`:

    - ``mcs``, ``ack``, ``bits``, ``tbler``: outcome of slot t
    - ``snr_db``: mean SNR of the episode [dB]: the drawn or fixed scenario SNR, or with
      ``snr_mode="link_budget"`` the mean SNR of the episode's slots
    - ``num_allocated_re``: data resource elements per slot (constant)
    - ``report``: the raw report that entered the observation (``slot``,
      ``sinr_wideband_db``, ``ack``, ``mcs``), or `None` if there is none yet
    - ``privileged``: per-PRB linear SINR of the next slot (``sinr_prb``,
      ``num_data_symbols``), or `None` after the last slot. It exists for the oracle
      baseline only and must not be used by agents.

    :meth:`reset` returns ``snr_db``, ``num_allocated_re``, ``report`` (`None`) and
    ``privileged`` (for slot 0).

    See docs/environment.md for the timing diagram and the scenario fields.

    :param config: Scenario; defaults to ``ScenarioConfig()``
    :param render_mode: `None` or ``"ansi"``; other modes only warn and render nothing
    :param kwargs: Overrides of ``config`` fields, e.g. ``speed=3.0``
    """

    # render_fps: one frame per slot at 30 kHz subcarrier spacing (0.5 ms)
    metadata = {"render_modes": ["ansi"], "render_fps": 2000}

    def __init__(
        self,
        config: ScenarioConfig | None = None,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = dataclasses.replace(config or ScenarioConfig(), **kwargs)
        # Unsupported modes only warn, as in gymnasium.make: SB3's make_vec_env passes
        # render_mode="rgb_array" by default. render() then returns None.
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            gymnasium.logger.warn(
                f"render_mode={render_mode!r} is not supported ({self.metadata['render_modes']})"
            )
        self.render_mode = render_mode

        c = self.config
        channel_source = None
        if c.channel == "trace":
            channel_source = TraceChannelSource(
                c.trace_path,
                num_prbs=c.num_prbs,
                subcarrier_spacing=c.subcarrier_spacing,
                carrier_frequency=c.carrier_frequency,
                slot_duration=slot_duration(c.subcarrier_spacing),
                num_slots=c.episode_length,
                splits=c.trace_splits,
                snr_mode=c.snr_mode,
                tx_power_dbm=c.tx_power_dbm,
                noise_figure_db=c.noise_figure_db,
            )
        self._sim = LinkSimulator(
            c.episode_length,
            snr_db=c.snr_db_range[0] if c.snr_db is None else c.snr_db,  # set in reset()
            speed=c.speed,
            tdl_model=c.tdl_model,
            delay_spread=c.delay_spread,
            carrier_frequency=c.carrier_frequency,
            num_prbs=c.num_prbs,
            subcarrier_spacing=c.subcarrier_spacing,
            num_data_symbols=c.num_data_symbols,
            channel_source=channel_source,
        )
        self.slot_duration = self._sim.slot_duration
        self.num_allocated_re = self._sim.num_allocated_re
        self.max_tb_size = int(tb_size_per_mcs(self.num_allocated_re)[MAX_MCS])

        self.action_space = spaces.Discrete(MAX_MCS - MIN_MCS + 1)
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(REPORT_SIZE * c.num_reports,), dtype=np.float32
        )

        self.snr_db = self._episode_snr_db()
        self._sinr_wideband_db = np.zeros(c.episode_length)
        self._reports: list[dict[str, Any]] = []
        self._last_result: LinkResult | None = None
        self._num_nacks = 0

    @property
    def last_result(self) -> LinkResult | None:
        """Simulator outcome of the last slot, a batch of one link.

        It includes ``ack_uniform``, the uniform draw behind the ACK. Meant for analysis
        and tests; it is not part of the observation or the info.
        """
        return self._last_result

    @property
    def channel_info(self) -> dict[str, Any]:
        """Channel of the current episode, e.g. the trace trajectory, offset and split.

        Always includes ``realized_snr_db``, the mean SNR of the episode's slots [dB]
        (10 log10 of the mean linear per-PRB SINR over all slots).
        """
        return {key: value[0] for key, value in self._sim.channel_info.items()}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        c = self.config
        channel_options = self._channel_options(options)
        # All randomness derives from self.np_random: the SNR, then the simulator seed,
        # from which the simulator derives the channel and ACK seeds. A pinned episode
        # makes the same draws; only the choice of trajectory and window is replaced.
        snr_db = None  # link budget: the channel source sets the SNR
        if c.snr_mode != "link_budget":
            if c.snr_db is None:
                snr_db = float(self.np_random.uniform(*c.snr_db_range))
            else:
                snr_db = float(c.snr_db)
        sim_seed = int(self.np_random.integers(np.iinfo(np.int64).max))
        self._sim.reset(sim_seed, snr_db=snr_db, channel_options=channel_options)
        self.snr_db = self._episode_snr_db()

        # Wideband SINR per slot [dB], independent of the MCS
        wideband = 10.0 * torch.log10(self._sim.sinr[0].mean(dim=-1))
        self._sinr_wideband_db = wideband.double().numpy()
        self._reports = []
        self._last_result = None
        self._num_nacks = 0
        return self._observation(), self._info(report=None)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r} for {self.action_space}")
        slot = self._sim.slot
        mcs = int(action) + MIN_MCS
        result = self._sim.step(mcs)
        self._last_result = result

        ack = bool(result.ack[0])
        bits = int(result.decoded_bits[0])
        self._num_nacks += not ack
        self._reports.append(
            {
                "slot": slot,
                "sinr_wideband_db": float(self._sinr_wideband_db[slot]),
                "ack": ack,
                "mcs": mcs,
            }
        )

        newest = self._newest_report_slot()
        report = self._reports[newest] if newest >= 0 else None
        info = {"mcs": mcs, "ack": ack, "bits": bits, "tbler": float(result.tbler[0])}
        info |= self._info(report)
        truncated = self._sim.slot == self.config.episode_length
        return self._observation(), bits / self.max_tb_size, False, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        if self._last_result is None:
            return "slot     - | no transmission yet"
        report = self._reports[-1]
        bits = int(self._last_result.decoded_bits[0])
        running_tbler = self._num_nacks / len(self._reports)
        return (
            f"slot {report['slot']:5d} | MCS {report['mcs']:2d} | "
            f"{'ACK ' if report['ack'] else 'NACK'} | {bits:6d} bits | "
            f"running TBLER {running_tbler:.3f}"
        )

    def _channel_options(self, options: dict[str, Any] | None) -> dict[str, Any] | None:
        """Reset options pinning the trace trajectory and window, as channel options."""
        if not options:
            return None
        unknown = sorted(set(options) - {"trajectory", "offset"})
        if unknown:
            raise ValueError(f"unknown reset options {unknown}; known: ['offset', 'trajectory']")
        if self.config.channel != "trace":
            raise ValueError("reset options 'trajectory' and 'offset' need channel='trace'")
        if set(options) != {"trajectory", "offset"}:
            raise ValueError("pin an episode with both 'trajectory' and 'offset'")
        return {"trajectories": [options["trajectory"]], "offsets": [options["offset"]]}

    def close(self) -> None:
        """Release the channel source's resources, e.g. close the trace file."""
        self._sim.close()

    def _episode_snr_db(self) -> float:
        """Mean SNR of the episode [dB] reported in the info."""
        if self.config.snr_mode == "link_budget":
            return float(self._sim.channel_info["realized_snr_db"][0])
        return float(self._sim.snr_db)

    def _newest_report_slot(self) -> int:
        """Slot of the most recent report available for the next decision (< 0: none)."""
        return self._sim.slot - self.config.feedback_delay

    def _observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        newest = self._newest_report_slot()
        for i in range(min(self.config.num_reports, max(newest + 1, 0))):
            report = self._reports[newest - i]
            obs[REPORT_SIZE * i : REPORT_SIZE * (i + 1)] = (
                scale_sinr_db(report["sinr_wideband_db"]),
                1.0 if report["ack"] else -1.0,
                scale_mcs(report["mcs"]),
            )
        return obs

    def _info(self, report: dict[str, Any] | None) -> dict[str, Any]:
        next_slot = self._sim.slot
        privileged = None
        if next_slot < self.config.episode_length:
            privileged = {
                "sinr_prb": self._sim.sinr[0, next_slot].numpy().copy(),
                "num_data_symbols": self.config.num_data_symbols,
            }
        return {
            "snr_db": self.snr_db,
            "num_allocated_re": self.num_allocated_re,
            "report": None if report is None else dict(report),
            "privileged": privileged,
        }
