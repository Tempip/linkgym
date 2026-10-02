"""Scenario configuration of the linkgym environments."""

from __future__ import annotations

import os
from dataclasses import dataclass

ENV_ID = "linkgym/LinkAdaptation-v0"
CHANNELS = ("tdl", "trace")
SNR_MODES = ("normalized", "link_budget")


@dataclass(frozen=True)
class ScenarioConfig:
    """Scenario of ``LinkAdaptation-v0``. Every field can also be passed to ``gymnasium.make``.

    :param episode_length: Slots per episode; the episode is truncated after it
    :param snr_db: Fixed mean SNR [dB]; if `None`, it is drawn per episode from ``snr_db_range``
    :param snr_db_range: Range (low, high) [dB] of the per-episode uniform SNR draw
    :param speed: UE speed [m/s] (TDL channel only)
    :param feedback_delay: Delay d >= 1 [slots] of the reports in the observation
    :param num_reports: Number K of reports in the observation, most recent first
    :param tdl_model: TR 38.901 TDL profile, e.g. "A" to "E" (TDL channel only)
    :param delay_spread: RMS delay spread [s] (TDL channel only)
    :param carrier_frequency: Carrier frequency [Hz]
    :param num_prbs: Number of allocated PRBs
    :param subcarrier_spacing: Subcarrier spacing [Hz]
    :param num_data_symbols: OFDM symbols per slot carrying data
    :param channel: "tdl" (3GPP TDL fading) or "trace" (gains read from ``trace_path``)
    :param trace_path: Trace file, required for ``channel="trace"``; see docs/channels.md
    :param trace_splits: Split labels of the trajectories to sample from (trace channel)
    :param snr_mode: "normalized" (scenario SNR, each trajectory scaled to unit mean gain)
        or "link_budget" (SNR from ``tx_power_dbm`` and ``noise_figure_db``; trace only)
    :param tx_power_dbm: Transmit power [dBm], required for ``snr_mode="link_budget"``
    :param noise_figure_db: Receiver noise figure [dB] (link budget)
    """

    episode_length: int = 1000
    snr_db: float | None = None
    snr_db_range: tuple[float, float] = (5.0, 20.0)
    speed: float = 15.0
    feedback_delay: int = 1
    num_reports: int = 4
    tdl_model: str = "A"
    delay_spread: float = 100e-9
    carrier_frequency: float = 3.5e9
    num_prbs: int = 52
    subcarrier_spacing: float = 30e3
    num_data_symbols: int = 12
    channel: str = "tdl"
    trace_path: str | None = None
    trace_splits: tuple[str, ...] = ("train",)
    snr_mode: str = "normalized"
    tx_power_dbm: float | None = None
    noise_figure_db: float = 7.0

    def __post_init__(self) -> None:
        if self.episode_length < 1:
            raise ValueError("episode_length must be >= 1")
        if self.feedback_delay < 1:
            raise ValueError("feedback_delay must be >= 1")
        if self.num_reports < 1:
            raise ValueError("num_reports must be >= 1")
        low, high = self.snr_db_range
        if low > high:
            raise ValueError("snr_db_range must satisfy low <= high")
        if self.channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}, got {self.channel!r}")
        if self.snr_mode not in SNR_MODES:
            raise ValueError(f"snr_mode must be one of {SNR_MODES}, got {self.snr_mode!r}")
        if self.channel == "trace" and self.trace_path is None:
            raise ValueError("channel='trace' requires trace_path")
        if self.snr_mode == "link_budget":
            if self.channel != "trace":
                raise ValueError("snr_mode='link_budget' requires channel='trace'")
            if self.tx_power_dbm is None:
                raise ValueError("snr_mode='link_budget' requires tx_power_dbm")
            if self.snr_db is not None:
                raise ValueError("snr_db is not used with snr_mode='link_budget'; leave it None")
        if self.trace_path is not None:
            object.__setattr__(self, "trace_path", os.fspath(self.trace_path))
        splits = self.trace_splits
        splits = (splits,) if isinstance(splits, str) else tuple(splits)
        if not splits or not all(isinstance(s, str) and s for s in splits):
            raise ValueError(f"trace_splits must be non-empty labels, got {self.trace_splits!r}")
        object.__setattr__(self, "trace_splits", splits)
