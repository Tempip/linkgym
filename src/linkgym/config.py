"""Scenario configuration of the linkgym environments."""

from __future__ import annotations

from dataclasses import dataclass

ENV_ID = "linkgym/LinkAdaptation-v0"


@dataclass(frozen=True)
class ScenarioConfig:
    """Scenario of ``LinkAdaptation-v0``. Every field can also be passed to ``gymnasium.make``.

    :param episode_length: Slots per episode; the episode is truncated after it
    :param snr_db: Fixed mean SNR [dB]; if `None`, it is drawn per episode from ``snr_db_range``
    :param snr_db_range: Range (low, high) [dB] of the per-episode uniform SNR draw
    :param speed: UE speed [m/s]
    :param feedback_delay: Delay d >= 1 [slots] of the reports in the observation
    :param num_reports: Number K of reports in the observation, most recent first
    :param tdl_model: TR 38.901 TDL profile, e.g. "A" to "E"
    :param delay_spread: RMS delay spread [s]
    :param carrier_frequency: Carrier frequency [Hz]
    :param num_prbs: Number of allocated PRBs
    :param subcarrier_spacing: Subcarrier spacing [Hz]
    :param num_data_symbols: OFDM symbols per slot carrying data
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
