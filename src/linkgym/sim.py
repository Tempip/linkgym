"""Minimal single-cell, single-user downlink link simulator built on Sionna SYS.

A :class:`LinkSimulator` pre-generates an episode of per-PRB SINR for ``batch_size``
independent links. Each :meth:`LinkSimulator.step` transmits one transport block per
link with the given MCS and returns the outcome of that slot. :func:`run_episode`
runs a link adaptation policy (oracle, ILLA, OLLA or fixed MCS) on top of it.

Fixed setup: PDSCH (MCS category 1), MCS table 1, SISO, perfect CSI, no interference,
no HARQ retransmissions. All computation runs on the CPU in float32.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
from collections.abc import Iterator
from dataclasses import dataclass, fields

import numpy as np
import torch

try:
    from sionna.phy import config as sionna_config
    from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
    from sionna.phy.channel.tr38901 import TDL
    from sionna.phy.nr import CarrierConfig
    from sionna.phy.nr.utils import MCSDecoderNR, TransportBlockNR
    from sionna.sys import EESM, InnerLoopLinkAdaptation, OuterLoopLinkAdaptation, PHYAbstraction
except ImportError as error:
    try:
        importlib.metadata.version("sionna-rt")
    except importlib.metadata.PackageNotFoundError:
        raise error from None
    # sionna-rt's sionna/__init__.py replaces sionna-no-rt's and imports Sionna RT, which
    # needs a CUDA GPU or LLVM, on every `import sionna`
    raise ImportError(
        f"Importing Sionna failed: {error}. Sionna RT (sionna-rt, installed by the "
        "linkgym[rt] extra) is installed in this environment; it makes every `import sionna` "
        "import Sionna RT, which needs a CUDA GPU or LLVM. Install linkgym[rt] in its own "
        "environment for trace generation and train in an environment without it "
        "(docs/channels.md)."
    ) from error

from linkgym.channels import ChannelEpisode, ChannelSource

DEVICE = "cpu"
MCS_CATEGORY = 1  # PDSCH
MCS_TABLE_INDEX = 1
MIN_MCS = 3  # Sionna 2.1 ships no BLER data for PDSCH table 1, MCS 0-2
MAX_MCS = 28
POLICIES = ("oracle", "illa", "olla", "fixed")


@contextlib.contextmanager
def seeded_sionna_rng(seed: int) -> Iterator[None]:
    """Seed Sionna's CPU generator inside the block and restore its previous state on exit.

    Sionna 2.1 keeps a single ``torch.Generator`` per device in its global config
    (sionna/phy/config.py:114-138), and TDL draws its Doppler shifts and phases from
    it (sionna/phy/channel/tr38901/tdl.py:602-651). That generator is not
    ``torch.default_generator``, so ``torch.random.fork_rng`` does not cover it. If a
    future Sionna version changes this, the reproducibility tests in tests/test_sim.py
    fail.
    """
    generator = sionna_config.torch_rng(DEVICE)
    state = generator.get_state()
    generator.manual_seed(seed)
    try:
        yield
    finally:
        generator.set_state(state)


def slot_duration(subcarrier_spacing: float) -> float:
    """Slot duration [s] of the 5G NR numerology with ``subcarrier_spacing`` [Hz]."""
    carrier = CarrierConfig(subcarrier_spacing=round(subcarrier_spacing / 1e3))
    return carrier.sub_frame_duration / carrier.num_slots_per_subframe


class TDLChannelGain:
    """3GPP TR 38.901 TDL fading, sampled once per slot and once per PRB.

    A :class:`~linkgym.channels.ChannelSource`. The TDL power delay profile is normalized
    to unit energy, so |h|^2 has unit mean over channel realizations and the episode uses
    the scenario SNR (``reference_snr_db`` is `None`).
    """

    def __init__(
        self,
        *,
        speed: float,
        tdl_model: str,
        delay_spread: float,
        carrier_frequency: float,
        num_prbs: int,
        subcarrier_spacing: float,
        slot_duration: float,
    ) -> None:
        self.num_prbs = num_prbs
        self._sampling_frequency = 1.0 / slot_duration
        self._tdl = TDL(
            tdl_model,
            delay_spread=delay_spread,
            carrier_frequency=carrier_frequency,
            min_speed=speed,
            device=DEVICE,
        )
        # One frequency point per PRB, spaced by the PRB bandwidth
        self._frequencies = subcarrier_frequencies(num_prbs, 12 * subcarrier_spacing, device=DEVICE)

    def generate(self, num_slots: int, batch_size: int, seed: int) -> ChannelEpisode:
        # TDL keeps time correlation only within one call (tdl.py:595-628),
        # so the whole episode is generated at once.
        with seeded_sionna_rng(seed):
            a, tau = self._tdl(
                batch_size=batch_size,
                num_time_steps=num_slots,
                sampling_frequency=self._sampling_frequency,
            )
        # [batch_size, 1, 1, 1, 1, num_slots, num_prbs]
        h = cir_to_ofdm_channel(self._frequencies, a, tau)
        return ChannelEpisode(gain=h[:, 0, 0, 0, 0].abs().square())


@dataclass(frozen=True)
class LinkResult:
    """Outcome per link: tensors of shape [B] for one slot, [B, num_slots] for an episode."""

    mcs: torch.Tensor  # int32
    sinr_eff: torch.Tensor  # linear effective SINR (EESM for the transmitted MCS)
    ack: torch.Tensor  # bool
    decoded_bits: torch.Tensor  # int32, TB information bits on ACK, else 0
    tbler: torch.Tensor  # transport block error probability; the ACK is drawn from it
    cb_bler: torch.Tensor  # code block error probability
    ack_uniform: torch.Tensor  # uniform draw u in [0, 1); ACK if u >= tbler


def sinr_grid(sinr_prb: torch.Tensor, num_data_symbols: int) -> torch.Tensor:
    """Per-RE SINR grid [B, num_data_symbols, 12 * num_prbs, 1, 1] from [B, num_prbs].

    Each PRB value is repeated over its 12 subcarriers and all data symbols, so Sionna
    counts exactly ``12 * num_prbs * num_data_symbols`` resource elements.
    """
    per_subcarrier = sinr_prb.repeat_interleave(12, dim=1)
    return per_subcarrier[:, None, :, None, None].expand(-1, num_data_symbols, -1, 1, 1)


def tb_size_per_mcs(num_allocated_re: int) -> torch.Tensor:
    """Transport block information bits for every MCS index 0-28, shape [29]."""
    mcs = torch.arange(MAX_MCS + 1, dtype=torch.int32)
    modulation_order, coderate = MCSDecoderNR(device=DEVICE)(mcs, MCS_TABLE_INDEX, MCS_CATEGORY)
    tb_size, _, _ = TransportBlockNR(device=DEVICE).transport_block_size(
        modulation_order, coderate, modulation_order * num_allocated_re
    )
    return tb_size


def transmit(
    phy: PHYAbstraction,
    tb_size: torch.Tensor,
    mcs: torch.Tensor,
    sinr_eff: torch.Tensor,
    num_allocated_re: int,
    u: torch.Tensor,
) -> LinkResult:
    """Transmit one transport block per link, drawing the ACK from the uniforms ``u``.

    ``tbler`` and ``cb_bler`` come from PHYAbstraction. The ACK rule ``u >= tbler`` is
    the one PHYAbstraction applies to its own draw (phy_abstraction.py:680-684), so with
    the same ``u`` both give the same ACK and the same delivered bits.

    :param tb_size: TB size per MCS index, from :func:`tb_size_per_mcs`
    :param mcs: [B] MCS index per link
    :param sinr_eff: [B] linear effective SINR per link
    :param u: [B] uniform draws in [0, 1)
    """
    num_re = torch.full((mcs.shape[0], 1), num_allocated_re, dtype=torch.int32)
    *_, tbler, cb_bler = phy(
        mcs[:, None],
        sinr_eff=sinr_eff[:, None],
        num_allocated_re=num_re,
        mcs_table_index=MCS_TABLE_INDEX,
        mcs_category=MCS_CATEGORY,
    )
    tbler, cb_bler = tbler[:, 0], cb_bler[:, 0]
    ack = u >= tbler
    decoded_bits = torch.where(ack, tb_size[mcs], 0)
    return LinkResult(mcs, sinr_eff, ack, decoded_bits, tbler, cb_bler, u)


class LinkSimulator:
    """Single-cell, single-user downlink over ``batch_size`` independent links.

    The SINR of all ``num_slots`` slots is generated at construction and at every
    :meth:`reset`. The seed fixes both the channel and the ACK draws; no global random
    state is read or left modified by the episode generation.

    The channel comes from ``channel_source`` (a :class:`~linkgym.channels.ChannelSource`),
    or from a :class:`TDLChannelGain` built from the TDL arguments if it is `None`. The
    SINR is ``10 ** (snr_db / 10) * gain`` with the episode's ``reference_snr_db`` if set,
    else the scenario ``snr_db``.

    :param num_slots: Episode length in slots
    :param batch_size: Number of independent links B
    :param seed: Seed of the episode
    :param snr_db: Mean SNR [dB]; can be changed per episode in :meth:`reset`
    :param speed: UE speed [m/s] (TDL)
    :param tdl_model: TDL profile, e.g. "A" to "E" (TDL)
    :param delay_spread: RMS delay spread [s] (TDL)
    :param carrier_frequency: Carrier frequency [Hz] (TDL)
    :param num_prbs: Number of allocated PRBs
    :param subcarrier_spacing: Subcarrier spacing [Hz]
    :param num_data_symbols: OFDM symbols per slot carrying data
    :param phy_abstraction: Shared PHYAbstraction instance; a new one is created if `None`
    :param channel_source: Channel source; a TDL source is built if `None`
    """

    def __init__(
        self,
        num_slots: int,
        *,
        batch_size: int = 1,
        seed: int = 0,
        snr_db: float = 15.0,
        speed: float = 3.0,
        tdl_model: str = "A",
        delay_spread: float = 100e-9,
        carrier_frequency: float = 3.5e9,
        num_prbs: int = 52,
        subcarrier_spacing: float = 30e3,
        num_data_symbols: int = 12,
        phy_abstraction: PHYAbstraction | None = None,
        channel_source: ChannelSource | None = None,
    ) -> None:
        self.num_slots = num_slots
        self.batch_size = batch_size
        self.snr_db = snr_db
        self.num_data_symbols = num_data_symbols
        # Data resource elements per slot (single layer)
        self.num_allocated_re = 12 * num_prbs * num_data_symbols
        self.slot_duration = slot_duration(subcarrier_spacing)

        if phy_abstraction is None:
            phy_abstraction = PHYAbstraction(device=DEVICE)
        self.phy_abstraction = phy_abstraction
        self._eesm = EESM(device=DEVICE)
        self._tb_size = tb_size_per_mcs(self.num_allocated_re)
        if channel_source is None:
            channel_source = TDLChannelGain(
                speed=speed,
                tdl_model=tdl_model,
                delay_spread=delay_spread,
                carrier_frequency=carrier_frequency,
                num_prbs=num_prbs,
                subcarrier_spacing=subcarrier_spacing,
                slot_duration=self.slot_duration,
            )
        elif channel_source.num_prbs != num_prbs:
            raise ValueError(
                f"channel_source has {channel_source.num_prbs} PRBs, the simulator {num_prbs}"
            )
        self._source: ChannelSource = channel_source
        self.reset(seed)

    def reset(self, seed: int, *, snr_db: float | None = None) -> None:
        """Generate a new episode from ``seed`` and restart at slot 0.

        :param snr_db: New mean SNR [dB]; if `None`, the current ``snr_db`` is kept. Not
            used for episodes whose channel sets ``reference_snr_db``.
        """
        if snr_db is not None:
            self.snr_db = snr_db
        # Independent seeds for the channel and the ACK draws
        channel_seed, ack_seed = (
            int(s) for s in np.random.SeedSequence(seed).generate_state(2, dtype=np.uint64)
        )
        episode = self._source.generate(self.num_slots, self.batch_size, channel_seed)
        expected = (self.batch_size, self.num_slots, self._source.num_prbs)
        if tuple(episode.gain.shape) != expected or episode.gain.dtype != torch.float32:
            raise ValueError(
                f"channel gain is {episode.gain.dtype} with shape {tuple(episode.gain.shape)}, "
                f"expected torch.float32 with shape {expected}"
            )
        reference = episode.reference_snr_db
        if reference is None:
            reference = self.snr_db
        elif isinstance(reference, np.ndarray):
            reference = torch.as_tensor(reference, dtype=torch.float32).reshape(-1, 1, 1)
        # [B, num_slots, num_prbs], linear
        self.sinr = 10.0 ** (reference / 10.0) * episode.gain
        mean_sinr = self.sinr.double().mean(dim=(1, 2))
        self.channel_info = episode.info | {
            "realized_snr_db": (10.0 * torch.log10(mean_sinr)).tolist()
        }
        self._ack_rng = torch.Generator(device=DEVICE).manual_seed(ack_seed)
        self.slot = 0

    def current_sinr_grid(self) -> torch.Tensor:
        """Per-RE SINR of the current slot, see :func:`sinr_grid`."""
        return sinr_grid(self.sinr[:, self.slot], self.num_data_symbols)

    def step(self, mcs: torch.Tensor | int) -> LinkResult:
        """Transmit one slot with ``mcs`` (scalar or [B]) and advance to the next slot."""
        if self.slot >= self.num_slots:
            raise RuntimeError("The episode is over; call reset()")
        mcs = torch.as_tensor(mcs, dtype=torch.int32).broadcast_to((self.batch_size,))
        if ((mcs < MIN_MCS) | (mcs > MAX_MCS)).any():
            raise ValueError(f"MCS must be in [{MIN_MCS}, {MAX_MCS}]")

        # [B, 1 symbol, num_prbs, 1 user, 1 stream] -> [B]
        sinr = self.sinr[:, self.slot, None, :, None, None]
        sinr_eff = self._eesm(sinr, mcs[:, None], MCS_TABLE_INDEX, MCS_CATEGORY)[:, 0]
        u = torch.rand(self.batch_size, generator=self._ack_rng)
        result = transmit(
            self.phy_abstraction, self._tb_size, mcs, sinr_eff, self.num_allocated_re, u
        )
        self.slot += 1
        return result

    def close(self) -> None:
        """Release the resources of the channel source, e.g. an open trace file."""
        close = getattr(self._source, "close", None)
        if close is not None:
            close()


def run_episode(
    sim: LinkSimulator,
    policy: str,
    *,
    bler_target: float = 0.1,
    feedback_delay: int = 1,
    fixed_mcs: int | None = None,
) -> LinkResult:
    """Run ``policy`` over the remaining slots of the current episode of ``sim``.

    Policies:

    - ``"fixed"``: always ``fixed_mcs``.
    - ``"illa"``: Sionna ILLA on the latest reported effective SINR.
    - ``"olla"``: Sionna OLLA on the latest reported HARQ feedback and effective SINR.
    - ``"oracle"``: Sionna ILLA on the current slot's per-RE SINR (no feedback delay).
      ILLA then computes the effective SINR of every candidate MCS as the PHY will.

    The report of slot t (ACK and effective SINR) reaches ILLA and OLLA at slot
    t + ``feedback_delay``. Until the first report arrives both assume an effective
    SINR of 0 dB, OLLA's default initial value.

    :output result: Stacked outcomes, tensors of shape [B, num_slots]
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    if feedback_delay < 1:
        raise ValueError("feedback_delay must be >= 1; use policy='oracle' for no delay")
    if policy == "fixed" and fixed_mcs is None:
        raise ValueError("policy='fixed' requires fixed_mcs")

    batch_size = sim.batch_size
    num_re = torch.full((batch_size, 1), sim.num_allocated_re, dtype=torch.int32)
    la_args = {"mcs_table_index": MCS_TABLE_INDEX, "mcs_category": MCS_CATEGORY}
    illa = olla = None
    if policy in ("illa", "oracle"):
        illa = InnerLoopLinkAdaptation(sim.phy_abstraction, bler_target=bler_target, device=DEVICE)
    elif policy == "olla":
        olla = OuterLoopLinkAdaptation(
            sim.phy_abstraction,
            num_ut=1,
            batch_size=batch_size,
            bler_target=bler_target,
            device=DEVICE,
        )

    reported_sinr_eff = torch.ones(batch_size)  # 0 dB until the first report
    results: list[LinkResult] = []
    for _ in range(sim.num_slots - sim.slot):
        k = len(results) - feedback_delay
        report = results[k] if k >= 0 else None

        if policy == "fixed":
            mcs = torch.full((batch_size,), fixed_mcs, dtype=torch.int32)
        elif policy == "oracle":
            mcs = illa(sinr=sim.current_sinr_grid(), **la_args)[:, 0]
        elif policy == "illa":
            if report is not None:
                reported_sinr_eff = report.sinr_eff
            mcs = illa(sinr_eff=reported_sinr_eff[:, None], num_allocated_re=num_re, **la_args)
            mcs = mcs[:, 0]
        else:
            if report is None:
                harq = torch.full((batch_size,), -1, dtype=torch.int32)  # missing
                sinr_eff = torch.zeros(batch_size)  # non-positive means missing
            else:
                harq, sinr_eff = report.ack.to(torch.int32), report.sinr_eff
            mcs = olla(num_re, harq[:, None], sinr_eff[:, None], **la_args)[:, 0]

        results.append(sim.step(mcs))

    return LinkResult(
        **{
            f.name: torch.stack([getattr(r, f.name) for r in results], dim=1)
            for f in fields(LinkResult)
        }
    )
