"""Link-level abstraction of one slot: effective SINR, transport block size and ACK.

Stability: stable (see docs/api_stability.md). The same functions run the simulator behind
``LinkAdaptation-v0``, so a link simulated with them outside the environment behaves
exactly like one inside it.

Fixed setup, as in the environment: PDSCH (MCS category 1), MCS table 1, single layer,
CPU, float32. Sionna 2.1 has no BLER data for MCS 0-2 of that table, hence
``MIN_MCS = 3``.

The random part of a transmission is the ACK: it is drawn from uniforms ``u`` that the
caller provides, so two policies (or two runs) given the same ``u`` face the same draws
(common random numbers). A transport block is acknowledged if ``u >= tbler``, the rule
Sionna's ``PHYAbstraction`` applies to its own draws.

    sinr_eff = effective_sinr(sinr_prb, mcs)           # [B] linear
    result = transmit(sinr_eff, mcs, u, num_allocated_re=12 * 52 * 12)
    result.ack, result.bits, result.tbler
"""

from __future__ import annotations

import functools
import importlib.metadata
from dataclasses import dataclass

import torch


def _sionna_import_error(error: ImportError) -> ImportError | None:
    """A clearer error if Sionna failed to import because sionna-rt is installed."""
    try:
        importlib.metadata.version("sionna-rt")
    except importlib.metadata.PackageNotFoundError:
        return None
    # sionna-rt's sionna/__init__.py replaces sionna-no-rt's and imports Sionna RT, which
    # needs a CUDA GPU or LLVM, on every `import sionna`
    return ImportError(
        f"Importing Sionna failed: {error}. Sionna RT (sionna-rt, installed by the "
        "linkgym[rt] extra) is installed in this environment; it makes every `import sionna` "
        "import Sionna RT, which needs a CUDA GPU or LLVM. Install linkgym[rt] in its own "
        "environment for trace generation and train in an environment without it "
        "(docs/channels.md)."
    )


try:
    from sionna.phy.nr.utils import MCSDecoderNR, TransportBlockNR
    from sionna.sys import EESM, PHYAbstraction
except ImportError as _error:
    _hint = _sionna_import_error(_error)
    if _hint is None:
        raise
    raise _hint from _error

__all__ = [
    "MAX_MCS",
    "MCS_CATEGORY",
    "MCS_TABLE_INDEX",
    "MIN_MCS",
    "Transmission",
    "effective_sinr",
    "tb_size_per_mcs",
    "transmit",
]

DEVICE = "cpu"
MCS_CATEGORY = 1  # PDSCH
MCS_TABLE_INDEX = 1
MIN_MCS = 3  # Sionna 2.1 ships no BLER data for PDSCH table 1, MCS 0-2
MAX_MCS = 28


@dataclass(frozen=True)
class Transmission:
    """Outcome of one transport block per link, tensors of shape [B].

    :param ack: bool, `True` if the transport block was decoded (``u >= tbler``)
    :param bits: int32, information bits delivered: the TB size on ACK, else 0
    :param tbler: transport block error probability the ACK was drawn from
    :param cb_bler: code block error probability
    """

    ack: torch.Tensor
    bits: torch.Tensor
    tbler: torch.Tensor
    cb_bler: torch.Tensor


def tb_size_per_mcs(num_allocated_re: int) -> torch.Tensor:
    """Transport block information bits for every MCS index 0-28, shape [29].

    :param num_allocated_re: Resource elements carrying data in the slot (single layer),
        e.g. 12 subcarriers x 52 PRBs x 12 data symbols
    """
    mcs = torch.arange(MAX_MCS + 1, dtype=torch.int32)
    modulation_order, coderate = MCSDecoderNR(device=DEVICE)(mcs, MCS_TABLE_INDEX, MCS_CATEGORY)
    tb_size, _, _ = TransportBlockNR(device=DEVICE).transport_block_size(
        modulation_order, coderate, modulation_order * num_allocated_re
    )
    return tb_size


def effective_sinr(sinr_prb: torch.Tensor, mcs: torch.Tensor) -> torch.Tensor:
    """EESM effective SINR of each link for its MCS, linear, shape [B].

    :param sinr_prb: [B, num_prbs] linear SINR per PRB (float32); every subcarrier of a PRB
        has the PRB's SINR
    :param mcs: [B] MCS index per link (int32)
    """
    sinr = sinr_prb[:, None, :, None, None]  # [B, 1 symbol, num_prbs, 1 user, 1 stream]
    return _eesm()(sinr, mcs[:, None], MCS_TABLE_INDEX, MCS_CATEGORY)[:, 0]


def transmit(
    sinr_eff: torch.Tensor,
    mcs: torch.Tensor,
    u: torch.Tensor,
    *,
    num_allocated_re: int,
    phy_abstraction: PHYAbstraction | None = None,
    tb_size: torch.Tensor | None = None,
) -> Transmission:
    """Transmit one transport block per link and draw its ACK from the uniforms ``u``.

    ``tbler`` and ``cb_bler`` come from Sionna's ``PHYAbstraction``
    (phy_abstraction.py:41); the ACK rule ``u >= tbler`` is the one it applies to its own
    draws (phy_abstraction.py:680-684), so with the same ``u`` both give the same ACK and
    the same delivered bits.

    :param sinr_eff: [B] linear effective SINR per link, e.g. from :func:`effective_sinr`
    :param mcs: [B] MCS index per link (int32), in [MIN_MCS, MAX_MCS]
    :param u: [B] uniform draws in [0, 1), one per link
    :param num_allocated_re: Resource elements carrying data in the slot (single layer)
    :param phy_abstraction: A ``PHYAbstraction`` to use; by default one shared instance
        (loading its BLER tables takes seconds)
    :param tb_size: TB size per MCS index from :func:`tb_size_per_mcs` for
        ``num_allocated_re``, to avoid recomputing it every slot
    """
    if not (sinr_eff.shape == mcs.shape == u.shape) or mcs.ndim != 1:
        raise ValueError(
            f"sinr_eff, mcs and u must have the same shape [B], got {tuple(sinr_eff.shape)}, "
            f"{tuple(mcs.shape)} and {tuple(u.shape)}"
        )
    if ((mcs < MIN_MCS) | (mcs > MAX_MCS)).any():
        raise ValueError(f"MCS must be in [{MIN_MCS}, {MAX_MCS}]")
    if phy_abstraction is None:
        phy_abstraction = _phy_abstraction()
    if tb_size is None:
        tb_size = _tb_size(num_allocated_re)
    num_re = torch.full((mcs.shape[0], 1), num_allocated_re, dtype=torch.int32)
    *_, tbler, cb_bler = phy_abstraction(
        mcs[:, None],
        sinr_eff=sinr_eff[:, None],
        num_allocated_re=num_re,
        mcs_table_index=MCS_TABLE_INDEX,
        mcs_category=MCS_CATEGORY,
    )
    tbler, cb_bler = tbler[:, 0], cb_bler[:, 0]
    ack = u >= tbler
    bits = torch.where(ack, tb_size[mcs], 0)
    return Transmission(ack=ack, bits=bits, tbler=tbler, cb_bler=cb_bler)


@functools.cache
def _eesm() -> EESM:
    return EESM(device=DEVICE)


@functools.cache
def _phy_abstraction() -> PHYAbstraction:
    return PHYAbstraction(device=DEVICE)


@functools.cache
def _tb_size(num_allocated_re: int) -> torch.Tensor:
    return tb_size_per_mcs(num_allocated_re)
