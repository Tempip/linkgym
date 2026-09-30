"""Link adaptation baselines that act through the ``LinkAdaptation-v0`` API.

Every policy has ``reset()`` and ``act(obs, info) -> action``. At step t, ``act`` gets the
observation and info returned by the previous ``step`` (or by ``reset``), i.e. exactly
what an agent sees before choosing the MCS of slot t. ILLA and OLLA read only
``info["report"]`` and the constant ``info["num_allocated_re"]``; the oracle reads
``info["privileged"]``, the SINR of slot t itself.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch
from sionna.sys import InnerLoopLinkAdaptation, OuterLoopLinkAdaptation, PHYAbstraction

from linkgym.sim import DEVICE, MAX_MCS, MCS_CATEGORY, MCS_TABLE_INDEX, MIN_MCS, sinr_grid

_LA_ARGS = {"mcs_table_index": MCS_TABLE_INDEX, "mcs_category": MCS_CATEGORY}


class Policy(Protocol):
    """Interface of the baselines; :func:`linkgym.evaluate` only requires ``act``."""

    def reset(self) -> None: ...

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int: ...


def _action(mcs: torch.Tensor) -> int:
    return int(mcs.reshape(-1)[0]) - MIN_MCS


def _db_to_lin(value_db: float) -> float:
    return 10.0 ** (value_db / 10.0)


class FixedMCSPolicy:
    """Always transmits with the same MCS."""

    def __init__(self, mcs: int) -> None:
        if not MIN_MCS <= mcs <= MAX_MCS:
            raise ValueError(f"mcs must be in [{MIN_MCS}, {MAX_MCS}]")
        self.mcs = mcs

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        return self.mcs - MIN_MCS


class ILLAPolicy:
    """Sionna ILLA on the latest wideband SINR report.

    Until the first report arrives, it assumes a SINR of 0 dB (OLLA's default initial
    value).
    """

    def __init__(
        self, bler_target: float = 0.1, *, phy_abstraction: PHYAbstraction | None = None
    ) -> None:
        if phy_abstraction is None:
            phy_abstraction = PHYAbstraction(device=DEVICE)
        self.bler_target = bler_target
        self._illa = InnerLoopLinkAdaptation(phy_abstraction, bler_target, device=DEVICE)
        self.reset()

    def reset(self) -> None:
        self._sinr = 1.0

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        report = info["report"]
        if report is not None:
            self._sinr = _db_to_lin(report["sinr_wideband_db"])
        mcs = self._illa(
            sinr_eff=torch.tensor([self._sinr]),
            num_allocated_re=torch.tensor([info["num_allocated_re"]], dtype=torch.int32),
            **_LA_ARGS,
        )
        return _action(mcs)


class OLLAPolicy:
    """Sionna OLLA fed the latest report: its ACK as HARQ feedback and its wideband SINR
    as effective SINR. Until the first report arrives the feedback is marked missing.

    :param bler_target: TBLER target
    :param delta_up: Offset increase [dB] after a NACK; the decrease after an ACK is
        ``delta_up * bler_target / (1 - bler_target)``. Defaults to Sionna's 1.0.
    """

    def __init__(
        self,
        bler_target: float = 0.1,
        *,
        delta_up: float = 1.0,
        phy_abstraction: PHYAbstraction | None = None,
    ) -> None:
        if phy_abstraction is None:
            phy_abstraction = PHYAbstraction(device=DEVICE)
        self.bler_target = bler_target
        self.delta_up = delta_up
        self._olla = OuterLoopLinkAdaptation(
            phy_abstraction, num_ut=1, bler_target=bler_target, delta_up=delta_up, device=DEVICE
        )

    def reset(self) -> None:
        self._olla.reset()

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        report = info["report"]
        if report is None:
            harq, sinr = -1, 0.0  # missing feedback; non-positive SINR means missing
        else:
            harq, sinr = int(report["ack"]), _db_to_lin(report["sinr_wideband_db"])
        mcs = self._olla(
            torch.tensor([info["num_allocated_re"]], dtype=torch.int32),
            torch.tensor([harq], dtype=torch.int32),
            torch.tensor([sinr]),
            **_LA_ARGS,
        )
        return _action(mcs)


class OraclePolicy:
    """Sionna ILLA on the per-RE SINR of the slot being decided (``info["privileged"]``).

    ILLA computes the effective SINR of every candidate MCS as the PHY will. This is an
    upper-bound reference and uses information no real scheduler has.
    """

    def __init__(
        self, bler_target: float = 0.1, *, phy_abstraction: PHYAbstraction | None = None
    ) -> None:
        if phy_abstraction is None:
            phy_abstraction = PHYAbstraction(device=DEVICE)
        self.bler_target = bler_target
        self._illa = InnerLoopLinkAdaptation(phy_abstraction, bler_target, device=DEVICE)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        privileged = info["privileged"]
        sinr_prb = torch.from_numpy(privileged["sinr_prb"])[None]
        grid = sinr_grid(sinr_prb, privileged["num_data_symbols"])
        return _action(self._illa(sinr=grid, **_LA_ARGS))


class SB3Policy:
    """Adapter for Stable-Baselines3 models or anything with ``predict(obs, deterministic)``."""

    def __init__(self, model: Any, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)
