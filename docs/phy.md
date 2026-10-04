# Link-level API: `linkgym.phy`

Since 0.3.0.

Status: **stable** (see [api_stability.md](api_stability.md)).

`linkgym.phy` turns the SINR of a slot into delivered bits, with the same functions the
simulator behind `LinkAdaptation-v0` uses. A link simulated with them outside the
environment, for example in your own multi-user or beam-management environment, behaves
exactly like one inside it.

Fixed setup, as in the environment: PDSCH (MCS category 1), MCS table 1 of TS 38.214
(`MIN_MCS` = 3 to `MAX_MCS` = 28; Sionna 2.1 has no BLER data for MCS 0-2 of that
table), a single layer, CPU and float32.

| name | what it does |
|---|---|
| `tb_size_per_mcs(num_allocated_re)` | transport block information bits for every MCS 0-28, shape [29] |
| `effective_sinr(sinr_prb, mcs)` | EESM effective SINR [B] (linear) from the per-PRB SINR [B, num_prbs] (linear) and the MCS [B] |
| `transmit(sinr_eff, mcs, u, *, num_allocated_re, phy_abstraction=None, tb_size=None)` | one transport block per link: `Transmission(ack, bits, tbler, cb_bler)`, tensors [B] |
| `MIN_MCS`, `MAX_MCS`, `MCS_TABLE_INDEX`, `MCS_CATEGORY` | the fixed setup |

**The ACK comes from your uniforms.** `transmit` draws nothing itself: the transport block
is acknowledged if `u >= tbler`, the rule Sionna's `PHYAbstraction` applies to its own
draws. Give two policies the same `u` and they face the same draws (common random
numbers); draw `u` from a generator seeded per episode to make runs reproducible.

`num_allocated_re` is the number of resource elements carrying data in the slot: 12
subcarriers x number of PRBs x data symbols per slot (the environment's default: 12 x 52 x
12 = 7488). Loading the BLER tables takes a few seconds, so `transmit` shares one
`PHYAbstraction` unless you pass your own; pass `tb_size` from `tb_size_per_mcs` to avoid
recomputing it every slot.

```python
import torch

from linkgym import phy

sinr_prb = torch.full((2, 52), 10.0)  # 2 links, linear SINR of every PRB (10 dB)
mcs = torch.tensor([10, 20], dtype=torch.int32)
u = torch.rand(2, generator=torch.Generator().manual_seed(0))  # your uniform draws

sinr_eff = phy.effective_sinr(sinr_prb, mcs)
result = phy.transmit(sinr_eff, mcs, u, num_allocated_re=12 * 52 * 12)
print(result.ack, result.bits, result.tbler)
```

The effective SINR is computed per link for its own MCS; to compare MCSs, call
`effective_sinr` once per MCS. The BLER data covers SINRs from -5 to 20 dB (table 1);
outside that range Sionna holds the BLER at the edge values (see
[benchmarks.md](benchmarks.md#known-limitations)).
