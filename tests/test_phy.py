"""linkgym.phy: the public link-level API, and that the simulator uses it."""

import pytest
import torch
from sionna.phy import config as sionna_config
from sionna.sys import EESM, PHYAbstraction

from linkgym import phy
from linkgym.sim import LinkSimulator

NUM_RE = 12 * 52 * 12


@pytest.fixture(scope="module")
def abstraction():
    return PHYAbstraction(device="cpu")


def links(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    sinr_prb = 10 ** (torch.empty(n, 52).uniform_(-0.5, 2.5, generator=g))  # -5 to 25 dB
    mcs = torch.randint(phy.MIN_MCS, phy.MAX_MCS + 1, (n,), generator=g, dtype=torch.int32)
    u = torch.rand(n, generator=g)
    return sinr_prb.float(), mcs, u


def test_constants():
    assert (phy.MIN_MCS, phy.MAX_MCS, phy.MCS_TABLE_INDEX, phy.MCS_CATEGORY) == (3, 28, 1, 1)


def test_tb_size_per_mcs():
    tb = phy.tb_size_per_mcs(NUM_RE)
    assert tb.shape == (29,) and tb.dtype == torch.int32
    # Non-decreasing in MCS, except 16 -> 17: in MCS table 1 of TS 38.214, MCS 17 (64-QAM,
    # rate 438/1024) has a slightly lower spectral efficiency than MCS 16 (16-QAM, 658/1024)
    decreases = [m for m in range(phy.MIN_MCS + 1, phy.MAX_MCS + 1) if tb[m] < tb[m - 1]]
    assert decreases == [17]
    assert int(tb[phy.MAX_MCS]) == int(tb.max())
    assert torch.all(phy.tb_size_per_mcs(NUM_RE // 2) <= tb)


def test_effective_sinr_is_sionna_eesm():
    sinr_prb, mcs, _ = links()
    reference = EESM(device="cpu")(sinr_prb[:, None, :, None, None], mcs[:, None], 1, 1)[:, 0]
    assert torch.equal(phy.effective_sinr(sinr_prb, mcs), reference)
    # A flat channel has its SINR as effective SINR
    flat = torch.full((3, 52), 10.0)
    assert torch.allclose(phy.effective_sinr(flat, mcs[:3]), torch.full((3,), 10.0), rtol=1e-5)


def test_transmit_matches_phy_abstraction_draws(abstraction):
    sinr_prb, mcs, _ = links()
    sinr_eff = phy.effective_sinr(sinr_prb, mcs)
    n = mcs.numel()
    # Draw the uniforms PHYAbstraction draws from Sionna's generator, rewind, let it draw
    generator = sionna_config.torch_rng("cpu")
    state = generator.get_state()
    try:
        u = torch.rand((n, 1), dtype=torch.float32, generator=generator)
        generator.set_state(state)
        ref_bits, ref_harq, _, ref_tbler, ref_cb_bler = abstraction(
            mcs[:, None],
            sinr_eff=sinr_eff[:, None],
            num_allocated_re=torch.full((n, 1), NUM_RE, dtype=torch.int32),
            mcs_table_index=1,
            mcs_category=1,
        )
    finally:
        generator.set_state(state)
    result = phy.transmit(sinr_eff, mcs, u[:, 0], num_allocated_re=NUM_RE)
    assert torch.equal(result.ack, ref_harq[:, 0] == 1)
    assert torch.equal(result.bits, ref_bits[:, 0])
    assert torch.equal(result.tbler, ref_tbler[:, 0])
    assert torch.equal(result.cb_bler, ref_cb_bler[:, 0])
    assert result.ack.any() and not result.ack.all()


def test_transmit_common_random_numbers(abstraction):
    sinr_prb, mcs, u = links()
    sinr_eff = phy.effective_sinr(sinr_prb, mcs)
    a = phy.transmit(sinr_eff, mcs, u, num_allocated_re=NUM_RE)
    b = phy.transmit(
        sinr_eff,
        mcs,
        u,
        num_allocated_re=NUM_RE,
        phy_abstraction=abstraction,
        tb_size=phy.tb_size_per_mcs(NUM_RE),
    )
    for name in ("ack", "bits", "tbler", "cb_bler"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    # The ACK is exactly u >= tbler; the delivered bits are the TB size on ACK
    assert torch.equal(a.ack, u >= a.tbler)
    tb = phy.tb_size_per_mcs(NUM_RE)
    assert torch.equal(a.bits, torch.where(a.ack, tb[mcs], 0))


def test_transmit_validates_inputs():
    sinr_prb, mcs, u = links(4)
    sinr_eff = phy.effective_sinr(sinr_prb, mcs)
    with pytest.raises(ValueError, match="same shape"):
        phy.transmit(sinr_eff, mcs, u[:3], num_allocated_re=NUM_RE)
    with pytest.raises(ValueError, match=r"MCS must be in \[3, 28\]"):
        phy.transmit(sinr_eff, torch.full((4,), 2, dtype=torch.int32), u, num_allocated_re=NUM_RE)
    with pytest.raises(TypeError):
        phy.transmit(sinr_eff, mcs, u)  # num_allocated_re is a required keyword


def test_simulator_step_is_phy_transmit():
    sim = LinkSimulator(5, batch_size=8, seed=3)
    state = sim._ack_rng.get_state()
    mcs = torch.full((8,), 12, dtype=torch.int32)
    stepped = sim.step(mcs)
    u = torch.rand(8, generator=torch.Generator().set_state(state))
    manual = phy.transmit(
        phy.effective_sinr(sim.sinr[:, 0], mcs), mcs, u, num_allocated_re=sim.num_allocated_re
    )
    assert torch.equal(stepped.ack, manual.ack)
    assert torch.equal(stepped.decoded_bits, manual.bits)
    assert torch.equal(stepped.tbler, manual.tbler)
    assert torch.equal(stepped.sinr_eff, phy.effective_sinr(sim.sinr[:, 0], mcs))
