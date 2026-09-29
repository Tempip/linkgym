import time
from dataclasses import fields

import pytest
import torch
from sionna.phy import config as sionna_config
from sionna.sys import PHYAbstraction

from linkgym.sim import (
    MAX_MCS,
    MCS_CATEGORY,
    MCS_TABLE_INDEX,
    MIN_MCS,
    LinkSimulator,
    run_episode,
    tb_size_per_mcs,
    transmit,
)


@pytest.fixture(scope="module")
def phy():
    return PHYAbstraction(device="cpu")


def _assert_results_equal(a, b):
    for f in fields(a):
        assert torch.equal(getattr(a, f.name), getattr(b, f.name)), f.name


def test_same_seed_gives_identical_outputs(phy):
    sim_a = LinkSimulator(40, batch_size=2, seed=7, phy_abstraction=phy)
    sim_b = LinkSimulator(40, batch_size=2, seed=7, phy_abstraction=phy)
    assert torch.equal(sim_a.sinr, sim_b.sinr)
    _assert_results_equal(run_episode(sim_a, "olla"), run_episode(sim_b, "olla"))


def test_different_seeds_differ(phy):
    sim_a = LinkSimulator(40, batch_size=2, seed=1, phy_abstraction=phy)
    sim_b = LinkSimulator(40, batch_size=2, seed=2, phy_abstraction=phy)
    assert not torch.equal(sim_a.sinr, sim_b.sinr)
    result_a, result_b = run_episode(sim_a, "olla"), run_episode(sim_b, "olla")
    assert not torch.equal(result_a.sinr_eff, result_b.sinr_eff)


def test_interleaved_instances_do_not_interact(phy):
    alone = LinkSimulator(30, seed=3, phy_abstraction=phy)
    reference = [alone.step(14) for _ in range(30)]

    sim = LinkSimulator(30, seed=3, phy_abstraction=phy)
    other = LinkSimulator(30, seed=4, phy_abstraction=phy)
    for expected in reference:
        other.step(20)
        _assert_results_equal(sim.step(14), expected)


def test_olla_run_is_fast_and_valid(phy):
    start = time.perf_counter()
    sim = LinkSimulator(50, batch_size=4, seed=0, phy_abstraction=phy)
    result = run_episode(sim, "olla")
    assert time.perf_counter() - start < 5.0

    assert result.mcs.shape == (4, 50)
    assert ((result.mcs >= MIN_MCS) & (result.mcs <= MAX_MCS)).all()
    for p in (result.tbler, result.cb_bler):
        assert ((p >= 0) & (p <= 1)).all()
    assert (result.decoded_bits >= 0).all()
    assert (result.sinr_eff > 0).all()


def test_episode_generation_leaves_sionna_rng_untouched(phy):
    sim = LinkSimulator(20, batch_size=2, seed=0, phy_abstraction=phy)
    generator = sionna_config.torch_rng("cpu")
    before = generator.get_state().clone()
    sim.reset(5)
    assert torch.equal(generator.get_state(), before)


def test_transmit_matches_phy_abstraction(phy):
    num_re = 7488
    mcs_values = torch.arange(MIN_MCS, MAX_MCS + 1, dtype=torch.int32)
    sinr_db = torch.arange(-6.0, 26.0, 0.5)
    mcs = mcs_values.repeat_interleave(len(sinr_db))
    sinr_eff = 10.0 ** (sinr_db.repeat(len(mcs_values)) / 10.0)
    n = mcs.numel()

    # Draw the uniforms PHYAbstraction draws from Sionna's generator
    # (phy_abstraction.py:674-679), rewind, and let PHYAbstraction draw them again.
    generator = sionna_config.torch_rng("cpu")
    state = generator.get_state()
    try:
        u = torch.rand((n, 1), dtype=torch.float32, generator=generator)
        generator.set_state(state)
        ref_bits, ref_harq, _, ref_tbler, ref_cb_bler = phy(
            mcs[:, None],
            sinr_eff=sinr_eff[:, None],
            num_allocated_re=torch.full((n, 1), num_re, dtype=torch.int32),
            mcs_table_index=MCS_TABLE_INDEX,
            mcs_category=MCS_CATEGORY,
        )
    finally:
        generator.set_state(state)

    ours = transmit(phy, tb_size_per_mcs(num_re), mcs, sinr_eff, num_re, u[:, 0])

    assert torch.equal(ours.ack, ref_harq[:, 0] == 1)
    assert torch.equal(ours.decoded_bits, ref_bits[:, 0])
    assert torch.equal(ours.tbler, ref_tbler[:, 0])
    assert torch.equal(ours.cb_bler, ref_cb_bler[:, 0])
    # The comparison covers ACKs, NACKs and the waterfall region
    assert ours.ack.any() and not ours.ack.all()
    assert ((ours.tbler > 0) & (ours.tbler < 1)).any()
