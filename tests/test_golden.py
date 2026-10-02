"""Golden outputs of the TDL channel path: simulator, environment and a baseline.

The hashes fix the exact float32 outputs, which can differ across operating systems, CPUs
and library versions. They are checked bit for bit only on the platform they were recorded
on (RECORDED_ON); elsewhere the tests are skipped. Do not update the values to make a
refactor pass: a changed hash means the TDL path changed.
"""

import hashlib
import platform
from dataclasses import fields
from importlib.metadata import version

import gymnasium
import numpy as np
import pytest
import torch

from linkgym import ENV_ID, evaluate
from linkgym.baselines import OLLAPolicy
from linkgym.sim import LinkSimulator, run_episode

RECORDED_ON = {
    "system": "Windows",
    "machine": "AMD64",
    "torch": "2.14.0+cpu",
    "cpu_capability": "AVX512",
    "sionna-no-rt": "2.1.0",
    "numpy": "2.5.3",
}

GOLDEN = {
    "sim_sinr": "072cdb0d12bc729fe443cf1493bdba4df18953bed344d6bd21e83f83756f9a57",
    "sim_olla": "bdd5ed7eead2aa51ee36f80f035eaa8d1ed68c0ce4162339cf7fb4365827346d",
    "sim_fixed": "0f28037c61d3a6640ffcde04f07c52329602b50b2351f9294e99fa116034d726",
    "sim_olla_decoded_bits": 8734168,
    "env_trajectory": "198211a7c929987533ab463f165e55d8de3ae84b77294c6b451ae1c012ae8ac4",
    "env_reward_sum_hex": "0x1.f30d740256f55p+3",
    "baseline_olla_tuned": "8a73ab5d7095b6e425db3d7a96f0f1e20859dda872a9067842fb8261c966ff05",
    "baseline_olla_tuned_goodput_hex": "0x1.03379fa97e133p+5",
}


def platform_signature() -> dict[str, str]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "sionna-no-rt": version("sionna-no-rt"),
        "numpy": np.__version__,
    }


def digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def result_digest(result) -> str:
    return digest(*(getattr(result, f.name).numpy() for f in fields(result)))


def compute_sim() -> dict:
    sim = LinkSimulator(200, batch_size=2, seed=0)
    sinr = digest(sim.sinr.numpy())
    olla = run_episode(sim, "olla")
    sim.reset(1)
    fixed = run_episode(sim, "fixed", fixed_mcs=14)
    return {
        "sim_sinr": sinr,
        "sim_olla": result_digest(olla),
        "sim_fixed": result_digest(fixed),
        "sim_olla_decoded_bits": int(olla.decoded_bits.sum()),
    }


def compute_env() -> dict:
    env = gymnasium.make(ENV_ID, episode_length=200)
    obs, info = env.reset(seed=3)
    observations, privileged = [obs], [info["privileged"]["sinr_prb"]]
    rewards, mcs, ack, bits, tbler, report_sinr = [], [], [], [], [], []
    for action in np.random.default_rng(0).integers(26, size=200):
        obs, reward, _, truncated, info = env.step(int(action))
        observations.append(obs)
        rewards.append(reward)
        mcs.append(info["mcs"])
        ack.append(info["ack"])
        bits.append(info["bits"])
        tbler.append(info["tbler"])
        report_sinr.append(info["report"]["sinr_wideband_db"])
        if info["privileged"] is not None:
            privileged.append(info["privileged"]["sinr_prb"])
    assert truncated
    trajectory = digest(
        np.stack(observations),
        np.array(rewards),
        np.array(mcs),
        np.array(ack),
        np.array(bits),
        np.array(tbler),
        np.array(report_sinr),
        np.stack(privileged),
        np.array([info["snr_db"]]),
    )
    return {"env_trajectory": trajectory, "env_reward_sum_hex": float(np.sum(rewards)).hex()}


def compute_baseline() -> dict:
    result = evaluate(OLLAPolicy(0.1, delta_up=0.25), {"episode_length": 200}, seeds=[1000])
    values = [
        [float(e[k]).hex() for k in ("snr_db", "goodput_mbps", "observed_tbler", "mean_mcs")]
        for e in result["per_episode"]
    ]
    return {
        "baseline_olla_tuned": digest(np.array(values)),
        "baseline_olla_tuned_goodput_hex": float(result["goodput_mbps"]["mean"]).hex(),
    }


@pytest.fixture
def recorded_platform():
    signature = platform_signature()
    if signature != RECORDED_ON:
        pytest.skip(f"golden outputs recorded on {RECORDED_ON}, this platform is {signature}")


def _check(computed: dict) -> None:
    for key, value in computed.items():
        assert value == GOLDEN[key], f"{key}: computed {value!r}, golden {GOLDEN[key]!r}"


def test_golden_simulator(recorded_platform):
    _check(compute_sim())


def test_golden_environment(recorded_platform):
    _check(compute_env())


def test_golden_baseline(recorded_platform):
    _check(compute_baseline())
