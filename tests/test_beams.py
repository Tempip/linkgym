"""linkgym.beams: DFT codebooks, steering vectors, beam gain, RSRP."""

import math
import subprocess
import sys

import numpy as np
import pytest
import torch
from sionna.phy.mimo import grid_of_beams_dft, grid_of_beams_dft_ula

from linkgym import beams

ARRAYS = [(1, 32, (1, 1)), (4, 8, (1, 1)), (1, 16, (1, 2)), (4, 8, (2, 2)), (2, 4, (1, 4))]


def visible_direction(u: float, v: float) -> torch.Tensor:
    return torch.tensor([math.sqrt(1 - u * u - v * v), u, v], dtype=torch.float64)


@pytest.mark.parametrize("rows, cols, oversampling", ARRAYS)
def test_codebook_shape_and_unit_norm(rows, cols, oversampling):
    cb = beams.dft_codebook(rows, cols, oversampling=oversampling)
    n = rows * cols
    assert cb.weights.shape == (n * oversampling[0] * oversampling[1], n)
    assert cb.weights.dtype == torch.complex64
    assert (cb.num_beams, cb.num_elements) == tuple(cb.weights.shape)
    assert torch.allclose(cb.weights.abs().square().sum(dim=1), torch.ones(cb.num_beams))


@pytest.mark.parametrize("rows, cols", [(1, 16), (1, 32), (4, 8), (8, 4)])
def test_orthonormal_without_oversampling(rows, cols):
    w = beams.dft_codebook(rows, cols).weights.to(torch.complex128)
    assert torch.allclose(w @ w.conj().T, torch.eye(rows * cols, dtype=torch.complex128), atol=1e-5)


@pytest.mark.parametrize("rows, cols, oversampling", ARRAYS)
def test_beam_gains_add_up_to_the_channel_energy(rows, cols, oversampling):
    cb = beams.dft_codebook(rows, cols, oversampling=oversampling)
    g = torch.Generator().manual_seed(0)
    h = torch.randn(5, 3, rows * cols, dtype=torch.complex128, generator=g)  # [slots, prbs, N]
    gains = beams.beam_gain(h, cb)
    assert gains.shape == (5, 3, cb.num_beams)
    energy = h.abs().square().sum(dim=-1)
    factor = oversampling[0] * oversampling[1]  # tight frame: B / N
    assert torch.allclose(gains.sum(dim=-1), factor * energy, rtol=1e-5)


@pytest.mark.parametrize("rows, cols, oversampling", ARRAYS)
def test_plane_wave_peaks_at_the_beam_pointing_there(rows, cols, oversampling):
    cb = beams.dft_codebook(rows, cols, oversampling=oversampling)
    n = rows * cols
    checked = 0
    for b in range(cb.num_beams):
        u, v = float(cb.u[b]), float(cb.v[b])
        if u * u + v * v >= 1:  # outside the visible region, no plane wave comes from there
            continue
        gains = beams.beam_gain(beams.steering_vector(rows, cols, visible_direction(u, v)), cb)
        assert int(beams.best_beam(gains)) == b
        assert math.isclose(float(gains[b]), n, rel_tol=1e-4)  # full array gain
        checked += 1
    assert checked >= n // 2


def test_ula_off_grid_direction_picks_the_nearest_beam():
    cb = beams.dft_codebook(1, 16, oversampling=(1, 2))
    for u in np.linspace(-0.95, 0.95, 39):
        gains = beams.beam_gain(beams.steering_vector(1, 16, visible_direction(u, 0.0)), cb)
        nearest = np.argmin(np.abs(np.angle(np.exp(1j * np.pi * (u - cb.u.numpy())))))
        assert int(beams.best_beam(gains)) == nearest


def test_upa_vertical_sign():
    # Rows are numbered from the top: a beam with v > 0 points up (+z)
    cb = beams.dft_codebook(4, 4)
    up = beams.beam_gain(beams.steering_vector(4, 4, visible_direction(0.0, 0.5)), cb)
    assert float(cb.v[beams.best_beam(up)]) == pytest.approx(0.5)
    assert float(cb.u[beams.best_beam(up)]) == pytest.approx(0.0)


def test_weights_are_sionna_grid_of_beams():
    assert torch.equal(beams.dft_codebook(1, 32).weights, grid_of_beams_dft_ula(32, 1))
    assert torch.equal(
        beams.dft_codebook(1, 16, oversampling=(1, 2)).weights, grid_of_beams_dft_ula(16, 2)
    )
    assert torch.equal(
        beams.dft_codebook(4, 8, oversampling=(2, 3)).weights,
        grid_of_beams_dft(4, 8, 2, 3).reshape(-1, 32),
    )


def test_front_back_ambiguity():
    k = torch.tensor([0.6, 0.48, 0.64], dtype=torch.float64)
    mirror = k * torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    assert torch.equal(beams.steering_vector(4, 8, k), beams.steering_vector(4, 8, mirror))


def test_rsrp_dbm():
    gains = torch.ones(52, 4)
    gains[:, 2] = 10.0
    rsrp = beams.rsrp_dbm(gains, 30.0)
    per_re = 30.0 - 10 * math.log10(12 * 52)
    assert torch.allclose(
        rsrp, torch.tensor([per_re, per_re, per_re + 10, per_re], dtype=torch.float64)
    )
    batched = beams.rsrp_dbm(torch.ones(7, 52, 4), 30.0)
    assert batched.shape == (7, 4)
    assert int(beams.best_beam(rsrp)) == 2


def test_input_validation():
    with pytest.raises(ValueError, match="single-row"):
        beams.dft_codebook(1, 8, oversampling=(2, 1))
    with pytest.raises(ValueError, match="antennas"):
        beams.beam_gain(torch.ones(7, dtype=torch.complex64), beams.dft_codebook(1, 8))
    with pytest.raises(ValueError, match=r"\[..., 3\]"):
        beams.steering_vector(1, 8, torch.ones(2))


def test_import_linkgym_does_not_import_torch():
    code = "import sys, linkgym; assert 'torch' not in sys.modules, 'torch imported'"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.rt
@pytest.mark.parametrize("rows, cols", [(1, 32), (4, 8)])
def test_matches_sionna_rt_planar_array_and_synthetic_phase(rows, cols):
    try:
        import mitsuba as mi
        import sionna.rt as rt
    except ImportError as e:
        pytest.skip(f"Sionna RT is not available: {e}")
    if not mi.variant().startswith("cuda"):
        pytest.skip(f"no CUDA GPU (Mitsuba variant {mi.variant()})")
    array = rt.PlanarArray(num_rows=rows, num_cols=cols, pattern="iso", polarization="V")
    y, z = beams._element_positions(rows, cols, (0.5, 0.5))
    assert np.allclose(array.normalized_positions.y.numpy(), y.numpy())
    assert np.allclose(array.normalized_positions.z.numpy(), z.numpy())

    # Line-of-sight path in an empty scene: the synthetic array's per-antenna coefficients
    # have the phases of linkgym's steering vector towards the receiver
    scene = rt.load_scene()
    scene.tx_array = array
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    rx_position = np.array([40.0, -25.0, 15.0])
    scene.add(rt.Transmitter(name="tx", position=mi.Point3f(0.0, 0.0, 0.0)))
    scene.add(rt.Receiver(name="rx", position=mi.Point3f(*(float(x) for x in rx_position))))
    paths = rt.PathSolver()(scene, max_depth=0, synthetic_array=True)
    a = (paths.a[0].numpy() + 1j * paths.a[1].numpy()).reshape(rows * cols, -1)[:, 0]
    k = torch.tensor(rx_position / np.linalg.norm(rx_position))
    sv = beams.steering_vector(rows, cols, k).numpy()
    assert np.allclose(a / a[0], sv / sv[0], atol=1e-4)
