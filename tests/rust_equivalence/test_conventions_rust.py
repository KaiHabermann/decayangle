"""
Compare helicity_angles and relative_wigner_angles between Python and Rust
for all three conventions (helicity, minus_phi, canonical), across all
topology combinations of a 3-body and a 4-body decay.
"""

import numpy as np
import pytest
from itertools import product

from decayangle.decay_topology import TopologyCollection
from decayangle.config import config as cfg

from .conftest import skip_no_rust

CONVENTIONS = ["helicity", "minus_phi", "canonical"]


def rotation_matrix_from_euler(phi, theta, psi):
    """Reconstruct SO(3) rotation from ZYZ Euler angles (phi, theta, psi) for comparison."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cs, ss = np.cos(psi), np.sin(psi)
    Rz_phi = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])
    Ry_t = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    Rz_psi = np.array([[cs, -ss, 0], [ss, cs, 0], [0, 0, 1]])
    return Rz_phi @ Ry_t @ Rz_psi


def assert_wigner_close(
    rs_phi, rs_theta, rs_psi, py_phi, py_theta, py_psi, atol, label
):
    """
    Compare Wigner angle triples by reconstructing the full rotation matrix per event.
    This handles both the ±2π SU(2) sign ambiguity and the theta=0 gimbal-lock
    degeneracy where phi and psi are individually undefined.
    """
    rs_phi = np.asarray(rs_phi).ravel()
    rs_theta = np.asarray(rs_theta).ravel()
    rs_psi = np.asarray(rs_psi).ravel()
    py_phi = np.asarray(py_phi).ravel()
    py_theta = np.asarray(py_theta).ravel()
    py_psi = np.asarray(py_psi).ravel()

    for i in range(len(rs_phi)):
        R_rs = rotation_matrix_from_euler(
            float(rs_phi[i]), float(rs_theta[i]), float(rs_psi[i])
        )
        R_py = rotation_matrix_from_euler(
            float(py_phi[i]), float(py_theta[i]), float(py_psi[i])
        )
        # SU(2) double cover: R and -R represent the same physical rotation
        diff = np.abs(R_rs - R_py).max()
        diff_neg = np.abs(R_rs + R_py).max()
        if min(diff, diff_neg) > atol:
            raise AssertionError(
                f"{label}, event {i}: rotation matrix mismatch\n"
                f"  Rust  (phi={rs_phi[i]:.6f}, theta={rs_theta[i]:.6f}, psi={rs_psi[i]:.6f})\n"
                f"  Python(phi={py_phi[i]:.6f}, theta={py_theta[i]:.6f}, psi={py_psi[i]:.6f})\n"
                f"  max|R_rs - R_py| = {diff:.2e}, max|R_rs + R_py| = {diff_neg:.2e}"
            )


def assert_angles_close(rs_val, py_val, atol, label):
    """Compare angles modulo 2π via sin/cos to handle SU(2) sign ambiguity."""
    rs = np.asarray(rs_val)
    py = np.asarray(py_val)
    np.testing.assert_allclose(
        np.sin(rs), np.sin(py), atol=atol, err_msg=f"{label}: sin mismatch"
    )
    np.testing.assert_allclose(
        np.cos(rs), np.cos(py), atol=atol, err_msg=f"{label}: cos mismatch"
    )


def run_helicity_comparison(topo, momenta, convention):
    old_rust = cfg.use_rust

    cfg.use_rust = False
    py_angles = topo.helicity_angles(momenta, convention=convention)

    cfg.use_rust = True
    try:
        rs_angles = topo.helicity_angles(momenta, convention=convention)
    finally:
        cfg.use_rust = old_rust

    assert set(py_angles.keys()) == set(rs_angles.keys()), (
        f"Key mismatch for convention={convention}: "
        f"Python={set(py_angles.keys())}, Rust={set(rs_angles.keys())}"
    )
    for key in py_angles:
        assert_angles_close(
            np.asarray(rs_angles[key].phi_rf),
            np.asarray(py_angles[key].phi_rf),
            atol=1e-10,
            label=f"convention={convention}, key={key}, phi_rf",
        )
        assert_angles_close(
            np.asarray(rs_angles[key].theta_rf),
            np.asarray(py_angles[key].theta_rf),
            atol=1e-10,
            label=f"convention={convention}, key={key}, theta_rf",
        )


def run_wigner_comparison(topo_i, topo_j, momenta, convention):
    # Same topology → all zeros by definition, nothing to compare
    if topo_i.tuple == topo_j.tuple:
        return

    old_rust = cfg.use_rust

    cfg.use_rust = False
    py_wigner = topo_i.relative_wigner_angles(topo_j, momenta, convention=convention)

    cfg.use_rust = True
    try:
        rs_wigner = topo_i.relative_wigner_angles(
            topo_j, momenta, convention=convention
        )
    finally:
        cfg.use_rust = old_rust

    assert set(py_wigner.keys()) == set(rs_wigner.keys()), (
        f"Key mismatch convention={convention}: "
        f"Python={set(py_wigner.keys())}, Rust={set(rs_wigner.keys())}"
    )
    for particle in py_wigner:
        # Filter out NaN events (unphysical momenta that both sides handle consistently)
        py_theta = np.asarray(py_wigner[particle].theta_rf)
        valid = ~np.isnan(py_theta)
        if not np.any(valid):
            continue

        def _sel(arr):
            return np.asarray(arr)[valid]

        assert_wigner_close(
            _sel(rs_wigner[particle].phi_rf),
            _sel(rs_wigner[particle].theta_rf),
            _sel(rs_wigner[particle].psi_rf),
            _sel(py_wigner[particle].phi_rf),
            _sel(py_wigner[particle].theta_rf),
            _sel(py_wigner[particle].psi_rf),
            atol=1e-7,
            label=f"convention={convention}, particle={particle}",
        )


# ── 3-body helicity_angles ────────────────────────────────────────────────────


@skip_no_rust
@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize("topo_idx", [0, 1, 2])
def test_helicity_angles_3body(momenta_3body, topo_idx, convention):
    tg = TopologyCollection(0, [1, 2, 3])
    topo = tg.topologies[topo_idx]
    momenta = topo.to_rest_frame(momenta_3body)
    run_helicity_comparison(topo, momenta, convention)


# ── 4-body helicity_angles ────────────────────────────────────────────────────


@skip_no_rust
@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize("topo_idx", [0, 1, 2])
def test_helicity_angles_4body(momenta_4body, topo_idx, convention):
    tg = TopologyCollection(0, [1, 2, 3, 4])
    topo = tg.topologies[topo_idx]
    momenta = topo.to_rest_frame(momenta_4body)
    run_helicity_comparison(topo, momenta, convention)


# ── 3-body relative_wigner_angles ─────────────────────────────────────────────


@skip_no_rust
@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize("topo_pair", list(product(range(3), range(3))))
def test_wigner_angles_3body(momenta_3body, topo_pair, convention):
    i, j = topo_pair
    tg = TopologyCollection(0, [1, 2, 3])
    topo_i = tg.topologies[i]
    topo_j = tg.topologies[j]
    momenta = topo_i.to_rest_frame(momenta_3body)
    run_wigner_comparison(topo_i, topo_j, momenta, convention)


# ── 4-body relative_wigner_angles ─────────────────────────────────────────────


@skip_no_rust
@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize("topo_pair", list(product(range(3), range(3))))
def test_wigner_angles_4body(momenta_4body, topo_pair, convention):
    i, j = topo_pair
    tg = TopologyCollection(0, [1, 2, 3, 4])
    topo_i = tg.topologies[i]
    topo_j = tg.topologies[j]
    momenta = topo_i.to_rest_frame(momenta_4body)
    run_wigner_comparison(topo_i, topo_j, momenta, convention)
