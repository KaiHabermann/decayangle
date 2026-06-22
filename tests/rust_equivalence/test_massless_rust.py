"""
Comprehensive Python vs Rust comparison for massless particles.

Compares helicity_angles and relative_wigner_angles for all topology pairs
and all conventions, with particle 2 declared massless. Collects all failures
and reports them together rather than stopping at the first mismatch.
"""

import numpy as np
import pytest
from itertools import product

from decayangle.decay_topology import TopologyCollection, Topology
from decayangle.config import config as cfg

from .conftest import skip_no_rust

CONVENTIONS = ["helicity", "minus_phi"]
ATOL = 1e-8


def _make_massless_momenta(n=500, seed=7):
    """
    Generate n 3-body events with particle 2 exactly massless (E = |p|).
    Momenta are in the approximate mother rest frame.
    """
    rng = np.random.default_rng(seed)
    # particle 1 — massive
    p1 = np.column_stack(
        [
            rng.uniform(-0.4, 0.4, n),
            rng.uniform(-0.4, 0.4, n),
            rng.uniform(-0.9, 0.9, n),
            np.ones(n),
        ]
    )
    p1[:, 3] = np.sqrt(0.04 + np.sum(p1[:, :3] ** 2, axis=1))  # m1 = 0.2

    # particle 2 — massless: pick a random direction, set E = |p|
    phi2 = rng.uniform(0, 2 * np.pi, n)
    theta2 = np.arccos(rng.uniform(-1, 1, n))
    pmag2 = rng.uniform(0.1, 0.8, n)
    p2 = np.column_stack(
        [
            pmag2 * np.sin(theta2) * np.cos(phi2),
            pmag2 * np.sin(theta2) * np.sin(phi2),
            pmag2 * np.cos(theta2),
            pmag2,  # E = |p| for massless
        ]
    )

    # particle 3 — momentum conservation, energy from m3 = 0.5
    p3_spatial = -p1[:, :3] - p2[:, :3]
    p3 = np.column_stack(
        [
            p3_spatial,
            np.sqrt(0.25 + np.sum(p3_spatial**2, axis=1)),  # m3 = 0.5
        ]
    )

    return {1: p1, 2: p2, 3: p3}


def _angles_close(a, b):
    """Check two angle arrays agree via sin/cos (handles ±2π)."""
    return np.allclose(np.sin(a), np.sin(b), atol=ATOL) and np.allclose(
        np.cos(a), np.cos(b), atol=ATOL
    )


def _max_angle_diff(a, b):
    ds = np.max(np.abs(np.sin(np.asarray(a)) - np.sin(np.asarray(b))))
    dc = np.max(np.abs(np.cos(np.asarray(a)) - np.cos(np.asarray(b))))
    return max(ds, dc)


@skip_no_rust
def test_massless_rust_comprehensive():
    raw_momenta = _make_massless_momenta()
    tg = TopologyCollection(0, [1, 2, 3])
    topologies = tg.topologies  # 3 topologies for 3-body decay

    failures = []

    for ref_idx, ref_topo in enumerate(topologies):
        momenta = ref_topo.to_rest_frame(raw_momenta)

        # ── helicity_angles ───────────────────────────────────────────────────
        for topo_idx, topo in enumerate(topologies):
            for convention in CONVENTIONS:
                cfg.use_rust = False
                py_ha = topo.helicity_angles(
                    momenta, convention=convention, massless=[2]
                )
                cfg.use_rust = True
                rs_ha = topo.helicity_angles(
                    momenta, convention=convention, massless=[2]
                )

                for (isobar, bachelor), py_angles in py_ha.items():
                    rs_angles = rs_ha.get((isobar, bachelor))
                    if rs_angles is None:
                        failures.append(
                            f"helicity_angles: ref={ref_idx} topo={topo_idx} conv={convention} "
                            f"key ({isobar},{bachelor}) missing in Rust output"
                        )
                        continue
                    for angle_name, py_val, rs_val in [
                        ("phi", py_angles.phi_rf, rs_angles.phi_rf),
                        ("theta", py_angles.theta_rf, rs_angles.theta_rf),
                    ]:
                        if not _angles_close(py_val, rs_val):
                            diff = _max_angle_diff(py_val, rs_val)
                            failures.append(
                                f"helicity_angles: ref={ref_idx} topo={topo_idx} "
                                f"conv={convention} key=({isobar},{bachelor}) "
                                f"angle={angle_name} max_diff={diff:.2e}"
                            )

        # ── relative_wigner_angles ────────────────────────────────────────────
        for i, j in product(range(len(topologies)), repeat=2):
            topo_i = topologies[i]
            topo_j = topologies[j]
            for convention in CONVENTIONS:
                cfg.use_rust = False
                py_wa = topo_i.relative_wigner_angles(
                    topo_j, momenta, convention=convention, massless=[2]
                )
                cfg.use_rust = True
                rs_wa = topo_i.relative_wigner_angles(
                    topo_j, momenta, convention=convention, massless=[2]
                )

                for particle in py_wa:
                    py_w = py_wa[particle]
                    rs_w = rs_wa.get(particle)
                    if rs_w is None:
                        failures.append(
                            f"wigner: ref={ref_idx} ({i},{j}) conv={convention} "
                            f"particle={particle} missing in Rust output"
                        )
                        continue

                    # Skip massless particle itself — angles are numerically undefined
                    if particle == 2:
                        continue

                    for angle_name, py_val, rs_val in [
                        ("phi_rf", py_w.phi_rf, rs_w.phi_rf),
                        ("theta_rf", py_w.theta_rf, rs_w.theta_rf),
                        ("psi_rf", py_w.psi_rf, rs_w.psi_rf),
                    ]:
                        if not _angles_close(py_val, rs_val):
                            diff = _max_angle_diff(py_val, rs_val)
                            failures.append(
                                f"wigner: ref={ref_idx} ({i},{j}) conv={convention} "
                                f"particle={particle} angle={angle_name} max_diff={diff:.2e}"
                            )

    cfg.use_rust = False  # restore

    if failures:
        report = f"{len(failures)} failure(s):\n" + "\n".join(
            f"  {f}" for f in failures
        )
        pytest.fail(report)
