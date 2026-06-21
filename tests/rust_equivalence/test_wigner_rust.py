"""
Compare relative_wigner_angles output between the Python and Rust implementations
for all topology pairs of a 3-body decay.
"""

import numpy as np
import pytest
from itertools import product

from decayangle.decay_topology import TopologyCollection
from decayangle.config import config as cfg

from .conftest import skip_no_rust


def assert_angles_close(rs_val, py_val, atol, label):
    """
    Compare two angle arrays modulo 2π by checking that sin and cos agree.
    This handles the ±2π ambiguity that can arise from SU(2) global sign differences.
    """
    rs = np.asarray(rs_val)
    py = np.asarray(py_val)
    np.testing.assert_allclose(
        np.sin(rs),
        np.sin(py),
        atol=atol,
        err_msg=f"{label}: sin mismatch",
    )
    np.testing.assert_allclose(
        np.cos(rs),
        np.cos(py),
        atol=atol,
        err_msg=f"{label}: cos mismatch",
    )


@skip_no_rust
@pytest.mark.parametrize("topo_pair", list(product(range(3), range(3))))
def test_wigner_angles_match(momenta_3body, topo_pair):
    i, j = topo_pair
    tg = TopologyCollection(0, [1, 2, 3])
    topo_i = tg.topologies[i]
    topo_j = tg.topologies[j]

    old_rust = cfg.use_rust
    cfg.use_rust = False
    momenta = topo_i.to_rest_frame(momenta_3body)

    # Python reference
    py_wigner = topo_i.relative_wigner_angles(topo_j, momenta, convention="helicity")

    # Rust result
    cfg.use_rust = True
    try:
        rs_wigner = topo_i.relative_wigner_angles(
            topo_j, momenta, convention="helicity"
        )
    finally:
        cfg.use_rust = old_rust

    assert set(py_wigner.keys()) == set(
        rs_wigner.keys()
    ), f"Key mismatch ({i},{j}): Python={set(py_wigner.keys())}, Rust={set(rs_wigner.keys())}"

    for particle in py_wigner:
        for angle_name, py_val, rs_val in [
            ("phi_rf", py_wigner[particle].phi_rf, rs_wigner[particle].phi_rf),
            ("theta_rf", py_wigner[particle].theta_rf, rs_wigner[particle].theta_rf),
            ("psi_rf", py_wigner[particle].psi_rf, rs_wigner[particle].psi_rf),
        ]:
            assert_angles_close(
                rs_val,
                py_val,
                atol=1e-10,
                label=f"topologies=({i},{j}), particle={particle}, angle={angle_name}",
            )
