"""
Compare helicity_angles output between the Python and Rust implementations
for all three topologies of a 3-body decay.
"""

import numpy as np
import pytest

from decayangle.decay_topology import TopologyCollection
from decayangle.config import config as cfg

from .conftest import skip_no_rust


@skip_no_rust
@pytest.mark.parametrize("topo_idx", [0, 1, 2])
def test_helicity_angles_match(momenta_3body, topo_idx):
    tg = TopologyCollection(0, [1, 2, 3])
    topo = tg.topologies[topo_idx]

    # Boost to rest frame once (Python implementation)
    old_rust = cfg.use_rust
    cfg.use_rust = False
    momenta = topo.to_rest_frame(momenta_3body)

    # Python reference
    py_angles = topo.helicity_angles(momenta, convention="helicity")

    # Rust result
    cfg.use_rust = True
    try:
        rs_angles = topo.helicity_angles(momenta, convention="helicity")
    finally:
        cfg.use_rust = old_rust

    assert set(py_angles.keys()) == set(
        rs_angles.keys()
    ), f"Key mismatch: Python={set(py_angles.keys())}, Rust={set(rs_angles.keys())}"

    for key in py_angles:
        py_phi = np.asarray(py_angles[key].phi_rf)
        py_theta = np.asarray(py_angles[key].theta_rf)
        rs_phi = np.asarray(rs_angles[key].phi_rf)
        rs_theta = np.asarray(rs_angles[key].theta_rf)

        np.testing.assert_allclose(
            rs_phi,
            py_phi,
            atol=1e-10,
            err_msg=f"topo={topo_idx}, key={key}: phi mismatch",
        )
        np.testing.assert_allclose(
            rs_theta,
            py_theta,
            atol=1e-10,
            err_msg=f"topo={topo_idx}, key={key}: theta mismatch",
        )
