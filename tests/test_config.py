import pytest
import numpy as np
from decayangle.config import config as cfg
from decayangle.decay_topology import TopologyCollection
from decayangle.lorentz import LorentzTrafo

def test_numerical_safety_checks():
    cfg.numerical_safety_checks = True

    momenta = {
        1: np.array([0.15, 0, 0, 1]),
        2: np.array([-0.5, 0.001, 0.2, 2]),
        3: np.array([0.1, 0.4, 0.2, 1.3]),
    }

    tc = TopologyCollection(0, [1, 2, 3])

    for topology in tc.topologies:
        pytest.raises(ValueError, topology.helicity_angles, momenta)

    cfg.numerical_safety_checks = False
    old_sorting = cfg.sorting = "off"
    for topology in tc.topologies:
        topology.helicity_angles(momenta)
    cfg.sorting = old_sorting
    cfg.numerical_safety_checks = True

def test_numerical_precision():
    cfg.numerical_safety_checks = True

    momenta = {
        1: np.array([0.15, 0, 0, 1]),
        2: np.array([-0.5, 0.001, 0.2, 2]),
        3: np.array([0.1, 0.4, 0.2, 1.3]),
    }

    tc = TopologyCollection(0, [1, 2, 3])
    topo = tc.topologies[0]
    # momenta = topo.to_rest_frame(momenta)
    try:
        cfg.gamma_tolerance = 1e-10
        for topology in tc.topologies:
            pytest.raises(ValueError, topology.helicity_angles, momenta)
        cfg.gamma_tolerance = 1

        for topology in tc.topologies:
            topology.helicity_angles(momenta)
        cfg.gamma_tolerance = 1e-10
    except Exception as e:
        raise e 
    finally:
        cfg.gamma_tolerance = 1e-10

    try:
        old_prec = cfg.shift_precision 
        cfg.shift_precision = 2
        phi, theta, xi, phi_rf, theta_rf,  psi_rf = -1.2, 2.3, 1.4, 2.1, 2.6, -2.7
        A = LorentzTrafo(phi, theta, xi, phi_rf, theta_rf,  psi_rf)

        pytest.raises(ValueError, A.decode)

        cfg.shift_precision = old_prec
        phi_, theta_, xi_, phi_rf_, theta_rf_,  psi_rf_ = A.decode()
        A_decoded = LorentzTrafo(phi_, theta_, xi_, phi_rf_, theta_rf_, psi_rf_)

        assert np.allclose(A_decoded.matrix_4x4, A.matrix_4x4)
        assert np.allclose(A_decoded.matrix_2x2, A.matrix_2x2) 
    except Exception as e:
        raise e
    finally:
        cfg.shift_precision = old_prec

if __name__ == "__main__":
    pytest.main(["-v", __file__])

