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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
