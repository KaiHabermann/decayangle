from decayangle.decay_topology import Topology
from decayangle.config import config as cfg
import numpy as np
def test_sorting():
    cfg.sorting = "off"
    root = 0
    topologies = [
        Topology(root, decay_topology=((1, 2), 3)),
        Topology(root, decay_topology=((3, 1), 2)), 
        Topology(root, decay_topology=((2, 3), 1)) 
    ]   

    momenta = {
        1: np.array([0, 0, -0.9, 1]),
        2: np.array([0, 0.15, 0.4, 1]),
        3: np.array([0, 0.3, 0.3, 1]),
    }
    momenta = topologies[0].to_rest_frame(momenta)

    assert (1,2) in topologies[0].helicity_angles(momenta)
    assert (3,1) in topologies[1].helicity_angles(momenta)
    assert (2,3) in topologies[2].helicity_angles(momenta)

if __name__ == "__main__":
    test_sorting()