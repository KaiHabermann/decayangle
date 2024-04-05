from decayangle.decay_topology import Topology, TopologyCollection
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

def test_circular():

    def circlular_sorting(value):
        if isinstance(value, int):
            return value
        
        if isinstance(value, list):
            if not all(isinstance(v, int) for v in value) or len(value) == 0:
                return value

            return list(circlular_sorting(tuple(value)))
        
        if isinstance(value, tuple):
            possibilites = [(1,2), (2,3), (3,1), (1,2,3)]
            ret = {
                tuple(sorted(v)): v for v in possibilites
            }[tuple(sorted(value))]
            return ret

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
    tc = TopologyCollection(topologies=topologies, sorting_fun = circlular_sorting)
    topologies = tc.topologies

    assert (1,2) in topologies[0].helicity_angles(momenta)
    assert (3,1) in topologies[1].helicity_angles(momenta)
    assert (2,3) in topologies[2].helicity_angles(momenta)


if __name__ == "__main__":
    test_circular()