import pytest
from decayangle.decay_topology import TopologyGroup, Node, Topology

def test_topology():
    tg = TopologyGroup(0, [1, 2, 3])

    def process_plane_sorting(plane):
        if isinstance(plane, tuple):
            return plane[0]
        if isinstance(plane, int):
            return -plane
        raise ValueError("Invalid plane type")

    representations = []
    for topology in tg.topologies:
        representations.append(str(topology))
        assert any( [topology.root.contains((1, 2)), 
                    topology.root.contains((2, 3)),
                    topology.root.contains((1, 3))
                     ] )
    tg.sorting_key = process_plane_sorting

    for i, topology in enumerate(tg.topologies):
        assert str(topology) != representations[i]


    

if __name__ == "__main__":
    test_topology()