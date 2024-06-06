import pytest
from decayangle.decay_topology import TopologyCollection, Node, Topology


def test_topology():
    tg = TopologyCollection(0, [1, 2, 3])

    def process_plane_sorting(plane):
        if isinstance(plane, tuple):
            return plane[::-1]
        return plane

    representations = []
    for topology in tg.topologies:
        representations.append(str(topology))
        assert any(
            [
                topology.root.contains((1, 2)),
                topology.root.contains((2, 3)),
                topology.root.contains((1, 3)),
            ]
        )

    root = 0
    topology = Topology(root, decay_topology=((1, 2), 3))
    (topology_from_group,) = tg.filter((1, 2))
    assert str(topology) == str(topology_from_group)

    tg.ordering_function = process_plane_sorting
    topology.ordering_function = process_plane_sorting
    assert str(topology) == str(topology_from_group)

    for i, topology in enumerate(tg.topologies):
        assert str(topology) != representations[i]


if __name__ == "__main__":
    test_topology()
