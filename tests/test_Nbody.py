from decayangle.decay_topology import (
    generate_topology_definitions,
    Node,
    TopologyCollection,
)
from jax import numpy as jnp
from jax import config as jax_cfg

jax_cfg.update("jax_enable_x64", True)
from decayangle.config import config as cfg

cb = cfg.backend

# cb = "jax"
# for static testing numpy is faster
# only if you compile the code, jax is faster
# but this is not really sensible for testing


def test_topology():
    p0 = 0
    p1 = 1
    p2 = 2
    p3 = 3
    p4 = 4

    tg = TopologyCollection(p0, [p1, p2, p3, p4])
    assert len(tg.topologies) == 15

    tg = TopologyCollection(p0, [p1, p2, p3])
    assert len(tg.topologies) == 3

    tg = TopologyCollection(p0, [p1, p2])
    assert len(tg.topologies) == 1

    tg = TopologyCollection(0, [1, 2, 3, 4])
    assert len(tg.topologies) == 15

    tg = TopologyCollection(0, [1, 2, 3])
    assert len(tg.topologies) == 3

    tg = TopologyCollection(0, [1, 2])
    assert len(tg.topologies) == 1

    tg = TopologyCollection(0, [1, 2, 3, 4, 5])
    assert len(tg.topologies) == 105

    momenta = {
        1: cb.array([0, 0, -0.9, 1]),
        2: cb.array([0, 0.15, 0.4, 1]),
        3: cb.array([0, 0.3, 0.3, 1]),
        4: cb.array([0, 0.25, 0.35, 1]),
        5: cb.array([0, 0.1, 0.8, 1]),
    }

    momenta = tg.topologies[0].to_rest_frame(momenta)
    first_node = tg.topologies[0].preorder()[0]
    base_tree = tg.topologies[0]

    # this test looks at what happens, if we do not have a net boost
    # here we have to be careful, because the boost is not unique
    # this is due two the structure of rotation boost rotation
    # with no net boosts, the rotations can not be uniquely determined
    for topology in tg.topologies[1:]:
        for node in [Node(1), Node(2), Node(3), Node(4), Node(5)]:
            tree1 = base_tree.boost(node, momenta)
            tree2 = topology.boost(node, momenta)

            difference = tree1 @ tree2.inverse()
            # we can't really assert things here, but if it runs through we at least know, that we can do the operations
            result = difference.decode()

            assert cb.isfinite(cb.array(result)).all()

    # you can filter for topologies, where specific nodes are in the topology
    # this is useful for cases, where resonances are only present in some of the possible isobars
    for topology in tg.filter(Node((1, 2)), Node((1, 2, 3))):
        for node in [Node(1), Node(2), Node(3), Node(4), Node(5)]:
            tree1 = base_tree.boost(node, momenta)
            tree2 = topology.boost(node, momenta)
            difference = tree1 @ tree2.inverse()
            base_tree.relative_wigner_angles(topology, momenta)
            # we can't really assert things here, but if it runs through we at least know, that we can do the operations
            result = difference.decode()
            assert cb.isfinite(cb.array(result)).all()

            _, _, xi1, phi1, theta1, xi_rf1 = base_tree.boost(node, momenta).decode(
                two_pi_aware=True
            )
            _, _, xi2, phi2, theta2, xi_rf2 = topology.boost(node, momenta).decode(
                two_pi_aware=True
            )
            assert cb.isfinite(
                cb.array([xi1, phi1, theta1, xi_rf1, xi2, phi2, theta2, xi_rf2])
            ).all()


if __name__ == "__main__":
    test_topology()
