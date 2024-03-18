from decayangle.DecayTopology import generateTreeDefinitions, Node, TopologyGroup
from jax import numpy as jnp
from decayangle.Decay import NBodyDecay
from jax import config as jax_cfg
jax_cfg.update("jax_enable_x64", True)
from decayangle.config import config
# config.backend = "jax"
# for static testing numpy is faster
# only if you compile the code, jax is faster
# but this is not really sensible for testing

def test_topology():
    p0 = 0
    p1 = 1
    p2 = 2
    p3 = 3
    p4 = 4

    tg = TopologyGroup(p0, [p1,p2,p3,p4])
    assert len(tg.trees) == 15

    tg = TopologyGroup(p0, [p1,p2,p3])
    assert len(tg.trees) == 3

    tg = TopologyGroup(p0, [p1,p2])
    assert len(tg.trees) == 1

    tg = TopologyGroup(0,[1,2,3,4])
    assert len(tg.trees) == 15

    tg = TopologyGroup(0,[1,2,3])
    assert len(tg.trees) == 3

    tg = TopologyGroup(0,[1,2])
    assert len(tg.trees) == 1

    tg = TopologyGroup(0,[1,2,3,4,5])
    assert len(tg.trees) == 105

    decay = NBodyDecay(0,1,2,3,4,5)

    momenta = {   1: config.backend.array([0, 0, -0.9, 1]),
                2: config.backend.array([0, 0.15, 0.4,1]),
                3: config.backend.array([ 0, 0.3, 0.3,1]),
                4: config.backend.array([ 0, 0.35, 0.4,1]),
                5: config.backend.array([ 0, 0.1, 0.8,1])}

    momenta = tg.trees[0].to_rest_frame(momenta)
    first_node = tg.trees[0].inorder()[0]
    base_tree = tg.trees[0]

    from tqdm import tqdm

    for tree in tqdm(tg.trees):
        for node in [Node(1), Node(2), Node(3), Node(4), Node(5)]:
            frame1 = base_tree.boost(node, momenta)
            frame2 = tree.boost(node, momenta)
            difference = frame1 @ tree.boost(node, momenta, inverse=True)
            # we cant really assert things here, but if it runs through we at least know, that we can do the operations
            result = difference.decode()
            assert config.backend.isfinite(config.backend.array(result)).all()
    
    # you can filter for topologies, where specific nodes are in the tree
    # this is useful for cases, where resonances are only present in some of the possible isobars
    for tree in tqdm(tg.filter(Node((1,2)), Node((1,2,3)))):
        for node in [Node(1), Node(2), Node(3), Node(4), Node(5)]:
            frame1 = base_tree.boost(node, momenta)
            frame2 = tree.boost(node, momenta)
            difference = frame1 @ tree.boost(node, momenta, inverse=True)
            base_tree.relative_wigner_angles(tree, node, momenta)
            # we cant really assert things here, but if it runs through we at least know, that we can do the operations
            result = difference.decode()
            assert config.backend.isfinite(config.backend.array(result)).all()

            _, _, xi1, theta1, phi1, _ = base_tree.boost(node, momenta).decode(two_pi_aware=True)
            _, _, xi2, theta2, phi2, _ = tree.boost(node, momenta).decode(two_pi_aware=True)
            def replace_pi(x):
                return config.backend.where(config.backend.isclose(x, config.backend.pi), 0., x)
            assert config.backend.allclose(replace_pi(xi1), replace_pi(xi2))
            assert config.backend.allclose(replace_pi(phi1), replace_pi(phi2))
            assert config.backend.allclose(replace_pi(theta1), replace_pi(theta2))

