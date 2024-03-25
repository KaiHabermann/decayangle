from decayangle.DecayTopology import TopologyGroup, Node
from jax import config as jax_cfg
jax_cfg.update("jax_enable_x64", True)
from decayangle.config import config as cfg
cb = cfg.backend

def test_decay():
    tg = TopologyGroup(0,[1,2,3])
    momenta = { 
        1: cb.array([0, 0, -0.9, 1]),
        2: cb.array([0, 0.15, 0.4,1]),
        3: cb.array([ 0, 0.3, 0.3,1]),
        4: cb.array([ 0, 0.25, -0.4,1]),
    }

    reference_tree = tg.trees[0]
    momenta = reference_tree.to_rest_frame(momenta)
    for tree in tg.trees:
        final_state_rotations = {
            target:reference_tree.relative_wigner_angles(tree, target, momenta)
            for target in [Node(1), Node(2), Node(3)]
        }


if __name__ == "__main__":
    test_decay()