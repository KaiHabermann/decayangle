from decayangle.DecayTopology import generateTreeDefinitions, Node, TopologyGroup
from jax import numpy as jnp
from decayangle.Decay import NBodyDecay

from jax import config
config.update("jax_enable_x64", True)

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


decay = NBodyDecay(0,1,2,3,4, 5)

momenta = {   1: jnp.array([0, 0, -0.9, 1]),
              2: jnp.array([0, 0.15, 0.4,1]),
              3: jnp.array([ 0, 0.3, 0.3,1]),
              4: jnp.array([ 0, 0.2, 0.4,1]),
              5: jnp.array([ 0, 0.1, 0.8,1])}

momenta = tg.trees[0].to_rest_frame(momenta)
first_node = tg.trees[0].inorder()[0]
tree = tg.trees[0]
tree2 = tg.trees[1]
# print(first_node.value ,first_node.momentum(momenta))
# exit(0)

frame1 = tree.boost(Node(4), momenta)
frame2 = tree2.boost(Node(4), momenta, inverse=True)

difference = frame1 @ frame2
print(tree2.boost(Node(4), momenta, inverse=True).M4 - frame2.inverse().M4)
assert jnp.allclose(tree2.boost(Node(4), momenta, inverse=True).M4, frame2.inverse().M4)

# print(difference)
# print(difference.decode(False))
# print(difference.decode())


# print(tree.boost(Node(4), momenta))
# print(first_node.boost(first_node, momenta)
exit(0)
for node in tg.filter(Node((2,1,3)), Node((1,2))) :
    # print(node.print_tree())
    print(node)    
