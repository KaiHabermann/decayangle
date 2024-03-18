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
              4: jnp.array([ 0, 0.35, 0.4,1]),
              5: jnp.array([ 0, 0.1, 0.8,1])}

momenta = tg.trees[0].to_rest_frame(momenta)
first_node = tg.trees[0].inorder()[0]
tree = tg.trees[0]
tree2 = tg.trees[1]

frame1 = tree.boost(Node(4), momenta)
frame2 = tree2.boost(Node(4), momenta)

difference = frame1 @ tree2.boost(Node(4), momenta, inverse=True)
assert jnp.allclose(tree2.boost(Node(4), momenta, inverse=True).M4, frame2.inverse().M4)
print(difference.decode())

print(frame1.decode())
print(frame2.decode())

for v1, v2 in zip(frame1.decode(), frame2.decode()):
    print(v1 - v2)

