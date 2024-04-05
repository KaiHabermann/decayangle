# decayangle

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/decayangle.svg)](https://pypi.org/project/decayangle)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/decayangle.svg)](https://pypi.org/project/decayangle) -->


[![PyPI - Version](https://img.shields.io/pypi/v/decayangle.svg)](https://test.pypi.org/project/decayangle/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/decayangle.svg)](https://test.pypi.org/project/decayangle/)
-----

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [On Topologies](#Topologies)
- [On Ordering](#ordering)
- [License](#license)

## Installation

```console
pip install decayangle
```
## Usage
First we define the decay we are working with. For example, we can define the decay of a particle 0 into particles 1, 2, and 3. We can then generate all possible decay topologies for this decay. We can then filter the topologies based on the intermediate states we are interesetd in. Finally, we can calculate the relative Wigner angles between the different topologies resulting from different rotations when boosting along different configurations.

Lets start with defining the decay and generating the decay topologies.
```python
from decayangle.decay_topology import TopologyCollection

tg = TopologyCollection(0, [1, 2, 3]) # generate all decay topologies for 0 -> 1 2 3
tg.topologies # list of all decay topologies
```
```tg.topologies``` now contains all possible decay topologies for the decay 0 -> 1 2 3. Each topology acts as a descritpor for the consecutive decays into intermediate states, unitl the final state is reached. For a three body decay these are
```python
for topology in tg.topologies:
    print(topology)
```
```console
( 0 -> ( (2, 3) -> 2, 3 ), 1 )
( 0 -> ( (1, 3) -> 1, 3 ), 2 )
( 0 -> ( (1, 2) -> 1, 2 ), 3 )
```

We get three topologies, where each topology contains a unique intermediate state.
To select specific topologies we can use the filter option of the ```TopologyCollection``` class. For example, we can filter for topologies where the intermediate state (2, 3) is present.
```python	
topology1, = tg.filter((2, 3)) # we filter for topologies where the state (2, 3) is present 
topology2, = tg.filter((1, 3)) # we filter for topologies where the state (1, 3) is present
topology3, = tg.filter((1, 2)) # we filter for topologies where the state (1, 2) is present
```

Finally, we can calculate the relative Wigner angles between the different topologies. For example, we can calculate the relative Wigner angles between topology1 and topology2 for the final state particle 1. Only for this last step 4 momenta are needed. 
The function expects these momenta to be in the form of an array of 4 vectors, where the last (index 3) element is the time component. The momenta variable is then a dict of the form {particle: momentum}. For example, for a 3 body decay we can define the momenta as
```python	
import numpy as np
momenta = {1: np.array([-1, 0, 0, 2]), 2: np.array([0, 2, 0, 4]), 3: np.array([0, 0, 0.3, 2])}
```
where the momentum of the initial state is [0, 0, 0, 0], the momentum of the final state particles are [1, 0, 0, 1], [0, 1, 0, 1], and [0, 0, 1, 1]. Then the relative Wigner angles can be calculated as

```python
# now we can get the rotation between each of the topologies and for each final state particle
rotation1_2_1 = topology1.relative_wigner_angles(topology2, momenta)
rotation1_2_2 = topology1.relative_wigner_angles(topology2, momenta)
rotation1_2_3 = topology1.relative_wigner_angles(topology2, momenta)
# etc.
```

## Topologies

Topologies are the central type of object for the genreation of angles. Topologies can be generated in two ways.

### Topologies from TopologyCollections
The easiest way to produce topologies is from a `TopologyCollection`
```python
from decayangle.decay_topology import TopologyCollection

tg = TopologyCollection(start_node = 0, final_state_node = [1, 2, 3]) # generate all decay topologies for 0 -> 1 2 3
tg.topologies # list of all decay topologies
```

Here all the possible decay topologies will be gernerated automatically and can be acessed by the `topologies` property. To find topologies, which include interesting intermediate states the `filter` method can be used.

```python
topology, = tg.filter((1, 2))
```
The output of `filter` is a list of all topologies, which include the given nodes. A composite node is represented by a tuple. The order of the values in the tuple can be arbitrary. Ordering is handeled internaly, to ensure constistency.

```python
tg = TopologyCollection(0, [1,2,3,4])
topologies = tg.filter((2, 1))
for topology in topologies:
    print(topology)
```
```
Topology: ( 0 -> ( (1, 2) -> 1, 2 ), ( (3, 4) -> 3, 4 ) )
Topology: ( 0 -> ( (1, 2, 4) -> ( (1, 2) -> 1, 2 ), 4 ), 3 )
Topology: ( 0 -> ( (1, 2, 3) -> ( (1, 2) -> 1, 2 ), 3 ), 4 )
```

### Topologies from decay descriptions
For larger sets of final state nodes the amount of topologies, which would be generated by the `TopologyCollection` may be verry large.
Thus it is also possible to generate a `Topology` based on a root node and a decay descriptior. There the decay descriptor is a tuple of tuples 
```python
from decayangle.decay_topology import Topology
root = 0
topologies = [
    Topology(root, decay_topology=((1,2), 3)),
    Topology(root, decay_topology=((1,3), 2)), 
    Topology(root, decay_topology=((2, 3), 1)) 
]
```
Here the ordering of the nodes inside a tuple is not relveant. Only the overall topology i.e. which particles form an intermediate state and how they decay.

Such a list of topologies can be fused into a `TopologyCollection` like

```python
tg = TopologyCollection(topologies=topologies)
```

### Angles from Topologies

The `Topology` class has two main methods to determine relevant angles. 
The angles are calculated based on four momentum vectors of the final state particles. Other momenta are calculated form these momenta. 
All momenta are expected to be in the mother particle rest frame and have the time component as the last (index 3) element. 
The code is fully vectorized, so the only requirement for the shape of momenta arrays is, that they all have the same shape and the last dimension is of size 4.


The first method will calculate the helicity angles for a given topology. 
The result will be a dict with keys being a tuple of the two decay daughters and the value being a named tuple of the two relevant angles $\phi$ and $\theta$.
```python
topology = topologies[0]

# momenta is a dict of momenta with 
# key=finalstate particle number
# value=np.ndarray or jax.numpy.ndarray with shape
# (..., 4) 
angles = topology.helicity_angles(momenta)
```

The second method is to calculate the relative rotations which relate the rest frames of the final state particles in which one arvies when following the different topologies. Here up to three angles can be nescessary. But for a three body decay only the $\theta$ angle is needed.

```python
reference = topologies[0]
other = topologies[1]

relative_angles = reference.relative_wigner_angles(other, momenta)
```

## Ordering
Final state particles are supposed to be represented as integers. Support for other data types may be added in the future, but for the time being integers are the most stable and easy to implement solution. 

Intermediate states are written as tuples of the particles they consist of. Each intermediate state is represented by a `Node` object, which holds the information on the daughters of said state aswell as its value (the aforementioned tuple). 

An ordering scheme is used to order the values and daughters, to ensure, that it is always clear to which particles momenta angles are calculated. 
In the case of the helicity angles it is allways the first particle listed in the key.

A different ordering scheme can be set at the 
`TopologyCollection` level. These schemes are functions, which are expected to take in integers or tuples and return an integer. This integer is then the value by which things are sorted.

```python
tg = TopologyCollection(0, [1,2,3])
tg.sorting_key = lambda x: x if not isinstance(x, tuple) else x[0]
```

The code above will now sort by value or first value in the tuple if the given value is a tuple.

The default sorting scheme prioritizes sortes by length first and then by value. 
To achieve this in the most simple way,  the maximum value for a final state particle (or rather the integer representing it) is limited to 10000. This should be enough for all realistic usecases though.

## Related projects

Amplitude analyses dealing with non-zero spin of final-state particles have to implement wigner rotations in some way.
However, there are a few projects addressing these rotations explicitly using analytic expressions in [DPD paper](https://inspirehep.net/literature/1758460), derived for three-body decays:
- [ThreeBodyDecays.jl](https://github.com/mmikhasenko/ThreeBodyDecays.jl), 
- [SymbolicThreeBodyDecays.jl](https://github.com/mmikhasenko/SymbolicThreeBodyDecays.jl),
- [ComPWA/ampform-dpd](https://github.com/ComPWA/ampform-dpd).
Consistency of the `decayangle` framework with these appoaches is validated in the tests.

## License

`decayangle` is distributed under the terms of the [Apache2.0](https://www.apache.org/licenses/LICENSE-2.0) license.
