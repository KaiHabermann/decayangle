# Welcome to the decayangle software Project

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/decayangle.svg)](https://pypi.org/project/decayangle)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/decayangle.svg)](https://pypi.org/project/decayangle) -->


[![PyPI - Version](https://img.shields.io/pypi/v/decayangle.svg)](https://test.pypi.org/project/decayangle/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/decayangle.svg)](https://test.pypi.org/project/decayangle/)
-----

**Table of Contents**

- [Installation](#installation)
- [Goal](#goal)
- [Usage](#usage)
- [On Topologies](#topologies)
- [On Ordering](#ordering)
- [Related projects](#related-projects)
- [License](#license)

## Installation

```console
pip install decayangle
```

## Goal

The software project `decayangle` provides a Python library for computing helicity angles and Wigner rotations in hadron physics, facilitating the analysis of particle decay processes. It enables the generation and manipulation of decay topologies, calculation of relative angles between different topologies. It supports amplitude analyses involving non-zero spin of final-state particles, while not being limited to three-body decays.


## Usage
First we define the decay we are working with. For example, we can define the decay of a particle 0 into particles 1, 2, and 3. We can then generate all possible decay topologies for this decay. We can then filter the topologies based on the intermediate states we are interested in. Finally, we can calculate the relative Wigner angles between the different topologies resulting from different rotations when boosting along different configurations.

Lets start with defining the decay and generating the decay topologies.
```python
from decayangle.decay_topology import TopologyCollection

tg = TopologyCollection(0, [1, 2, 3]) # generate all decay topologies for 0 -> 1 2 3
tg.topologies # list of all decay topologies
```
```tg.topologies``` now contains all possible decay topologies for the decay 0 -> 1 2 3. Each topology acts as a descriptor for the consecutive decays into intermediate states, until the final state is reached. For a three-body decay these are
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
To select specific topologies we can use the filter option of the `TopologyCollection` class. For example, we can filter for topologies where the intermediate state (2, 3) is present.
```python	
topology1, = tg.filter((2, 3)) # we filter for topologies where the state (2, 3) is present 
topology2, = tg.filter((1, 3)) # we filter for topologies where the state (1, 3) is present
topology3, = tg.filter((1, 2)) # we filter for topologies where the state (1, 2) is present
```

Finally, we can calculate the relative Wigner angles between the different topologies. For example, we can calculate the relative Wigner angles between `topology1` and `topology2` for the final-state particle 1. Only for this last step 4 momenta are needed. 
The function expects these momenta to be in the form of an array of 4 vectors, where the last (index 3) element is the time component. The momenta variable is then a dict of the form `{particle: momentum}`. For example, for a 3 body decay we can define the momenta as
```python	
import numpy as np
momenta = {1: np.array([-1, 0, 0, 2]), 2: np.array([0, 2, 0, 4]), 3: np.array([0, 0, 0.3, 2])}
```
where the momentum of the initial state is [0, 0, 0, 0], the momentum of the final-state particles are [1, 0, 0, 1], [0, 1, 0, 1], and [0, 0, 1, 1]. Then, the relative Wigner angles can be calculated as

```python
# now we can get the rotation between each of the topologies and for each final-state particle
rotation1_2_1 = topology1.relative_wigner_angles(topology2, momenta)
rotation1_2_2 = topology1.relative_wigner_angles(topology2, momenta)
rotation1_2_3 = topology1.relative_wigner_angles(topology2, momenta)
# etc.
```

## Topologies

Topologies are the central type of object for the generation of angles.
Topologies can be generated in two ways.

### Topologies from TopologyCollections
The easiest way to produce topologies is from a `TopologyCollection`
```python
from decayangle.decay_topology import TopologyCollection

tg = TopologyCollection(start_node = 0, final_state_node = [1, 2, 3]) # generate all decay topologies for 0 -> 1 2 3
tg.topologies # list of all decay topologies
```

Here all the possible decay topologies will be generated automatically and can be accessed by the `topologies` property. To find topologies, which include interesting intermediate states the `filter` method can be used.

```python
topology, = tg.filter((1, 2))
# one can pass multiple nodes
topology, = tg.filter((1, 2), 1) # the same since every topology includes one
```
The output of `filter` is a list of all topologies, which include the given nodes. A composite node is represented by a tuple. The order of the values in the tuple can be arbitrary. Tuples are sorted (see [ordering](#ordering)) before the search to ensure consistency. When multiple arguments are passed, only topologies that include all provided nodes are returned.

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

For larger sets of final state nodes the amount of topologies, which would be generated by the `TopologyCollection` may be very large.
Thus it is also possible to generate a `Topology` based on a root node and a decay descriptor. There the decay descriptor is a tuple of tuples 
```python
from decayangle.decay_topology import Topology

root = 0
topologies = [
    Topology(root, decay_topology=((1, 2), 3)),
    Topology(root, decay_topology=((1, 3), 2)), 
    Topology(root, decay_topology=((2, 3), 1)) 
]
```
Here the ordering of the nodes inside a tuple is not relevant. Only the overall topology i.e. which particles form an intermediate state and how they decay.

To change this behaviour and keep the ordering as it is given in the topology descriptors, one can change the config setting to disable sorting. This should be done before the topologies are created. 
More on sorting can be found in the [ordering](#ordering) section.

```python
from decayangle.decay_topology import Topology
from decayangle.config import config as cfg
cfg.sorting = "off"

root = 0
topologies = [
    Topology(root, decay_topology=((1, 2), 3)),
    Topology(root, decay_topology=((1, 3), 2)), 
    Topology(root, decay_topology=((2, 3), 1)) 
]
```

A list of topologies can be fused into a `TopologyCollection` like

```python
tg = TopologyCollection(topologies=topologies)
```

### Angles from Topologies

The angles are calculated from four-vectors of the particles by rotating them and boosting throughout the topology tree.
The input momenta are expected to be in the mother particle rest frame and have the time component as the last (index 3) element. 
The code is fully vectorized, so the only requirement for the shape of momenta arrays is, that they all have the same shape and the last dimension is of size 4.

The `Topology` class has two main methods to determine relevant angles.
The first method, `helicity_angles` calculates the helicity angles for a given topology. It returns a dict with keys in a form of $(A,B)$ where `A` and `B` are node values (a tuple of integers or an integer).
The values of the dictionary returned by the `helicity_angles` are named tuples of the azimuthal and polar angles, $\phi$ and $\theta$, respectively.

```python
topology = topologies[0]

# `momenta` is a dict of particle momenta with 
#  - key:  the final-state particle number
#  - value: np.ndarray or jax.numpy.ndarray with shape
# (..., 4) 
angles = topology.helicity_angles(momenta)
```

The second method is to calculate the relative rotations which relate the rest frames of the final-state particles in which one arrives when following the different topologies. Here up to three angles can be necessary. But for a three-body decay only the $\theta$ angle is needed.

```python
reference = topologies[0]
other = topologies[1]

relative_angles = reference.relative_wigner_angles(other, momenta)
```

## Ordering

Final-state particles are supposed to be represented as integers. 
Support for other data types may be added in the future, but for the time being integers are the most stable and easy to implement solution. 

Intermediate node is written as tuples of the particles they consist of.
Each intermediate state is represented by a `Node` object, which holds the information on the daughters of said state as well as its value (the aforementioned tuple). 

A particular scheme is used to order daughters, and determine node names.
Helicity angle are always calculated with respect to the first daughter.

The ordering scheme can be customized at `TopologyCollection` level.
The schemes is specified by a function, which is <expected to take in integers or tuples and return an integer used to sort the daughters, and integers in the node name. 

```python
tg = TopologyCollection(0, [1,2,3])
tg.ordering_function = lambda x: x
```
The code above will just leave the object as it comes. Thus applying no sorting.

The default sorting scheme puts longest node first and then sorts by value.
The maximum value for a final-state particle (or rather the integer representing it) is limited to 10000. This should be enough for all realistic use cases.

To change the default sorting scheme one can use the config.

```python
from decayangle.config import config as cfg
cfg.sorting = "off" # turns sorting off
cfg.sorting = "value" # default sorting

```
At this time only `"off"` and `"value"` are supported. For more sophisticated sorting algorithms the user has to write custom functions.
## Related projects

Amplitude analyses dealing with non-zero spin of final-state particles have to implement wigner rotations in some way.
However, there are a few projects addressing these rotations explicitly using analytic expressions in [DPD paper](https://inspirehep.net/literature/1758460), derived for three-body decays:
- [ThreeBodyDecays.jl](https://github.com/mmikhasenko/ThreeBodyDecays.jl), 
- [SymbolicThreeBodyDecays.jl](https://github.com/mmikhasenko/SymbolicThreeBodyDecays.jl),
- [ComPWA/ampform-dpd](https://github.com/ComPWA/ampform-dpd).
Consistency of the `decayangle` framework with these appoaches is validated in the tests.

## License

`decayangle` is distributed under the terms of the [Apache2.0](https://www.apache.org/licenses/LICENSE-2.0) license.
