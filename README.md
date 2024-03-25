# decayangle

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/decayangle.svg)](https://pypi.org/project/decayangle)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/decayangle.svg)](https://pypi.org/project/decayangle) -->


[![PyPI - Version](https://img.shields.io/pypi/v/decayangle.svg)](https://test.pypi.org/project/decayangle/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/decayangle.svg)](https://test.pypi.org/project/decayangle/)
-----

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

```console
pip install decayangle
```

## Usage
First we define the decay we are working with. For example, we can define the decay of a particle 0 into particles 1, 2, and 3. We can then generate all possible decay trees for this decay. We can then filter the trees based on the intermediate states we are interesetd in. Finally, we can calculate the relative Wigner angles between the different frames resulting from different rotations when boosting along different configurations.

Lets start with defining the decay and generating the decay trees.
```python
from decayangle.decay_topology import TopologyGroup

tg = TopologyGroup(0, [1, 2, 3]) # generate all decay trees for 0 -> 1 2 3
tg.trees # list of all decay trees
```
```tg.trees``` now contains all possible decay trees for the decay 0 -> 1 2 3. Each tree acts as a descritpor for the consecutive decays into intermediate states, unitl the final state is reached. For a three body decay these are
```python
for tree in tg.trees:
    print(tree)
```
```console
( 0 -> ( (2, 3) -> 2, 3 ), 1 )
( 0 -> ( (1, 3) -> 1, 3 ), 2 )
( 0 -> ( (1, 2) -> 1, 2 ), 3 )
```

We get three trees, where each tree contains a unique intermediate state.
To select specific trees we can use the filter option of the ```TopologyGroup``` class. For example, we can filter for trees where the intermediate state (2, 3) is present.
```python	
frame1, = tg.filter((2, 3)) # we filter for trees where the state (2, 3) is present 
frame2, = tg.filter((1, 3)) # we filter for trees where the state (1, 3) is present
frame3, = tg.filter((1, 2)) # we filter for trees where the state (1, 2) is present
```

Finally, we can calculate the relative Wigner angles between the different frames. For example, we can calculate the relative Wigner angles between frame1 and frame2 for the final state particle 1. Only for this last step 4 momenta are needed. 
The function expects these momenta to be in the form of an array of 4 vectors, where the last (index 3) element is the time component. The momenta variable is then a dict of the form {particle: momentum}. For example, for a 3 body decay we can define the momenta as
```python	
import numpy as np
momenta = {1: np.array([-1, 0, 0, 2]), 2: np.array([0, 2, 0, 4]), 3: np.array([0, 0, 0.3, 2])}
```
where the momentum of the initial state is [0, 0, 0, 0], the momentum of the final state particles are [1, 0, 0, 1], [0, 1, 0, 1], and [0, 0, 1, 1]. Then the relative Wigner angles can be calculated as

```python
# now we can get the rotation between each of the frames and for each final state particle
rotation1_2_1 = frame1.relative_wigner_angles(frame2, 1, momenta)
rotation1_2_2 = frame1.relative_wigner_angles(frame2, 2, momenta)
rotation1_2_3 = frame1.relative_wigner_angles(frame2, 3, momenta)
# etc.
```


## License

`decayangle` is distributed under the terms of the [Apache2.0](https://www.apache.org/licenses/LICENSE-2.0) license.
