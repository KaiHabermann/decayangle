import numpy as np
from jax import jit, vmap
from jax import numpy as jnp
from functools import cached_property
from decayangle.DecayTopology import TopologyGroup, Topology


class NBodyDecay:

    def __init__(self, parent: int, *children: int):
        self.__children = children
        self.__parent = parent
        self.__particles = [parent, *children]
    
    @property
    def children(self):
        return self.__children
    
    @property
    def parent(self):
        return self.__parent
    
    @property
    def particles(self):
        return self.__particles

    @property
    def n_children(self):
        return len(self.children)
    
    @property
    def n_body(self):
        return self.n_children + 1

    @cached_property
    def topologies(self):
        return TopologyGroup(self.parent, list(self.children)).trees

    