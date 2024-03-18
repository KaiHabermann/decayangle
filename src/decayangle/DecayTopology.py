
import numpy as np
from jax import numpy as jnp
from typing import List, Tuple, Optional, Union, Any
from functools import cached_property
from decayangle.lorentz import LorentzTrafo
from decayangle import kinematics as akm
import networkx as nx
from decayangle.config import config


class Node:
    def __init__(self, value: Union[Any, tuple]):
        self.value = value
        if isinstance(value, tuple):
            self.value = tuple(sorted(value))
        self.daughters = []
        self.parent = None
    
    def add_daughter(self, daughter):
        self.daughters.append(daughter)
        daughter.parent = self
    
    def __repr__(self):
        if len(self.daughters) == 0:
            return str(self.value)
        return f"( {self.value} -> " + f"{', '.join([str(d) for d in self.daughters])} )"
    
    def __str__(self):
        return self.__repr__()
    
    def print_tree(self):
        for d in self.daughters:
            d.print_tree()
        print(f"\n {self.value}" )

    def contains(self, contained_node:'Node'):
        if self.value == contained_node.value:
            return True
        for d in self.daughters:
            if d.contains(contained_node):
                return True
        return False
    
    def inorder(self):
        if len(self.daughters) == 0:
            return [self]
        return [self] + [node for d in self.daughters for node in d.inorder()]
    
    def momentum(self, momenta:dict):
        """Get a particles momentum

        Args:
            momenta (dict): the momenta of the final state particles

        Returns:
            the momentum of the particle, as set by the momenta dictionary
            This expects the momenta to be jax or numpy compatible
        """
        if len(self.daughters) == 0:
            return momenta[self.value]
        return sum([d.momentum(momenta) for d in self.daughters])
    
    def transform(self, trafo:LorentzTrafo, momenta:dict):
        return {k: trafo.M4 @ v for k,v in momenta.items()}

    def boost(self, target: 'Node', momenta: dict):
        """ Get the boost from this node to a target node
            The momenta dictionary will define the initial configuration.
            It is expected, that the momenta are jax or numpy compatible and that the momenta are given in the rest frame of this node.
        """
        if not config.backend.allclose(akm.gamma(self.momentum(momenta)), config.backend.ones_like(self.momentum(momenta))):
            gamma = akm.gamma(self.momentum(momenta))
            raise ValueError(f"gamma = {gamma} For the time being only particles at rest are supported as start nodes for a boost. This will be fixed in the future.")
        zero = config.backend.zeros_like(akm.time_component(self.momentum(momenta)))
        one = config.backend.ones_like(zero)
        if self.value == target.value:
            return LorentzTrafo(zero ,zero, zero, zero, zero, zero)
        
        if not target in self.daughters:
            raise ValueError(f"Target node {target} is not a direct daughter of this node {self}")
        
        # rotate so that the target momentum is aligned with the 
        psi_rf, theta_rf = akm.rotate_to_z_axis(target.momentum(momenta))
        rotation = LorentzTrafo(zero, zero, zero, theta_rf, zero, psi_rf)
        rotated_momenta = self.transform(rotation, momenta)
        # assert the rotation worked as expected (TODO: remove this in the future, but for now, this gives security while debugging other parts of the code)
        assert config.backend.allclose(akm.y_component(target.momentum(rotated_momenta)), config.backend.zeros_like(akm.y_component(target.momentum(rotated_momenta))))
        assert config.backend.allclose(akm.x_component(target.momentum(rotated_momenta)), config.backend.zeros_like(akm.x_component(target.momentum(rotated_momenta))))

        # boost to the rest frame of the target
        xi = -akm.rapidity(target.momentum(rotated_momenta))
        boost = LorentzTrafo(zero, zero, xi, zero, zero, zero)
        # assert the boost worked as expected (TODO: remove this in the future, but for now, this gives security while debugging other parts of the code)
        assert config.backend.allclose(akm.gamma(target.momentum(self.transform(boost, rotated_momenta))), one)

        return boost @ rotation

class Tree:
    def __init__(self, root:Node):
        self.root = root
    
    def __repr__(self):
        return str(self.root)
    
    def contains(self, contained_node:'Node'):
        return self.root.contains(contained_node)
    
    def to_rest_frame(self, momenta:dict):
        momentum = self.root.momentum(momenta)
        return {k: akm.boost_to_rest(v, momentum) for k,v in momenta.items()}

    def __build_boost_tree(self, momenta:dict):
        boost_tree = nx.DiGraph()
        node_dict = {}
        for node in self.inorder():
            boost_tree.add_node(node.value)
            node_dict[node.value] = node
        for node in self.inorder():
            for d in node.daughters:
                boost_tree.add_edge(node.value, d.value)
        return boost_tree, node_dict
    
    def boost(self, target: 'Node', momenta: dict, inverse:bool = False) -> LorentzTrafo:
        boost_tree, node_dict = self.__build_boost_tree(momenta)
        path = nx.shortest_path(boost_tree, self.root.value, target.value)[1:]
        trafo = self.root.boost(node_dict[path[0]], momenta)
        momenta = self.root.transform(trafo, momenta)
        trafos = [trafo]
        for i in range(1, len(path)):
            boost = node_dict[path[i-1]].boost(node_dict[path[i]], momenta) 
            momenta = node_dict[path[i-1]].transform(boost, momenta)
            trafo = boost @ trafo
            trafos.append(boost)
        if inverse:
            # this is more precise then the naive inverse
            inverse_trafo = trafos[0].inverse()
            for trafo in trafos[1:]:
                inverse_trafo = inverse_trafo @ trafo.inverse()
            return inverse_trafo
        return trafo
    
    def relative_wigner_angles(self, other:'Tree', target: 'Node', momenta: dict) -> Tuple[Union[jnp.ndarray, np.array], Union[jnp.ndarray, np.array]]:
        # invert self, since this final state is seen as the reference
        boost1_inv = self.boost(target, momenta, inverse=True) 
        boost2 = other.boost(target, momenta)
        return (boost2 @ boost1_inv).wigner_angles()
    
    def __getattr__(self, name):
        return getattr(self.root, name)

def split(nodes:List[Node], split:int) -> Tuple[Tuple[Node], Tuple[Node]]:
    """
    Split a list of nodes into two lists of nodes.
    Parameters: nodes: List of nodes to split
                split: Index at which to split the list
    Returns: Tuple of lists of nodes
    """
    left = []
    right = []
    for i,n in enumerate(nodes):
        if split & (1 << i):
            left.append(n)
        else:
            right.append(n)
    return  tuple(left), tuple(right)


def generateTreeDefinitions(nodes:List[int]) -> List[Node]:
    """
    Generate all possible tree definitions for a given list of nodes.
    Parameters: nodes: List of nodes to generate tree definitions for
    Returns: List of tree definitions
    """
    trees = []
    if len(nodes) == 1:
        return [(None, None)]
    for i in range(1,1 << len(nodes) - 1):
        left, right = split(nodes, i)
        for l,r in generateTreeDefinitions(left):
            if len(left) == 1:
                lNode = Node(left[0])
            else:
                lNode = Node(left)
            if l is not None:
                lNode.add_daughter(l)
                lNode.add_daughter(r)
            for l2,r2 in generateTreeDefinitions(right):
                if len(right) == 1:
                    rNode = Node(right[0])
                else:
                    rNode = Node(right)
                if l2 is not None:
                    rNode.add_daughter(l2)
                    rNode.add_daughter(r2)
                trees.append((lNode, rNode))
    return trees

class Topology:

    def __init__(self, tree:Node):
        """
        Class to represent the topology of an N-body decay.
        Parameters: topology: List of integers representing the topology of the decay
        """
        self.__tree = tree

    @property
    def tree(self):
        """
        Returns: Tree representation of the topology
        """
        return Tree(self.__tree)
    
    def __repr__(self) -> str:
        return str(self.tree)
    
    def contains(self, contained_node:'Node'):
        """
        Check if a given node is contained in this topology.
        Parameters: contained_node: Node to check if it is contained
        Returns: True if the given node is contained in this topology, False otherwise
        """
        return self.tree.contains(contained_node)
    
    def generate_boost_graph(self, momenta:dict):
        """
        Generate the boost graph for this topology.
        Parameters: momenta: Dictionary of momenta for the final state particles
        Returns: Boost graph for this topology
        """
        return 

class TopologyGroup:
    @staticmethod
    def filter_list(trees:List[Node], contained_node: Node):
        """
        Filter the topologies based on the number of contained steps.

        Args:
            contained_step (list): sub topology for which to filter
        """
        return [t for t in trees if t.contains(contained_node)]

    def __init__(self, start_node:int, final_state_nodes:List[int]):
        self.start_node = start_node
        self.final_state_nodes = final_state_nodes
        self.node_numbers = {i:node for i,node in enumerate([start_node] + final_state_nodes)}
    
    @cached_property
    def trees(self) -> List[Tree]:
        trees = generateTreeDefinitions(self.final_state_nodes)    
        trees_with_root_node = []
        for l,r in trees:
            root = Node(self.start_node)
            root.add_daughter(l)
            root.add_daughter(r)
            trees_with_root_node.append(root)
        return [Tree(node) for node in trees_with_root_node]
    
    @cached_property
    def topologies(self):
        return [Topology(tree) for tree in self.trees]
    
    def filter(self, *contained_nodes: Node):
        """
        Filter the topologies based on the number of contained steps.

        Args:
            contained_nodes (tuple[Node]): nodes which should be contained in the trees
        """
        trees = self.trees
        for contained_node in contained_nodes:
            trees = self.filter_list(trees, contained_node)
        return trees
        