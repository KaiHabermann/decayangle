from typing import List, Tuple, Union, Any, Dict, Optional
from functools import cached_property
from collections import namedtuple
import numpy as np
from jax import numpy as jnp
import networkx as nx
from decayangle.lorentz import LorentzTrafo
from decayangle import kinematics as akm
from decayangle.numerics_helpers import matrix_vector_product
from decayangle.config import config as cfg

cb = cfg.backend

HelicityAngles = namedtuple("HelicityAngles", ["theta_rf", "psi_rf"])


def flat(l):
    """Flatten a nested list

    Args:
        l (list): the list to flatten

    Returns:
        list: the flattened list
    """
    if isinstance(l, (tuple, list)):
        for el in l:
            yield from flat(el)
    else:
        yield l


class Node:

    @staticmethod
    def construct_topology(node: "Node", topology: Tuple[Union[int, tuple]]):
        """Construct a topology from a tuple of integers and tuples, in the form like the string representation of a topology

        i.e. ((1,2), 3)
        or ((1, (2, 3)), 4) for a four body decay

        Args:
            node (Node): the node to add the daughters to
            topology (List[Union[int, tuple]]): the topology to construct
        """
        if not isinstance(topology, tuple):
            return
        left = Node(tuple(flat(topology[0])))
        right = Node(tuple(flat(topology[1])))
        node.add_daughter(left)
        node.add_daughter(right)
        Node.construct_topology(left, topology[0])
        Node.construct_topology(right, topology[1])

    @classmethod
    def get_node(cls, value: Union[int, Tuple[int], "Node"]) -> "Node":
        """Get a node from a value or return the value if a node is given

        Args:
            value (Union[int, Tuple[int], Node]): the value of the node

        Returns:
            Node: the node
        """
        if isinstance(value, Node):
            return value
        return cls(value)

    def __init__(self, value: Union[Any, tuple], sorting_key=None):
        if sorting_key is not None:
            self.__sorting_key = sorting_key
        else:
            self.__sorting_key = cfg.sorting_key

        if isinstance(value, tuple):
            if len(value) == 0:
                raise ValueError(
                    "Node value has to be an integer or a tuple of integers"
                )
            if len(value) == 1:
                # a single element in a tuple should have the value of the element
                # tuples are only for composites
                self.value = value[0]
            else:
                self.value = tuple(sorted(value, key=self.sorting_key))
        else:
            if not isinstance(value, int):
                raise ValueError(
                    "Node value has to be an integer or a tuple of integers"
                )
            if value < 0:
                raise ValueError("Node value has to be a positive integer or 0")
            if value > 10000:
                raise ValueError(
                    "Node value has to be smaller than 10000 to ensure consistent sorting of daughters"
                )
            self.value = value
        self.__daughters = []
        self.parent = None

    @property
    def sorting_key(self):
        """Get the sorting key of the node.
        This is used to sort the daughters and make sure, that the order of the daughters is consistent.

        Returns:
            int: the sorting key
        """
        return self.__sorting_key

    @sorting_key.setter
    def sorting_key(self, value):
        if not isinstance(value((1, 2, 3)), int):
            raise ValueError(
                "Sorting key has to be a function returning an integer and accepting an integer or tuple as input"
            )
        if not isinstance(value(1), int):
            raise ValueError(
                "Sorting key has to be a function returning an integer and accepting an integer or tuple as input"
            )

        self.__sorting_key = value
        self.__daughters = sorted(
            self.__daughters, key=lambda x: self.sorting_key(x.value)
        )
        if isinstance(self.value, tuple):
            self.value = tuple(sorted(self.value, key=self.sorting_key))
        for d in self.__daughters:
            d.sorting_key = value

    def add_daughter(self, daughter: "Node"):
        """Add a daughter to the node

        Args:
            daughter (Node): the daughter to add
        """
        self.__daughters.append(daughter)
        self.__daughters = sorted(
            self.__daughters, key=lambda x: self.sorting_key(x.value)
        )
        daughter.parent = self

    @property
    def daughters(self) -> List["Node"]:
        """Get the daughters of the node

        Returns:
            List[Node]: the daughters of the node
        """
        return self.__daughters

    def __repr__(self):
        if len(self.daughters) == 0:
            return str(self.value)
        return (
            f"( {self.value} -> " + f"{', '.join([str(d) for d in self.daughters])} )"
        )

    @property
    def final_state(self):
        """Check if the node is a final state node

        Returns:
            bool: True if the node is a final state node, False otherwise
        """
        return len(self.daughters) == 0

    def __str__(self):
        return self.__repr__()

    def print_tree(self):
        """Print the tree below from this node"""
        for d in self.daughters:
            d.print_tree()
        print(f"\n {self.value}")

    def contains(self, contained_node: "Node") -> bool:
        """Check if a node is contained in the topology

        Args:
            contained_node (Node): the node to check for

        Returns:
            bool: True if the node is contained in the topology, False otherwise
        """
        contained_node = Node.get_node(contained_node)
        contained_node.sorting_key = self.sorting_key
        if self.value == contained_node.value:
            return True
        for d in self.daughters:
            if d.contains(contained_node):
                return True
        return False

    def inorder(self):
        """Get the nodes in the tree in inorder

        Returns:
            list: the nodes in the tree in inorder
        """
        if len(self.daughters) == 0:
            return [self]
        return [self] + [node for d in self.daughters for node in d.inorder()]

    def momentum(
        self, momenta: Dict[str, Union[jnp.ndarray, np.array]]
    ) -> Union[jnp.ndarray, np.array]:
        """Get a particles momentum

        Args:
            momenta (dict): the momenta of the final state particles

        Returns:
            the momentum of the particle, as set by the momenta dictionary
            This expects the momenta to be jax or numpy compatible
        """
        if len(self.daughters) == 0:
            return momenta[self.value]
        return sum(d.momentum(momenta) for d in self.daughters)

    def mass(
        self, momenta: Dict[str, Union[jnp.ndarray, np.array]]
    ) -> Union[jnp.ndarray, np.array]:
        """Get the mass of the particle

        Args:
            momenta (dict): the momenta of the final state particles

        Returns:
            the mass of the particle, as set by the momenta dictionary
            This expects the momenta to be jax or numpy compatible
        """
        return akm.mass(self.momentum(momenta))

    def transform(self, trafo: LorentzTrafo, momenta: dict) -> dict:
        """Transform the momenta of the final state particles

        Args:
            trafo (LorentzTrafo): transformation to apply
            momenta (dict): the momenta of the final state particles

        Returns:
            dict: the transformed momenta
        """
        return {
            k: matrix_vector_product(trafo.matrix_4x4, v) for k, v in momenta.items()
        }

    def boost(self, target: Union["Node", int], momenta: dict):
        """Get the boost from this node to a target node
        The momenta dictionary will define the initial configuration.
        It is expected, that the momenta are jax or numpy compatible and that the momenta are given in the rest frame of this node.
        """
        if not cb.allclose(
            akm.gamma(self.momentum(momenta)),
            cb.ones_like(akm.gamma(self.momentum(momenta))),
        ):
            gamma = akm.gamma(self.momentum(momenta))
            raise ValueError(
                f"gamma = {gamma} For the time being only particles at rest are supported as start nodes for a boost. This will be fixed in the future."
            )
        target = Node.get_node(target)
        zero = cb.zeros_like(akm.time_component(self.momentum(momenta)))
        one = cb.ones_like(zero)
        if self.value == target.value:
            return LorentzTrafo(zero, zero, zero, zero, zero, zero)

        if target not in self.daughters:
            raise ValueError(
                f"Target node {target} is not a direct daughter of this node {self}"
            )

        # rotate so that the target momentum is aligned with the
        rotation, _, _ = self.rotate_to(target, momenta)
        rotated_momenta = self.transform(rotation, momenta)

        # boost to the rest frame of the target
        xi = -akm.rapidity(target.momentum(rotated_momenta))
        boost = LorentzTrafo(zero, zero, xi, zero, zero, zero)

        return boost @ rotation

    def align_with_daughter(
        self, momenta: Dict[int, Union[np.array, jnp.array]], nth_daughter: int = 0
    ) -> Dict[int, Union[np.array, jnp.array]]:
        """Align the momenta with the nth daughter

        Args:
            momenta (dict): the momenta of the final state particles
            nth_daughter (int, optional): the daughter to align with. Defaults to 0.

        Returns:
            dict: the aligned momenta
        """
        if nth_daughter >= len(self.daughters):
            raise ValueError(
                f"Node {self} does not have a daughter with index {nth_daughter}"
            )
        rotation, _, _ = self.rotate_to(self.daughters[nth_daughter], momenta)
        return self.transform(rotation, momenta)

    def helicity_angles(self, momenta: dict):
        """
        Get the helicity angles for every internal node

        Parameters:
            momenta: Dictionary of momenta for the final state particles

        Returns:
            Helicity angles for the final state particles

        """

        # define the daughter for which the momentum should be aligned with the positive z after the rotation
        positive_z = self.daughters[0]
        _, theta_rf, psi_rf = self.rotate_to(positive_z, momenta)
        return HelicityAngles(theta_rf, psi_rf)

    def rotate_to(
        self, target: "Node", momenta: dict
    ) -> Tuple[LorentzTrafo, float, float]:
        """Get the rotation from this node to a target node
        The momenta dictionary will define the initial configuration.
        It is expected, that the momenta are jax or numpy compatible and that the momenta are given in the rest frame of this node.

        Returns:
            rotation: The rotation to align the target momentum with the z-axis
            psi_rf: The angle of the target momentum in the rest frame of this node
            theta_rf: The angle of the target momentum in the rest frame of this node
        """
        if not cb.allclose(
            akm.gamma(self.momentum(momenta)),
            cb.ones_like(akm.gamma(self.momentum(momenta))),
        ):
            gamma = akm.gamma(self.momentum(momenta))
            raise ValueError(
                f"gamma = {gamma} For the time being only particles at rest are supported as start nodes for a boost. This will be fixed in the future."
            )
        zero = cb.zeros_like(akm.time_component(self.momentum(momenta)))
        if self.value == target.value:
            return LorentzTrafo(zero, zero, zero, zero, zero, zero)

        if not target in self.daughters:
            raise ValueError(
                f"Target node {target} is not a direct daughter of this node {self}"
            )

        # rotate so that the target momentum is aligned with the z axis
        psi_rf, theta_rf = akm.rotate_to_z_axis(target.momentum(momenta))
        rotation = LorentzTrafo(zero, zero, zero, zero, theta_rf, psi_rf)

        return rotation, theta_rf, psi_rf


class Topology:
    def __init__(
        self,
        root: Union[Node, int],
        decay_topology: Optional[List[Union[int, tuple]]] = None,
        sorting_key=None,
    ):
        self.__root = Node.get_node(root)
        if sorting_key is not None:
            self.__sorting_key = sorting_key
            self.__root.sorting_key = sorting_key
        else:
            self.__sorting_key = cfg.sorting_key

        if decay_topology is not None:
            if len(self.root.daughters) != 0:
                raise ValueError(
                    "If a decay topology is given, then the root node should not already have daughters!"
                    f"Root: {self.root}"
                )
            Node.construct_topology(self.root, decay_topology)

    @property
    def root(self) -> Node:
        """The root node of the topology

        Returns:
            Node: the root node of the topology
        """
        return self.__root

    @property
    def final_state_nodes(self) -> List[Node]:
        """The final state nodes of the topology

        Returns:
            List[Node]: the final state nodes of the topology
        """
        return [n for n in self.inorder() if n.final_state]

    @property
    def sorting_key(self):
        """The sorting key of the topology

        Returns:
            int: the sorting key of the topology
        """
        return self.__sorting_key

    @sorting_key.setter
    def sorting_key(self, value):
        if not isinstance(value((1, 2, 3)), int):
            raise ValueError(
                "Sorting key has to be a function returning an integer and accepting an integer or tuple as input"
            )
        if not isinstance(value(1), int):
            raise ValueError(
                "Sorting key has to be a function returning an integer and accepting an integer or tuple as input"
            )

        self.__sorting_key = value
        self.__root.sorting_key = value

    def __repr__(self) -> str:
        return f"Topology: {self.root}"

    def contains(self, contained_node: Union["Node", int]) -> bool:
        """Check if a node is contained in the topology

        Args:
            contained_node (Node): the node to check for

        Returns:
            bool: True if the node is contained in the topology, False otherwise
        """
        contained_node = Node.get_node(contained_node)
        contained_node.sorting_key = self.sorting_key
        return self.root.contains(contained_node)

    def to_rest_frame(self, momenta: dict):
        """Transform the momenta to the rest frame of the root node

        Args:
            momenta (dict): the momenta of the final state particles

        Returns:
            dict: the momenta in the rest frame of the root node
        """
        momentum = self.root.momentum(momenta)
        return {k: akm.boost_to_rest(v, momentum) for k, v in momenta.items()}

    def __build_boost_tree(self) -> Tuple[nx.DiGraph, Dict[int, Node]]:
        boost_tree = nx.DiGraph()
        node_dict = {}
        for node in self.inorder():
            boost_tree.add_node(node.value)
            node_dict[node.value] = node
        for node in self.inorder():
            for d in node.daughters:
                boost_tree.add_edge(node.value, d.value)
        return boost_tree, node_dict

    @property
    def nodes(self) -> Dict[Union[tuple, int], Node]:
        """nodes of the tree

        Returns:
            Dict[Union[tuple, int], Node]: A dict of the nodes with the node value as key
        """
        return {n.value: n for n in self.inorder()}

    def helicity_angles(self, momenta: dict) -> Dict[Tuple[int, int], HelicityAngles]:
        """
        Get a tree with the helicity angles for every internal node

        Parameters:
            momenta: Dictionary of momenta for the final state particles

        Returns:
            Helicity angles for the final state particles

        """
        helicity_angles = {}

        for node in self.root.inorder():
            if not node.final_state:
                if node != self.root:
                    boost_to_node = self.boost(node, momenta)
                    momenta_in_node_frame = self.root.transform(boost_to_node, momenta)
                else:
                    momenta_in_node_frame = momenta
                isobar, spectator = node.daughters
                helicity_angles[(isobar.value, spectator.value)] = node.helicity_angles(
                    momenta_in_node_frame
                )
        return helicity_angles

    def boost(
        self, target: Union["Node", int], momenta: dict, inverse: bool = False
    ) -> LorentzTrafo:
        """
        Get the boost from the root node to a target node.

        Parameters:
            target: Node to boost to
            momenta: Dictionary of momenta for the final state particles
            inverse: If True, return the inverse of the boost

        Returns:
            Boost from the root node to the target node

        """
        target = Node.get_node(target)
        boost_tree, node_dict = self.__build_boost_tree()
        path = nx.shortest_path(boost_tree, self.root.value, target.value)[1:]
        trafo = self.root.boost(node_dict[path[0]], momenta)
        momenta = self.root.transform(trafo, momenta)
        trafos = [trafo]
        for i in range(1, len(path)):
            boost = node_dict[path[i - 1]].boost(node_dict[path[i]], momenta)
            momenta = node_dict[path[i - 1]].transform(boost, momenta)
            trafo = boost @ trafo
            trafos.append(boost)
        if inverse:
            # this is more precise then the naive inverse
            inverse_trafo = trafos[0].inverse()
            for trafo in trafos[1:]:
                inverse_trafo = inverse_trafo @ trafo.inverse()
            return inverse_trafo
        return trafo

    def relative_wigner_angles(
        self, other: "Topology", target: Union["Node", int], momenta: dict
    ) -> Tuple[Union[jnp.ndarray, np.array], Union[jnp.ndarray, np.array]]:
        """Get the relative Wigner angles between two topologies

        Parameters:
            other: Topology to compare to
            target: Node to compare to
            momenta: Dictionary of momenta for the final state particles

        Returns:
            Tuple of the relative Wigner angles
        """
        target = Node.get_node(target)
        # invert self, since this final state is seen as the reference
        boost1_inv = self.boost(target, momenta, inverse=True)
        boost2 = other.boost(target, momenta)
        return (boost2 @ boost1_inv).wigner_angles()

    def __getattr__(self, name):
        return getattr(self.root, name)


def split(nodes: List[Node], splitter: int) -> Tuple[Tuple[Node], Tuple[Node]]:
    """
    Split a list of nodes into two lists of nodes.
    Parameters: nodes: List of nodes to split
                split: Index at which to split the list
    Returns: Tuple of lists of nodes
    """
    left = []
    right = []
    for i, n in enumerate(nodes):
        if splitter & (1 << i):
            left.append(n)
        else:
            right.append(n)
    return tuple(left), tuple(right)


def generate_tree_definitions(nodes: List[int]) -> List[Node]:
    """
    Generate all possible topology definitions for a given list of nodes.
    Parameters: nodes: List of nodes to generate topology definitions for
    Returns: List of topology definitions
    """
    topologies = []
    if len(nodes) == 1:
        return [(None, None)]
    for i in range(1, 1 << len(nodes) - 1):
        left, right = split(nodes, i)
        for l, r in generate_tree_definitions(left):
            if len(left) == 1:
                l_node = Node(left[0])
            else:
                l_node = Node(left)
            if l is not None:
                l_node.add_daughter(l)
                l_node.add_daughter(r)
            for l2, r2 in generate_tree_definitions(right):
                if len(right) == 1:
                    r_node = Node(right[0])
                else:
                    r_node = Node(right)
                if l2 is not None:
                    r_node.add_daughter(l2)
                    r_node.add_daughter(r2)
                topologies.append((l_node, r_node))
    return topologies


class TopologyCollection:
    """
    A group of topologies with the same start node and final state nodes
    """

    @staticmethod
    def filter_list(topologies: List[Node], contained_node: Node):
        """
        Filter the topologies based on the number of contained steps.

        Args:
            contained_step (list): sub topology for which to filter
        """
        return [t for t in topologies if t.contains(contained_node)]

    def __init__(
        self,
        start_node: int = None,
        final_state_nodes: List[int] = None,
        topologies: List[Topology] = None,
        sorting_key=None,
    ):
        if topologies is not None:
            self.__topologies = topologies
            self.start_node = topologies[0].root.value
            self.final_state_nodes = [n.value for n in topologies[0].final_state_nodes]
            for topology in topologies:
                if topology.root.value != self.start_node:
                    raise ValueError("All topologies have to have the same start node")
                if any(
                    [
                        n.value not in self.final_state_nodes
                        for n in topology.final_state_nodes
                    ]
                ):
                    raise ValueError(
                        "All topologies have to have the same final state nodes"
                    )
        elif start_node is not None and final_state_nodes is not None:
            self.__topologies = None
            self.start_node = start_node
            self.final_state_nodes = final_state_nodes
        else:
            raise ValueError(
                "Either topologies or start_node and final_state_nodes have to be given"
            )

        self.node_numbers = dict(enumerate([start_node] + final_state_nodes))
        if sorting_key is not None:
            self.__sorting_key = sorting_key
        else:
            self.__sorting_key = cfg.sorting_key

    @property
    def sorting_key(self):
        """The sorting key of the topology, used to sort the daughters of the nodes and the values of the composite nodes

        Returns:
            A function returning an integer and accepting an integer or tuple as input
        """
        return self.__sorting_key

    @sorting_key.setter
    def sorting_key(self, value):
        if not isinstance(value((1, 2, 3)), int):
            raise ValueError(
                "Sorting key has to be a function returning an integer and accepting an integer or tuple as input"
            )
        if not isinstance(value(1), int):
            raise ValueError(
                "Sorting key has to be a function returning an integer and accepting an integer or tuple as input"
            )

        self.__sorting_key = value
        for topology in self.topologies:
            topology.sorting_key = value

    def __generate_topologies(self) -> List[Topology]:
        """returns all possible topologies for the given final state nodes

        Returns:
            List[Topology]: all possible topologies for the given final state nodes
        """
        topologies = generate_tree_definitions(self.final_state_nodes)
        topologies_with_root_node = []
        for l, r in topologies:
            root = Node(self.start_node)
            root.add_daughter(l)
            root.add_daughter(r)
            topologies_with_root_node.append(root)
        return [
            Topology(node, sorting_key=self.sorting_key)
            for node in topologies_with_root_node
        ]

    @property
    def topologies(self) -> List[Topology]:
        """returns all possible topologies for the given final state nodes

        Returns:
            List[Topology]: all possible topologies for the given final state nodes
        """
        if self.__topologies is None:
            self.__topologies = self.__generate_topologies()
        return self.__topologies

    def filter(self, *contained_nodes: Node):
        """
        Filter the topologies based on the number of contained steps.

        Args:
            contained_nodes (tuple[Node]): nodes which should be contained in the topologies
        """
        topologies = self.topologies
        for contained_node in contained_nodes:
            topologies = self.filter_list(topologies, contained_node)
        return topologies
