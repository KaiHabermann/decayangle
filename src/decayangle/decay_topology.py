from __future__ import annotations
from typing import List, Tuple, Union, Any, Dict, Optional, Generator, Callable, Literal
from collections import namedtuple
import numpy as np
from jax import numpy as jnp
import networkx as nx
from decayangle.lorentz import LorentzTrafo, WignerAngles
from decayangle import kinematics as akm
from decayangle.numerics_helpers import matrix_vector_product
from decayangle.config import config as cfg

cb = cfg.backend

HelicityAngles = namedtuple("HelicityAngles", ["phi_rf", "theta_rf"])


def flat(l) -> Generator:
    """Flatten a nested list

    Args:
        l (list): the list to flatten

    Returns:
        list: the flattened list as a generator
    """
    if isinstance(l, (tuple, list)):
        for el in l:
            yield from flat(el)
    else:
        yield l


class Node:
    """
    Class to represent a node in a decay topology. The node can have daughters, which are also nodes.
    The value of a node is either an integer or a tuple of integers. A tuple implies that the node will finally decay into the particles given in the tuple.
    The ordering function is used to sort the daughters and the values of the composite nodes.

    Attributes:
        value (Union[int, Tuple[int]]): the value of the node
        parent (Node): the parent node of the node
        daughters (Tuple[Node]): the daughters of the node
        ordering_function (function): function ordering the daughters and node values of the node
    """

    @staticmethod
    def construct_topology(node: Node, topology: Tuple[Union[int, tuple]]):
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
    def get_node(
        cls, value: Union[int, Tuple[int], Node], add_daughters: bool = True
    ) -> Node:
        """Get a node from a value or return the value if a node is given

        Args:
            value (Union[int, Tuple[int], Node]): the value of the node

        Returns:
            Node: the node
        """
        if isinstance(value, Node):
            return value
        return cls(value, add_daughters=add_daughters)

    def __init__(
        self,
        value: Union[Any, tuple],
        ordering_function=None,
        add_daughters: bool = False,
    ):
        if ordering_function is not None:
            self.__sorting_fun = ordering_function
        else:
            self.__sorting_fun = cfg.ordering_function

        if isinstance(value, tuple):
            if len(value) == 0:
                raise ValueError(
                    f"Node value has to be an integer or a tuple of integers not {type(value)}"
                )
            if len(value) == 1:
                # a single element in a tuple should have the value of the element
                # tuples are only for composites
                self.value = value[0]
            else:
                self.value = tuple(self.ordering_function(value))
        else:
            if not isinstance(value, int):
                raise ValueError(
                    f"Node value has to be an integer or a tuple of integers not {type(value)}"
                )
            if value < 0:
                raise ValueError("Node value has to be a positive integer or 0")
            if value > 10000:
                raise ValueError(
                    "Node value has to be smaller than 10000 to ensure consistent sorting of daughters"
                )
            self.value = value
        self.__daughters = tuple()
        self.parent = None
        if add_daughters:
            self.add_daughters()

    def add_daughters(self):
        """Add the daughters to the node"""
        if isinstance(self.value, tuple) and len(self.daughters) == 0:
            for v in self.value:
                self.add_daughter(Node.get_node(v, add_daughters=True))

    @property
    def ordering_function(self) -> Callable:
        """Get the sorting key of the node.
        This is used to sort the daughters and make sure, that the order of the daughters is consistent.

        Returns:
            int: the sorting key
        """
        return self.__sorting_fun

    @property
    def tuple(self):
        """
        Tuple representation of the node. When called on the root node of a tree, the result can be used to reconstruct the tree.

        Returns:
            tuple: the tuple representation of the node
        """
        if self.final_state:
            return self.value
        return tuple((d.tuple for d in self.daughters))

    def root(self):
        if self.parent is not None:
            return self.parent.root()
        return self

    @ordering_function.setter
    def ordering_function(self, value):
        """
        Set the sorting function for the node and all daughters
        Sorting functions are expected to return the same data type as the input
        They need to accept lists, tuples and integers as input
        """

        if not isinstance(value((1, 2, 3)), tuple):
            raise ValueError(
                "Sorting function has to be a function returning the sorted value of the same datatype and accepting tupels and lists of integers"
            )
        if not isinstance(value(1), int):
            raise ValueError(
                "Sorting function has to be a function returning the sorted value of the same datatype and accepting tupels and lists of integers"
            )

        self.__sorting_fun = value
        self.__daughters = self.__sorted_daughters()

        if isinstance(self.value, tuple):
            self.value = self.ordering_function(self.value)
        for d in self.__daughters:
            d.ordering_function = value

    def __sorted_daughters(self) -> Tuple[Node]:
        """
        Sort the daughters of the node, by passing the values to the sorting function
        Then return the daughters in the order of the sorted values
        """
        daughter_values = [d.value for d in self.__daughters]
        sorted_values = self.ordering_function(daughter_values)
        return tuple(self.__daughters[daughter_values.index(v)] for v in sorted_values)

    def add_daughter(self, daughter: Node):
        """Add a daughter to the node.
        The daughter has to be of type Node, since this function should only be called when constructing a topology.
        No checks are made to ensure that the daughter is not already a daughter of the node.
        Daughters are re-sorted after adding a new daughter.

        Args:
            daughter (Node): the daughter to add
        """
        if not isinstance(daughter, Node):
            raise ValueError("Daughter has to be a Node")

        self.__daughters = self.__daughters + (daughter,)
        self.__daughters = self.__sorted_daughters()
        daughter.parent = self

    @property
    def daughters(self) -> Tuple[Node]:
        """Get the daughters of the node

        Returns:
            List[Node]: the daughters of the node
        """
        return self.__daughters

    def __repr__(self) -> str:
        if len(self.daughters) == 0:
            return str(self.value)
        return (
            f"( {self.value} -> " + f"{', '.join([str(d) for d in self.daughters])} )"
        )

    @property
    def final_state(self) -> bool:
        """Check if the node is a final state node by checking if it has daughters

        Returns:
            bool: True if the node is a final state node, False otherwise
        """
        return len(self.daughters) == 0

    def __str__(self) -> str:
        return self.__repr__()

    def contains(self, contained_node: Node) -> bool:
        """Check if a node is contained in the topology

        Args:
            contained_node (Node): the node to check for

        Returns:
            bool: True if the node is contained in the topology, False otherwise
        """
        contained_node = Node.get_node(contained_node)
        if isinstance(self.value, int):
            if self.value == contained_node.value:
                return True
        elif set(self.value) == set(contained_node.value):
            return True
        for d in self.daughters:
            if d.contains(contained_node):
                return True
        return False

    def preorder(self) -> List[Node]:
        """Get the nodes in the tree in preorder.
        This is a recursive function, which will return the nodes in the tree in the order of the preorder traversal.
        This means, root first, then daughters in the order they were added.

        Returns:
            list: the nodes in the tree in preorder
        """
        if len(self.daughters) == 0:
            return [self]
        return [self] + [node for d in self.daughters for node in d.preorder()]

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

    def transform(
        self, trafo: LorentzTrafo, momenta: Dict[str, Union[np.array, jnp.array]]
    ) -> Dict[int, Union[np.array, jnp.array]]:
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

    def boost(
        self,
        target: Union[Node, int],
        momenta: Dict[str, Union[np.array, jnp.array]],
        tol: Optional[float] = None,
        convention: Literal["helicity", "minus_phi", "canonical"] = "helicity",
        frame_for_massless: Optional[Union[Node, int]] = None,
    ) -> LorentzTrafo:
        """Get the boost from this node to a target node
        The momenta dictionary will define the initial configuration.
        It is expected, that the momenta are jax or numpy compatible and that the momenta are given in the rest frame of this node.

        Args:
            target (Union[Node, int]): the target node to boost to
            momenta (dict): the momenta of the final state particles
            tol (float, optional): tolerance for the gamma check. Defaults to the value in the config.
            convention (Literal["helicity", "minus_phi", "canonical"], optional): the convention to use for the boost. Defaults to "helicity".
                helicity: we just rotate to be aligned with the target and boost
                minus_phi: we rotate to be aligned, boost and then roate the azimutal angle (phi or psi) back
                canonical: We rotate, boost and rotate back. Thus the action is a pure boost
            frame_for_massless (Optional[Union[Node, int]], optional): the frame to use for massless particles. Defaults to the root node.
        """
        if tol is None:
            tol = cfg.gamma_tolerance
        if not cb.allclose(
            akm.gamma(self.momentum(momenta)),
            cb.ones_like(akm.gamma(self.momentum(momenta))),
            rtol=tol,
        ):
            gamma = akm.gamma(self.momentum(momenta))
            cfg.raise_if_safety_on(
                ValueError(
                    f"Trying to boost {self} to {target} but the momentum is not at rest. gamma = {gamma} For the time being only particles at rest are supported as start nodes for a boost. To boost to the rest frame of the root node, use method ```momenta = topology.to_rest_frame(momenta)```."
                )
            )
        target = Node.get_node(target)

        zero = cb.zeros_like(akm.time_component(self.momentum(momenta)))
        if self.value == target.value:
            return LorentzTrafo(zero, zero, zero, zero, zero, zero)

        frame_for_massless = (
            Node.get_node(frame_for_massless)
            if frame_for_massless is not None
            else self.root()
        )

        if target not in self.daughters and target != frame_for_massless:
            raise ValueError(
                f"Target node {target} is not a direct daughter of this node {self}"
            )

        # boost to the rest frame of the target
        xi = -akm.rapidity(target.momentum(momenta))
        boost = LorentzTrafo(zero, zero, xi, zero, zero, zero)
        masses = cb.nan_to_num(akm.mass(target.momentum(momenta)), nan=0)
        if cb.all(masses < 1e-6):
            # target.parent = self
            if frame_for_massless == self:
                boost_to_root = LorentzTrafo(zero, zero, zero, zero, zero, zero)
            else:
                boost_to_root = self.boost(
                    frame_for_massless,
                    momenta,
                    convention="canonical",
                    frame_for_massless=frame_for_massless,
                )  # canonical convention is a pure boost
            boost = boost_to_root

        if convention == "canonical":
            rotation, minus_theta_rf, minus_psi_rf = self.rotate_to(
                target, momenta, tol=tol
            )
            # return here, since we do not need particle 2 convention
            return rotation.inverse() @ boost @ rotation

        # particle 2 convention requires, that we move to particle 2 as Lambda_1 * Rot_y(-pi)
        if convention == "helicity":
            rotation_daughter1, minus_theta_rf, minus_psi_rf = self.rotate_to(
                self.daughters[0], momenta, tol=tol
            )
            rotation = rotation_daughter1

        elif convention == "minus_phi":
            _, minus_theta_rf, minus_psi_rf = self.rotate_to(
                self.daughters[0], momenta, tol=tol
            )
            rotation = LorentzTrafo(
                zero, zero, zero, -minus_psi_rf, minus_theta_rf, minus_psi_rf
            )

        else:
            raise ValueError(
                f"Convention {convention} not supported. Use 'helicity', 'minus_phi' or 'canonical'."
            )

        if target != self.daughters[0]:
            rotation = LorentzTrafo(0, 0, 0, 0, -cb.pi, 0) @ rotation
        full_transformation = boost @ rotation

        return full_transformation

    def align_with_daughter(
        self,
        momenta: Dict[int, Union[np.array, jnp.array]],
        nth_daughter: int = 0,
        tol: Optional[float] = None,
    ) -> Dict[int, Union[np.array, jnp.array]]:
        """Align the momenta with the nth daughter. It is written in this way, to highlight, that one can only align with the direct daughters of a node.
        This requires the momenta to be in the rest frame of the node.

        Args:
            momenta (dict): the momenta of the final state particles
            nth_daughter (int, optional): the daughter to align with. Defaults to 0.
            tol (float, optional): tolerance for the gamma check. Defaults to the value in the config.

        Returns:
            dict: the aligned momenta
        """
        if nth_daughter >= len(self.daughters):
            raise ValueError(
                f"Node {self} does not have a daughter with index {nth_daughter}"
            )
        rotation, _, _ = self.rotate_to(self.daughters[nth_daughter], momenta, tol=tol)
        return self.transform(rotation, momenta)

    def helicity_angles(
        self,
        momenta: Dict[str, Union[np.array, jnp.array]],
        tol: Optional[float] = None,
    ) -> HelicityAngles:
        """
        Get the helicity angles for the daughters of this node.
        The angles are with respect to the first daughter.
        Here the ordering scheme can be important.

        Parameters:
            momenta: Dictionary of momenta for the final state particles
            tol: Tolerance for the gamma check

        Returns:
            Helicity angles for the final state particles

        """

        # define the daughter for which the momentum should be aligned with the positive z after the rotation
        positive_z = self.daughters[0]
        _, minus_theta_rf, minus_phi_rf = self.rotate_to(positive_z, momenta, tol=tol)
        return HelicityAngles(
            -minus_phi_rf,
            -minus_theta_rf,
        )

    def rotate_to(
        self,
        target: Node,
        momenta: Dict[str, Union[np.array, jnp.array]],
        tol: Optional[float] = None,
    ) -> Tuple[
        LorentzTrafo,
        Union[float, np.array, jnp.array],
        Union[float, np.array, jnp.array],
    ]:
        """Get the rotation from this node to a target node
        The momenta dictionary will define the initial configuration.
        It is expected, that the momenta are jax or numpy compatible and that the momenta are given in the rest frame of this node.

        Args:
            target (Node): the target node to rotate to
            momenta (dict): the momenta of the final state particles
            tol (float, optional): tolerance for the gamma check. Defaults to the value in the config.

        Returns:
            rotation (LorentzTrafo): the rotation to apply
            theta_rf (Union[float, np.array, jnp.array]): the polar angle of the rotation
            psi_rf (Union[float, np.array, jnp.array]): the azimuthal angle of the rotation
        """

        if tol is None:
            tol = cfg.gamma_tolerance
        if not cb.allclose(
            akm.gamma(self.momentum(momenta)),
            cb.ones_like(akm.gamma(self.momentum(momenta))),
            rtol=tol,
        ):
            gamma = akm.gamma(self.momentum(momenta))
            cfg.raise_if_safety_on(
                ValueError(
                    f"Trying to rotate {self} to {target} but the momentum is not at rest. gamma = {gamma} For the time being only particles at rest are supported as start nodes for a boost. To boost to the rest frame of the root node, use method ```momenta = topology.to_rest_frame(momenta)```."
                )
            )
        zero = cb.zeros_like(akm.time_component(self.momentum(momenta)))
        if self.value == target.value:
            return LorentzTrafo(zero, zero, zero, zero, zero, zero)

        # if target not in self.daughters and target != self.root():
        #     raise ValueError(
        #         f"Target node {target} is not a direct daughter of this node {self}"
        #     )

        # rotate so that the target momentum is aligned with the z axis
        minus_phi_rf, minus_theta_rf = akm.rotate_to_z_axis(target.momentum(momenta))
        rotation = LorentzTrafo(zero, zero, zero, zero, minus_theta_rf, minus_phi_rf)
        return rotation, minus_theta_rf, minus_phi_rf


class Topology:
    """A class to represent a decay topology as a tree of nodes.
    The tree is constructed from a root node, which has daughters, which can have daughters and so on.
    The final state nodes are the nodes without daughters.
    The ordering function is used to sort the daughters of the nodes and the values of the composite nodes.

    Properties:
        root (Node): the root node of the topology
        final_state_nodes (List[Node]): the final state nodes of the topology
        ordering_function (function): function ordering the daughters and node values of the topology

    """

    def __init__(
        self,
        root: Union[Node, int],
        decay_topology: Optional[List[Union[int, tuple]]] = None,
        ordering_function=None,
    ):
        """
        Constructor for the Topology class
        Topologies can be constructed from a root node and a decay topology. Or only be initialized with the root node.
        A tree then has to be construced via adding daughters to the root node.

        Args:
            root (Union[Node, int]): the root node of the topology
            decay_topology (Optional[List[Union[int, tuple]]], optional): the decay topology of the root node. Defaults to None.
            ordering_function ([type], optional): the ordering function for the nodes in the topology. Defaults to None.

        Raises:
            ValueError: If the root node already has daughters and a decay topology is given

        Examples:
        ```python
        # Initialize a topology with a root node and a decay topology
        topology = Topology(0, decay_topology=((1, 2), 3))
        # Initialize a topology with a root node
        topology = Topology(0)
        topology.root.add_daughter(Node(1))
        ```
        """
        self.__root = Node.get_node(root)
        if ordering_function is not None:
            self.__sorting_fun = ordering_function
            self.__root.ordering_function = ordering_function
        else:
            self.__sorting_fun = cfg.ordering_function

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
        return [n for n in self.preorder() if n.final_state]

    @property
    def tuple(self):
        """
        Tuple representation of the topology. When called on the root node of a tree, the result can be used to reconstruct the tree.

        Returns:
            tuple: the tuple representation of the topology root node
        """
        return self.root.tuple

    @property
    def ordering_function(self) -> Callable:
        """The sorting key of the topology

        Returns:
            int: the sorting key of the topology
        """
        return self.__sorting_fun

    @ordering_function.setter
    def ordering_function(self, value):
        if not isinstance(value((1, 2, 3)), tuple):
            raise ValueError(
                "Sorting function has to be a function returning the sorted value of the same datatype and accepting tupels and lists of integers"
            )
        if not isinstance(value(1), int):
            raise ValueError(
                "Sorting function has to be a function returning the sorted value of the same datatype and accepting tupels and lists of integers"
            )

        self.__sorting_fun = value
        self.__root.ordering_function = value

    def __repr__(self) -> str:
        return f"Topology: {self.root}"

    def contains(self, contained_node: Union[Node, int]) -> bool:
        """Check if a node is contained in the topology

        Args:
            contained_node (Node): the node to check for

        Returns:
            bool: True if the node is contained in the topology, False otherwise
        """
        contained_node = Node.get_node(contained_node)
        contained_node.ordering_function = self.ordering_function
        return self.root.contains(contained_node)

    def to_rest_frame(
        self,
        momenta: Dict[str, Union[np.array, jnp.array]],
        tol: Optional[float] = None,
    ) -> Dict[int, Union[np.array, jnp.array]]:
        """Transform the momenta to the rest frame of the root node

        Args:
            momenta (dict): the momenta of the final state particles
            tol (float, optional): tolerance for the gamma check. Defaults to the value in the config. When the original gamma is close to 1, the momenta are assumed to be in the rest frame of the root node.

        Returns:
            dict: the momenta in the rest frame of the root node
        """
        if tol is None:
            tol = cfg.gamma_tolerance

        momentum = self.root.momentum(momenta)
        gamma = akm.gamma(momentum)
        if cb.allclose(gamma, cb.ones_like(gamma), rtol=tol):
            return momenta
        return {k: akm.boost_to_rest(v, momentum) for k, v in momenta.items()}

    def __build_boost_tree(self) -> Tuple[nx.DiGraph, Dict[int, Node]]:
        """Build a boost tree from the topology

        Returns:
            nx.DiGraph: the boost tree as a directed graph
            Dict[int, Node]: a dictionary of the nodes with the node value as key
        """

        boost_tree = nx.DiGraph()
        node_dict = {}
        for node in self.preorder():
            boost_tree.add_node(node.value)
            node_dict[node.value] = node
        for node in self.preorder():
            for d in node.daughters:
                boost_tree.add_edge(node.value, d.value)
        return boost_tree, node_dict

    def path_to(
        self, target: Union[Node, int]
    ) -> Tuple[Union[tuple, int], Dict[int, Node]]:
        """Get the path to a target node

        Args:
            target (Union[Node, int]): the target node to get the path to

        Returns:
            list: the path to the target node
        """
        target = Node.get_node(target)
        boost_tree, node_dict = self.__build_boost_tree()
        return (
            nx.shortest_path(boost_tree, source=self.root.value, target=target.value),
            node_dict,
        )

    @property
    def nodes(self) -> Dict[Union[tuple, int], Node]:
        """nodes of the tree

        Returns:
            Dict[Union[tuple, int], Node]: A dict of the nodes with the node value as key
        """
        return {n.value: n for n in self.preorder()}

    def helicity_angles(
        self,
        momenta: Dict[str, Union[np.array, jnp.array]],
        tol: Optional[float] = None,
        convention: Literal["helicity", "minus_phi", "canonical"] = "helicity",
    ) -> Dict[Tuple[Union[tuple, int], Union[tuple, int]], HelicityAngles]:
        """
        Get a tree with the helicity angles for every internal node

        Parameters:
            momenta(Dictionary): Dictionary of momenta for the final state particles
            tol(float): Tolerance for the gamma check. Defaults to the value in the config.
            convention(Literal["helicity", "minus_phi", "canonical"]): the convention to use.
            Defaults to "helicity".
                helicity: we just rotate to be aligned with the target and boost
                minus_phi: we rotate to be aligned, boost and then roate the azimutal angle (phi or psi) back
                canonical: We rotate, boost and rotate back. Thus the action is a pure boost


        Returns:
            Helicity angles for the final state particles

        """
        helicity_angles = {}

        for node in self.root.preorder():
            if not node.final_state:
                if node != self.root:
                    # TODO: this is a slow, but clean approach, where we only arrive in internal node frames vial the boost method. Maybe we could speed this up by not boosting from the root all the time
                    boost_to_node = self.boost(
                        node, momenta, tol=tol, convention=convention
                    )
                    momenta_in_node_frame = self.root.transform(boost_to_node, momenta)
                else:
                    momenta_in_node_frame = momenta
                isobar, spectator = node.daughters
                helicity_angles[(isobar.value, spectator.value)] = node.helicity_angles(
                    momenta_in_node_frame
                )
        return helicity_angles

    def boost(
        self,
        target: Union[Node, int],
        momenta: Dict[str, Union[np.array, jnp.array]],
        inverse: bool = False,
        tol: Optional[float] = None,
        convention: Literal["helicity", "minus_phi", "canonical"] = "helicity",
        frame_for_massless: Optional[Union[Node, int]] = None,
    ) -> LorentzTrafo:
        """
        Get the boost from the root node to a target node.

        Parameters:
            target: Node to boost to
            momenta: Dictionary of momenta for the final state particles
            inverse: If True, return the inverse of the boost
            tol: Tolerance for the gamma check. Defaults to the value in the config.
            convention: The convention to use for the boost. Defaults to "helicity".
            frame_for_massless: The frame to use for massless particles. Defaults to the root node.
        Returns:
            Boost from the root node to the target node

        """
        target = Node.get_node(target)
        path, node_dict = self.path_to(target)
        trafo = self.root.boost(
            node_dict[path[0]],
            momenta,
            tol=tol,
            convention=convention,
            frame_for_massless=frame_for_massless,
        )
        momenta = self.root.transform(trafo, momenta)
        trafos = [trafo]
        for i in range(1, len(path)):
            boost = node_dict[path[i - 1]].boost(
                node_dict[path[i]],
                momenta,
                tol=tol,
                convention=convention,
                frame_for_massless=frame_for_massless,
            )
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

    def rotate_between_topologies(
        self,
        other: "Topology",
        target: Union[Node, int],
        momenta: Dict[str, Union[np.array, jnp.array]],
        tol: Optional[float] = None,
        convention: Literal["helicity", "minus_phi", "canonical"] = "helicity",
        frame_for_massless: Optional[Union[Node, int]] = None,
    ) -> LorentzTrafo:
        """Get the relative Wigner angles between two topologies

        Parameters:
            other: Topology to compare to
            target: Node to compare to
            momenta: Dictionary of momenta for the final state particles
            tol: Tolerance for the gamma check. Defaults to the value in the config.
            convention: The convention to use for the boost. Defaults to "helicity".
            frame_for_massless: The frame to use for massless particles. Defaults to the root node.
        Returns:
            The rotation between the two rest frames for the target node, one arrives at by boosting from the mother rest frame to the target rest frame as described by the two topologies
        """
        if convention not in ["helicity", "minus_phi", "canonical"]:
            raise ValueError(
                f"Convention {convention} not supported. Use 'helicity', 'minus_phi' or 'canonical'."
            )
        if self == other:
            return LorentzTrafo(0, 0, 0, 0, 0, 0)
        target = Node.get_node(target)
        # invert self, since this final state is seen as the reference
        boost1_inv = self.boost(
            target,
            momenta,
            tol=tol,
            convention=convention,
            inverse=True,
            frame_for_massless=frame_for_massless,
        )
        boost2 = other.boost(
            target,
            momenta,
            tol=tol,
            convention=convention,
            frame_for_massless=frame_for_massless,
        )
        return boost2 @ boost1_inv

    def relative_wigner_angles(
        self,
        other: "Topology",
        momenta: Dict[str, Union[np.array, jnp.array]],
        tol: Optional[float] = None,
        convention: Literal["helicity", "minus_phi", "canonical"] = "helicity",
        frame_for_massless: Optional[Union[Node, int]] = None,
    ) -> Dict[int, Tuple[Union[jnp.ndarray, np.array], Union[jnp.ndarray, np.array]]]:
        """Get the relative Wigner angles between two topologies

        Parameters:
            other: Topology to compare to
            target: Node to compare to
            momenta: Dictionary of momenta for the final state particles
            tol: Tolerance for the gamma check. Defaults to the value in the config.
            convention: The convention to use for the boost. Defaults to "helicity".
            frame_for_massless: The frame to use for massless particles. Defaults to the root node.
        Returns:
            Dict of the relative Wigner angles with the final state node as key
        """
        return {
            target.value: self.rotate_between_topologies(
                other,
                target,
                momenta,
                tol=tol,
                convention=convention,
                frame_for_massless=frame_for_massless,
            ).wigner_angles()
            for target in self.final_state_nodes
        }

    def align_with_daughter(
        self,
        momenta: Dict[int, Union[np.array, jnp.array]],
        node: Optional[Union[int, Node]] = None,
    ) -> Dict[int, Union[np.array, jnp.array]]:
        """Align the momenta with the node passed as argument. If no node is passed, the first daughter is used.
        If the node is not a daughter of the root node, a ValueError is raised.

        Args:
            momenta (dict): the momenta of the final state particles
            node (int, optional): the daughter to align with. Defaults to the first daughter.

        Returns:
            dict: the aligned momenta
        """
        if node is None:
            nth_daughter = 0
        else:
            node = Node.get_node(node)
            node.ordering_function = self.ordering_function
            try:
                (nth_daughter,) = [
                    i
                    for i, d in enumerate(self.root.daughters)
                    if d.value == node.value
                ]
            except ValueError:
                raise ValueError(
                    f"Node {node} is not a daughter of the root node {self.root}"
                )
        return self.root.align_with_daughter(momenta, nth_daughter)

    def preorder(self) -> List[Node]:
        """Get the nodes in the tree in preorder. This only calls the preorder function of the root node.
        For more details see the preorder function of the Node class.

        Returns:
            list: the nodes in the tree in preorder
        """
        return self.root.preorder()


def split(nodes: List[Node], splitter: int) -> Tuple[Tuple[Node], Tuple[Node]]:
    """
    Split a list of nodes into two lists of nodes.
    Parameters: nodes: List of nodes to split
                splitter: Bitmask to split the nodes 1 -> left, 0 -> right
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


def generate_topology_definitions(nodes: List[int]) -> List[Node]:
    """
    Generate all possible topology definitions for a given list of nodes.

    Parameters:
        nodes: List of nodes to generate topology definitions for

    Returns:
        List of topology definitions
    """
    topologies = []
    if len(nodes) == 1:
        return [(None, None)]
    for i in range(1, 1 << len(nodes) - 1):
        left, right = split(nodes, i)
        for l, r in generate_topology_definitions(left):
            if len(left) == 1:
                l_node = Node(left[0])
            else:
                l_node = Node(left)
            if l is not None:
                l_node.add_daughter(l)
                l_node.add_daughter(r)
            for l2, r2 in generate_topology_definitions(right):
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
    A group of topologies with the same start node and final state nodes. Mostly used to filter topologies based on the internal nodes they contain.
    Also ensures, that all topologies are for the same final state nodes and start node.

    Attributes:
        start_node (int): the start node of the topologies. Has to be the same for all topologies.
        final_state_nodes (list): the final state nodes of the topologies. Have to be the same for all topologies.
        topologies (list): the topologies of the collection
        ordering_function (function): function ordering the daughters and node values of the topologies. Will overwrite the ordering function of the topologies.
    """

    @staticmethod
    def filter_list(topologies: List[Node], contained_node: Node) -> List[Topology]:
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
        ordering_function=None,
    ):
        """
        Initialize the topology collection with either a list of topologies or a start node and final state nodes.
        If topologies are given, the start node and final state nodes are taken from the first topology.
        If start node and final state nodes are given, the topologies are generated from the final state nodes.

        Args:
            start_node (int): the start node of the topologies
            final_state_nodes (list): the final state nodes of the topologies
            topologies (list): the topologies of the collection
            ordering_function (function): function ordering the daughters and node values of the topologies

        Raises:
            ValueError: if neither topologies nor start_node and final_state_nodes are given
            ValueError: if the topologies have different start nodes. Only applies if a list of topologies is given.

        Examples:

        ```python
        from decayangle.decay_topology import Topology, TopologyCollection
        # Create a topology collection from a list of topologies
        topologies = [Topology(start_node=0, (1, (2, 3))), Topology((start_node=0, (2, (1, 3)))]
        collection = TopologyCollection(topologies=topologies)

        # Create a topology collection from a start node and final state nodes
        collection = TopologyCollection(start_node=0, final_state_nodes=[2, 3])
        ```
        """
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

        self.node_numbers = dict(enumerate([self.start_node] + self.final_state_nodes))
        if ordering_function is not None:
            self.ordering_function = ordering_function
        else:
            self.ordering_function = cfg.ordering_function

    @property
    def ordering_function(self) -> Callable:
        """The sorting key of the topology, used to sort the daughters of the nodes and the values of the composite nodes

        Returns:
            A function returning an integer and accepting an integer or tuple as input
        """
        return self.__sorting_fun

    @ordering_function.setter
    def ordering_function(self, value: Callable):
        """
        Set the sorting function for the TopologyCollection and all topologies in the collection
        Sorting functions are expected to return the same data type as the input
        They need to accept lists, tuples and integers as input

        Args:
            value (Callable): the sorting function with the signature (Union[int, Tuple[int]]) -> Union[int, Tuple[int]]

        Raises:
            ValueError: if the sorting function does not have the correct signature
        """

        if not isinstance(value((1, 2, 3)), tuple):
            raise ValueError(
                "Sorting function has to be a function returning the sorted value of the same datatype and accepting tupels and lists of integers"
            )
        if not isinstance(value(1), int):
            raise ValueError(
                "Sorting function has to be a function returning the sorted value of the same datatype and accepting tupels and lists of integers"
            )

        self.__sorting_fun = value
        for topology in self.topologies:
            topology.ordering_function = value

    def __generate_topologies(self) -> List[Topology]:
        """returns all possible topologies for the given final state nodes

        Returns:
            List[Topology]: all possible topologies for the given final state nodes
        """
        topologies = generate_topology_definitions(self.final_state_nodes)
        topologies_with_root_node = []
        for l, r in topologies:
            root = Node(self.start_node)
            root.add_daughter(l)
            root.add_daughter(r)
            topologies_with_root_node.append(root)
        return [
            Topology(node, ordering_function=self.ordering_function)
            for node in topologies_with_root_node
        ]

    @property
    def topologies(self) -> List[Topology]:
        """Returns all possible topologies for the given final state nodes or the topologies provided at initialization

        Returns:
            List[Topology]: the topologies of the collection
        """
        if self.__topologies is None:
            self.__topologies = self.__generate_topologies()
        return self.__topologies

    def filter(self, *contained_nodes: Node) -> List[Topology]:
        """
        Filter the topologies based on the number of contained intermediate nodes.

        Args:
            contained_nodes (tuple[Node]): nodes which should be contained in the topologies
        """
        topologies = self.topologies
        for contained_node in contained_nodes:
            topologies = self.filter_list(topologies, contained_node)
        return topologies
