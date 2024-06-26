from itertools import chain
from typing import List, Sequence, Dict, Any

import numpy as np
from print_tree import print_tree

from renormalizer import Op
from renormalizer.model.basis import BasisSet, BasisDummy
from renormalizer.tn.node import NodeUnion, TreeNodeBasis, copy_connection


class Tree:
    def __init__(self, root: NodeUnion):
        assert root.parent is None
        self.root = root
        self.node_list = self.preorder_list()
        self.node_idx = {node: i for i, node in enumerate(self.node_list)}

    def preorder_list(self, func=None) -> List[NodeUnion]:
        def recursion(node: NodeUnion):
            if func is None:
                ret = [node]
            else:
                ret = [func(node)]
            if not node.children:
                return ret
            for child in node.children:
                ret += recursion(child)
            return ret

        return recursion(self.root)

    def postorder_list(self) -> List[NodeUnion]:
        def recursion(node: NodeUnion):
            if not node.children:
                return [node]
            ret = []
            for child in node.children:
                ret += recursion(child)
            ret.append(node)
            return ret

        return recursion(self.root)

    @staticmethod
    def find_path(node1: NodeUnion, node2: NodeUnion) -> List[NodeUnion]:
        """Find the path from node1 to node2. Not most efficient but simple to implement"""
        assert node1 != node2
        ancestors1 = node1.ancestors
        ancestors2 = node2.ancestors
        ancestors2_set = set(ancestors2)
        common_ancestors = [ancestor for ancestor in ancestors1 if ancestor in ancestors2_set]
        common_ancestor = common_ancestors[0]
        path1 = ancestors1[:ancestors1.index(common_ancestor) + 1]
        path2 = ancestors2[:ancestors2.index(common_ancestor)]
        return path1 + path2[::-1]

    @property
    def size(self):
        return len(self.node_list)

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.node_list)


class BasisTree(Tree):
    """Tree of basis sets."""

    @classmethod
    def linear(cls, basis_list: List[BasisSet]):
        node_list = [TreeNodeBasis([basis]) for basis in basis_list]
        for i in range(len(node_list) - 1):
            node_list[i].add_child(node_list[i + 1])
        return cls(node_list[0])

    @classmethod
    def binary(cls, basis_list: List[BasisSet]):
        node_list = [TreeNodeBasis([basis]) for basis in basis_list]

        def binary_recursion(node: TreeNodeBasis, offspring: List[TreeNodeBasis]):
            if len(offspring) == 0:
                return
            node.add_child(offspring[0])
            if len(offspring) == 1:
                return
            node.add_child(offspring[1])
            new_offspring = offspring[2:]
            mid_idx = len(new_offspring) // 2
            binary_recursion(offspring[0], new_offspring[:mid_idx])
            binary_recursion(offspring[1], new_offspring[mid_idx:])

        binary_recursion(node_list[0], node_list[1:])
        return cls(node_list[0])

    @classmethod
    def general_mctdh(
        cls,
        basis_list: List[BasisSet],
        tree_order: int,
        contract_primitive: bool = False,
        contract_label: Sequence[bool] = None,
        dummy_label="MCTDH virtual",
    ):
        # `contract_primitive` means a 2-index tensor for each primitive basis
        #       o
        #      / \
        #      o  o
        # d -> |  | <- d
        # not contract primitive
        #        o
        # d ->  / \ <- d
        # d means physical bond
        # `contract_label` decides whether we do contraction for a particular basis
        assert len(basis_list) > 1

        # prepare elementary nodes
        elementary_nodes: List[TreeNodeBasis] = []
        if not contract_primitive:
            assert contract_label is None, "providing label makes sense only when primitives are contracted"
            while tree_order < len(basis_list):
                node = TreeNodeBasis(basis_list[:tree_order])
                elementary_nodes.append(node)
                basis_list = basis_list[tree_order:]
            elementary_nodes.append(TreeNodeBasis(basis_list))
        else:
            if contract_label is None:
                for basis in basis_list:
                    node1 = TreeNodeBasis([basis])
                    elementary_nodes.append(node1)
            else:
                assert len(contract_label) == len(basis_list)
                i = 0
                while i != len(basis_list):
                    if contract_label[i]:
                        elementary_nodes.append(TreeNodeBasis([basis_list[i]]))
                        i += 1
                    else:
                        for j in range(1, tree_order + 1):
                            if i + j == len(contract_label) or contract_label[i + j]:
                                break
                        elementary_nodes.append(TreeNodeBasis(basis_list[i : i + j]))
                        i += j

        # recursive tree construction
        def recursion(elementary_nodes_: List[TreeNodeBasis]) -> TreeNodeBasis:
            nonlocal dummy_i
            node = TreeNodeBasis([BasisDummy((dummy_label, dummy_i))])
            dummy_i += 1
            if len(elementary_nodes_) <= tree_order:
                node.add_child(elementary_nodes_)
                return node
            for group in approximate_partition(elementary_nodes_, tree_order):
                node.add_child(recursion(group))
            return node

        dummy_i = 0
        root = recursion(elementary_nodes)
        return cls(root)

    @classmethod
    def binary_mctdh(
        cls, basis_list: List[BasisSet], contract_primitive=False, contract_label=None, dummy_label="MCTDH virtual"
    ):
        return cls.general_mctdh(basis_list, 2, contract_primitive, contract_label, dummy_label)

    @classmethod
    def ternary_mctdh(
        cls, basis_list: List[BasisSet], contract_primitive=False, contract_label=None, dummy_label="MCTDH virtual"
    ):
        return cls.general_mctdh(basis_list, 3, contract_primitive, contract_label, dummy_label)

    @classmethod
    def t3ns(cls, basis_list: List[BasisSet], t3ns_label="T3NS virtual"):
        def recursion(parent, basis_list_: List[BasisSet]):
            nonlocal dummy_i
            if len(basis_list_) == 0:
                return
            if len(basis_list_) == 1:
                parent.add_child(TreeNodeBasis(basis_list_))
                return
            if len(basis_list_) == 2:
                node1 = TreeNodeBasis(basis_list_[:1])
                parent.add_child(node1)
                node2 = TreeNodeBasis(basis_list_[1:])
                node1.add_child(node2)
                return
            node1 = TreeNodeBasis(basis_list_[:1])
            parent.add_child(node1)
            node2 = TreeNodeBasis([BasisDummy((t3ns_label, dummy_i))])
            dummy_i += 1
            node1.add_child(node2)
            for partition_ in approximate_partition(basis_list_[1:], 2):
                recursion(node2, partition_)

        dummy_i = 0
        root = TreeNodeBasis([BasisDummy((t3ns_label, dummy_i))])
        dummy_i += 1
        for partition in approximate_partition(basis_list, 3):
            recursion(root, partition)
        return cls(root)

    def __init__(self, root: TreeNodeBasis):
        super().__init__(root)
        for node in self.node_list:
            assert isinstance(node, TreeNodeBasis)
        qn_size_list = [n.qn_size for n in self.node_list]
        if len(set(qn_size_list)) != 1:
            raise ValueError(f"Inconsistent quantum number size: {set(qn_size_list)}")
        self.qn_size: int = qn_size_list[0]

        # map basis to node index
        self.basis2idx: Dict[BasisSet, int] = {}
        # map dof to node index
        self.dof2idx: Dict[Any, int] = {}
        # map dof to basis
        self.dof2basis: Dict[Any, BasisSet] = {}
        for i, node in enumerate(self.node_list):
            for b in node.basis_sets:
                self.basis2idx[b] = i
                for d in b.dofs:
                    self.dof2idx[d] = i
                    self.dof2basis[d] = b

        # identity operator
        self.identity_op: Op = Op("I", self.root.dofs[0][0])
        # identity ttno
        self.identity_ttno = None
        # dummy ttno. Same tree topology but only has dummy basis
        # used as a dummy operator for calculating norm, etc
        self.dummy_ttno = None

    def print(self, print_function=None):
        class print_tn_basis(print_tree):
            def get_children(self, node):
                return node.children

            def get_node_str(self, node):
                return str([b.dofs for b in node.basis_sets])

        tree = print_tn_basis(self.root)
        if print_function is not None:
            for row in tree.rows:
                print_function(row)

    @property
    def basis_list(self) -> List[BasisSet]:
        return list(chain(*[n.basis_sets for n in self.node_list]))

    @property
    def dof_list(self) -> List[Any]:
        return list(chain(*[b.dofs for b in self.basis_list]))

    @property
    def basis_list_postorder(self) -> List[BasisSet]:
        return list(chain(*[n.basis_sets for n in self.postorder_list()]))

    @property
    def pbond_dims(self) -> List[List[int]]:
        return [b.pbond_dims for b in self.node_list]

    def add_auxiliary_space(self, auxiliary_label="Q") -> "BasisTree":
        # make a new basis tree with auxiliary basis
        node2_list = []
        for node in self:
            basis_set2_list = []
            for basis in node.basis_sets:
                # the P space
                basis_set2_list.append(basis)
                if not isinstance(basis, BasisDummy):
                    # the Q space
                    basis_q: BasisSet = basis.copy((auxiliary_label, basis.dofs))
                    # set to zero for know. could change to more complicated case in the future
                    basis_q.sigmaqn = np.zeros_like(basis.sigmaqn)
                    basis_set2_list.append(basis_q)
            node2_list.append(TreeNodeBasis(basis_set2_list))
        copy_connection(self.node_list, node2_list)
        basis_tree2 = BasisTree(node2_list[0])
        return basis_tree2


def approximate_partition(sequence, ngroups):
    size = (len(sequence) - 1) // ngroups + 1
    ret = []
    for i in range(ngroups):
        start = i * size
        end = min((i + 1) * size, len(sequence))
        ret.append(sequence[start:end])
    return ret
