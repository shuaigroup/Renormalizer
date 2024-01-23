from itertools import chain
from typing import List

from print_tree import print_tree

from renormalizer import Op
from renormalizer.model.basis import BasisSet, BasisDummy
from renormalizer.tn.node import NodeUnion, TreeNodeBasis


class Tree:
    def __init__(self, root: NodeUnion):
        assert root.parent is None
        self.root = root
        self.node_list = self.preorder_list()
        self.node_idx = {node:i for i, node in enumerate(self.node_list)}

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
            node_list[i].add_child(node_list[i+1])
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
    def general_mctdh(cls, basis_list: List[BasisSet], tree_order:int, contract_primitive=False):
        assert len(basis_list) > 1

        elementary_nodes = []
        if not contract_primitive:
            while tree_order < len(basis_list):
                node = TreeNodeBasis(basis_list[:tree_order])
                elementary_nodes.append(node)
                basis_list = basis_list[tree_order:]
            elementary_nodes.append(TreeNodeBasis(basis_list))
        else:
            dummy_i = 0
            for basis in basis_list:
                node1 = TreeNodeBasis([basis])
                elementary_nodes.append(node1)

        def recursion(elementary_nodes_: List[TreeNodeBasis]) -> TreeNodeBasis:
            nonlocal dummy_i
            node = TreeNodeBasis([BasisDummy(("MCTDH virtual", dummy_i))])
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
    def binary_mctdh(cls, basis_list: List[BasisSet], contract_primitive=False):
        return cls.general_mctdh(basis_list, 2, contract_primitive)

    @classmethod
    def ternary_mctdh(cls, basis_list: List[BasisSet], contract_primitive=False):
        return cls.general_mctdh(basis_list, 3, contract_primitive)

    def __init__(self, root: TreeNodeBasis):
        super().__init__(root)
        for node in self.node_list:
            assert isinstance(node, TreeNodeBasis)
        qn_size_list = [n.qn_size for n in self.node_list]
        if len(set(qn_size_list)) != 1:
            raise ValueError(f"Inconsistent quantum number size: {set(qn_size_list)}")
        self.qn_size: int = qn_size_list[0]
        # identity operator
        self.identity_op: Op = Op("I", self.root.dofs[0][0])
        # identity ttno
        self.identity_ttno = None

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
    def basis_list_postorder(self) -> List[BasisSet]:
        return list(chain(*[n.basis_sets for n in self.postorder_list()]))


def approximate_partition(sequence, ngroups):
    size = (len(sequence) - 1) // ngroups + 1
    ret = []
    for i in range(ngroups):
        start = i*size
        end = min((i+1)*size, len(sequence))
        ret.append(sequence[start:end])
    return ret