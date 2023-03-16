from typing import List, Dict

import numpy as np
import scipy
import opt_einsum as oe
from print_tree import print_tree

from renormalizer import Op, Mps, Model
from renormalizer.model.basis import BasisSet
from renormalizer.mps.symbolic_mpo import symbolic_mo_to_numeric_mo
from renormalizer.tn.node import TreeNodeTensor, TreeNodeBasis, NodeUnion, copy_connection, TreeNodeEnviron


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


class TreeWorkspace:

    def __init__(self, tree: Tree):
        self.tree: Tree = tree

    def __enter__(self):
        for node in self.tree.node_list:
            assert node.workspace is None

    def __exit__(self, exc_type, exc_val, exc_tb):
        for node in self.tree.node_list:
            node.workspace = None


class BasisTree(Tree):
    """Tree of basis sets."""
    @classmethod
    def linear(cls, basis_list: List[BasisSet]):
        node_list = [TreeNodeBasis(basis) for basis in basis_list]
        for i in range(len(node_list) - 1):
            node_list[i].add_child(node_list[i+1])
        return cls(node_list[0])

    @classmethod
    def binary(cls, basis_list: List[BasisSet]):
        node_list = [TreeNodeBasis(basis) for basis in basis_list]
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


    def __init__(self, root: TreeNodeBasis):
        super().__init__(root)
        for node in self.node_list:
            assert isinstance(node, TreeNodeBasis)
        qn_size_list = [n.basis_set.sigmaqn.shape[1] for n in self.node_list]
        if len(set(qn_size_list)) != 1:
            raise ValueError(f"Inconsistent quantum number size: {set(qn_size_list)}")
        self.qn_size: int = qn_size_list[0]

    def split_tensor_tree(self):
        def recursion(node: TreeNodeBasis):
            new_node = TreeNodeTensor(node.workspace)
            for child in node.children:
                new_node.add_child(recursion(child))
            return new_node
        root = recursion(self.root)
        return root

    def print(self):
        class print_tn_basis(print_tree):

            def get_children(self, node):
                return node.children

            def get_node_str(self, node):
                return str(node.basis_set.dofs)

        print_tn_basis(self.root)

    @property
    def basis_list(self) -> List[BasisSet]:
        return [n.basis_set for n in self.node_list]


class TensorTreeOperator(Tree):
    def __init__(self, basis:BasisTree, ham_terms: List[Op]):
        self.basis: BasisTree = basis
        self.dtype = np.float64
        # temporary solution to avoid cyclic import
        from renormalizer.tn.symbolic_mpo import construct_symbolic_mpo
        symbolic_mpo = construct_symbolic_mpo(basis, ham_terms)
        #from renormalizer.mps.symbolic_mpo import _format_symbolic_mpo
        #print(_format_symbolic_mpo(symbolic_mpo))
        node_list_basis = self.basis.postorder_list()
        node_list_op = []
        for impo, mo in enumerate(symbolic_mpo):
            node_basis: TreeNodeBasis = node_list_basis[impo]
            mo_mat = symbolic_mo_to_numeric_mo(node_basis.basis_set, mo, self.dtype)
            node_list_op.append(TreeNodeTensor(mo_mat))
        root = copy_connection(node_list_basis, node_list_op)
        super().__init__(root)
        # tensor node to basis node
        self.tn2bn = {tn: bn for tn, bn in zip(self.node_list, self.basis.node_list)}
        self.tn2dofs = {tn: bn.basis_set.dofs for tn, bn in self.tn2bn.items()}

    def todense(self, order:List[BasisSet]=None) -> np.ndarray:
        _id = str(id(self))
        args = self.to_contract_args(_id, _id)
        if order is None:
            order = [n.basis_set for n in self.basis.node_list]
        indices_up = []
        indices_down = []
        for basis in order:
            indices_up.append((_id, str(basis.dofs), "up"))
            indices_down.append((_id, str(basis.dofs), "down"))
        output_indices = [(_id, "root", str(self.tn2dofs[self.root]))] + indices_up + indices_down
        args.append(output_indices)
        res = oe.contract(*args)
        assert res.shape[0] == 1
        res = res[0]
        dim = round(np.sqrt(np.prod(res.shape)))
        return res.reshape(dim, dim)

    def to_contract_args(self, prefix_up, prefix_down):
        args = []
        for node in self.node_list:
            assert isinstance(node, TreeNodeTensor)
            indices = self.get_node_indices(node, prefix_up, prefix_down)
            args.extend([node.tensor, indices])
        return args

    def get_node_indices(self, node, prefix_up, prefix_down):
        _id = str(id(self))
        dofs = self.tn2dofs[node]
        indices = []
        for child in node.children:
            indices.append((_id, str(dofs), str(self.tn2dofs[child])))
        indices.append((prefix_up, str(dofs), "up"))
        indices.append((prefix_down, str(dofs), "down"))
        if node.parent is None:
            indices.append((_id, "root", str(dofs)))
        else:
            indices.append((_id, str(self.tn2dofs[node.parent]), str(dofs)))
        assert len(indices) == node.tensor.ndim
        return indices


class TensorTreeState(Tree):
    def __init__(self, basis:BasisTree, condition:Dict=None):
        self.basis = basis
        if condition is None:
            condition = {}
        mps = Mps.hartree_product_state(Model(basis.basis_list, []), condition)
        node_list_basis = basis.node_list
        node_list_state = []

        for i, ms in enumerate(mps):
            node_basis: TreeNodeBasis = node_list_basis[i]
            ms_mat = ms.array.reshape([1] * len(node_basis.children) + [-1, 1])
            node_list_state.append(TreeNodeTensor(ms_mat))

        root = copy_connection(node_list_basis, node_list_state)
        super().__init__(root)
        self.check_shape()
        # tensor node to basis node
        self.tn2bn = {tn: bn for tn, bn in zip(self.node_list, self.basis.node_list)}
        self.tn2dofs = {tn: bn.basis_set.dofs for tn, bn in self.tn2bn.items()}

    def check_shape(self):
        for i, node in enumerate(self.node_list):
            assert node.tensor.ndim == len(node.children) + 2
            assert node.tensor.shape[-2] == self.basis.basis_list[i].nbas

    def update_2site(self, snode, tensor, m:int, cano_parent=True):
        """cano_parent: set canonical center at parent"""
        parent = snode.parent
        assert parent is not None
        dim1 = np.prod(snode.tensor.shape[:-1])
        tensor = tensor.reshape(dim1, -1)
        # u for snode and vt for parent
        u, s, vt = scipy.linalg.svd(tensor, full_matrices=False)
        if m < len(s):
            u = u[:, :m]
            s = s[:m]
            vt = vt[:m, :]
        if cano_parent:
            vt = s.reshape(-1, 1) * vt
        else:
            u = u * s.reshape(1, -1)
        snode.tensor = u.reshape(list(snode.tensor.shape[:-1]) + [-1])
        shape = list(parent.tensor.shape)
        ichild = parent.children.index(snode)
        del shape[ichild]
        shape = [-1] + shape
        parent.tensor = np.moveaxis(vt.reshape(shape), 0, ichild)

    def expectation(self, tto: TensorTreeOperator):
        args = self.to_contract_args("ket")
        args.extend(self.to_contract_args("bra", conj=True))
        args.extend(tto.to_contract_args("bra", "ket"))
        return oe.contract(*args).ravel()[0]

    def to_contract_args(self, prefix, conj=False):
        args = []
        for node in self.node_list:
            assert isinstance(node, TreeNodeTensor)
            indices = self.get_node_indices(node, prefix, conj)
            args.extend([node.tensor, indices])
        return args

    def get_node_indices(self, node, prefix, conj=False):
        if not conj:
            ud = "down"
            _id = str(id(self))
        else:
            ud = "up"
            _id = str(id(self)) + "_conj"

        dofs = self.tn2dofs[node]
        indices = []
        for child in node.children:
            indices.append((_id, str(dofs), str(self.tn2dofs[child])))
        indices.append((prefix, str(dofs), ud))
        if node.parent is None:
            indices.append((_id, "root", str(dofs)))
        else:
            indices.append((_id, str(self.tn2dofs[node.parent]), str(dofs)))
        assert len(indices) == node.tensor.ndim
        return indices


class TensorTreeEnviron(Tree):
    def __init__(self, tts:TensorTreeState, tto:TensorTreeOperator):
        self.basis =  tts.basis
        enodes: List[TreeNodeEnviron] = [TreeNodeEnviron() for _ in range(tts.size)]
        copy_connection(tts.node_list, enodes)
        super().__init__(enodes[0])
        assert self.root.parent is None
        self.root.environ_parent = np.array([1]).reshape([1, 1, 1])
        # tensor node to basis node. todo: remove duplication?
        self.tn2bn = {tn: bn for tn, bn in zip(self.node_list, self.basis.node_list)}
        self.tn2dofs = {tn: bn.basis_set.dofs for tn, bn in self.tn2bn.items()}
        self.build_children_environ(tts, tto)
        self.build_parent_environ(tts, tto)


    def build_children_environ(self, tts, tto):
        # first run, children environment to the parent.
        # set enode.environ_children
        snodes: List[TreeNodeTensor] = tts.postorder_list()
        for snode in snodes:
            self.build_children_environ_node(snode, tts, tto)

    def build_parent_environ(self, tts, tto):
        # second run, parent environment to children
        # set enode.environ_parent
        snodes: List[TreeNodeTensor] = tts.node_list
        for snode in snodes:
            for ichild in range(len(snode.children)):
                self.build_parent_environ_node(snode, ichild, tts, tto)

    def update_2site(self, snode, tts, tto):
        # update environ based on snode and its parent
        self.build_children_environ_node(snode, tts, tto)
        self.build_children_environ_node(snode.parent, tts, tto)
        for ichild in range(len(snode.parent.children)):
            self.build_parent_environ_node(snode.parent, ichild, tts, tto)
        for ichild in range(len(snode.children)):
            self.build_parent_environ_node(snode, ichild, tts, tto)

    def build_children_environ_node(self, snode:TreeNodeTensor, tts: TensorTreeState, tto: TensorTreeOperator):
        if snode.parent is None:
            return
        enode = self.node_list[tts.node_idx[snode]]
        onode = tto.node_list[tts.node_idx[snode]]
        args = []
        for i, child_tensor in enumerate(enode.environ_children):
            indices = self.get_child_indices(enode, i, tts, tto)
            args.extend([child_tensor, indices])

        args.append(snode.tensor.conj())
        args.append(tts.get_node_indices(snode, "bra", conj=True))

        args.append(onode.tensor)
        args.append(tto.get_node_indices(onode, "bra", "ket"))

        args.append(snode.tensor)
        args.append(tts.get_node_indices(snode, "ket"))

        # indices for the resulting tensor
        indices = self.get_parent_indices(enode, tts, tto)
        args.append(indices)
        res = oe.contract(*args)
        if len(enode.parent.environ_children) != len(enode.parent.children):
            # first run
            enode.parent.environ_children.append(res)
        else:
            # updating
            ichild = snode.parent.children.index(snode)
            enode.parent.environ_children[ichild] = res

    def build_parent_environ_node(self, snode:TreeNodeTensor, ichild: int, tts: TensorTreeState, tto: TensorTreeOperator):
        # build the environment for the ith child of snode
        enode = self.node_list[tts.node_idx[snode]]
        onode = tto.node_list[tts.node_idx[snode]]
        args = []
        # children tensor
        for j, child_tensor in enumerate(enode.environ_children):
            if j == ichild:
                continue
            indices = self.get_child_indices(enode, j, tts, tto)
            args.extend([child_tensor, indices])

        # parent tensor
        indices = self.get_parent_indices(enode, tts, tto)
        args.extend([enode.environ_parent, indices])

        args.append(snode.tensor.conj())
        args.append(tts.get_node_indices(snode, "bra", conj=True))

        args.append(onode.tensor)
        args.append(tto.get_node_indices(onode, "bra", "ket"))

        args.append(snode.tensor)
        args.append(tts.get_node_indices(snode, "ket"))

        # indices for the resulting tensor
        indices = self.get_child_indices(enode, ichild, tts, tto)

        args.append(indices)
        res = oe.contract(*args)
        enode.children[ichild].environ_parent = res

    def get_child_indices(self, enode, i, tts, tto):
        dofs = self.tn2dofs[enode]
        dofs_child = self.tn2dofs[enode.children[i]]
        indices = [
            (str(id(tts)) + "_conj", str(dofs), str(dofs_child)),
            (str(id(tto)), str(dofs), str(dofs_child)),
            (str(id(tts)), str(dofs), str(dofs_child)),
        ]
        return indices

    def get_parent_indices(self, enode, tts, tto):
        dofs = self.tn2dofs[enode]
        if enode.parent is not None:
            dofs_parent = self.tn2dofs[enode.parent]
        else:
            dofs_parent = "root"
        indices = [
            (str(id(tts)) + "_conj", str(dofs_parent), str(dofs)),
            (str(id(tto)), str(dofs_parent), str(dofs)),
            (str(id(tts)), str(dofs_parent), str(dofs)),
        ]
        return indices