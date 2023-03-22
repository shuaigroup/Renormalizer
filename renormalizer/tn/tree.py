from itertools import chain
from typing import List, Dict

import numpy as np
import scipy
import opt_einsum as oe
from print_tree import print_tree

from renormalizer import Op, Mps, Model
from renormalizer.model.basis import BasisSet
from renormalizer.mps.svd_qn import add_outer, svd_qn, blockrecover, get_qn_mask
from renormalizer.mps.lib import select_basis
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


    def __init__(self, root: TreeNodeBasis):
        super().__init__(root)
        for node in self.node_list:
            assert isinstance(node, TreeNodeBasis)
        qn_size_list = [n.qn_size for n in self.node_list]
        if len(set(qn_size_list)) != 1:
            raise ValueError(f"Inconsistent quantum number size: {set(qn_size_list)}")
        self.qn_size: int = qn_size_list[0]

    def print(self):
        class print_tn_basis(print_tree):

            def get_children(self, node):
                return node.children

            def get_node_str(self, node):
                return str([b.dofs for b in node.basis_sets])

        print_tn_basis(self.root)

    @property
    def basis_list(self) -> List[BasisSet]:
        return list(chain(*[n.basis_sets for n in self.node_list]))

    @property
    def basis_list_postorder(self) -> List[BasisSet]:
        return list(chain(*[n.basis_sets for n in self.postorder_list()]))


class TensorTreeOperator(Tree):
    def __init__(self, basis:BasisTree, ham_terms: List[Op]):
        self.basis: BasisTree = basis
        self.dtype = np.float64
        # temporary solution to avoid cyclic import
        from renormalizer.tn.symbolic_mpo import construct_symbolic_mpo, symbolic_mo_to_numeric_mo_general
        symbolic_mpo, mpoqn = construct_symbolic_mpo(basis, ham_terms)
        #from renormalizer.mps.symbolic_mpo import _format_symbolic_mpo
        #print(_format_symbolic_mpo(symbolic_mpo))
        node_list_basis = self.basis.postorder_list()
        node_list_op = []
        for impo, (mo, qn) in enumerate(zip(symbolic_mpo, mpoqn)):
            node_basis: TreeNodeBasis = node_list_basis[impo]
            mo_mat = symbolic_mo_to_numeric_mo_general(node_basis.basis_sets, mo, self.dtype)
            node_list_op.append(TreeNodeTensor(mo_mat, qn))
        root = copy_connection(node_list_basis, node_list_op)
        super().__init__(root)
        # tensor node to basis node
        self.tn2bn = {tn: bn for tn, bn in zip(self.node_list, self.basis.node_list)}
        self.tn2dofs = {tn: bn.dofs for tn, bn in self.tn2bn.items()}

    def todense(self, order:List[BasisSet]=None) -> np.ndarray:
        _id = str(id(self))
        args = self.to_contract_args("up", "down")
        if order is None:
            order = self.basis.basis_list
        indices_up = []
        indices_down = []
        for basis in order:
            indices_up.append(("up", str(basis.dofs)))
            indices_down.append(("down", str(basis.dofs)))
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
        all_dofs = self.tn2dofs[node]
        indices = []
        for child in node.children:
            indices.append((_id, str(all_dofs), str(self.tn2dofs[child])))
        for dofs in all_dofs:
            indices.append((prefix_up, str(dofs)))
            indices.append((prefix_down, str(dofs)))
        if node.parent is None:
            indices.append((_id, "root", str(all_dofs)))
        else:
            indices.append((_id, str(self.tn2dofs[node.parent]), str(all_dofs)))
        assert len(indices) == node.tensor.ndim
        return indices

    @property
    def qntot(self):
        # duplicate with tts
        return self.root.qn[0]


class TensorTreeState(Tree):

    @classmethod
    def random(cls, basis:BasisTree, qntot, m_max, percent=1.0):
        tts = cls(basis)
        if isinstance(qntot, int):
            qntot = np.array([qntot])
        qn_size = len(qntot)
        assert basis.qn_size == qn_size

        for node in tts.postorder_list()[:-1]:
            qnbigl, _, _ = tts.get_qnmat(node, include_parent=False)
            qnbigl_shape = qnbigl.shape
            qnbigl = qnbigl.reshape(-1, qn_size)
            u_list = []
            s_list = []
            qn_list = []

            for iblock in set([tuple(t) for t in qnbigl]):
                if np.all(np.array(qntot) < np.array(iblock)):
                    continue
                # find the quantum number index
                indices = [i for i, x in enumerate(qnbigl) if tuple(x) == iblock]
                assert len(indices) != 0
                a = np.random.random([len(indices), len(indices)]) - 0.5
                a = a + a.T
                s, u = scipy.linalg.eigh(a=a)
                u_list.append(blockrecover(indices, u, len(qnbigl)))
                s_list.append(s)
                qn_list += [iblock] * len(indices)

            u = np.concatenate(u_list, axis=1)
            s = np.concatenate(s_list)
            mt, mpsdim, mpsqn, nouse = select_basis(
                u, s, qn_list, u, m_max, percent=percent
            )
            node.tensor = mt.reshape(list(qnbigl_shape)[:-1] + [mpsdim])
            node.qn = mpsqn
        # deal with root
        tts.root.qn = np.ones((1, qn_size), dtype=int) * qntot
        _, _, qnmat = tts.get_qnmat(tts.root, include_parent=False)
        qn_mask = get_qn_mask(qnmat, tts.qntot)
        tts.root.tensor = np.random.random(qn_mask.shape) - 0.5
        tts.root.tensor[~qn_mask] = 0
        tts.root.tensor /= np.linalg.norm(tts.root.tensor.ravel())
        tts.check_shape()
        return tts

    def __init__(self, basis:BasisTree, condition:Dict=None):
        self.basis = basis
        if condition is None:
            condition = {}
        basis_list = basis.basis_list_postorder
        mps = Mps.hartree_product_state(Model(basis_list, []), condition, len(basis_list))
        # can't directly use MPS qn because the topology is different
        site_qn = [mps.qn[i+1] - mps.qn[i] for i in range(len(mps))]
        node_list_state = []

        for node_basis in basis.node_list:
            mps_indices = [basis_list.index(b) for b in node_basis.basis_sets]
            assert mps_indices
            tensor = np.eye(1)
            # here only the site qn (rather than bond qn) is set
            qn = 0
            for i in mps_indices:
                tensor = np.tensordot(tensor, mps[i].array, axes=1)
                qn += site_qn[i]
            tensor = tensor.reshape([1] * len(node_basis.children) + list(tensor.shape)[1:-1] + [1])
            node_list_state.append(TreeNodeTensor(tensor, qn))

        root = copy_connection(basis.node_list, node_list_state)

        super().__init__(root)

        # summing up the site qn
        for node in self.postorder_list():
            for child in node.children:
                node.qn += child.qn
        self.check_shape()
        # tensor node to basis node
        self.tn2bn = {tn: bn for tn, bn in zip(self.node_list, self.basis.node_list)}
        self.tn2dofs = {tn: bn.dofs for tn, bn in self.tn2bn.items()}

    def check_shape(self):
        for snode, bnode in zip(self.node_list, self.basis.node_list):
            assert snode.tensor.ndim == len(snode.children) + bnode.n_sets + 1
            assert snode.qn.shape[1] == bnode.qn_size
            for i, b in enumerate(bnode.basis_sets):
                assert snode.tensor.shape[len(snode.children) + i] == b.nbas

    def get_qnmat(self, node, include_parent=True):
        qnbigl = np.zeros(self.basis.qn_size, dtype=int)
        for child in node.children:
            qnbigl = add_outer(qnbigl, child.qn)
        for b in self.tn2bn[node].basis_sets:
            qnbigl = add_outer(qnbigl, b.sigmaqn)
        if not include_parent:
            qnbigr = self.qntot - node.qn
            # single site
            qnmat = add_outer(qnbigl, qnbigr)
            return qnbigl, qnbigr, qnmat
        # two site
        qnbigr = np.zeros(self.basis.qn_size, dtype=int)
        assert node.parent is not None
        for child in node.parent.children:
            if child is node:
                continue
            qnbigr = add_outer(qnbigr, child.qn)
        for b in self.tn2bn[node.parent].basis_sets:
            qnbigr = add_outer(qnbigr, b.sigmaqn)
        qnbigr = add_outer(qnbigr, self.qntot - node.parent.qn)
        qnmat = add_outer(qnbigl, qnbigr)
        return qnbigl, qnbigr, qnmat

    def update_2site(self, node, tensor, m:int, percent:float=0, cano_parent:bool=True):
        """cano_parent: set canonical center at parent. to_right = True"""
        parent = node.parent
        assert parent is not None
        qnbigl, qnbigr, _ = self.get_qnmat(node)
        dim1 = np.prod(qnbigl.shape)
        tensor = tensor.reshape(dim1, -1)
        # u for snode and v for parent
        # duplicate with MatrixProduct._udpate_mps. Should consider merging when doing e.g. state averaged algorithm.
        u_list, su_list, qnlnew, v_list, sv_list, qnrnew = svd_qn(tensor, qnbigl, qnbigr, self.qntot)
        if cano_parent:
            m_node, msdim, msqn, m_parent = select_basis(
                u_list, su_list, qnlnew, v_list, m, percent=percent
            )
            m_parent = m_parent.T
        else:
            m_parent, msdim, msqn, m_node = select_basis(
                v_list, sv_list, qnrnew, u_list, m, percent=percent
            )
            m_node = m_node.T
        node.tensor = m_node.reshape(list(node.tensor.shape[:-1]) + [-1])
        if cano_parent:
            node.qn = msqn
        else:
            node.qn = self.qntot - msqn
        assert len(node.qn) == node.tensor.shape[-1]
        shape = list(parent.tensor.shape)
        ichild = parent.children.index(node)
        del shape[ichild]
        shape = [-1] + shape
        parent.tensor = np.moveaxis(m_parent.reshape(shape), 0, ichild)

    def update_2site2(self, node, tensor, m:int, cano_parent=True):
        """cano_parent: set canonical center at parent"""
        parent = node.parent
        assert parent is not None
        dim1 = np.prod(node.tensor.shape[:-1])
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
        node.tensor = u.reshape(list(node.tensor.shape[:-1]) + [-1])
        shape = list(parent.tensor.shape)
        ichild = parent.children.index(node)
        del shape[ichild]
        shape = [-1] + shape
        parent.tensor = np.moveaxis(vt.reshape(shape), 0, ichild)

    def expectation(self, tto: TensorTreeOperator):
        args = self.to_contract_args()
        args.extend(self.to_contract_args(conj=True))
        args.extend(tto.to_contract_args("up", "down"))
        return oe.contract(*args).ravel()[0]

    def to_contract_args(self, conj=False):
        args = []
        for node in self.node_list:
            assert isinstance(node, TreeNodeTensor)
            indices = self.get_node_indices(node, conj)
            args.extend([node.tensor, indices])
        return args

    def get_node_indices(self, node, conj=False):
        if not conj:
            ud = "down"
            _id = str(id(self))
        else:
            ud = "up"
            _id = str(id(self)) + "_conj"

        all_dofs = self.tn2dofs[node]
        indices = []
        for child in node.children:
            indices.append((_id, str(all_dofs), str(self.tn2dofs[child])))
        for dofs in all_dofs:
            indices.append((ud, str(dofs)))
        if node.parent is None:
            indices.append((_id, "root", str(all_dofs)))
        else:
            indices.append((_id, str(self.tn2dofs[node.parent]), str(all_dofs)))
        assert len(indices) == node.tensor.ndim
        return indices

    @property
    def qntot(self):
        # duplicate with tto
        return self.root.qn[0]


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
        self.tn2dofs = {tn: bn.dofs for tn, bn in self.tn2bn.items()}
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
        args.append(tts.get_node_indices(snode, conj=True))

        args.append(onode.tensor)
        args.append(tto.get_node_indices(onode, "up", "down"))

        args.append(snode.tensor)
        args.append(tts.get_node_indices(snode))

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
        args.append(tts.get_node_indices(snode, conj=True))

        args.append(onode.tensor)
        args.append(tto.get_node_indices(onode, "up", "down"))

        args.append(snode.tensor)
        args.append(tts.get_node_indices(snode))

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
