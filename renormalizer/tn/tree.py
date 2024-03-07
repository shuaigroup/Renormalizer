from typing import List, Dict, Tuple, Union

import scipy
import opt_einsum as oe
from print_tree import print_tree

from renormalizer import Op, Mps, Model
from renormalizer.model.basis import BasisSet, BasisDummy
from renormalizer.mps.backend import np, backend
from renormalizer.mps.matrix import asnumpy, asxp_oe_args, tensordot
from renormalizer.mps.svd_qn import add_outer, svd_qn, blockrecover, get_qn_mask
from renormalizer.mps.lib import select_basis
from renormalizer.utils.configs import CompressConfig, OptimizeConfig, EvolveConfig, EvolveMethod
from renormalizer.tn.node import TreeNodeTensor, TreeNodeBasis, copy_connection, TreeNodeEnviron
from renormalizer.tn.treebase import Tree, BasisTree
from renormalizer.tn.symbolic_mpo import construct_symbolic_mpo, symbolic_mo_to_numeric_mo_general


class TTNBase(Tree):
    # A tree whose tree node is TreeNodeTensor
    def print_shape(self, full=False, print_function=None):
        class print_tn_basis(print_tree):

            def get_children(self, node):
                return node.children

            def get_node_str(self, node: TreeNodeTensor):
                if full:
                    return str(node.tensor.shape)
                else:
                    return str(node.tensor.shape[-1])

        tree = print_tn_basis(self.root)
        if print_function is not None:
            for row in tree.rows:
                print_function(row)


class TTNO(TTNBase):
    @classmethod
    def identity(cls, basis:BasisTree):
        if not basis.identity_ttno:
            basis.identity_ttno = cls(basis, [basis.identity_op])
        return basis.identity_ttno

    def __init__(self, basis: BasisTree, terms: Union[List[Op], Op], root: TreeNodeTensor=None):
        self.basis: BasisTree = basis
        if isinstance(terms, Op):
            terms = [terms]
        self.terms: List[Op] = terms

        if not root:
            symbolic_mpo, mpoqn = construct_symbolic_mpo(basis, terms)
            #from renormalizer.mps.symbolic_mpo import _format_symbolic_mpo
            #print(_format_symbolic_mpo(symbolic_mpo))
            node_list_basis = self.basis.postorder_list()
            node_list_op = []
            for impo, (mo, qn) in enumerate(zip(symbolic_mpo, mpoqn)):
                node_basis: TreeNodeBasis = node_list_basis[impo]
                mo_mat = symbolic_mo_to_numeric_mo_general(node_basis.basis_sets, mo, backend.real_dtype)
                node_list_op.append(TreeNodeTensor(mo_mat, qn))
            root: TreeNodeTensor = copy_connection(node_list_basis, node_list_op)
        super().__init__(root)
        # tensor node to basis node
        self.tn2bn = {tn: bn for tn, bn in zip(self.node_list, self.basis.node_list)}
        self.tn2dofs = {tn: bn.dofs for tn, bn in self.tn2bn.items()}

    def apply(self, ttns: "TTNS", canonicalise: bool=False) -> "TTNS":
        # todo: apply to mpdm. Allow partial apply and ignore some indices
        new = ttns.metacopy()

        for snode1, snode2, onode in zip(new, ttns, self):
            assert len(snode2.children) == len(onode.children)

            bnode = self.tn2bn[onode]
            indices1 = ttns.get_node_indices(snode2)
            indices2 = self.get_node_indices(onode, "up", "down")
            output_indices = []
            output_shape = []
            # children indices
            for i in range(len(snode2.children)):
                output_shape.append(snode2.shape[i] * onode.shape[i])
                output_indices.extend([indices1[i], indices2[i]])
            # physical indices
            for i in range(bnode.n_sets):
                j = len(snode2.children) + 2 * i
                output_shape.append(onode.shape[j])
                output_indices.append(indices2[j])
            # parent indices
            output_shape.append(snode2.shape[-1] * onode.shape[-1])
            output_indices.extend([indices1[-1], indices2[-1]])
            # do contraction
            args = [snode2.tensor, indices1, onode.tensor, indices2, output_indices]
            res = oe.contract(*(asxp_oe_args(args))).reshape(output_shape)
            snode1.tensor = res
            snode1.qn = add_outer(snode2.qn, onode.qn).reshape(output_shape[-1], ttns.basis.qn_size)

        new.check_shape()
        if canonicalise:
            new.canonicalise()
        return new

    def contract(self, ttns: "TTNS", algo="svd") -> "TTNS":
        assert algo == "svd", "variational compress not supported yet"
        new_ttns = self.apply(ttns)
        new_ttns.canonicalise()
        new_ttns.compress()
        return new_ttns

    def todense(self, order:List[BasisSet]=None) -> np.ndarray:
        _id = str(id(self))
        args = self.to_contract_args("up", "down")
        if order is None:
            order = self.basis.basis_list
        indices_up = []
        indices_down = []
        for basis in order:
            if isinstance(basis, BasisDummy):
                continue
            indices_up.append(("up", str(basis.dofs)))
            indices_down.append(("down", str(basis.dofs)))
        output_indices = indices_up + indices_down
        args.append(output_indices)
        res = oe.contract(*asxp_oe_args(args))
        # to be consistent with the behavior of MPS/MPO
        res = asnumpy(res)
        dim = round(np.sqrt(np.prod(res.shape)))
        return res.reshape(dim, dim)

    def to_contract_args(self, prefix_up, prefix_down):
        args = []
        for node in self.node_list:
            assert isinstance(node, TreeNodeTensor)
            indices = self.get_node_indices(node, prefix_up, prefix_down)
            indices = [indices[i] for i, s in enumerate(node.tensor.shape) if s != 1]
            tensor = node.tensor.squeeze()
            assert len(indices) == tensor.ndim
            args.extend([tensor, indices])
        return args

    def get_node_indices(self, node, prefix_up, prefix_down):
        _id = str(id(self))
        all_dofs = self.tn2dofs[node]
        indices = []
        for child in node.children:
            indices.append((_id, str(all_dofs), str(self.tn2dofs[child])))
        for dofs in all_dofs:
            # interleaved up and down
            indices.append((prefix_up, str(dofs)))
            indices.append((prefix_down, str(dofs)))
        if node.parent is None:
            indices.append((_id, "root", str(all_dofs)))
        else:
            indices.append((_id, str(self.tn2dofs[node.parent]), str(all_dofs)))
        assert len(indices) == node.tensor.ndim
        return indices

    @property
    def bond_dims(self):
        # duplicate with ttns
        return [node.tensor.shape[-1] for node in self]

    @property
    def bond_dims_mean(self) -> int:
        # duplicate with ttns and Mps
        return int(round(np.mean(self.bond_dims)))

    @property
    def qntot(self):
        # duplicate with ttns
        return self.root.qn[0]

    def __matmul__(self, other):
        # duplicate with Mpo
        return self.apply(other)

# values are set in time_evolution.py
EVOLVE_METHODS = {}


class TTNS(TTNBase):

    @classmethod
    def random(cls, basis: BasisTree, qntot, m_max, percent=1.0):
        ttns = cls(basis)
        if isinstance(qntot, int):
            qntot = np.array([qntot])
        qn_size = len(qntot)
        assert basis.qn_size == qn_size

        for node in ttns.postorder_list()[:-1]:
            qnbigl, _, _ = ttns.get_qnmat(node, include_parent=False)
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
        ttns.root.qn = np.ones((1, qn_size), dtype=int) * qntot
        qn_mask = ttns.get_qnmask(ttns.root, include_parent=False)
        ttns.root.tensor = np.random.random(qn_mask.shape) - 0.5
        ttns.root.tensor[~qn_mask] = 0
        ttns.root.tensor /= np.linalg.norm(ttns.root.tensor.ravel())
        ttns.check_shape()
        ttns.check_canonical()
        return ttns

    @classmethod
    def from_tensors(cls, template: "TTNS", tensors):
        """QN is taken into account"""
        ttns = template.metacopy()
        cursor = 0
        for node, tnode in zip(ttns.node_list, template.node_list):
            qnmask = template.get_qnmask(tnode)
            length = np.sum(qnmask)
            node.tensor = np.zeros(tnode.shape, dtype=tensors.dtype)
            node.tensor[qnmask] = asnumpy(tensors[cursor:cursor+length])
            node.qn = tnode.qn
            cursor += length
        assert len(tensors) == cursor
        ttns.check_shape()
        return ttns

    def __init__(self, basis: BasisTree, condition:Dict=None, root:TreeNodeTensor=None):
        self.basis = basis
        if not root:
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

            root: TreeNodeTensor = copy_connection(basis.node_list, node_list_state)

            super().__init__(root)

            # summing up the site qn
            for node in self.postorder_list():
                for child in node.children:
                    node.qn += child.qn
        else:
            super().__init__(root)

        self.coeff = 1
        self.check_shape()
        # tensor node to basis node. make a property?
        self.tn2bn = {tn: bn for tn, bn in zip(self.node_list, self.basis.node_list)}
        self.tn2dofs = {tn: bn.dofs for tn, bn in self.tn2bn.items()}

        self.compress_config = CompressConfig()
        self.optimize_config = OptimizeConfig()
        self.evolve_config = EvolveConfig(EvolveMethod.tdvp_vmf, force_ovlp=False)

    def check_shape(self):
        for snode, bnode in zip(self.node_list, self.basis.node_list):
            assert snode.tensor.ndim == len(snode.children) + bnode.n_sets + 1
            assert snode.qn.shape[0] == snode.tensor.shape[-1]
            assert snode.qn.shape[1] == bnode.qn_size
            for i, b in enumerate(bnode.basis_sets):
                assert snode.shape[len(snode.children) + i] == b.nbas

    def check_canonical(self, atol=None):
        for node in self.node_list[1:]:
            node.check_canonical(atol)
        return True

    def is_canonical(self, atol=None):
        for node in self.node_list[1:]:
            if not node.check_canonical(atol, assertion=False):
                return False
        return True

    def canonicalise(self):
        for node in self.postorder_list()[:-1]:
            self.push_cano_to_parent(node)
        return self

    def compress(self, temp_m_trunc=None, ret_s=False):
        if self.compress_config.bonddim_should_set:
            self.compress_config.set_bonddim(len(self.node_list)+1)
        s_dict: Dict[TreeNodeTensor, np.ndarray] = {self.root: np.array([1])}
        compress_recursion(self.root, self, s_dict, temp_m_trunc)
        self.check_shape()
        self.check_canonical()
        if not ret_s:
            return self
        else:
            s_list = [s_dict[n] for n in self.node_list]
            return self, s_list

    def expectation1(self, ttno: TTNO, bra: "TTNS"=None):
        if bra is None:
            bra = self
        args = self.to_contract_args()
        args.extend(bra.to_contract_args(conj=True))
        args.extend(ttno.to_contract_args("up", "down"))
        val = oe.contract(*asxp_oe_args(args), optimize="greedy").ravel()[0]

        if np.isclose(float(val.imag), 0):
            return float(val.real)
        else:
            return complex(val)

    def expectation(self, ttno: TTNO, bra: "TTNS"=None):
        assert bra is None # not implemented yet
        basis_node = TreeNodeBasis([BasisDummy("expectation dummy")])
        basis_node.add_child(self.basis.root)
        basis_tree = BasisTree(basis_node)
        snode = TreeNodeTensor(np.ones((1, 1, 1)), qn=np.zeros((1, basis_tree.qn_size)))
        snode.add_child(self.root)
        onode = TreeNodeTensor(np.ones((1, 1, 1, 1)), qn=np.zeros((1, basis_tree.qn_size)))
        onode.add_child(ttno.root)

        ttns_extended = TTNS(basis_tree, root=snode)
        ttno_extended = TTNO(basis_tree, [], root=onode)
        environ = TTNEnviron(ttns_extended, ttno_extended, build_environ=False)
        environ.build_children_environ(ttns_extended, ttno_extended)
        val = environ.root.environ_children[0].ravel()[0]

        for node in [self.basis.root, self.root, ttno.root]:
            node.parent = None

        if np.isclose(float(val.imag), 0):
            return float(val.real)
        else:
            return complex(val)

    def add(self, other: "TTNS") -> "TTNS":
        new = self.metacopy()
        for new_node, node1, node2 in zip(new, self, other):
            new_shape = []
            indices1 = []
            indices2 = []
            for i, (dim1, dim2) in enumerate(zip(node1.shape, node2.shape)):
                is_physical_idx = len(node1.children) <= i and i != node1.tensor.ndim - 1
                is_parent_idx = i == node1.tensor.ndim - 1
                if is_physical_idx or (is_parent_idx and node1 is self.root):
                    assert dim1 == dim2
                    new_shape.append(dim1)
                    indices1.append(slice(0, dim1))
                    indices2.append(slice(0, dim1))
                else:
                    # virtual indices
                    new_shape.append(dim1 + dim2)
                    indices1.append(slice(0, dim1))
                    indices2.append(slice(dim1, dim1 + dim2))
            dtype = np.promote_types(node1.tensor.dtype, node2.tensor.dtype)
            new_node.tensor = np.zeros(new_shape, dtype=dtype)
            indices1 = tuple(indices1)
            indices2 = tuple(indices2)
            new_node.tensor[indices1] = node1.tensor
            new_node.tensor[indices2] = node2.tensor
            if node1 is self.root:
                np.testing.assert_allclose(node1.qn, node2.qn)
                new_node.qn = node1.qn.copy()
            else:
                new_node.qn = np.concatenate([node1.qn, node2.qn], axis=0)
        new.check_shape()
        #assert new.check_canonical()
        return new

    def normalize(self, kind):
        r''' normalize the wavefunction

        Parameters
        ----------
        kind: str
            "ttns_only": the ttns part is normalized and coeff is not modified;
            "ttns_norm_to_coeff": the ttns part is normalized and the norm is multiplied to coeff;
            "ttns_and_coeff": both ttns and coeff is normalized

        Returns
        -------
        ``self`` is overwritten.
        '''

        if kind == "mps_only":
            new_coeff = self.coeff
        elif kind == "mps_and_coeff":
            new_coeff = self.coeff / np.linalg.norm(self.coeff)
        elif kind == "mps_norm_to_coeff":
            new_coeff = self.coeff * self.ttns_norm
        else:
            raise ValueError(f"kind={kind} is not valid.")
        new_mps = self.scale(1.0 / self.ttns_norm, inplace=True)
        new_mps.coeff = new_coeff

        return new_mps

    def evolve(self, ttno:TTNO, tau:Union[complex, float], normalize:bool=True):
        imag_time = np.iscomplex(tau)
        # trick to avoid complex algebra
        # exp{coeff * H * tau}
        # coef and tau are different from MPS implementation
        if imag_time:
            coeff = 1
            tau = tau.imag
            ttns = self
        else:
            coeff = -1j
            ttns = self.to_complex()
        method = EVOLVE_METHODS[self.evolve_config.method]
        new_ttns = method(ttns, ttno, coeff, tau)
        if normalize:
            if imag_time:
                new_ttns.normalize("mps_and_coeff")
            else:
                new_ttns.normalize("mps_only")
        return new_ttns

    def metacopy(self):
        # node tensor and qn not set
        new = self.__class__(self.basis)
        new.coeff = self.coeff
        new.optimize_config = self.optimize_config.copy()
        new.evolve_config = self.evolve_config.copy()
        new.compress_config = self.compress_config.copy()
        return new

    def copy(self):
        new = self.metacopy()
        for node1, node2 in zip(new, self):
            node1.tensor = node2.tensor.copy()
            node1.qn = node2.qn.copy()
        return new

    def to_complex(self, inplace=False):
        if inplace:
            new = self
        else:
            new = self.metacopy()
        for node1, node2 in zip(self, new):
            node2.tensor = np.array(node1.tensor, dtype=complex)
            node2.qn = node1.qn.copy()
        return new

    def todense(self, order:List[BasisSet]=None) -> np.ndarray:
        _id = str(id(self))
        args = self.to_contract_args()
        if order is None:
            order = self.basis.basis_list
        indices_up = []
        for basis in order:
            indices_up.append(("down", str(basis.dofs)))
        output_indices = indices_up
        args.append(output_indices)
        res = oe.contract(*asxp_oe_args(args))
        # to be consistent with the behavior of MPS/MPO
        res = asnumpy(res)
        return res

    def to_contract_args(self, conj=False):
        args = []
        for node in self.node_list:
            assert isinstance(node, TreeNodeTensor)
            indices = self.get_node_indices(node, conj)
            tensor = node.tensor
            if conj:
                tensor = tensor.conj()
            indices = [indices[i] for i, s in enumerate(tensor.shape) if s != 1]
            tensor = tensor.squeeze()
            assert len(indices) == tensor.ndim
            args.extend([tensor, indices])
        return args

    def decompose_to_parent(self, node: TreeNodeTensor) -> np.ndarray:
        assert node.parent
        qnbigl, qnbigr, _ = self.get_qnmat(node, include_parent=False)
        tensor = node.tensor.reshape(-1, node.shape[-1])
        u, qnlnew, v, qnrnew = svd_qn(tensor, qnbigl, qnbigr, self.qntot, QR=True, system="L", full_matrices=False)
        # could shrink during QR
        node.tensor = u.reshape(list(node.shape[:-1]) + [u.shape[1]])
        node.qn = np.array(qnlnew)
        return v

    def merge_to_parent(self, node: TreeNodeTensor, v: np.ndarray) -> np.ndarray:
        # contract parent
        parent_indices = self.get_node_indices(node.parent)
        args = [node.parent.tensor, parent_indices]
        child_idx1 = parent_indices[node.idx_as_child]  # old parent->child index
        child_idx2 = tuple(list(child_idx1) + ["_idx2"])  # new parent->child index
        args.extend([v, (child_idx1, child_idx2)])
        output_indices = parent_indices.copy()
        output_indices[node.idx_as_child] = child_idx2
        args.append(output_indices)
        node.parent.tensor = oe.contract(*asxp_oe_args(args))

    def push_cano_to_parent(self, node: TreeNodeTensor):
        assert node.parent
        # move the cano center to parent
        v = self.decompose_to_parent(node)
        self.merge_to_parent(node, v)


    def decompose_to_child(self, node: TreeNodeTensor, ichild):
        qnbigl, qnbigr, tensor, shape = moveaxis(self, node, ichild)

        # u for node and v for child
        u, qnl, v, qnr = svd_qn(
            tensor,
            qnbigl,
            qnbigr,
            self.qntot,
            QR=True,
            system="L",
            full_matrices=False
        )
        # XXX: temp debug code for consistency with MPS
        """
        # u for node and v for child
        u, qnl, v, qnr = svd_qn(
            tensor.T,
            qnbigr,
            qnbigl,
            self.qntot,
            QR=True,
            system="R",
            full_matrices=False
        )
        v, qnr, u, qnl = u, qnl, v, qnr
        """
        shape[-1] = u.shape[-1]
        node.tensor = np.moveaxis(u.reshape(shape), -1, ichild)
        node.children[ichild].qn = qnr
        return v

    def merge_to_child(self, node:TreeNodeTensor, ichild, v: np.ndarray):
        child = node.children[ichild]
        child.tensor = tensordot(child.tensor, v, axes=[-1, 0])

    def push_cano_to_child(self, node:TreeNodeTensor, ichild):
        v = self.decompose_to_child(node, ichild)
        self.merge_to_child(node, ichild, v)

    def compress_node(self, node:TreeNodeTensor, ichild:int, temp_m_trunc:int=None, cano_child:bool=True) -> np.ndarray:
        """Compress the bond between node and ichild"""
        qnbigl, qnbigr, tensor, shape = moveaxis(self, node, ichild)
        # u for node and v for child
        u, s, qnl, v, s, qnr = svd_qn(
            tensor,
            qnbigl,
            qnbigr,
            self.qntot,
            full_matrices=False,
        )
        if temp_m_trunc is None:
            idx = self.node_idx[node.children[ichild]]
            m_trunc = self.compress_config.compute_m_trunc(s, idx, left=False)
        else:
            m_trunc = min(temp_m_trunc, len(s))
        orig_s = s.copy()
        u, s, v, qnl, qnr = truncate_tensors(u, s, v, qnl, qnr, m_trunc)

        if cano_child:
            v *= s.reshape(1, -1)
        else:
            u *= s.reshape(1, -1)

        shape[-1] = min(m_trunc, u.shape[-1])
        node.tensor = np.moveaxis(u.reshape(shape), -1, ichild)
        child = node.children[ichild]
        child.tensor = tensordot(child.tensor, v, axes=[-1, 0])
        child.qn = qnr
        return orig_s


    def get_qnmat(self, node:TreeNodeTensor, include_parent:bool=False):
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

    def get_qnmask(self, node, include_parent=False):
        qnmat = self.get_qnmat(node, include_parent)[-1]
        return get_qn_mask(qnmat, self.qntot)

    def update_2site(self, node, tensor, m:int, percent:float=0, cano_parent:bool=True):
        """cano_parent: set canonical center at parent. to_right = True"""
        parent = node.parent
        assert parent is not None
        qnbigl, qnbigr, _ = self.get_qnmat(node, include_parent=True)
        dim1 = np.prod(qnbigl.shape)
        tensor = asnumpy(tensor.reshape(dim1, -1))
        # u for snode and v for parent
        # duplicate with MatrixProduct._udpate_mps. Should consider merging when doing e.g. state averaged algorithm.
        u, su, qnlnew, v, sv, qnrnew = svd_qn(tensor, qnbigl, qnbigr, self.qntot)
        if cano_parent:
            m_node, msdim, msqn, m_parent = select_basis(
                u, su, qnlnew, v, m, percent=percent
            )
        else:
            m_parent, msdim, msqn, m_node = select_basis(
                v, sv, qnrnew, u, m, percent=percent
            )
        m_parent = m_parent.T
        node.tensor = m_node.reshape(list(node.shape[:-1]) + [-1])
        if cano_parent:
            node.qn = msqn
        else:
            node.qn = self.qntot - msqn
        assert len(node.qn) == node.shape[-1]
        shape = list(parent.tensor.shape)
        ichild = parent.children.index(node)
        del shape[ichild]
        shape = [-1] + shape
        parent.tensor = np.moveaxis(m_parent.reshape(shape), 0, ichild)

    def get_node_indices(self, node:TreeNodeTensor, conj:bool=False, include_parent:bool=False) -> List[Tuple[str]]:
        if include_parent:
            snode_indices = self.get_node_indices(node, conj)
            parent_indices = self.get_node_indices(node.parent, conj)
            indices = snode_indices + parent_indices
            shared_bond = snode_indices[-1]
            for i in range(2):
                indices.remove(shared_bond)
            return indices

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

    def merge_with_parent(self, node):
        # merge a node with its parent
        args = []
        snode_indices = self.get_node_indices(node)
        parent_indices = self.get_node_indices(node.parent)
        args.extend([node.tensor, snode_indices])
        args.extend([node.parent.tensor, parent_indices])
        output_indices = self.get_node_indices(node, include_parent=True)
        args.append(output_indices)
        return oe.contract(*asxp_oe_args(args))

    @property
    def norm(self):
        '''duplicate with Mps
        '''
        return np.linalg.norm(self.coeff) * self.ttns_norm

    @property
    def qntot(self):
        # duplicate with ttno
        return self.root.qn[0]

    @property
    def ttns_norm(self):
        res = self.expectation(TTNO.identity(self.basis))

        if res < 0:
            assert np.abs(res) < 1e-8
            res = 0
        res = np.sqrt(res)
        return float(res)

    @property
    def bond_dims(self):
        return [node.tensor.shape[-1] for node in self]

    @property
    def bond_dims_mean(self) -> int:
        # duplicate with Mps
        return int(round(np.mean(self.bond_dims)))

    def scale(self, val, inplace=False):
        # no need to be canonical
        # self.check_canonical()
        if inplace:
            new_mp = self
        else:
            new_mp = self.copy()
        if np.iscomplex(val):
            new_mp.to_complex(inplace=True)
        else:
            val = val.real

        new_mp.root.tensor *= val
        return new_mp

    def __add__(self, other: "TTNS"):
        return self.add(other)


class TTNEnviron(Tree):
    def __init__(self, ttns:TTNS, ttno:TTNO, build_environ=True):
        self.basis =  ttns.basis
        enodes: List[TreeNodeEnviron] = [TreeNodeEnviron() for _ in range(ttns.size)]
        copy_connection(ttns.node_list, enodes)
        super().__init__(enodes[0])
        assert self.root.parent is None
        self.root.environ_parent = np.array([1]).reshape([1, 1, 1])
        # tensor node to basis node. todo: remove duplication?
        self.tn2bn = {tn: bn for tn, bn in zip(self.node_list, self.basis.node_list)}
        self.tn2dofs = {tn: bn.dofs for tn, bn in self.tn2bn.items()}
        if build_environ:
            self.build_children_environ(ttns, ttno)
            self.build_parent_environ(ttns, ttno)

    def build_children_environ(self, ttns, ttno):
        # first run, children environment to the parent.
        # set enode.environ_children
        snodes: List[TreeNodeTensor] = ttns.postorder_list()
        for snode in snodes:
            self.build_children_environ_node(snode, ttns, ttno)

    def build_parent_environ(self, ttns, ttno):
        # second run, parent environment to children
        # set enode.environ_parent
        snodes: List[TreeNodeTensor] = ttns.node_list
        for snode in snodes:
            for ichild in range(len(snode.children)):
                self.build_parent_environ_node(snode, ichild, ttns, ttno)

    def update_1bond(self, snode: TreeNodeTensor, ttns: TTNS, ttno: TTNO):
        # update environ for the bond between snode and snode.parent
        self.build_children_environ_node(snode, ttns, ttno)
        self.build_parent_environ_node(snode.parent, snode.idx_as_child, ttns, ttno)

    def update_1site(self, snode: TreeNodeTensor, ttns: TTNS, ttno: TTNO):
        # update environ based on snode
        self.build_children_environ_node(snode, ttns, ttno)
        for ichild in range(len(snode.children)):
            self.build_parent_environ_node(snode, ichild, ttns, ttno)

    def update_2site(self, snode, ttns, ttno):
        # update environ based on snode and its parent
        self.build_children_environ_node(snode, ttns, ttno)
        self.build_children_environ_node(snode.parent, ttns, ttno)
        for ichild in range(len(snode.parent.children)):
            self.build_parent_environ_node(snode.parent, ichild, ttns, ttno)
        for ichild in range(len(snode.children)):
            self.build_parent_environ_node(snode, ichild, ttns, ttno)

    def build_children_environ_node(self, snode:TreeNodeTensor, ttns: TTNS, ttno: TTNO):
        # build the environment from snode to its parent
        if snode.parent is None:
            return
        enode = self.node_list[ttns.node_idx[snode]]
        onode = ttno.node_list[ttns.node_idx[snode]]
        args = []
        for i, child_tensor in enumerate(enode.environ_children):
            indices = self.get_child_indices(enode, i, ttns, ttno)
            args.extend([child_tensor, indices])

        args.append(snode.tensor.conj())
        args.append(ttns.get_node_indices(snode, conj=True))

        args.append(onode.tensor)
        args.append(ttno.get_node_indices(onode, "up", "down"))

        args.append(snode.tensor)
        args.append(ttns.get_node_indices(snode))

        # indices for the resulting tensor
        indices = self.get_parent_indices(enode, ttns, ttno)
        args.append(indices)
        res = oe.contract(*asxp_oe_args(args))
        if len(enode.parent.environ_children) != len(enode.parent.children):
            # first run
            enode.parent.environ_children.append(asnumpy(res))
        else:
            # updating
            ichild = snode.parent.children.index(snode)
            enode.parent.environ_children[ichild] = asnumpy(res)

    def build_parent_environ_node(self, snode:TreeNodeTensor, ichild: int, ttns: TTNS, ttno: TTNO):
        # build the environment from snode to the ith child of snode
        enode = self.node_list[ttns.node_idx[snode]]
        onode = ttno.node_list[ttns.node_idx[snode]]
        args = []
        # children tensor
        for j, child_tensor in enumerate(enode.environ_children):
            if j == ichild:
                continue
            indices = self.get_child_indices(enode, j, ttns, ttno)
            args.extend([child_tensor, indices])

        # parent tensor
        indices = self.get_parent_indices(enode, ttns, ttno)
        args.extend([enode.environ_parent, indices])

        args.append(snode.tensor.conj())
        args.append(ttns.get_node_indices(snode, conj=True))

        args.append(onode.tensor)
        args.append(ttno.get_node_indices(onode, "up", "down"))

        args.append(snode.tensor)
        args.append(ttns.get_node_indices(snode))

        # indices for the resulting tensor
        indices = self.get_child_indices(enode, ichild, ttns, ttno)

        args.append(indices)
        res = oe.contract(*asxp_oe_args(args))
        enode.children[ichild].environ_parent = asnumpy(res)

    def get_child_indices(self, enode, i, ttns, ttno):
        dofs = self.tn2dofs[enode]
        dofs_child = self.tn2dofs[enode.children[i]]
        indices = [
            (str(id(ttns)) + "_conj", str(dofs), str(dofs_child)),
            (str(id(ttno)), str(dofs), str(dofs_child)),
            (str(id(ttns)), str(dofs), str(dofs_child)),
        ]
        return indices

    def get_parent_indices(self, enode, ttns, ttno):
        dofs = self.tn2dofs[enode]
        if enode.parent is not None:
            dofs_parent = self.tn2dofs[enode.parent]
        else:
            dofs_parent = "root"
        indices = [
            (str(id(ttns)) + "_conj", str(dofs_parent), str(dofs)),
            (str(id(ttno)), str(dofs_parent), str(dofs)),
            (str(id(ttns)), str(dofs_parent), str(dofs)),
        ]
        return indices


def from_mps(mps: Mps) -> Tuple[BasisTree, TTNS, TTNO]:
    # useful function for comparing results with MPS
    mps = mps.copy()
    mps.ensure_left_canonical()
    mps.move_qnidx(len(mps) + 1)
    # take reverse because in `node` the order of the indices is
    # children + physical + parent
    # |    |    |     |
    # o -> o -> o -> root (canonical center)
    basis = BasisTree.linear(mps.model.basis[::-1])
    ttns = TTNS(basis)
    for i in range(len(mps)):
        node = ttns.node_list[::-1][i]
        node.tensor = mps[i].array
        node.qn = mps.qn[i + 1]
        if i == 0:
            # remove the empty children index
            node.tensor = node.tensor[0, ...]
    ttns.check_shape()
    ttns.check_canonical()
    ttno = TTNO(basis, mps.model.ham_terms)
    return basis, ttns, ttno


def compress_recursion(snode: TreeNodeTensor, ttns: TTNS, s_dict:Dict, temp_m_trunc: int = None):
    assert snode.children, "can't compress a single tree node"
    for ichild, child in enumerate(snode.children):
        cano_child = bool(child.children)
        # compress snode - child
        s = ttns.compress_node(snode, ichild, temp_m_trunc, cano_child)
        s_dict[child] = s

        if cano_child:
            compress_recursion(child, ttns, s_dict, temp_m_trunc)
            # cano to snode
            ttns.push_cano_to_parent(child)


def truncate_tensors(u, s, v, qnl, qnr, m):
    u = u[:, :m]
    s = s[:m]
    v = v[:, :m]
    qnl = qnl[:m]
    qnr = qnr[:m]
    return u, s, v, qnl, qnr


def moveaxis(ttns:TTNS, node:TreeNodeTensor, ichild:int):
    # move one of the children indices to the end

    # left indices: other children + physical bonds + parent
    qnbigl = np.zeros(ttns.basis.qn_size, dtype=int)
    # other children
    for child in node.children:
        if child == node.children[ichild]:
            continue
        qnbigl = add_outer(qnbigl, child.qn)
    # physical bonds
    for b in ttns.tn2bn[node].basis_sets:
        qnbigl = add_outer(qnbigl, b.sigmaqn)
    # parent
    qnbigl = add_outer(qnbigl, ttns.qntot - node.qn)
    # right indices: the ith child
    qnbigr = node.children[ichild].qn
    # 2d tensor (node, child)
    tensor = np.moveaxis(node.tensor, ichild, -1)
    shape = list(tensor.shape)
    tensor = tensor.reshape(-1, node.shape[ichild])
    return qnbigl, qnbigr, tensor, shape