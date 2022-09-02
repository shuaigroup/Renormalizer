import logging
import itertools
from copy import deepcopy
from typing import List, Union

import numpy as np
import scipy
import scipy.sparse

from renormalizer.model import Model, HolsteinModel
from renormalizer.mps.backend import xp
from renormalizer.mps.matrix import moveaxis, tensordot
from renormalizer.mps.mp import MatrixProduct
from renormalizer.mps import svd_qn
from renormalizer.mps.lib import update_cv
from renormalizer.mps.symbolic_mpo import construct_symbolic_mpo, _terms_to_table, symbolic_mo_to_numeric_mo, swap_site
from renormalizer.utils import Quantity
from renormalizer.model.op import Op
from renormalizer.utils.elementop import (
    construct_ph_op_dict,
)


logger = logging.getLogger(__name__)


class Mpo(MatrixProduct):
    """
    Matrix product operator (MPO)
    """

    @classmethod
    def exact_propagator(cls, model: HolsteinModel, x, space="GS", shift=0.0):
        """
        construct the GS space propagator e^{xH} exact MPO
        H=\\sum_{in} \\omega_{in} b^\\dagger_{in} b_{in}
        fortunately, the H is local. so e^{xH} = e^{xh1}e^{xh2}...e^{xhn}
        the bond dimension is 1
        shift is the a constant for H+shift
        """
        assert space in ["GS", "EX"]

        mpo = cls()
        if np.iscomplex(x):
            mpo.to_complex(inplace=True)
        mpo.model = model

        for imol, mol in enumerate(model):
            if model.scheme < 4:
                mo = np.eye(2).reshape(1, 2, 2, 1)
                mpo.append(mo)
            elif model.scheme == 4:
                if len(mpo) == model.order[0]:
                    n = model.mol_num
                    mpo.append(np.eye(n+1).reshape(1, n+1, n+1, 1))
            else:
                assert False

            for ph in mol.ph_list:

                if space == "EX":
                    ph_pbond = ph.pbond
                    # construct the matrix exponential by diagonalize the matrix first
                    phop = construct_ph_op_dict(ph_pbond)

                    h_mo = (
                        phop[r"b^\dagger b"] * ph.omega[0]
                        + phop[r"b^\dagger + b"] * ph.term10
                    )

                    w, v = scipy.linalg.eigh(h_mo)
                    h_mo = np.diag(np.exp(x * w))
                    h_mo = v.dot(h_mo)
                    h_mo = h_mo.dot(v.T)
                    mo = h_mo.reshape(1, ph_pbond, ph_pbond, 1)

                    mpo.append(mo)

                elif space == "GS":
                    # for the ground state space
                    ph_pbond = ph.pbond
                    d = np.exp(
                            x
                            * ph.omega[0]
                            * np.arange(ph_pbond)
                        )
                    mo = np.diag(d).reshape(1, ph_pbond, ph_pbond, 1)
                    mpo.append(mo)
                else:
                    assert False
        # shift the H by plus a constant

        mpo.qn = [[0]] * (len(mpo) + 1)
        mpo.qnidx = len(mpo) - 1
        mpo.qntot = 0

        # np.exp(shift * x) is usually very large
        mpo = mpo.scale(np.exp(shift * x), inplace=True)

        return mpo

    @classmethod
    def onsite(cls, model: Model, opera, dipole=False, dof_set=None):
        if dof_set is None:
            if model.n_edofs == 0:
                raise ValueError("No electronic DoF present in the model.")
            dof_set = model.e_dofs
        ops = []
        for idx in dof_set:
            if dipole:
                factor = model.dipole[idx]
            else:
                factor = 1.
            ops.append(Op(opera, idx, factor))

        return cls(model, ops)

    @classmethod
    def ph_onsite(cls, model: HolsteinModel, opera: str, mol_idx:int, ph_idx=0):
        assert opera in ["b", r"b^\dagger", r"b^\dagger b"]
        if not isinstance(model, HolsteinModel):
            raise TypeError("ph_onsite only supports HolsteinModel")
        return cls(model, Op(opera, (mol_idx, ph_idx)))

    @classmethod
    def intersite(cls, model: HolsteinModel, e_opera: dict, ph_opera: dict, scale:
            Quantity=Quantity(1.)):
        r""" construct the inter site MPO
        
        Parameters
        ----------
        model : HolsteinModel
            the molecular information
        e_opera:
            the electronic operators. {imol: operator}, such as {1:"a", 3:r"a^\dagger"}
        ph_opera:
            the vibrational operators. {(imol, iph): operator}, such as {(0,5):"b"}
        scale: Quantity
            scalar to scale the mpo

        Note
        -----
        the operator index starts from 0,1,2...
        
        """

        ops = []
        for e_key, e_op in e_opera.items():
            ops.append(Op(e_op, e_key))
        for v_key, v_op in ph_opera.items():
            ops.append(Op(v_op, v_key))
        op = scale.as_au() * Op.product(ops)
        return cls(model, op)

    @classmethod
    def finiteT_cv(cls, model, nexciton, m_max, spectratype, percent=1.0):
        np.random.seed(0)

        X = cls()
        X.model = model
        if spectratype == "abs":
            # quantum number index, |1><0|
            tag_1, tag_2 = 0, 1
        elif spectratype == "emi":
            # quantum number index, |0><1|
            tag_1, tag_2 = 1, 0
        X.qn = [[[0, 0]]]
        for ix in range(model.nsite - 1):
            X.qn.append(None)
        X.qn.append([[0, 0]])
        dim_list = [1]

        for ix in range(model.nsite - 1):
            sigmaqn = model.basis[ix].sigmaqn
            sigmaqn = np.array(list(itertools.product(sigmaqn, repeat=2)))
            qn1 = np.add.outer(np.array(X.qn[ix])[:, 0], sigmaqn[:, 0]).ravel()
            qn2 = np.add.outer(np.array(X.qn[ix])[:, 1], sigmaqn[:, 1]).ravel()
            qnbig = np.stack([qn1, qn2], axis=1)
            # print('qnbig', qnbig)
            u_set = []
            s_set = []
            qnset = []
            if spectratype != "conductivity":
                fq = list(itertools.chain.from_iterable([y[tag_1]] for y in qnbig))
                for iblock in range(min(fq), nexciton+1):
                    indices = [i for i, y in enumerate(qnbig) if
                               ((y[tag_1] == iblock) and (y[tag_2] == 0))]
                    if len(indices) != 0:
                        np.random.seed(0)
                        a: np.ndarray = np.random.random([len(indices), len(indices)]) - 0.5
                        a = a + a.T
                        s, u = scipy.linalg.eigh(a=a)
                        u_set.append(svd_qn.blockrecover(indices, u, len(qnbig)))
                        s_set.append(s)
                        if spectratype == "abs":
                            qnset += [iblock, 0] * len(indices)
                        elif spectratype == "emi":
                            qnset += [0, iblock] * len(indices)
            else:
                fq1 = list(itertools.chain.from_iterable([y[0]] for y in qnbig))
                fq2 = list(itertools.chain.from_iterable([y[1]] for y in qnbig))
                # print('fq1, fq2', fq1, fq2)
                for iblock in range(min(fq1), nexciton+1):
                    for jblock in range(min(fq2), nexciton+1):
                        # print('iblock', iblock, jblock)
                        indices = [i for i, y in enumerate(qnbig) if
                                   ((y[0] == iblock) and (y[1] == jblock))]
                        # print('indices', indices)
                        if len(indices) != 0:
                            a: np.ndarray = np.random.random([len(indices), len(indices)]) - 0.5
                            a = a + a.T
                            s, u = scipy.linalg.eigh(a=a)
                            u_set.append(svd_qn.blockrecover(indices, u, len(qnbig)))
                            s_set.append(s)
                            qnset += [iblock, jblock] * len(indices)
                            # print('iblock', iblock)
            list_qnset = []
            for i in range(0, len(qnset), 2):
                list_qnset.append([qnset[i], qnset[i + 1]])
            qnset = list_qnset
            # print('qnset', qnset)
            u_set = np.concatenate(u_set, axis=1)
            s_set = np.concatenate(s_set)
            # print('uset', u_set.shape)
            # print('s_set', s_set.shape)
            x, xdim, xqn, compx = update_cv(u_set, s_set, qnset, None, nexciton, m_max, spectratype, percent=percent)
            dim_list.append(xdim)
            X.qn[ix + 1] = xqn
            x = x.reshape(dim_list[-2], model.pbond_list[ix], model.pbond_list[ix], dim_list[ix + 1])
            X.append(x)
        dim_list.append(1)
        X.append(np.random.random([dim_list[-2], model.pbond_list[-1],
                                   model.pbond_list[-1], dim_list[-1]]))
        X.qnidx = len(X) - 1
        X.to_right = False
        X.qntot = nexciton
        # print('dim', [X[i].shape for i in range(len(X))])
        return X

    @classmethod
    def identity(cls, model: Model):
        mpo = cls()
        mpo.model = model
        for p in model.pbond_list:
            mpo.append(np.eye(p).reshape(1, p, p, 1))
        mpo.build_empty_qn()
        return mpo

    def __init__(self, model: Model = None, terms: Union[Op, List[Op]] = None, offset: Quantity = Quantity(0), ):

        """
        todo: document
        """
        super(Mpo, self).__init__()
        # leave the possibility to construct MPO by hand
        if model is None:
            return
        if not isinstance(offset, Quantity):
            raise ValueError(f"offset must be Quantity object. Got {offset} of {type(offset)}.")

        self.offset = offset.as_au()
        if terms is None:
            terms = model.ham_terms
        elif isinstance(terms, Op):
            terms = [terms]

        if len(terms) == 0:
            raise ValueError("Terms contain nothing.")
        terms = model.check_operator_terms(terms)
        if len(terms) == 0:
            raise ValueError("Terms all have factor 0.")

        table, factor = _terms_to_table(model, terms, -self.offset)

        self.dtype = factor.dtype

        mpo_symbol, self.qn, self.qntot, self.qnidx, self.symbolic_out_ops_list, self.primary_ops = construct_symbolic_mpo(table, factor)
        # print(_format_symbolic_mpo(mpo_symbol))
        self.model = model
        self.to_right = False

        # evaluate the symbolic mpo
        assert model.basis is not None

        for impo, mo in enumerate(mpo_symbol):
            mo_mat = symbolic_mo_to_numeric_mo(model.basis[impo], mo, self.dtype)
            self.append(mo_mat)


    def _get_sigmaqn(self, idx):
        array_up = self.model.basis[idx].sigmaqn
        return np.subtract.outer(array_up, array_up)

    @property
    def is_mps(self):
        return False

    @property
    def is_mpo(self):
        return True

    @property
    def is_mpdm(self):
        return False

    def metacopy(self):
        new = super().metacopy()
        # some mpo may not have these things
        attrs = ["scheme", "offset", "symbolic_out_ops_list", "primary_ops"]
        for attr in attrs:
            if hasattr(self, attr):
                setattr(new, attr, deepcopy(getattr(self, attr)))
        return new

    @property
    def dummy_qn(self):
        return [[0] * dim for dim in self.bond_dims]

    @property
    def digest(self):
        return np.array([mt.var() for mt in self]).var()

    def promote_mt_type(self, mp):
        if self.is_complex and not mp.is_complex:
            mp.to_complex(inplace=True)
        return mp

    def apply(self, mp: MatrixProduct, canonicalise: bool=False) -> MatrixProduct:
        # todo: use meta copy to save time, could be subtle when complex type is involved
        # todo: inplace version (saved memory and can be used in `hybrid_exact_propagator`)
        # the model is the same as the mps.model
        new_mps = self.promote_mt_type(mp.copy())
        if mp.is_mps:
            # mpo x mps
            for i, (mt_self, mt_other) in enumerate(zip(self, mp)):
                assert mt_self.shape[2] == mt_other.shape[1]
                # mt=np.einsum("apqb,cqd->acpbd",mpo[i],mps[i])
                mt = xp.moveaxis(
                    tensordot(mt_self.array, mt_other.array, axes=([2], [1])), 3, 1
                )
                mt = mt.reshape(
                    (
                        mt_self.shape[0] * mt_other.shape[0],
                        mt_self.shape[1],
                        mt_self.shape[-1] * mt_other.shape[-1],
                    )
                )
                new_mps[i] = mt
        elif mp.is_mpo or mp.is_mpdm:
            # mpo x mpo
            for i, (mt_self, mt_other) in enumerate(zip(self, mp)):
                assert mt_self.shape[2] == mt_other.shape[1]
                # mt=np.einsum("apqb,cqrd->acprbd",mt_s,mt_o)
                mt = xp.moveaxis(
                    tensordot(mt_self.array, mt_other.array, axes=([2], [1])),
                    [-3, -2],
                    [1, 3],
                )
                mt = mt.reshape(
                    (
                        mt_self.shape[0] * mt_other.shape[0],
                        mt_self.shape[1],
                        mt_other.shape[2],
                        mt_self.shape[-1] * mt_other.shape[-1],
                    )
                )
                new_mps[i] = mt
        else:
            assert False
        orig_idx = new_mps.qnidx
        new_mps.move_qnidx(self.qnidx)
        new_mps.qn = [
            np.add.outer(np.array(qn_o), np.array(qn_m)).ravel().tolist()
            for qn_o, qn_m in zip(self.qn, new_mps.qn)
        ]
        new_mps.qntot += self.qntot
        new_mps.move_qnidx(orig_idx)
        # concerns about whether to canonicalise:
        # * canonicalise helps to keep mps in a truly canonicalised state
        # * canonicalise comes with a cost. Unnecessary canonicalise (for example in P&C evolution and
        #   expectation calculation) hampers performance.
        if canonicalise:
            new_mps.canonicalise()
        return new_mps

    def contract(self, mps, algo="svd"):
        r""" an approximation of mpo @ mps/mpdm/mpo
        
        Parameters
        ----------
        mps : `Mps`, `Mpo`, `MpDm`
        algo: str, optional
            The algorithm to compress mpo @ mps/mpdm/mpo.  It could be ``svd``
            (default) and ``variational``. 
        
        Returns
        -------
        new_mps : `Mps`
            an approximation of mpo @ mps/mpdm/mpo. The input ``mps`` is not
            overwritten.

        See Also
        --------
        renormalizer.mps.mp.MatrixProduct.compress : svd compression.
        renormalizer.mps.mp.MatrixProduct.variational_compress : variational
            compression.


        """
        if algo == "svd":
            # mapply->canonicalise->compress
            new_mps = self.apply(mps)
            new_mps.canonicalise()
            new_mps.compress()
        elif algo == "variational":
            new_mps = mps.variational_compress(self)
        else:
            assert False

        return new_mps

    def try_swap_site(self, new_model: Model, swap_jw: bool):
        # in place swapping.
        # if swap_jw is set to True, then self.primary_ops is modified in place
        diffs = []
        for i, (b1, b2) in enumerate(zip(self.model.basis, new_model.basis)):
            if b1.dofs != b2.dofs:
                diffs.append(i)
        if len(diffs) == 0:
            logger.debug("MPO: No need to swap")
            return
        assert len(diffs) == 2
        i, j = min(diffs), max(diffs)
        assert j - i == 1
        logger.debug(f"MPO: swaping {i} and {j}")
        # although usually the `model` of MPO does not store `mpos`
        new_model.mpos.clear()

        out_ops2, out_ops3, mo1, mo2, qn = swap_site(self.symbolic_out_ops_list[i:i+3], self.primary_ops, swap_jw)

        self.symbolic_out_ops_list[i+1] = out_ops2
        self.symbolic_out_ops_list[i+2] = out_ops3
        self.model = new_model
        self.qn[i+1] = qn

        for impo, mo in zip([i, j], [mo1, mo2]):
            self[impo] = symbolic_mo_to_numeric_mo(new_model.basis[impo], mo, self.dtype)
        logger.debug(self)

    def conj_trans(self):
        new_mpo = self.metacopy()
        for i in range(new_mpo.site_num):
            new_mpo[i] = moveaxis(self[i], (1, 2), (2, 1)).conj()
        new_mpo.qn = [[-i for i in mt_qn] for mt_qn in new_mpo.qn]
        return new_mpo

    def todense(self):
        dim = np.prod(self.pbond_list)
        if 20000 < dim:
            raise ValueError("operator too large")
        res = np.ones((1, 1, 1, 1))
        for mt in self:
            dim1 = res.shape[1] * mt.shape[1]
            dim2 = res.shape[2] * mt.shape[2]
            dim3 = mt.shape[-1]
            res = np.tensordot(res, mt.array, axes=1).transpose((0, 1, 3, 2, 4, 5)).reshape(1, dim1, dim2, dim3)
        return res[0, :, :, 0]

    def is_hermitian(self):
        full = self.todense()
        return np.allclose(full.conj().T, full, atol=1e-7)

    def __matmul__(self, other):
        return self.apply(other)

    @classmethod
    def from_mp(cls, model, mp):
        # mpo from matrix product
        mpo = cls()
        mpo.model = model
        for mt in mp:
            mpo.append(mt)
        mpo.build_empty_qn()
        return mpo
