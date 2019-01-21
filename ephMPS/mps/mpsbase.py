from __future__ import absolute_import, print_function, unicode_literals

import itertools

import numpy as np
import scipy

from ephMPS.mps import svd_qn
from ephMPS.mps.lib import updatemps
from ephMPS.mps.matrix import MatrixState
from ephMPS.mps.mp import MatrixProduct


class MpsBase(MatrixProduct):

    @classmethod
    def random(cls, mpo, nexciton, m_max, percent=0):
        mps = cls()
        mps.mol_list = mpo.mol_list
        mps.qn = [[0], ]
        dim_list = [1, ]
        nmps = len(mpo)

        for imps in range(nmps - 1):

            # quantum number
            if mps.ephtable.is_electron(imps):
                # e site
                qnbig = list(itertools.chain.from_iterable([x, x + 1] for x in mps.qn[imps]))
            else:
                # ph site
                qnbig = list(itertools.chain.from_iterable([x] * mps.pbond_list[imps] for x in mps.qn[imps]))

            u_set = []
            s_set = []
            qnset = []

            for iblock in range(min(qnbig), nexciton + 1):
                # find the quantum number index
                indices = [i for i, x in enumerate(qnbig) if x == iblock]

                if len(indices) != 0:
                    a = np.random.random([len(indices), len(indices)]) - 0.5
                    a = a + a.T
                    s, u = scipy.linalg.eigh(a=a)
                    u_set.append(svd_qn.blockrecover(indices, u, len(qnbig)))
                    s_set.append(s)
                    qnset += [iblock] * len(indices)

            u_set = np.concatenate(u_set, axis=1)
            s_set = np.concatenate(s_set)
            mt, mpsdim, mpsqn, nouse = updatemps(u_set, s_set, qnset, u_set, nexciton, m_max, percent=percent)
            # add the next mpsdim
            dim_list.append(mpsdim)
            mps.append(mt.reshape(dim_list[imps], mps.pbond_list[imps], dim_list[imps + 1]))
            mps.qn.append(mpsqn)

        # the last site
        mps.qn.append([0])
        dim_list.append(1)
        mps.append(np.random.random([dim_list[-2], mps.pbond_list[-1], dim_list[-1]]) - 0.5)

        mps.qnidx = len(mps) - 1
        mps.qntot = nexciton

        # print("self.dim", self.dim)
        mps._left_canon = True

        mps.wfn = []
        for mol in mps.mol_list:
            for ph in mol.hartree_phs:
                mps.wfn.append(np.random.random(ph.n_phys_dim))
        mps.wfn.append(1.0)

        return mps

    @classmethod
    def gs(cls, mol_list, max_entangled):
        """
        T = \infty maximum entangled GS state
        electronic site: pbond 0 element 1.0
                         pbond 1 element 0.0
        phonon site: digonal element sqrt(pbond) for normalization
        """
        mps = cls()
        mps.mol_list = mol_list
        mps.qn = [[0]] * (len(mps.ephtable) + 1)
        mps.qnidx = len(mps.ephtable) - 1
        mps.qntot = 0
        
        for mol in mol_list:
            # electron mps
            mps.append(np.array([1, 0]).reshape(1, 2, 1))
            # ph mps
            for ph in mol.dmrg_phs:
                for iboson in range(ph.nqboson):
                    ms = np.zeros((1, ph.base, 1))
                    if max_entangled:
                        ms[0, :, 0] = 1.0 / np.sqrt(ph.base)
                    else:
                        ms[0, 0, 0] = 1.0
                    mps.append(ms)

        return mps

    def __init__(self):
        super(MpsBase, self).__init__()
        self.mtype = MatrixState
        self.wfn = []

    @property
    def digest(self):
        if 10 < self.site_num:
            return None
        prod = np.eye(1).reshape(1, 1, 1)
        for ms in self:
            prod = np.tensordot(prod, ms, axes=1)
            prod = prod.reshape((prod.shape[0], -1, prod.shape[-1]))
        return {'var': prod.var(), 'mean': prod.mean(), 'ptp': prod.ptp()}


    @property
    def nexciton(self):
        return self.qntot

    def hartree_wfn_diff(self, other):
        res = []
        for wfn1, wfn2 in zip(self.wfn, other.wfn):
            res.append(scipy.linalg.norm(np.tensordot(wfn1, wfn1, axes=0) - np.tensordot(wfn2, wfn2, axes=0)))
        return np.array(res)
