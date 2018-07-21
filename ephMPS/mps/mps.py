from __future__ import absolute_import, print_function, unicode_literals

import itertools

import numpy as np
import scipy

from ephMPS.mps.lib import updatemps
from ephMPS.mps.matrix import MatrixState
from ephMPS.mps.mp import MatrixProduct
from ephMPS.mps.ephtable import EphTable
from ephMPS.utils import svd_qn


class Mps(MatrixProduct):

    @classmethod
    def from_mpo(cls, mpo, nexciton, m_max, thresh=1e-3, percent=0):
        mps = cls()
        mps.ephtable = mpo.ephtable
        mps.pbond_list = mpo.pbond_list
        mps.thresh = thresh
        mps.qn = [[0], ]
        mps.dim_list = [1, ]
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
            mps.dim_list.append(mpsdim)
            mps.append(mt.reshape(mps.dim_list[imps], mps.pbond_list[imps], mps.dim_list[imps + 1]))
            mps.qn.append(mpsqn)

        # the last site
        mps.qn.append([0])
        mps.dim_list.append(1)
        mps.append(np.random.random([mps.dim_list[-2], mps.pbond_list[-1], mps.dim_list[-1]]) - 0.5)

        mps.qnidx = len(mps) - 1
        mps.qntot = nexciton

        # print("self.dim", self.dim)
        mps._left_canon = True
        return mps

    @classmethod
    def max_entangled_gs(cls, mol_list, pbond_list, normalize=True):
        """
        T = \infty maximum entangled GS state
        electronic site: pbond 0 element 1.0
                         pbond 1 element 0.0
        phonon site: digonal element sqrt(pbond) for normalization
        """
        mps = cls()
        mps.dim_list = [1] * (len(pbond_list) + 1)
        mps.qn = [[0]] * (len(pbond_list) + 1)
        mps.qnidx = len(pbond_list) - 1
        mps.qntot = 0
        mps.ephtable = EphTable.from_mol_list(mol_list)

        imps = 0
        for mol in mol_list:
            ms = np.zeros([mps.dim_list[imps], pbond_list[imps], mps.dim_list[imps + 1]])
            for ibra in range(pbond_list[imps]):
                if ibra == 0:
                    ms[0, ibra, 0] = 1.0
                else:
                    ms[0, ibra, 0] = 0.0

            mps.append(ms)
            imps += 1

            for ph in mol.phs:
                for iboson in range(ph.nqboson):
                    ms = np.zeros([mps.dim_list[imps], pbond_list[imps], mps.dim_list[imps + 1]])
                    if normalize:
                        ms[0, :, 0] = 1.0 / np.sqrt(pbond_list[imps])
                    else:
                        ms[0, :, 0] = 1.0

                    mps.append(ms)
                    imps += 1

        return mps

    def __init__(self):
        super(Mps, self).__init__()
        self.dim_list = None
        self.pbond_list = None
        self.mtype = MatrixState

    @property
    def nexciton(self):
        return self.qntot
