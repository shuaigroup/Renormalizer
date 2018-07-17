from __future__ import absolute_import, print_function, unicode_literals

import itertools

import numpy as np
import scipy

from ephMPS.mps.lib import updatemps
from ephMPS.mps.matrix import MatrixState
from ephMPS.mps.mp import MatrixProduct
from ephMPS.utils import svd_qn


class Mps(MatrixProduct):

    @classmethod
    def from_mpo(cls, mpo, nexciton, Mmax, thresh=1e-3, percent=0):
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

            Uset = []
            Sset = []
            qnset = []

            for iblock in range(min(qnbig), nexciton + 1):
                # find the quantum number index
                indices = [i for i, x in enumerate(qnbig) if x == iblock]

                if len(indices) != 0:
                    a = np.random.random([len(indices), len(indices)]) - 0.5
                    a = a + a.T
                    S, U = scipy.linalg.eigh(a=a)
                    Uset.append(svd_qn.blockrecover(indices, U, len(qnbig)))
                    Sset.append(S)
                    qnset += [iblock] * len(indices)

            Uset = np.concatenate(Uset, axis=1)
            Sset = np.concatenate(Sset)
            mt, mpsdim, mpsqn, nouse = updatemps(Uset, Sset, qnset, Uset, nexciton, Mmax, percent=percent)
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

    def __init__(self):
        super(Mps, self).__init__()
        self.dim_list = None
        self.pbond_list = None
        self.mtype = MatrixState


    @property
    def nexciton(self):
        return self.qntot






