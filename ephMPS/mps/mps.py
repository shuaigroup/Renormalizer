from __future__ import absolute_import, print_function, unicode_literals

import itertools

import numpy as np
import scipy

from ephMPS.mps.lib import updatemps
from ephMPS.mps.mp import MatrixProduct
from ephMPS.mps.matrix import MatrixState
from ephMPS.utils import svd_qn
from ephMPS.mps.ephtable import electron



class Mps(MatrixProduct):

    def __init__(self, domain, mpo, nexciton, Mmax, percent=0):

        super(Mps, self).__init__()
        self.mtype = MatrixState
        self.nexciton = nexciton
        self.Mmax = Mmax

        self.pbond_list = mpo.pbond_list
        self.ephtable = mpo.ephtable

        self.qn = [[0], ]
        self.dim = [1, ]

        nmps = len(mpo)

        for imps in range(nmps - 1):

            # quantum number
            if self.ephtable[imps] is electron:
                # e site
                qnbig = list(itertools.chain.from_iterable([x, x + 1] for x in self.qn[imps]))
            else:
                # ph site
                qnbig = list(itertools.chain.from_iterable([x] * self.pbond_list[imps] for x in self.qn[imps]))

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
            mt, mpsdim, mpsqn, nouse = updatemps(Uset, Sset, qnset, Uset, self.nexciton, self.Mmax, percent=percent)
            # add the next mpsdim
            self.dim.append(mpsdim)
            self.append(mt.reshape(self.dim[imps], self.pbond_list[imps], self.dim[imps + 1]))
            self.qn.append(mpsqn)

        # the last site
        self.qn.append([0])
        self.dim.append(1)
        self.append(np.random.random([self.dim[-2], self.pbond_list[-1], self.dim[-1]]) - 0.5)

        self.qnidx = len(self) - 1
        self.qntot = self.nexciton

        #print("self.dim", self.dim)
        self._left_domain = True







