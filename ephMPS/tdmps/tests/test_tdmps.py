# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
from __future__ import absolute_import, print_function, division

import unittest
import os

from ddt import ddt, data
import numpy as np

from ephMPS.tdmps import tdmps
from ephMPS.mps.mpo import Mpo
from ephMPS.tests.parameter import custom_mol_list, j_matrix
from ephMPS import constant
from ephMPS.mps import solver

cur_dir = os.path.dirname(os.path.abspath(__file__))


@ddt
class Test_tMPS(unittest.TestCase):
    # todo: variational method not implemented yet
    @data(
        [1, "svd", True, [[4, 4]], 1e-3],
        [2, "svd", True, [[4, 4]], 1e-3],
        #[1, "svd", None, [[4, 4]], 1e-3],
        #[2, "svd", None, [[4, 4]], 1e-3],
        #[1, "variational", None, [[4, 4]], 1e-2],
        #[2, "variational", None, [[4, 4]], 1e-2],
        #[1, "svd", True, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
        #[2, "svd", True, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
        #[1, "svd", None, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
        #[2, "svd", None, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
        #[1, "variational", None, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
        #[2, "variational", None, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2]
    )
    def test_ZeroTcorr(self, value):
        np.random.seed(0)
        #print "data", value
        nexciton = 0
        procedure = [[1, 0], [1, 0], [1, 0]]

        mol_list = custom_mol_list(*value[3])

        mps, mpo= solver.construct_MPS_MPO_2(mol_list, j_matrix, procedure[0][0], nexciton)

        solver.optimization(mps, mpo, procedure, method="2site")
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in range(mpo.pbond_list[0]):
            mpo[0][0, ibra, ibra, 0] -= 2.28614053 / constant.au2ev

        enable_qn = value[2]
        if enable_qn:
            mps.enable_qn = mpo.enable_qn = True
        else:
            mps.enable_qn = mpo.enable_qn = False

        dipole_mpo = Mpo.onsite(mol_list, mpo.pbond_list, "a^\dagger", dipole=True)

        nsteps = 100
        dt = 30.0

        autocorr = tdmps.zero_t_corr(mps, mpo, dipole_mpo, nsteps, dt, algorithm=value[0],
                                  compress_method=value[1])
        autocorr = np.array(autocorr)
        with open(os.path.join(cur_dir, 'ZeroTabs_' + str(value[0]) + str(value[1]) + '.npy'), 'rb') as f:
            ZeroTabs_std = np.load(f)
        self.assertTrue(np.allclose(autocorr, ZeroTabs_std, rtol=value[4]))





        

if __name__ == "__main__":
    print("Test tMPS")
    unittest.main()
