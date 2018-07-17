# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
from __future__ import absolute_import

import unittest
import os

import numpy as np
from ddt import ddt, data

from ephMPS.spectra import spectra
from ephMPS.tests.parameter import mol_list, j_matrix
from ephMPS import constant
from ephMPS.mps import solver
from ephMPS.mps.mpo import Mpo


cur_dir = os.path.dirname(os.path.abspath(__file__))


@ddt
class TestSpectra(unittest.TestCase):

    @data([[[4, 4]], 1e-3])
    def test_ZeroExactEmi(self, value):
        # print "data", value
        nexciton = 1
        procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]

        mps, mpo = solver.construct_MPS_MPO_2(mol_list, j_matrix, procedure[0][0], nexciton)
        energy = solver.optimization(mps, mpo, procedure, method="2site")

        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in range(mpo.pbond_list[0]):
            mpo[0][0, ibra, ibra, 0] -= 2.28614053 / constant.au2ev

        dipole_mpo = Mpo.onsite(mol_list, mpo.pbond_list, "a", dipole=True)
        nsteps = 3000
        dt = 30.0
        temperature = 0
        autocorr = spectra.Exact_Spectra("emi", mol_list, mps, dipole_mpo, nsteps, dt, temperature)
        autocorr = np.array(autocorr)
        with open(os.path.join(cur_dir, 'ZeroExactEmi.npy'), 'rb') as f:
            ZeroExactEmi_std = np.load(f)

        self.assertTrue(np.allclose(autocorr, ZeroExactEmi_std, rtol=value[1]))


if __name__ == "__main__":
    print("Test tMPS")
    unittest.main()
