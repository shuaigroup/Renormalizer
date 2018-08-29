# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
from __future__ import absolute_import, print_function, division

import unittest

import numpy as np
from ddt import ddt, data

from ephMPS.mps import Mpo
from ephMPS.mps import solver
from ephMPS.tests import parameter
from ephMPS.utils import constant

nexciton = 1
procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]


@ddt
class TestQuasiBoson(unittest.TestCase):
    def test_quasiboson_constructMPO(self):

        mol_list1 = parameter.custom_mol_list([4, 4])
        mps1, mpo1 = solver.construct_mps_mpo_2(mol_list1, parameter.j_matrix, procedure[0][0], nexciton, scheme=2)

        mol_list2 = mol_list = parameter.custom_mol_list([4, 4], [2, 2], [1e-7, 1e-7])
        mps2, mpo2 = solver.construct_mps_mpo_2(mol_list2, parameter.j_matrix, procedure[0][0], nexciton, scheme=2)

        # merge the decomposed MPO
        mpo_merge = Mpo()
        impo = 0
        for mol in mol_list:
            mpo_merge.append(mpo2[impo])
            impo += 1
            for _ in mol.phs:
                mo = np.einsum("abcd, defg -> abecfg", mpo2[impo], mpo2[impo + 1]) \
                    .reshape(mpo2[impo].shape[0], 4, 4, mpo2[impo + 1].shape[-1])
                mpo_merge.append(mo)
                impo += 2

        self.assertAlmostEqual(mpo1.distance(mpo_merge), 0.0)

        energy = solver.optimize_mps(mps2, mpo2, procedure, method="2site")
        self.assertAlmostEqual(np.min(energy) * constant.au2ev, 2.28614053133)

        energy = solver.optimize_mps(mps2, mpo2, procedure, method="1site")
        self.assertAlmostEqual(np.min(energy) * constant.au2ev, 2.28614053133)

    @data([[[64, 64]], [[64, 64], [6, 6], [1e-7, 1e-7]], [[64, 64], [6, 1], [1e-7, 1e-7]]],
          [[[27, 27]], [[27, 27], [3, 3], [1e-7, 1e-7]], [[27, 27], [3, 1], [1e-7, 1e-7]]]
          )
    def test_quasiboson_solver(self, value):
        np.random.seed(0)
        # normal boson
        mol_list1 = parameter.custom_mol_list(*value[0])
        mps1, mpo1 = solver.construct_mps_mpo_2(mol_list1, parameter.j_matrix, procedure[0][0], nexciton, scheme=2)

        # quasiboson
        mol_list2 = parameter.custom_mol_list(*value[1])
        mps2, mpo2 = solver.construct_mps_mpo_2(mol_list2, parameter.j_matrix, procedure[0][0], nexciton, scheme=2)

        # quasiboson + normal boson
        mol_list3 = parameter.custom_mol_list(*value[2])
        mps3, mpo3 = solver.construct_mps_mpo_2(mol_list3, parameter.j_matrix, procedure[0][0], nexciton, scheme=2)

        for method in ['1site', '2site']:
            energy1 = solver.optimize_mps(mps1, mpo1, procedure, method=method)
            energy2 = solver.optimize_mps(mps2, mpo2, procedure, method=method)
            energy3 = solver.optimize_mps(mps3, mpo3, procedure, method=method)
            self.assertAlmostEqual(np.min(energy1), np.min(energy2))
            self.assertAlmostEqual(np.min(energy2), np.min(energy3))


if __name__ == '__main__':
    # TestQuasiBoson.test_quasiboson_solver()
    unittest.main()
