# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import unittest

from ddt import ddt, data, unpack
import numpy as np

from ephMPS.transport import ChargeTransport
from ephMPS.model import Phonon, Mol, MolList
from ephMPS import constant


@ddt
class TestBandLimitZeroT(unittest.TestCase):

    @data(
        [13, 0.8, 3.87e-3, [[1e-10, 1e-10]], 4, 2, 15]
    )
    @unpack
    def test(self, mol_num, j_constant, elocalex, ph_info, ph_phys_dim, evolve_dt, nsteps):
        ph_list = [Phonon.simple_phonon(omega * constant.cm2au, displacement, ph_phys_dim)
                   for omega, displacement in ph_info]
        mol_list = MolList([Mol(elocalex, ph_list)] * mol_num)
        ct = ChargeTransport(mol_list, j_constant).evolve(evolve_dt, nsteps)
        analytical_r_square = 2 * (j_constant / constant.au2ev) ** 2 * ct.evolve_times_array ** 2
        self.assertTrue(np.allclose(analytical_r_square, ct.r_square_array, rtol=1e-3))

@ddt
class TestBandLimitFiniteT(unittest.TestCase):

    @data(
        [13, 0.8, 3.87e-3, [[1e-10, 1e-10]], 4, 2, 15]
    )
    @unpack
    def test(self, mol_num, j_constant, elocalex, ph_info, ph_phys_dim, evolve_dt, nsteps):
        ph_list = [Phonon.simple_phonon(omega * constant.cm2au, displacement, ph_phys_dim)
                   for omega, displacement in ph_info]
        mol_list = MolList([Mol(elocalex, ph_list)] * mol_num)
        ct = ChargeTransport(mol_list, j_constant, temperature=298).evolve(evolve_dt, nsteps)
        analytical_r_square = 2 * (j_constant / constant.au2ev) ** 2 * ct.evolve_times_array ** 2
        self.assertTrue(np.allclose(analytical_r_square, ct.r_square_array, rtol=1e-3))

@ddt
class TestHoppingLimitZeroT(unittest.TestCase):
    @data(
        [13, 0.02, 3.87e-3, [[1345.6738910804488, 16.274571056529368]], 10, 20, 100]
    )
    @unpack
    def test(self, mol_num, j_constant, elocalex, ph_info, ph_phys_dim, evolve_dt, nsteps):
        ph_list = [Phonon.simple_phonon(omega * constant.cm2au, displacement, ph_phys_dim)
                   for omega, displacement in ph_info]
        mol_list = MolList([Mol(elocalex, ph_list)] * mol_num)
        ct = ChargeTransport(mol_list, j_constant).evolve(evolve_dt, nsteps)
        pass


@ddt
class TestHoppingLimitFiniteT(unittest.TestCase):
    @data(
        [13, 0.02, 3.87e-3, [[1345.6738910804488, 16.274571056529368]], 10, 20, 100]
    )
    @unpack
    def test(self, mol_num, j_constant, elocalex, ph_info, ph_phys_dim, evolve_dt, nsteps):
        ph_list = [Phonon.simple_phonon(omega * constant.cm2au, displacement, ph_phys_dim)
                   for omega, displacement in ph_info]
        mol_list = MolList([Mol(elocalex, ph_list)] * mol_num)
        ct = ChargeTransport(mol_list, j_constant, temperature=100).evolve(evolve_dt, nsteps)
        pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHoppingLimitZeroT)
    unittest.TextTestRunner().run(suite)
    #unittest.main()
