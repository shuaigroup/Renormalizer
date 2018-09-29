# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import unittest

import numpy as np
from ddt import ddt, data, unpack

from ephMPS.model import Phonon, Mol, MolList
from ephMPS.transport import ChargeTransport, EDGE_THRESHOLD
from ephMPS.utils import Quantity


@ddt
class TestBandLimitZeroT(unittest.TestCase):

    @data(
        [13, 0.8, 3.87e-3, [[1e-10, 1e-10]], 4, 2, 50]
    )
    @unpack
    def test(self, mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps):
        ph_list = [Phonon.simple_phonon(Quantity(omega, 'cm^{-1}'), Quantity(displacement, 'a.u.'), ph_phys_dim)
                   for omega, displacement in ph_info]
        mol_list = MolList([Mol(Quantity(elocalex_value, 'a.u.'), ph_list)] * mol_num)
        j_constant = Quantity(j_constant_value, 'eV')
        ct = ChargeTransport(mol_list, j_constant)
        ct.stop_at_edge = True
        ct.evolve(evolve_dt, nsteps)
        analytical_r_square = 2 * (j_constant.as_au()) ** 2 * ct.evolve_times_array ** 2
        # has evolved to the edge
        self.assertGreater(ct.latest_mps.e_occupations[0], EDGE_THRESHOLD)
        # value OK
        self.assertTrue(np.allclose(analytical_r_square, ct.r_square_array, rtol=1e-3))

@ddt
class TestSimilar(unittest.TestCase):

    @data(
        [5, 0.8, 3.87e-3, [[1400, 17]], 4, 2, 50]
    )
    @unpack
    def test(self, mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps):
        ph_list = [Phonon.simple_phonon(Quantity(omega, 'cm^{-1}'), Quantity(displacement, 'a.u.'), ph_phys_dim)
                   for omega, displacement in ph_info]
        mol_list = MolList([Mol(Quantity(elocalex_value, 'a.u.'), ph_list)] * mol_num)
        j_constant = Quantity(j_constant_value, 'eV')
        ct1 = ChargeTransport(mol_list, j_constant)
        ct1.evolve(evolve_dt, nsteps)
        ct2 = ChargeTransport(mol_list, j_constant)
        ct2.evolve(evolve_dt + 1e-5, nsteps)
        self.assertTrue(ct1.is_similar(ct2))


@ddt
class TestBandLimitFiniteT(unittest.TestCase):
    @data(
        [3, 1, 3.87e-3, [[1e-5, 1e-5]], 2, 2, 50]
    )
    @unpack
    def test(self, mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps):
        ph_list = [Phonon.simple_phonon(Quantity(omega, 'cm^{-1}'), Quantity(displacement, 'a.u.'), ph_phys_dim)
                   for omega, displacement in ph_info]
        mol_list = MolList([Mol(Quantity(elocalex_value, 'a.u.'), ph_list)] * mol_num)
        j_constant = Quantity(j_constant_value, 'eV')
        ct1 = ChargeTransport(mol_list, j_constant)
        ct1.evolve(evolve_dt, nsteps)
        ct2 = ChargeTransport(mol_list, j_constant, temperature=Quantity(1e-10, 'K'))
        ct2.evolve(evolve_dt, nsteps)
        self.assertTrue(ct1.is_similar(ct2))

@ddt
class TestHoppingLimitZeroT(unittest.TestCase):
    @data(
        [9, 0.02, 3.87e-3, [[1345.6738910804488, 16.274571056529368]], 10, 20, 300]
    )
    @unpack
    def test(self, mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps):
        ph_list = [Phonon.simple_phonon(Quantity(omega, 'cm^{-1}'), Quantity(displacement, 'a.u.'), ph_phys_dim)
                   for omega, displacement in ph_info]
        mol_list = MolList([Mol(Quantity(elocalex_value, 'a.u.'), ph_list)] * mol_num)
        j_constant = Quantity(j_constant_value, 'eV')
        ct = ChargeTransport(mol_list, j_constant)
        ct.stop_at_edge = True
        ct.evolve(evolve_dt, nsteps)
        pass


@ddt
class TestHoppingLimitFiniteT(unittest.TestCase):
    @data(
        [13, 0.02, 3.87e-3, [[1345.6738910804488, 16.274571056529368]], 10, 20, 100]
    )
    @unpack
    def test(self, mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps):
        ph_list = [Phonon.simple_phonon(Quantity(omega, 'cm^{-1}'), Quantity(displacement, 'a.u.'), ph_phys_dim)
                   for omega, displacement in ph_info]
        mol_list = MolList([Mol(Quantity(elocalex_value, 'a.u.'), ph_list)] * mol_num)
        j_constant = Quantity(j_constant_value, 'eV')
        ct = ChargeTransport(mol_list, j_constant, temperature=298).evolve(evolve_dt, nsteps)
        pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBandLimitFiniteT)
    unittest.TextTestRunner().run(suite)
    #unittest.main()
