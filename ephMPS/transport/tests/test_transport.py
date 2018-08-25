# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import unittest

from ddt import ddt, data, unpack

from ephMPS.transport import ChargeTransport
from ephMPS.model import Phonon, Mol, MolList
from ephMPS import constant


@ddt
class TestChargeTransport(unittest.TestCase):

    @data(
        [33, 0.8, 3.87e-3, [[1345.6738910804488, 16.274571056529368]], 4, 0.4, 200]
    )
    @unpack
    def test_zero_t_charge_transport(self, mol_num, j_constant, elocalex, ph_info, ph_phys_dim, evolve_dt, nsteps):
        ph_list = [Phonon.simple_phonon(omega * constant.cm2au, displacement, ph_phys_dim)
                   for omega, displacement in ph_info]
        mol_list = MolList([Mol(elocalex, ph_list)] * mol_num)
        ct = ChargeTransport(mol_list, j_constant)
        ct.evolve(evolve_dt, nsteps)
        pass

if __name__ == '__main__':
    unittest.main()