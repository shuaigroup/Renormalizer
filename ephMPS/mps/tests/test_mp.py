# -*- coding: utf-8 -*-

import os

import numpy as np

from ephMPS.mps import Mps, Mpo
from ephMPS.tests import parameter

def test_save_load():
    mps = Mpo.onsite(parameter.mol_list, "a^\dagger", mol_idx_set={0}).apply(Mps.gs(parameter.mol_list, False))
    mpo = Mpo(parameter.mol_list)
    mps1 = mps.copy()
    for i in range(2):
        mps1 = mps1.evolve(mpo, 10)
    mps2 = mps.evolve(mpo, 10)
    fname = "test.npz"
    mps2.dump(fname)
    mps2 = Mps.load(parameter.mol_list, fname)
    mps2 = mps2.evolve(mpo, 10)
    assert np.allclose(mps1.e_occupations, mps2.e_occupations)
    os.remove(fname)
