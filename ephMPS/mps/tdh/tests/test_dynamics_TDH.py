# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import logging
import os

import numpy as np

from ephMPS.mps.tdh import tdh
from ephMPS.tests import parameter_PBI
from ephMPS.mps.tdh.tests import cur_dir
from ephMPS.utils import log, Quantity


def test_ZT_dynamics_TDH():

    log.init_log(logging.WARNING)

    mol_list = parameter_PBI.construct_mol(4, dmrg_nphs=0, hartree_nphs=10)

    operators = []
    for imol in range(len(mol_list)):
        dipoleO = tdh.construct_onsiteO(
            mol_list, r"a^\dagger a", dipole=False, mol_idx_set={imol}
        )
        operators.append(dipoleO)

    nsteps = 100 - 1
    dt = 10.0
    dynamics = tdh.Dynamics(mol_list, property_ops=operators)
    dynamics.evolve(dt, nsteps)
    with open(os.path.join(cur_dir, "ZT_occ10.npy"), "rb") as f:
        std = np.load(f)
    assert np.allclose(dynamics.properties, std)


def test_FT_dynamics_TDH():

    log.init_log(logging.WARNING)

    mol_list = parameter_PBI.construct_mol(4, dmrg_nphs=0, hartree_nphs=10)

    operators = []
    for imol in range(len(mol_list)):
        dipoleO = tdh.construct_onsiteO(
            mol_list, r"a^\dagger a", dipole=False, mol_idx_set={imol}
        )
        operators.append(dipoleO)

    T = Quantity(2000, "K")
    insteps = 1
    dynamics = tdh.Dynamics(
        mol_list, property_ops=operators, temperature=T, insteps=insteps
    )
    nsteps = 300 - 1
    dt = 10.0
    dynamics.evolve(dt, nsteps)

    with open(os.path.join(cur_dir, "FT_occ10.npy"), "rb") as f:
        std = np.load(f)
    assert np.allclose(dynamics.properties, std)
