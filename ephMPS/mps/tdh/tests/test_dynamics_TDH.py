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

    log.init_log(logging.INFO)
    sites = 4
    mol_list = parameter_PBI.construct_mol(sites, dmrg_nphs=0, hartree_nphs=10)

    operators = []
    nsteps = 100 - 1
    dt = 10.0
    dynamics = tdh.Dynamics(mol_list, property_ops=operators, init_idx=0)
    dynamics.info_interval = 50
    dynamics.evolve(dt, nsteps)
    electrons = np.array([wfn[0] for wfn in dynamics.tdmps_list])
    electrons = (electrons.conj() * electrons).real
    # debug code to calculate r
    # r_list = []
    # for electron in electrons:
    #     r = np.average(np.arange(sites)**2, weights=electron) - np.average(np.arange(sites), weights=electron) ** 2
    #     r_list.append(r)
    with open(os.path.join(cur_dir, "ZT_occ10.npy"), "rb") as f:
        std = np.load(f)
    assert np.allclose(electrons.T, std, atol=1e-2)


#from matplotlib import pyplot as plt
#plt.plot(r_list); plt.show()
#import seaborn as sns
#sns.heatmap(electrons, cmap="ocean_r")
#plt.show()

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
    assert np.allclose(dynamics.properties, std, atol=1e-2)
