# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import logging
import os

import numpy as np

from renormalizer.mps.tdh import tdh
from renormalizer.tests import parameter_PBI
from renormalizer.mps.tdh.tests import cur_dir
from renormalizer.utils import log, Quantity


def test_ZT_dynamics_TDH():

    log.init_log(logging.INFO)
    sites = 4
    mol_list = parameter_PBI.construct_mol(sites, dmrg_nphs=0, hartree_nphs=10)

    nsteps = 100 - 1
    dt = 10.0
    dynamics = tdh.Dynamics(mol_list, init_idx=0)
    dynamics.info_interval = 50
    dynamics.evolve(dt, nsteps)
    with open(os.path.join(cur_dir, "ZT_occ10.npy"), "rb") as f:
        std = np.load(f)
    assert np.allclose(dynamics.e_occupations_array, std.T, atol=1e-2)


#from matplotlib import pyplot as plt
#plt.plot(r_list); plt.show()
#import seaborn as sns
#sns.heatmap(electrons, cmap="ocean_r")
#plt.show()

def test_FT_dynamics_TDH():

    log.init_log(logging.WARNING)

    mol_list = parameter_PBI.construct_mol(4, dmrg_nphs=0, hartree_nphs=10)

    T = Quantity(2000, "K")
    insteps = 1
    dynamics = tdh.Dynamics(mol_list, temperature=T, insteps=insteps)
    nsteps = 300 - 1
    dt = 10.0
    dynamics.evolve(dt, nsteps)

    with open(os.path.join(cur_dir, "FT_occ10.npy"), "rb") as f:
        std = np.load(f)
    assert np.allclose(dynamics.e_occupations_array, std.T, atol=1e-2)
