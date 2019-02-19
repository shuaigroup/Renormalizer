# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import os

import numpy as np
import pytest

from ephMPS.spectra import SpectraOneWayPropZeroT, SpectraTwoWayPropZeroT, SpectraExact
from ephMPS.spectra.tests import cur_dir
from ephMPS.tests import parameter
from ephMPS.utils import Quantity


@pytest.mark.parametrize("algorithm, mol_list, std_fname, rtol",
                         (
                                 # [1, parameter.hybrid_mol_list,"hybrid_ZTabs.npy",1e-2],
                                 # [1, parameter.mol_list,"ZeroTabs_2svd.npy",1e-2],
                                 [2, parameter.hybrid_mol_list, "hybrid_ZTabs.npy", 1e-3],
                                 [2, parameter.mol_list, "ZeroTabs_2svd.npy", 1e-2],
                         )
                         )
def test_hybrid_abs(algorithm, mol_list, std_fname, rtol):
    np.random.seed(0)
    # print "data", value

    if algorithm == 1:
        SpectraZeroT = SpectraOneWayPropZeroT
    else:
        SpectraZeroT = SpectraTwoWayPropZeroT

    zero_t_corr = SpectraZeroT(mol_list, "abs", offset=Quantity(2.28614053, 'ev'))
    zero_t_corr.info_interval = 30
    nsteps = 100
    dt = 30.0
    zero_t_corr.evolve(dt, nsteps)
    with open(os.path.join(cur_dir, std_fname), 'rb') as f:
        std = np.load(f)
    assert np.allclose(zero_t_corr.autocorr[:nsteps], std[:nsteps], rtol=rtol)


# from matplotlib import pyplot as plt
# plt.plot(zero_t_corr.autocorr)
# plt.plot(std)
# plt.show()

@pytest.mark.parametrize("algorithm, mol_list, std_fname, rtol",
                         (
                                 [1, parameter.hybrid_mol_list, "hybrid_ZTemi_prop.npy", 1e-3],
                                 [1, parameter.mol_list, "ZeroExactEmi.npy", 1e-2],
                                 [2, parameter.hybrid_mol_list, "hybrid_ZTemi_prop.npy", 1e-3],
                                 [2, parameter.mol_list, "ZeroExactEmi.npy", 1e-2]
                         )
                         )
def test_hybrid_emi(algorithm, mol_list, std_fname, rtol):
    np.random.seed(0)
    # print "data", value
    if algorithm == 1:
        SpectraZeroT = SpectraOneWayPropZeroT
    else:
        SpectraZeroT = SpectraTwoWayPropZeroT

    zero_t_corr = SpectraZeroT(mol_list, "emi")
    zero_t_corr.info_interval = 100
    nsteps = 1000
    dt = 30.0
    zero_t_corr.evolve(dt, nsteps)
    with open(os.path.join(cur_dir, std_fname), 'rb') as f:
        std = np.load(f)
    assert np.allclose(zero_t_corr.autocorr[:nsteps], std[:nsteps], rtol=rtol)


def test_Exact_Spectra_hybrid_TDDMRG_TDH():
    # print "data", value
    exact_emi = SpectraExact(parameter.hybrid_mol_list, spectratype='emi')
    exact_emi.info_interval = 100
    nsteps = 3000
    # nsteps = 50
    dt = 30.0
    exact_emi.evolve(dt, nsteps)
    with open(os.path.join(cur_dir, 'hybrid_ZTemi_exact.npy'), 'rb') as fin:
        std = np.load(fin)

    assert np.allclose(exact_emi.autocorr[:nsteps], std[:nsteps], rtol=1e-5)


@pytest.mark.parametrize("algorithm",
                         (
                                 "pure",
                                 "hybrid")
                         )
def test_1mol_Exact_Spectra_hybrid_TDDMRG_TDH(algorithm):
    nmols = 1
    J = np.zeros([1, 1])

    if algorithm == "pure":
        mol_list = parameter.custom_mol_list(J, nmols=nmols)
    elif algorithm == "hybrid":
        mol_list = parameter.custom_mol_list(J, hartrees=[True, False], nmols=nmols)
    else:
        assert False

    E_offset = - mol_list[0].elocalex - mol_list[0].reorganization_energy
    exact_abs = SpectraExact(mol_list, spectratype='abs', offset=Quantity(E_offset))
    exact_abs.info_interval = 100
    nsteps = 1000
    dt = 30.0
    exact_abs.evolve(dt, nsteps)

    with open("1mol_ZTabs.npy", 'rb') as f:
        mol1_ZTabs_std = np.load(f)

    assert np.allclose(exact_abs.autocorr[:nsteps], mol1_ZTabs_std[:nsteps], rtol=1e-3)
