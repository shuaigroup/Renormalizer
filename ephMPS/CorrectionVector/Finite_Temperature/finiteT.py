# -*- coding: utf-8 -*-
# Author: Tong Jiang <tongjiang1000@gmail.com>
# Excutive file for finite temperature Green Function

from ephMPS.mps import Mpo, MpDm, ThermalProp
from ephMPS.tests.parameter import mol_list
from ephMPS.utils import (constant, Quantity)
import numpy as np
import cv_solver


beta = Quantity(298, unit='K').to_beta()
h_mpo = Mpo(mol_list)
dipole_mpo = Mpo.onsite(mol_list, r"a^\dagger", dipole=True)
i_mpo = MpDm.max_entangled_gs(mol_list)
tp = ThermalProp(i_mpo, h_mpo, exact=True, space='GS')
tp.evolve(None, 1, beta / 2j)
ket_mpo = tp.latest_mps
# ket_mpo.canonical_normalize()
a_ket_mpo = dipole_mpo.apply(ket_mpo, canonicalise=True)
a_ket_mpo.canonical_normalize()
a_bra_mpo = a_ket_mpo.copy()

m_max = 10
eta = 1.e-3
spectratype = 'abs'
method = '1site'
X = Mpo.finiteT_cv(mol_list, 1, m_max,
                   spectratype, percent=1.0)

OMEGA = np.arange(0.05, 0.11, 1.e-3)
spectra = []
spectra = cv_solver.main(OMEGA, X, h_mpo, a_ket_mpo, method,
                         eta, m_max, spectratype)
np.save('spectra.npy', spectra)

