# -*- coding: utf-8 -*-
# Author: Tong Jiang <tongjiang1000@gmail.com>
# Excutive file for zero temperature Green Function

from ephMPS.mps import Mpo, Mps
from ephMPS.mps.solver import construct_mps_mpo_2, optimize_mps
from ephMPS.tests.parameter import mol_list
import cv_solver
from ephMPS.utils import constant
import numpy as np
# import multiprocessing


# nexciton = 0  # absorption"
procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
mps, mpo = construct_mps_mpo_2(mol_list,
                               procedure[0][0], 0)
mps.optimize_config.procedure = procedure
mps.optimize_config.method = "2site"
E_0 = optimize_mps(mps, mpo)

spectratype = 'abs'
if spectratype == "abs":
    dipole_type = r"a^\dagger"
elif spectratype == "emi":
    dipole_type = "a"

dipole_mpo = Mpo.onsite(mol_list, dipole_type, dipole=True)
ketmps = dipole_mpo.apply(mps, canonicalise=True)
ketmps.canonical_normalize()
eta = 5.e-5
Mmax = 5
B = ketmps.scale(-eta)

X = Mps.random(mol_list, 1, Mmax, percent=1.0)
OMEGA = np.linspace(1.9, 2.7, num=500) / constant.au2ev
OMEGA = np.arange(0.05, 0.11, 5.e-5)
spectra = []
#cores = multiprocessing.cpu_count()
#pool = multiprocessing.Pool(processes=cores)
#OMEGA_list = [OMEGA[i: i+5] for i in range(0, len(OMEGA), 5)]
spectra = cv_solver.main(OMEGA, X, mps, mpo, dipole_mpo, ketmps, E_0, B,
                         '1site', eta, Mmax, spectratype)
np.save('spectra.npy', spectra)
