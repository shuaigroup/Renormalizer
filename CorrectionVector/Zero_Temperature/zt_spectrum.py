# -*- coding: utf-8 -*-

# Excutive file for zero temperature Green Function

import numpy as np
from ephMPS import MPSsolver
from ephMPS import tMPS
from parameter_zt import *
from ephMPS.lib import mps as mpslib
import zt_lib
from ephMPS import constant


procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
    MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
gs_e = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, ephtable, pbond,
                              nexciton, procedure, method="2site")
E_0 = np.min(gs_e)

if spectratype == "abs":
    dipole_type = "a^\dagger"
elif spectratype == "emi":
    dipole_type = "a"
dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond, dipole_type,
                                                   dipole=True)
ketMPS = mpslib.mapply(dipoleMPO, MPS)
B = mpslib.scale(ketMPS, -eta)

X, Xdim, XQN = MPSsolver.construct_MPS("L", ephtable, pbond, 1-nexciton, Mmax)
with open('X.npy', 'wb') as x_handle:
    np.save(x_handle, X)
with open('Xdim.npy', 'wb') as dim_handle:
    np.save(dim_handle, Xdim)
with open('XQN.npy', 'wb') as qn_handle:
    np.save(qn_handle, XQN)

OMEGA = np.linspace(1.9, 2.7, num=500) / constant.au2ev
RESULT = []
MIN_L = []
OVERLAP = []

result, min_L, overlap = zt_lib.main(OMEGA, MPS, MPO, dipoleMPO, ketMPS, E_0, B,
                                     ephtable, pbond, method, Mmax, spectratype)
RESULT.append(result)
MIN_L.append(min_L)
OVERLAP.append(overlap)
with open('result.npy', 'wb') as f_handle:
    np.save(f_handle, RESULT)
with open('min_l.npy', 'wb') as a_handle:
    np.save(a_handle, MIN_L)
with open('overlap.npy', 'wb') as o_handle:
    np.save(o_handle, OVERLAP)
