# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import parameter_PBI
from ephMPS import MPSsolver
from ephMPS import RK
from ephMPS import hybrid_TDDMRG_TDH
from ephMPS import tMPS
from ephMPS import TDH
from ephMPS.lib import mps as mpslib


mol, J = parameter_PBI.construct_mol(1,nphs=10)
TDH.construct_Ham_vib(mol, hybrid=True)

nexciton = 1
dmrg_procedure = [[20,0.5],[20,0.3],[40,0.2],[40,0],[40,0]]

MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
    MPSsolver.construct_MPS_MPO_2(mol, J, dmrg_procedure[0][0], nexciton)

MPS, MPSQN, WFN, Etot = hybrid_TDDMRG_TDH.hybrid_DMRG_H_SCF(mol, J, \
        nexciton, dmrg_procedure, 20, DMRGthresh=1e-7, Hthresh=1e-7)

WFN = [wfn.astype(np.complex128) for wfn in WFN[:-1]]+[WFN[-1]]
MPS = mpslib.MPSdtype_convert(MPS)

MPOs = []
dipoleMPO, dipoleMPOdim = MPSsolver.construct_onsiteMPO(mol, pbond, "a^\dagger a", sitelist=[0])
MPOs.append(dipoleMPO)

rk = RK.Runge_Kutta(method="RKF45", rtol=1e-4, adaptive=True)
setup = tMPS.prop_setup(rk)

nsteps = 100
dt = 0.1

tlist, data = hybrid_TDDMRG_TDH.dynamics_hybrid_TDDMRG_TDH(setup, mol, J, MPS, \
        WFN, nsteps, dt, ephtable, thresh=1e-3, property_MPOs=MPOs)
        
np.save("data",data)
np.save("tlist",tlist)
