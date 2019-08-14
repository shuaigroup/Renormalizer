import ft_lib
import numpy as np
import MPSsolver
from ephMPS.lib import mps as mpslib
from ephMPS import tMPS
import constant
from parameter_ft import *
import multiprocessing

# Construct MPO
procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
    MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], 0)


# Contruct dipoleMPO and density operator
if spectratype == "abs":

    dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond, "a^\dagger", dipole=True)

    GSMPS, GSMPSdim = tMPS.Max_Entangled_GS_MPS(mol, pbond)
    GSMPO = tMPS.hilbert_to_liouville(GSMPS)
    thermalMPO, thermalMPOdim = tMPS.ExactPropagatorMPO(mol, pbond, -beta/2.0)
    ketMPO = mpslib.mapply(thermalMPO, GSMPO)

else:
    dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond, "a", dipole=True)
    dipoleMPO = mpslib.MPSdtype_convert(dipoleMPO)

    iMPO, iMPOdim = tMPS.Max_Entangled_EX_MPO(mol, pbond, norm=True)
    iMPO = mpslib.MPSdtype_convert(iMPO)
    insteps = 50
    ketMPO = tMPS.thermal_prop(iMPO, MPO, insteps, ephtable,
                               prop_method="C_RK4",
                               thresh=1e-3, temperature=temperature,
                               compress_method="svd")
ket = mpslib.norm(ketMPO)
ketMPO = mpslib.scale(ketMPO, 1./ket)
dipoleMPO = mpslib.mapply(dipoleMPO, ketMPO)


Identity, DIM = tMPS.Max_Entangled_MPS(mol, pbond)
Identity = tMPS.hilbert_to_liouville(Identity)


XX, XXdim, XXQN = ft_lib.construct_X(ephtable, pbond, 1, Mmax, spectratype)
with open('XX.npy', 'wb') as f_handle:
    np.save(f_handle, XX)
with open('XXdim.npy', 'wb') as f_handle:
    np.save(f_handle, XXdim)
with open('XXQN.npy', 'wb') as f_handle:
    np.save(f_handle, XXQN)

RE = []
OMEGA = np.linspace(constant.nm2au(610),constant.nm2au(580),num=100)
RESULT = ft_lib.main(OMEGA, MPO, dipoleMPO, ephtable, pbond, method, Mmax, spectratype)
np.save('RESULT.npy', RESULT)


# cores = multiprocessing.cpu_count()
# pool = multiprocessing.Pool(processes=cores)
# OMEGA_list = [OMEGA[i:i+5] for i in range(0, len(OMEGA), 5)]
# OMEGA = OMEGA_list
# for re in pool.imap(mod_run_no3.main, OMEGA):
#     RE.append(re)






