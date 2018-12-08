# -*- coding: utf-8 -*-
#  Use  Chebyshve expansion to expand the delta function in green function

import numpy as np
import scipy
from ephMPS.lib import mps as mpslib
from ephMPS import tMPS
from ephMPS import obj
import copy
# from parameter import *
import constant
import MPSsolver
import matplotlib.pyplot as plt
elocalex = 2.13/constant.au2ev
dipole_abs = 1.0
nmols = 2


J = np.zeros((2,2))
J += np.diag([-500.0]*1,k=1)
J += np.diag([-500.0]*1,k=-1)
J = J * constant.cm2au
print "J=", J


omega_value = np.array([206.0, 211.0, 540.0, 552.0, 751.0, 1325.0, 1371.0, 1469.0, 1570.0, 1628.0])*constant.cm2au
omega = [{0:x,1:x} for x in omega_value]

S_value = np.array([0.197, 0.215, 0.019, 0.037, 0.033, 0.010, 0.208, 0.042, 0.083, 0.039])
D_value = np.sqrt(S_value)/np.sqrt(omega_value/2.)
D = [{0:0.0,1:x} for x in D_value]

print "omega", omega_value*constant.au2ev
print "J", J*constant.au2ev

nphs = 10
nlevels =  [5]*nphs

phinfo = [list(a) for a in zip(omega, D, nlevels)]


mol = []
for imol in xrange(nmols):
    mol_local = obj.Mol(elocalex, nphs, dipole_abs)
    mol_local.create_ph(phinfo)
    mol.append(mol_local)

elocalex = 1.0
dipole_abs = 1.0
nmols = 1
nphs = 1
nexciton = 0
'''
'#J = np.array([[0.0,0.1],[0.1,0.0]])'
'#J = np.array([[0.0,-0.1],[-0.1,0.0]])'
'''
J = np.array([[0.0, 0.1], [0.1, 0.0]])
# J = 0
omega_value = np.array([0.1])
omega = [{0: x, 1: x} for x in omega_value]
S_value = np.array([1.0])
D_value = np.sqrt(S_value) / np.sqrt(omega_value / 2.0)
D = [{0: 0.0, 1: x} for x in D_value]
nlevels = [2]
phinfo = [list(a) for a in zip(omega, D, nlevels)]
spectratype = "abs"
assert spectratype in ["abs", "emi"]
temperature = 0.1 * constant.au2K
# temperature = 298.0
mol = []
for imol in range(nmols):
    mol_local = obj.Mol(elocalex, nphs, dipole_abs)
    mol_local.create_ph(phinfo)
    mol.append(mol_local)


temperature = 298.0
spectratype = "abs"
nexciton = 0
# Construct MPO
procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
    MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
# MPO = mpslib.MPSdtype_convert(MPO)
print('pbond', pbond)
print('MPOdim', MPOdim)

if spectratype == "abs":
    beta = constant.T2beta(temperature)
    dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond, "a^\dagger", dipole=True)
    dipoleMPO = mpslib.MPSdtype_convert(dipoleMPO)
    GSMPS, GSMPSdim = tMPS.Max_Entangled_GS_MPS(mol, pbond)
    GSMPO = tMPS.hilbert_to_liouville(GSMPS)
    GSMPO = mpslib.MPSdtype_convert(GSMPO)
    thermalMPO, thermalMPOdim = tMPS.ExactPropagatorMPO(mol, pbond, -beta/2.0)
    ketMPO = mpslib.mapply(thermalMPO, GSMPO)
else:
    dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond, "a", dipole="True")
    iMPO, iMPOdim = tMPS.Max_Entangled_EX_MPO(mol, pbond, norm=True)
    iMPO = mpslib.MPSdtype_convert(iMPO)
    insteps = 50
    ketMPO = tMPS.thermal_prop(iMPO, MPO, insteps, ephtable, prop_method="C_RK4",
                            thresh=1e-3, temperature=temperature, compress_method="svd")
# construct the L operator and scale it into interval [-1, 1]
L = copy.deepcopy(MPO)
# set scaling factors, assuming interval [-W/2, W/2]
#OMEGA = np.linspace(1.4, 3.0, num=1000) / constant.au2ev
#OMEGA = OMEGA[300:680, ]
OMEGA = np.arange(0.1, 3.0, 0.01)

W = OMEGA[-1] - OMEGA[0]
epsi = 0.025
W_prime = 1 - 0.5 * epsi
a = W / (W_prime)
OMEGA_prime = (OMEGA - OMEGA[0]) / a

for ibra in range(pbond[0]):
    L[0][0, ibra, ibra, 0] -= OMEGA[0]
L_prime = mpslib.scale(L, 1 / a)
# for ibra in range(pbond[0]):
#     L_prime[0][0, ibra, ibra, 0] -= W_prime

def L_apply(x):
    Lx = mpslib.contract(L_prime, x, 'l', 1.e-3, compress_method='svd')
    xL = mpslib.contract(x, L_prime, 'l', 1.e-3, compress_method='svd')
    L_x = mpslib.add(Lx, mpslib.scale(xL, -1))
    L_x = mpslib.canonicalise(L_x, 'l')
    L_x = mpslib.compress(L_x, 'l', trunc=1.e-3)
    return L_x


# use recursion relation to generate series |t_n\rangle
t_overlap = []
t_0 = mpslib.mapply(dipoleMPO, ketMPO)
t_1 = L_apply(t_0)
t_00 = mpslib.dot(mpslib.conj(t_0), t_0)
t_01 = mpslib.dot(mpslib.conj(t_0), t_1)
t_overlap.append(t_00)
t_overlap.append(t_01)
t_nm1 = copy.deepcopy(t_1)
t_nm2 = copy.deepcopy(t_0)


# we try to set interval to test the neccesity of going on increasing N
def calc_tn(len_interval, t_overlap, t_nm1, t_nm2):
    start = len(t_overlap)
    if start == 2:
        len_interval -=  2
    for i in range(start, start + len_interval):
        print('now generate t_n series', i)
        temp = mpslib.scale(L_apply(t_nm1), 2)
        t_n = mpslib.add(temp, mpslib.scale(t_nm2, -1))
        t_n = mpslib.canonicalise(t_n, 'l')
        t_n = mpslib.compress(t_n, 'l', trunc=1.e-3)
        t_nm2 = copy.deepcopy(t_nm1)
        t_nm1 = copy.deepcopy(t_n)
        t_0i = mpslib.dot(mpslib.conj(t_0), t_n)
        t_overlap.append(t_0i)
    print('Bonddim')
    print[x.shape for x in t_n]
    return t_overlap, t_nm1, t_nm2


# now we calculate the Green function
max_N = 100000
len_interval = 100

G = np.zeros(shape=(len(OMEGA_prime), max_N / len_interval))
# t_overlap, t_nm1, t_nm2 = calc_tn(len_interval, t_overlap, t_nm1, t_nm2)
num_omega = 0
for i_column in range(G.shape[1]):
    print('generating' , i_column, " * 2000 now")
    t_overlap, t_nm1, t_nm2 = calc_tn(len_interval, t_overlap, t_nm1, t_nm2)
    g = []
    for i in range(len(t_overlap)):
        g_i = (len(t_overlap) - i + 1) * np.cos(i * np.pi / (len(t_overlap) + 1)) + np.sin(i * np.pi / (len(t_overlap) + 1)) / np.tan(np.pi / (len(t_overlap) + 1))
        g.append(g_i / (len(t_overlap) + 1))
    num_omega = 0
    for omega in OMEGA_prime:
        print('calculationg omega', omega)
        t_omega = [1, omega]
        for n in range(2, max_N):
            t_omega.append(2 * omega * t_omega[n - 1] - t_omega[n - 2])
        G_old = g[0] * t_overlap[0]
        for i in range(1, len(t_overlap)):
            G_new = G_old + 2 * g[i] * t_overlap[i] * t_omega[i]
            G_old = G_new
        G[num_omega][i_column] = (1. / np.sqrt(1 - omega * omega) * G_new)
        num_omega += 1
    reno = np.zeros(shape=(len(OMEGA_prime), (i_column + 1)))
    for i in range(reno.shape[1]):
        reno[:, i] = G[:, i] / max(G[:, i])
    with open('cheb_interval.npy', 'wb') as f_handle:
        np.save(f_handle, reno)









