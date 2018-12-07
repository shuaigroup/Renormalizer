# -*- coding: utf-8 -*-

import numpy as np
import constant
from ephMPS import obj
# parameters: testing system 3mol+2phs
'''
elocalex = 2.67/constant.au2ev
dipole_abs = 15.45
nmols = 3
# eV
J = np.array([[0.0, -0.1, -0.2], [-0.1, 0.0, -0.3],
              [-0.2, -0.3, 0.0]])/constant.au2ev
# cm^-1
omega_value = np.array([106.51, 1555.55])*constant.cm2au
Omega = [{0: omega_value[0], 1: omega_value[0]},
         {0: omega_value[1], 1: omega_value[1]}]
# a.u.
D_value = np.array([30.1370, 8.7729])
D = [{0: 0.0, 1: D_value[0]},
     {0: 0.0, 1: D_value[1]}]
nphs = 2
nlevels = [4, 4]

phinfo = [list(a) for a in zip(Omega, D, nlevels)]

mol = []
for imol in xrange(nmols):
    mol_local = obj.Mol(elocalex, nphs, dipole_abs)
    mol_local.create_ph(phinfo)
    mol.append(mol_local)
'''
# parameters: testing system 1mol+1ph
'''
elocalex = 1.0
dipole_abs = 1.0
nmols = 2
nphs = 1
J = np.array([[0.0, 0.1], [0.1, 0.0]])
omega_value = np.array([0.1])
Omega = [{0: x, 1: x} for x in omega_value]
S_value = np.array([1.0])
D_value = np.sqrt(S_value) / np.sqrt(omega_value / 2.0)
D = [{0: 0.0, 1: x} for x in D_value]
nlevels = [4]
phinfo = [list(a) for a in zip(Omega, D, nlevels)]
mol = []
for imol in range(nmols):
    mol_local = obj.Mol(elocalex, nphs, dipole_abs)
    mol_local.create_ph(phinfo)
    mol.append(mol_local)
'''

elocalex = 2.13/constant.au2ev
dipole_abs = 1.0
nmols = 2

# cm^-1
J = np.zeros((2,2))
J += np.diag([-500.0]*1,k=1)
J += np.diag([-500.0]*1,k=-1)
J = J * constant.cm2au


# cm^-1
omega_value = np.array([206.0, 211.0, 540.0, 552.0, 751.0, 1325.0, 1371.0, 1469.0, 1570.0, 1628.0])*constant.cm2au
Omega = [{0:x,1:x} for x in omega_value]

S_value = np.array([0.197, 0.215, 0.019, 0.037, 0.033, 0.010, 0.208, 0.042, 0.083, 0.039])
D_value = np.sqrt(S_value)/np.sqrt(omega_value/2.)
D = [{0:0.0,1:x} for x in D_value]



nphs = 10
nlevels =[5]*nphs


phinfo = [list(a) for a in zip(Omega, D, nlevels)]



mol = []
for imol in range(nmols):
    mol_local = obj.Mol(elocalex, nphs, dipole_abs)
    mol_local.create_ph(phinfo)
    mol.append(mol_local)
spectratype = "abs"
nexciton = 0
eta = 5.e-5
Mmax = 20
method = '1site'
# 2site is better, and not expensive either
