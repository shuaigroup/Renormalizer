#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import exact_solver
from obj import *
import matplotlib.pyplot as plt

elocalex = 2.67/27.211
dipole_abs = 15.45
nmols = 1
# eV
J = np.zeros((1))

# cm^-1
omega1 = np.array([106.51])

# a.u.
D1 = np.array([30.1370])

# 1
S1 = np.array([0.2204])

# transfer all these parameters to a.u
# ev to a.u.
J = J/27.2107
# cm^-1 to a.u.
omega1 = omega1 / 219474.63
print "omega1", omega1*27.211

nphcoup1 = np.sqrt(omega1/2.0)*D1

print "Huang", S1
print nphcoup1**2


nphs = 1
nlevels =  [5]

phinfo = [list(a) for a in zip(omega1, nphcoup1, nlevels)]

print phinfo

mol = []
for imol in xrange(nmols):
    mol_local = Mol(elocalex, nphs, dipole_abs)
    mol_local.create_ph(phinfo)
    mol.append(mol_local)

ix, iy, ie, ic, iph_dof_list, inconfigs, iHmat = exact_solver.Exact_Diagonalization_Solver(0, mol, J)
fx, fy, fe, fc, fph_dof_list, fnconfigs, fHmat = exact_solver.Exact_Diagonalization_Solver(1, mol, J)

# exact method
dipdip = exact_solver.full_diagonalization_spectrum(ic,ie,ix,iy,fc,fe,fx,fy,iph_dof_list,mol)
exact_solver.absorption(dipdip, 298, ie)
exact_solver.absorption(dipdip, 0, ie)
#exact_solver.emission(dipdip, 0.01, fe)
#exact_solver.emission(dipdip, 0, fe)

# lanczos method T=0
dipolemat = exact_solver.construct_dipoleMat(inconfigs,ix,iy,fnconfigs,fx,fy,iph_dof_list,mol)

AC = exact_solver.dipoleC(mol, ic[:,0], inconfigs, iph_dof_list, ix, iy, \
        fnconfigs, fph_dof_list, fx, fy)

dyn_omega = np.linspace(2.6, 2.8, num=500)
dyn_omega /=  27.211
dyn_corr = exact_solver.dyn_lanczos(0.0, AC, dipolemat, iHmat, fHmat, dyn_omega,\
        ie[0], eta=0.00005)


# lanczos method T>0
dyn_corr = exact_solver.dyn_lanczos(298.0, AC, dipolemat, iHmat, fHmat,\
        dyn_omega, ie[0], eta=0.00005)

plt.plot(dyn_omega * 27.211, dyn_corr*dyn_omega)
plt.show()

