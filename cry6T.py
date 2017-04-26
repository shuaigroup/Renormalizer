#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import exact_solver
from obj import *

elocalex = 2.67/27.211
dipole_abs = 15.45
nmols = 1
# eV
#J = np.diag([0.096435, 0.096435, 0.096435, 0.096435, 0.096435, 0.096435, \
#    0.096435, 0.096435], k=-1)
#J += np.diag([0.030193, 0.088265, 0.030452, 0.041970, 0.030193, 0.096435, \
#    0.041970], k=-2)
#J += np.diag([0.039978, 0.039978, 0.030193, 0.030452, 0.039978, 0.096435], k=-3)
#J += np.diag([0.030193, 0.030193, 0.007255, 0.030193, 0.088265], k=-4)
#J += np.diag([0.039978, 0.030452, 0.030452, 0.096435], k=-5)
#J += np.diag([0.041970, 0.041970, 0.041970], k=-6)
#J += np.diag([0.096435, 0.096435], k=-7)
#J += np.diag([0.088265], k=-8)
#J = np.diag([0.096435], k=-1)
#J = J + np.transpose(J)
J = np.zeros((1))

# cm^-1
omega1 = np.array([102.50, 106.51, 1487.35, 1555.55, 1570.37])

# a.u.
D1 = np.array([-0.2274, 30.1370, -0.0948, 8.7729, 8.6450])

# 1
S1 = np.array([0.0000, 0.2204, 0.0000, 0.2727, 0.2674])

# transfer all these parameters to a.u
# ev to a.u.
J = J/27.2107
# cm^-1 to a.u.
omega1 = omega1 / 219474.63
print "omega1", omega1*27.211

nphcoup1 = np.sqrt(omega1/2.0)*D1

print "Huang", S1
print nphcoup1**2


nphs = 5
nlevels =  [2,4,2,4,4]

phinfo = [list(a) for a in zip(omega1, nphcoup1, nlevels)]

print phinfo

mol = []
for imol in xrange(nmols):
    mol_local = Mol(elocalex, nphs, dipole_abs)
    mol_local.create_ph(phinfo)
    mol.append(mol_local)

ix, iy, ie, ic, iph_dof_list, inconfigs = exact_solver.Exact_Diagonalization_Solver(0, mol, J)
fx, fy, fe, fc, fph_dof_list, fnconfigs = exact_solver.Exact_Diagonalization_Solver(1, mol, J)

#print ic[:,0]
#AC =  exact_solver.dipoleC(mol, ic[:,0], inconfigs, iph_dof_list, ix, iy, \
#        fnconfigs, fph_dof_list, fx, fy)
#print "AC=", AC

exact_solver.full_diagonalization_spectrum(ic,ie,ix,iy,fc,fe,fx,fy,iph_dof_list,mol)
