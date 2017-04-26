#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import exact_solver
from obj import *


nmols = 3
mol = []
elocalex = 0.0
nphs = 2
dipole = 2.0
#phinfo = [[2.0,3.0,2],[4.0,5.0,2]]
phinfo = [[1.1,0.5/1.1,3],[1.1,0.5/1.1,3]]
#phinfo = [[3,3,3]]
J = -1.1*np.ones((nmols,nmols))

for imol in xrange(nmols):
    mol_local = Mol(elocalex, nphs, dipole)
    mol_local.create_ph(phinfo)
    mol.append(mol_local)

ix, iy, ie, ic, iph_dof_list, inconfigs = exact_solver.Exact_Diagonalization_Solver(0, mol, J)
fx, fy, fe, fc, fph_dof_list, fnconfigs = exact_solver.Exact_Diagonalization_Solver(1, mol, J)

print ic[:,0]
AC =  exact_solver.dipoleC(mol, ic[:,0], inconfigs, iph_dof_list, ix, iy, \
        fnconfigs, fph_dof_list, fx, fy)
print "AC=", AC

exact_solver.full_diagonalization_spectrum(ic,ie,ix,iy,fc,fe,fx,fy,iph_dof_list,mol)
