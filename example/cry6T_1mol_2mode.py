#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import exact_solver
from obj import *
import matplotlib.pyplot as plt
import scipy.constants 
from constant import * 
import benchmark

elocalex = 2.67/au2ev
dipole_abs = 15.45
nmols = 1
# eV
J = np.zeros((1))

# cm^-1
omega1 = np.array([106.51, 1555.55])

# a.u.
D1 = np.array([30.1370, 8.7729])

# 1
S1 = np.array([0.2204, 0.2727])

# transfer all these parameters to a.u
# ev to a.u.
J = J/au2ev
# cm^-1 to a.u.
omega1 = omega1 * 1.0E2 * \
scipy.constants.physical_constants["inverse meter-hertz relationship"][0] / \
scipy.constants.physical_constants["hartree-hertz relationship"][0]

print "omega1", omega1*au2ev

nphcoup1 = np.sqrt(omega1/2.0)*D1

print "Huang", S1
print nphcoup1**2


nphs = 2
nlevels =  [4,4]

phinfo = [list(a) for a in zip(omega1, nphcoup1, nlevels)]

print phinfo

mol = []
for imol in xrange(nmols):
    mol_local = Mol(elocalex, nphs, dipole_abs)
    mol_local.create_ph(phinfo)
    mol.append(mol_local)

dyn_omega = np.linspace(2.2, 3.1, num=500)
benchmark.benchmark(mol, J, dyn_omega, T=298.0, eta=0.00005, nsamp=300, M=100,
        outfile="1mol_2mode.eps")
