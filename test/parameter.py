import numpy as np
from ephMPS import obj
from ephMPS import constant


elocalex = 2.67/constant.au2ev
dipole_abs = 15.45
nmols = 3
# eV
J = np.array([[0.0,-0.1,-0.2],[-0.1,0.0,-0.3],[-0.2,-0.3,0.0]])/constant.au2ev
# cm^-1
omega = np.array([106.51, 1555.55])*constant.cm2au
# a.u.
D = np.array([30.1370, 8.7729])

nphcoup = np.sqrt(omega/2.0)*D
nphs = 2
nlevels =  [4,4]

phinfo = [list(a) for a in zip(omega, nphcoup, nlevels)]

mol = []
for imol in xrange(nmols):
    mol_local = obj.Mol(elocalex, nphs, dipole_abs)
    mol_local.create_ph(phinfo)
    mol.append(mol_local)
