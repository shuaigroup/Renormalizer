import numpy as np
from ephMPS import obj
from ephMPS import constant
import scipy.linalg

elocalex = 2.67/constant.au2ev
dipole_abs = 15.45
nmols = 3
# eV
J = np.array([[0.0,-0.1,-0.2],[-0.1,0.0,-0.3],[-0.2,-0.3,0.0]])/constant.au2ev
w, v = scipy.linalg.eigh(a=J)
print J, v, w

# cm^-1
omega_value = np.array([1555.55])*constant.cm2au
omega = [{0:omega_value[0],1:omega_value[0]}]
# a.u.
D_value = np.array([8.7729])

D = [{0:0.0,1:D_value[0]}]
nphs = 1
nlevels =  [4]

phinfo = [list(a) for a in zip(omega, D, nlevels)]

mol = []
for imol in xrange(nmols):
    mol_local = obj.Mol(elocalex, nphs, dipole_abs)
    mol_local.create_ph(phinfo)
    mol.append(mol_local)


def construct_mol(nlevels, nqboson=[1], qbtrunc=[0.0], force3rd=[None], D_value=D_value):
    
    D = [{0:0.0,1:D_value[0]}]
    phinfo = [list(a) for a in zip(omega, D, nlevels, force3rd, nqboson, qbtrunc)]
    mol = []
    for imol in xrange(nmols):
        mol_local = obj.Mol(elocalex, nphs, dipole_abs)
        mol_local.create_ph(phinfo)
        mol.append(mol_local)
    
    return mol
