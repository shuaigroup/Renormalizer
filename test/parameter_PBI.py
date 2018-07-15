import numpy as np
from ephMPS import obj
from ephMPS.constant import *

def construct_mol(nmols, nphs=10):
    elocalex = 2.13/au2ev
    dipole_abs = 1.0
    
    # cm^-1
    J = np.zeros((nmols,nmols))
    J += np.diag([-500.0]*(nmols-1),k=1)
    J += np.diag([-500.0]*(nmols-1),k=-1)
    J = J * cm2au
    print "J=", J
    
    # cm^-1
    omega_value = np.array([206.0, 211.0, 540.0, 552.0, 751.0, 1325.0, 1371.0, 1469.0, 1570.0, 1628.0])*cm2au
    S_value = np.array([0.197, 0.215, 0.019, 0.037, 0.033, 0.010, 0.208, 0.042, 0.083, 0.039])
    
    # sort from large to small
    gw = np.sqrt(S_value)*omega_value
    idx = np.argsort(gw)[::-1]
    omega_value = omega_value[idx]
    S_value = S_value[idx]

    omega = [{0:x,1:x} for x in omega_value]
    D_value = np.sqrt(S_value)/np.sqrt(omega_value/2.)
    D = [{0:0.0,1:x} for x in D_value]
    
    nphs_hybrid = 10-nphs
    nlevels =  [5]*10
    
    print nphs, nphs_hybrid
    phinfo = [list(a) for a in zip(omega[:nphs], D[:nphs], nlevels[:nphs])]
    phinfo_hybrid = [list(a) for a in zip(omega[nphs:], D[nphs:], nlevels[nphs:])]
    
    mol = []
    for imol in xrange(nmols):
        mol_local = obj.Mol(elocalex, nphs, dipole_abs, nphs_hybrid=nphs_hybrid)
        mol_local.create_ph(phinfo)
        mol_local.create_ph(phinfo_hybrid, phtype="hybrid")
        mol.append(mol_local)

    return mol, J
