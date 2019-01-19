import numpy as np
from obj import *
import scipy.constants 
from constant import * 
import exact_solver
from MPSsolver import *
import time
from tMPS import *

def main():
    starttime = time.time()
    
    elocalex = 2.13/au2ev
    dipole_abs = 1.0
    nmols = 2
    
    # cm^-1
    J = np.zeros((2,2))
    J += np.diag([-500.0]*1,k=1)
    J += np.diag([-500.0]*1,k=-1)
    J = J * cm2au
    print "J=", J
    
    # cm^-1
    omega_value = np.array([206.0, 211.0, 540.0, 552.0, 751.0, 1325.0, 1371.0, 1469.0, 1570.0, 1628.0])*cm2au
    omega = [{0:x,1:x} for x in omega_value]
    
    S_value = np.array([0.197, 0.215, 0.019, 0.037, 0.033, 0.010, 0.208, 0.042, 0.083, 0.039])
    D_value = np.sqrt(S_value)/np.sqrt(omega_value/2.)
    D = [{0:0.0,1:x} for x in D_value]
    
    print "omega", omega_value*au2ev
    print "J", J*au2ev
    
    nphs = 10
    nlevels =  [5]*nphs
    
    phinfo = [list(a) for a in zip(omega, D, nlevels)]
    
    
    mol = []
    for imol in xrange(nmols):
        mol_local = Mol(elocalex, nphs, dipole_abs)
        mol_local.create_ph(phinfo)
        mol.append(mol_local)
    
    
    nexciton = 0
    procedure = [[20,0.5],[20,0.3],[10,0.2],[5,0],[1,0]]
    
    iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
    
    for ibra in xrange(pbond[0]):
        HMPO[0][0,ibra,ibra,0] -= 0.0766943466108
    
    QNargs = [ephtable, False]
    HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
    
    dipoleMPO, dipoleMPOdim = construct_onsiteMPO(mol, pbond, "a^\dagger",
            dipole=True, QNargs=QNargs)
    GSMPS, GSMPSdim = Max_Entangled_GS_MPS(mol, pbond, QNargs=QNargs)
    GSMPO = hilbert_to_liouville(GSMPS, QNargs=QNargs)
    GSMPO = mpslib.MPSdtype_convert(GSMPO, QNargs=QNargs)
    
    nsteps = 5
    dt = 20.0
    print "energy dE", 1.0/dt/nsteps * au2ev * 2.0 * np.pi
    print "energy E", 1.0/dt * au2ev * 2.0 * np.pi
    
    starttime = time.time()
    autocorr = FiniteT_spectra("abs", mol, pbond, GSMPO, HMPO, dipoleMPO, nsteps, dt, \
            ephtable, thresh=100, temperature=298, algorithm=2,compress_method="svd", QNargs=QNargs)
    endtime = time.time()
    print "Running time=", endtime-starttime
main()
