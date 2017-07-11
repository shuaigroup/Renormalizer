#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from configidx import *
import itertools
from obj import *

# n-particle approximation
# (n-1) GS state has phonon excitation

def construct_config_dic(mol, nexciton,  nparticle=False):
# construct the configuration dictionary
    '''
    The structure of the string is [0/1,0/1] nexciton items
    ((el1ph1,el1ph2,...),(el2ph1,el2ph2,...)...)
    1. exact diagonalization
    '''
    
    nmols = len(mol)
    assert nparticle <= nmols

    if nparticle == False:
        ngs_vib_allow = nmols - nexciton
    else:
        ngs_vib_allow = nparticle - nexciton
    
    config_dic = bidict({})
    config_dic_key = -1
    
    for exlist in itertools.combinations(range(nmols), nexciton):
        
        exiconfig = [1 if x in exlist else 0 for x in xrange(nmols)]

        ex_vib_allow = [i for i,j in enumerate(exiconfig) if j != 0]
        gs_vib = [i for i,j in enumerate(exiconfig) if j == 0]
        for gs_vib_allow in itertools.combinations(gs_vib, ngs_vib_allow):
            vib_allow = sorted(ex_vib_allow + list(gs_vib_allow))
            
            phlist = []
            for imol in xrange(nmols):
                for iph in xrange(mol[imol].nphs):
                    if imol in vib_allow:
                        phlist.append(range(mol[imol].ph[iph].nlevels)) 
                    else:
                        phlist.append([0]) 
            for phiconfig in itertools.product(*phlist):
                newconfig = tuple(exiconfig + list(phiconfig))
                if newconfig not in config_dic.inverse:
                    config_dic_key += 1
                    config_dic[config_dic_key] = newconfig

    assert len(config_dic) == len(config_dic.inverse)
    
    return config_dic
    

if __name__ == '__main__':
    
    import exact_solver
    from constant import * 
    from obj import *

    elocalex = 2.67/au2ev
    dipole_abs = 15.45
    nmols = 3
    # eV
    #J = np.zeros((2,2))
    #J += np.diag([0.1],k=1)
    #J += np.diag([0.1],k=-1)
    J = np.zeros((3,3))
    print "J=", J
    
    # cm^-1
    omega1 = np.array([106.51])
    
    # a.u.
    D1 = np.array([30.1370])
    
    # 1
    S1 = np.array([0.2204])

    # cm^-1
    #omega1 = np.array([106.51, 1555.55])
    #
    ## a.u.
    #D1 = np.array([30.1370, 8.7729])
    #
    ## 1
    #S1 = np.array([0.2204, 0.2727])
    
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
    
    
    nphs = 1
    nlevels =  [2]
    
    phinfo = [list(a) for a in zip(omega1, nphcoup1, nlevels)]
    
    mol = []
    for imol in xrange(nmols):
        mol_local = Mol(elocalex, nphs, dipole_abs)
        mol_local.create_ph(phinfo)
        mol.append(mol_local)
    
    fx, fy, fph_dof_list, fnconfigs = exact_solver.pre_Hmat(1, mol)
    
    construct_config_dic(mol, 1, nparticle=1)


        



