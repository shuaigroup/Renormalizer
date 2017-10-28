# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
Map the Holstein Hamiltonian from star-representation to chain representation
from a Unitary transformation by Lanczos threee diagonalization
'''
import numpy as np
import ephMPS.lib.mathutils as mathutils
import scipy.linalg
import copy


def Chain_Map_discrete(mol):
    '''
    algorithm see Phys. Rev. B 92, 155126
    '''
    nmols = len(mol)
    Chain = []
    for imol in xrange(nmols):
        v0 = np.zeros(mol[imol].nphs)
        tot = 0.0
        for iph in xrange(mol[imol].nphs):
            Vcoup = mol[imol].ph[iph].ephcoup *\
            mol[imol].ph[iph].omega[1]**2/mol[imol].ph[iph].omega[0]
            tot += Vcoup**2
            v0[iph] = Vcoup  
        
        tot = np.sqrt(tot)
        v0 /= tot
        Omega = np.array([mol[imol].ph[iph].omega[0] for iph in xrange(mol[imol].nphs)])
        
        Alpha, Beta = tri_lanczos(Omega, v0)
        Chain.append((Alpha, Beta, tot))
    
    return Chain
    

def tri_lanczos(omega, v0):
    Alpha = []
    Beta = [0.0,]
    V = [v0, ]
    for iv in xrange(len(omega)):
        alpha = np.sum(omega * V[iv]**2)
        Alpha.append(alpha)
        if iv != len(omega)-1:
            if iv != 0:
                r = omega * V[iv] - Alpha[iv]*V[iv] - Beta[iv]*V[iv-1]
            else:
                r = omega * V[iv] - Alpha[iv]*V[iv]
            r = mathutils.Gram_Schmit(r, V)
            beta = scipy.linalg.norm(r)
            Beta.append(beta)
            V.append(r/beta)

    return Alpha, Beta


def Chain_Mol(Chain, mol):
    '''
    reconstruct the object mol in the chain representation
    '''
    molnew = copy.deepcopy(mol)

    for imol in xrange(len(molnew)):
        Alpha, Beta, tot = Chain[imol]
        molnew[imol].phhop = np.diag(np.array(Beta[1:]),k=1) + \
                np.diag(np.array(Beta[1:]),k=-1)
        
        for iph in xrange(molnew[imol].nphs):
            # only linear e-ph coupling is possible in chain mapping
            for key in molnew[imol].ph[iph].omega.keys():
                molnew[imol].ph[iph].omega[key] = Alpha[iph]
            
            if iph == 0:
                molnew[imol].ph[iph].ephcoup= tot / Alpha[iph]
            else:
                molnew[imol].ph[iph].ephcoup= 0.0
    
    return molnew
