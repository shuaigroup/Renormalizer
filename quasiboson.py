# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
'''
decompose boson site to several quasi-boson site like a n bit integral
then the boson levels cutoff is 2^n
Eric Jeckelmann Phys. Rev. B 57, 6376
'''

import numpy as np
import scipy.linalg
import lib.mps as mpslib
from ephMPS.utils.utils import roundrobin

def baseConvert(n, base):
    '''
    convert 10 base number to any base number
    '''
    result = ''
    while True:
        tup = divmod(n, base)
        result += str(tup[1])
        if tup[0] == 0:
            return result[::-1]
        else:
            n = tup[0]


def Quasi_Boson_MPO(opera, nqb, trunc, base=2):
    '''
    nqb : # of quasi boson sites
    opera : operator to be decomposed
            "b + b^\dagger"
    '''
    assert opera in ["b + b^\dagger","b^\dagger b", "b", "b^\dagger"]
    
    # the structure is [bra_highest_bit, ket_highest_bit,..., bra_lowest_bit,
    # ket_lowest_bit]
    mat = np.zeros([base,]*nqb*2) 
    
    if opera == "b + b^\dagger" or opera == "b^\dagger" or opera == "b":
        if opera == "b + b^\dagger" or opera == "b^\dagger":
            for i in xrange(1,base**nqb):
                # b^+
                lstring = np.array(map(int, baseConvert(i, base).zfill(nqb)))
                rstring = np.array(map(int, baseConvert(i-1, base).zfill(nqb)))
                pos = tuple(roundrobin(lstring,rstring))
                mat[pos] = np.sqrt(i)

        if opera == "b + b^\dagger" or opera == "b":
            for i in xrange(0,base**nqb-1):
                # b
                lstring = np.array(map(int, baseConvert(i, base).zfill(nqb)))
                rstring = np.array(map(int, baseConvert(i+1, base).zfill(nqb)))
                pos = tuple(roundrobin(lstring,rstring))
                mat[pos] = np.sqrt(i+1)

    elif opera == "b^\dagger b":
        # actually Identity operator can be constructed directly
        for i in xrange(0,base**nqb):
            # I
            lstring = np.array(map(int, baseConvert(i, base).zfill(nqb)))  
            rstring = np.array(map(int, baseConvert(i, base).zfill(nqb))) 
            pos = tuple(roundrobin(lstring,rstring))
            mat[pos] = float(i)
    
    # check the original mat
    # mat = np.moveaxis(mat,range(1,nqb*2,2),range(nqb,nqb*2))
    # print mat.reshape(base**nqb,base**nqb)
    
    # decompose canonicalise
    MPO = []
    mat = mat.reshape(1,-1)
    for idx in xrange(nqb-1):
        U, S, Vt = scipy.linalg.svd(mat.reshape(mat.shape[0]*base**2,-1), \
                full_matrices=False)
        U = U.reshape(mat.shape[0],base,base,-1)
        MPO.append(U)
        mat = np.einsum("i, ij -> ij", S, Vt)
    
    MPO.append(mat.reshape(-1,base,base,1))
    print "original MPO shape:", [i.shape[0] for i in MPO] + [1]
    
    # compress
    MPOnew = mpslib.compress(MPO,'l',trunc=trunc)
    print "trunc", trunc, "distance", mpslib.distance(MPO,MPOnew)
    fidelity = mpslib.dot(mpslib.conj(MPOnew), MPO) / mpslib.dot(mpslib.conj(MPO), MPO)
    print "compression fidelity:: ", fidelity
    print "compressed MPO shape", [i.shape[0] for i in MPOnew] + [1]
    
    return MPOnew
