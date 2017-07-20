#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def GetLR(domain, siteidx, MPS, MPSconj, MPO, itensor=np.ones((1,1,1)), method="Scratch"):
    
    '''
    get the L/R Hamiltonian matrix at a random site: 3d tensor
    S-
    O-
    S-
    enviroment part from disc,  system part from one step calculation
    support from scratch calculation: from two open boundary
    '''
    
    assert domain == "L" or domain == "R"
    assert method == "Enviro" or method == "System" or method == "Scratch"
    
    if siteidx not in range(len(MPS)):
        return np.ones((1,1,1))

    if method == "Scratch":
        itensor = np.ones((1,1,1))
        if domain == "L":
            sitelist = range(siteidx+1)
        else:
            sitelist = range(len(MPS)-1,siteidx-1,-1)
        for imps in sitelist:
            itensor = addone(itensor, MPS, MPSconj, MPO, imps, domain)
    elif method == "Enviro" :
        itensor = Enviro_read(domain, siteidx)
    elif method == "System" :
        itensor = addone(itensor, MPS, MPSconj, MPO, siteidx, domain)
        Enviro_write(domain,siteidx,itensor)
    
    return itensor


def addone(intensor, MPS, MPSconj, MPO, isite, domain):
    '''
    add one MPO/MPS(MPO) site 
    S-S-
    O-O-
    S-S-
    '''
    assert domain in ["L","R","l","r"]
    
    if domain == "L" or domain == "l":
        assert intensor.shape[0] == MPSconj[isite].shape[0] 
        assert intensor.shape[1] == MPO[isite].shape[0] 
        assert intensor.shape[2] == MPS[isite].shape[0] 
        '''
                       l 
        S-a-S-f    O-a-O-f
            d          d
        O-b-O-g or O-b-O-g  
            e          e
        S-c-S-h    O-c-O-h
                       l  
        '''
        tmp1 = np.tensordot(intensor,  MPO[isite], axes=([1],[0])) 
        tmp2 = np.tensordot(tmp1,  MPSconj[isite], axes=([0,2],[0,1])) 
        
        if MPS[isite].ndim == 3:
            # version 1: very slow
            #outtensor = np.einsum("abc, daf, debg, ech -> fgh", intensor, MPS[isite],
            #        MPO[isite], MPS[isite]) 
            # version 2: not bad
            #tmp1 = np.einsum("abc, bdeg -> acdeg", intensor,  MPO[isite]) 
            #tmp2 = np.einsum("acdeg, adf -> cegf", tmp1,  MPSconj[isite]) 
            #outtensor = np.einsum("cegf, ceh -> fgh", tmp2,  MPS[isite]) 
            outtensor = np.tensordot(tmp2,  MPS[isite],axes=([0,1],[0,1]))
        elif MPS[isite].ndim == 4:
            #tmp1 = np.einsum("abc, bdeg -> acdeg", intensor,  MPO[isite]) 
            #tmp2 = np.einsum("acdeg, adlf -> ceglf", tmp1,  MPSconj[isite]) 
            #outtensor = np.einsum("ceglf, celh -> fgh", tmp2,  MPS[isite]) 
            outtensor = np.tensordot(tmp2,  MPS[isite],axes=([0,1,3],[0,1,2]))
    else:
        assert intensor.shape[0] == MPSconj[isite].shape[-1] 
        assert intensor.shape[1] == MPO[isite].shape[-1]
        assert intensor.shape[2] == MPS[isite].shape[-1] 
        '''
                       l
        -f-S-a-S    -f-S-a-S
           d           d
        -g-O-b-O or -g-O-b-O
           e           e
        -h-S-c-S    -h-S-c-S
                       l
        '''
        tmp1 = np.tensordot(intensor,  MPO[isite], axes=([1],[-1])) 
        tmp2 = np.tensordot(tmp1,  MPSconj[isite], axes=([0,3],[-1,1])) 
        if MPS[isite].ndim == 3:
            # version 1: very slow
            #outtensor = np.einsum("abc, dfa, degb, ehc -> fgh", intensor, MPS[isite],
            #        MPO[isite], MPS[isite]) 
            # version 2: not bad
            #tmp1 = np.einsum("abc, gdeb -> acgde", intensor,  MPO[isite]) 
            #tmp2 = np.einsum("acgde, fda -> cgef", tmp1,  MPSconj[isite]) 
            #outtensor = np.einsum("cgef, hec -> fgh", tmp2,  MPS[isite]) 

            outtensor = np.tensordot(tmp2,  MPS[isite], axes=([0,2],[2,1])) 
        elif MPS[isite].ndim == 4:
            #tmp1 = np.einsum("abc, gdeb -> acgde", intensor,  MPO[isite]) 
            #tmp2 = np.einsum("acgde, fdla -> cgefl", tmp1,  MPSconj[isite]) 
            #outtensor = np.einsum("cgefl;, helc -> fgh", tmp2,  MPS[isite]) 
            outtensor = np.tensordot(tmp2,  MPS[isite], axes=([0,2,4],[-1,1,2])) 
            
    outtensor = np.moveaxis(outtensor,1,0)
    return outtensor


def construct_enviro(MPS, MPSconj, MPO, domain):
    tensor = np.ones((1,1,1))
    assert domain in ["L", "R", "l", "r"]
    if domain == "L" or domain == "l":
        start, end, inc = 0,len(MPS)-1,1
    else:
        start, end, inc = len(MPS)-1,0,-1

    for idx in xrange(start, end, inc):
        tensor = addone(tensor, MPS, MPSconj, MPO, idx, domain)
        Enviro_write(domain,idx,tensor)    


def Enviro_write(domain, siteidx, tensor):
    with open(domain+str(siteidx)+".npy", 'wb') as f:
        np.save(f,tensor)


def Enviro_read(domain, siteidx):
    with open(domain + str(siteidx)+".npy", 'rb') as f:
        return np.load(f)
