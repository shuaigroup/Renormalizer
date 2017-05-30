import numpy as N
#import autograd.numpy as N
import math
import mps

def create(ops):
    """
    Create MPO operator from a
    tensor product of single site operators e.g.
    I otimes c otimes d otimes ...
    """
    pdim=ops[0].shape[0]
    assert ops[0].shape[1]==pdim

    return [N.reshape(op,[1,pdim,pdim,1]) for op in ops]

def to_mps(mpo):
    """
    flatten physical indices of MPO to MPS
    """
    pdim=mpo[0].shape[1]
    assert pdim==mpo[0].shape[2]
    
    return [N.reshape(mt,[mt.shape[0],mt.shape[1]*mt.shape[2],mt.shape[3]]) for mt in mpo]


def from_mps(mps):
    """
    squaren physical indices of MPS to MPO
    """
    MPO = []
    for imps in mps:
        pdim=int(math.sqrt(imps.shape[1]))
        assert pdim*pdim==imps.shape[1]
        MPO.append(N.reshape(imps,[imps.shape[0],pdim,pdim,imps.shape[2]]))

    return MPO


def contract(mpo,mpsb,side,thresh,ncanonical=1):
    """
    mapply->canonicalise->compress
    """
    ret=mapply(mpo,mpsb)
    # roundoff can cause problems, 
    # so do multiple canonicalisations
    for i in xrange(ncanonical):
        ret=mps.canonicalise(ret,side)
    ret=mps.compress(ret,side,thresh)
    return ret

def mapply(mpo,mps):
    """
    apply mpo to mps, or apply mpo to mpo
    """
    nsites=len(mpo)
    assert len(mps)==nsites

    ret=[None]*nsites

    if len(mps[0].shape)==3: 
        # mpo x mps
        for i in xrange(nsites):
            assert mpo[i].shape[2]==mps[i].shape[1]
            mt=N.einsum("apqb,cqd->acpbd",mpo[i],mps[i])
            mt=N.reshape(mt,[mpo[i].shape[0]*mps[i].shape[0],mpo[i].shape[1],
                             mpo[i].shape[-1]*mps[i].shape[-1]])
            ret[i]=mt
    elif len(mps[0].shape)==4: 
        # mpo x mpo
        for i in xrange(nsites):
            assert mpo[i].shape[2]==mps[i].shape[1]
            mt=N.einsum("apqb,cqrd->acprbd",mpo[i],mps[i])
            mt=N.reshape(mt,[mpo[i].shape[0]*mps[i].shape[0],
                             mpo[i].shape[1],mps[i].shape[2],
                             mpo[i].shape[-1]*mps[i].shape[-1]])
            ret[i]=mt
    
    return ret

def trace(mpo):
    """
    \sum_{n1n2...} A[n1n1] A[n2n2] A[nknk]
    """
    traced_mts=[N.einsum("innj",mt) for mt in mpo]
    val=traced_mts[0]
    nsites=len(mpo)
    for i in xrange(1,nsites):
        val=N.dot(val,traced_mts[i])
    return N.trace(val)#[0,0]



