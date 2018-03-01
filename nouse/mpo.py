import numpy as N
import scipy.linalg
import MPS as mps

#@profile
def contract(mpo,mpsb,side,thresh,ncanonical=1,mpoqn=None, mpsqn=None, ephtable=None, pbond=None, nexciton=None):
    """
    mapply->canonicalise->compress
    """
    ret, mposqn = mapply (mpo,mpsb, mpoqn=mpoqn, mpsqn=mpsqn)
    # roundoff can cause problems, 
    # so do multiple canonicalisations
    for i in xrange(ncanonical):
        ret, mposqn = mps.canonicalise(ret,side,mpsqn=mposqn, ephtable=ephtable,\
                pbond=pbond, nexciton=nexciton)
    ret, mposqn = mps.compress(ret,side,thresh,mpsqn=mposqn, ephtable=ephtable,pbond=pbond, nexciton=nexciton)

    ret, mposqn = mps.compress(ret,'r',thresh,mpsqn=mposqn, ephtable=ephtable,pbond=pbond, nexciton=nexciton)

    return ret, mposqn


def mapply(mpo,mps, mpoqn=None, mpsqn=None):
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
    
    if mpoqn != None and mpsqn != None:
        mposqn = []
        for i in xrange(len(mps)+1):
            oqn = N.array(mpoqn[i])
            sqn = N.array(mpsqn[i])
            osqn = N.add.outer(oqn,sqn).ravel()
            mposqn.append(osqn.tolist())
        print "mposqn", [len(i) for i in mposqn]
        print "ret", [imps.shape[0] for imps in ret]
        
        print "mps", [imps.shape[0] for imps in mps]
        print "mpsqn", [len(i) for i in mpsqn]
        print "mpo", [impo.shape[0] for impo in mpo]
        print "mpoqn", [len(i) for i in mpoqn]
        return ret, mposqn
    else:
        return ret, None


