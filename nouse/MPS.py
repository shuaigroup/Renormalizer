import numpy as N
import scipy.linalg
import copy
from qnsvd import *
import mps as mpsbase
import utils

def add(mpsa,mpsb,mpsaqn=None,mpsbqn=None):
    """
    add two mps
    """
    if mpsa==None:
        return [mt.copy() for mt in mpsb], mpsbqn
    elif mpsb==None:
        return [mt.copy() for mt in mpsa], mpsaqn

    assert len(mpsa)==len(mpsb)

    nsites=len(mpsa)

    mpsab=[None]*nsites
    
    mpsab[0]=N.dstack([mpsa[0],mpsb[0]])
    for i in xrange(1,nsites-1):
        mta=mpsa[i]
        mtb=mpsb[i]
        pdim = mta.shape[1]
        assert pdim==mtb.shape[1]
        mpsab[i]=utils.zeros([mta.shape[0]+mtb.shape[0],pdim,
                              mta.shape[2]+mtb.shape[2]])
        mpsab[i][:mta.shape[0],:,:mta.shape[2]]=mta[:,:,:]
        mpsab[i][mta.shape[0]:,:,mta.shape[2]:]=mtb[:,:,:]

    mpsab[-1]=N.vstack([mpsa[-1],mpsb[-1]])
    
    if mpsaqn != None and mpsbqn != None:
        mpsabqn = []
        for i in xrange(len(mpsaqn)):
            mpsabqn.append(mpsaqn[i]+mpbqn[i])
        return mpsab, mpsabqn
    else:
        return mpsab, None


def canonicalise(mps,side, mpsqn=None, ephtable=None, pbond=None, nexciton=None):
    """
    create canonical MPS
    """
    if side=='l':
        return compress(mps,'r',0, mpsqn=mpsqn, ephtable=ephtable, \
                pbond=pbond, nexciton=nexciton)
    else:
        return compress(mps,'l',0, mpsqn=mpsqn, ephtable=ephtable, \
                pbond=pbond, nexciton=nexciton)


#@profile
def compress(mps,side,trunc=1.e-12, check_canonical=False, mpsqn=None, ephtable=None, pbond=None, nexciton=None):
    """
    inp: canonicalise MPS (or MPO)

    trunc=0: just canonicalise
    0<trunc<1: sigma threshold
    trunc>1: number of renormalised vectors to keep

    side='l': compress LEFT-canonicalised MPS 
              by sweeping from RIGHT to LEFT
              output MPS is right canonicalised i.e. CRRR

    side='r': reverse of 'l'
   
    returns:
         truncated or canonicalised MPS
    """

    if mpsqn != None:
        mpsqnnew = copy.deepcopy(mpsqn)
        print [imps.shape[0] for imps in mps]
        print [len(j) for j in mpsqn]

    assert side in ["l","r"]

    # if trunc==0, we are just doing a canonicalisation,
    # so skip check, otherwise, ensure mps is canonicalised
    if trunc != 0 and check_canonical:
        if side=="l":
            assert mpsbase.is_left_canonical(mps)
        else:
            assert mpsbase.is_right_canonical(mps)

    ret_mps=[]
    nsites=len(mps)

    if side=="l":
        res=mps[-1]
    else:
        res=mps[0]

    for i in xrange(1,nsites):
        # physical indices exclude first and last indices
        pdim=list(res.shape[1:-1])

        if mpsqn == None:
            if side=="l":
                res=N.reshape(res,(res.shape[0],N.prod(res.shape[1:])))
            else:
                res=N.reshape(res,(N.prod(res.shape[:-1]),res.shape[-1]))
            u,sigma,vt=scipy.linalg.svd(res,full_matrices=False)
        else:
            if side == "l":
                addlist = [nsites-i]
            else:
                addlist = [i-1]

            qnmat, qnl, qnr, qnsigmalist = construct_qnmat(mpsqnnew, ephtable, pbond, addlist)

            if side == "l":
                qnbigl = qnl
                qnbigr = N.add.outer(qnsigmalist[-1], qnr)
            else:
                qnbigl = N.add.outer(qnl,qnsigmalist[0])
                qnbigr = qnr
            
            print [len(j) for j in mpsqnnew] 
            print "qnbigl, qnbigr", qnbigl, qnbigr
            bsigma, bu, buqn, bv, bvqn = Csvd(res, qnbigl, qnbigr, nexciton, full_matrices=False)
            
            # sort singular value
            sidx = N.argsort(bsigma)[::-1]
            sigma = N.sort(bsigma)[::-1]
            u = np.zeros_like(bu)
            v = np.zeros_like(bv)
            uqn = []
            vqn = []
            for idx in xrange(len(sidx)):
                u[:,idx] = bu[:,sidx[idx]]
                v[:,idx] = bv[:,sidx[idx]]
                uqn.append(buqn[sidx[idx]])
                vqn.append(bvqn[sidx[idx]])
            vt = v.T
        
        if trunc==0:
            m_trunc=len(sigma)
        elif trunc<1.:
            # count how many sing vals < trunc            
            normed_sigma=sigma/scipy.linalg.norm(sigma)
            m_trunc=len([s for s in normed_sigma if s >trunc])
        else:
            m_trunc=int(trunc)
            m_trunc=min(m_trunc,len(sigma))
        
        u=u[:,0:m_trunc]
        vt=vt[0:m_trunc,:]
        sigma=N.diag(sigma[0:m_trunc])


        if mpsqn != None:
            if side == "l":
                mpsqnnew[nsites-i] = vqn[0:m_trunc]
            else:
                mpsqnnew[i] = uqn[0:m_trunc]

        if side=="l":
            u=N.dot(u,sigma)
            res=N.dot(mps[nsites-i-1],u)
            ret_mpsi=N.reshape(vt,[m_trunc]+pdim+[vt.shape[1]/N.prod(pdim)])
        else:
            vt=N.dot(sigma,vt)
            res=N.tensordot(vt,mps[i],1)
            ret_mpsi=N.reshape(u,[u.shape[0]/N.prod(pdim)]+pdim+[m_trunc])
                
        ret_mps.append(ret_mpsi)

    ret_mps.append(res)
    if side=="l":
        ret_mps.reverse()

    #fidelity = mpsbase.dot(ret_mps, mps)/mpsbase.dot(mps, mps)
    #print "compression fidelity:: ", fidelity
    
    if mpsqn != None:
        return ret_mps, mpsqnnew
    else:
        return ret_mps, None


