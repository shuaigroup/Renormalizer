import numpy as N
import numpy.random
#import numpy.linalg.linalg
import scipy.linalg
import utils
import copy
import fci

#import autograd.numpy as N
#import autograd.numpy.linalg
#svd = autograd.numpy.linalg.svd

def check_lortho(tens):
    tensm=N.reshape(tens,[N.prod(tens.shape[:-1]),tens.shape[-1]])
    s=N.dot(N.conj(tensm.T),tensm)
    return scipy.linalg.norm(s-N.eye(s.shape[0]))

def check_rortho(tens):
    tensm=N.reshape(tens,[tens.shape[0],N.prod(tens.shape[1:])])
    s=N.dot(tensm,N.conj(tensm.T))
    return scipy.linalg.norm(s-N.eye(s.shape[0]))

def conj(mps):
    """
    complex conjugate
    or 
    conjugated transpose of MPO bra to ket
    a[lbond,upbond,downbond,rbond] -> a[lbond,downbond,upbond,rbond]*
    """

    assert mps[0].ndim in [3,4]
    if mps[0].ndim == 3:
        return [N.conj(mt) for mt in mps]
    if mps[0].ndim == 4:
        return [impo.transpose(0,2,1,3).conj() for impo in mps]

def is_left_canonical(mps,thresh=1.e-8):
    ret=True
    for mt in mps[:-1]:
        #print check_lortho(mt)
        ret*=check_lortho(mt)<thresh
    return ret

def is_right_canonical(mps,thresh=1.e-8):
    ret=True
    for mt in mps[1:]:
        #print check_rortho(mt)
        ret*=check_rortho(mt)<thresh
    return ret

def shape(mps):
    """
    shapes of tensors
    """
    return [mt.shape for mt in mps]

def zeros(nsites,pdim,m):
    mps=[None]*nsites
    mps[0]=numpy.zeros([1,pdim,m])
    mps[-1]=numpy.zeros([m,pdim,1])
    for i in xrange(1,nsites-1):
        mps[i]=numpy.zeros([m,pdim,m])
    return mps
    
def random(nsites,pdim,m):
    """
    create random MPS for nsites, with m states,
    and physical dimension pdim
    """
    mps=[None]*nsites
    mps[0]=numpy.random.random([1,pdim,m])
    mps[-1]=numpy.random.random([m,pdim,1])
    for i in xrange(1,nsites-1):
        mps[i]=numpy.random.random([m,pdim,m])
    return mps

# get each config's coefficient
def ceval(mps,config):
    """
    Evaluates mps at given config
    """
    mps_mats=[None]*len(config)
    nsites=len(config)
    for i, pval in enumerate(config):
        mps_mats[i]=mps[i][:,pval,:]
    
    # multiply "backwards" from right to left
    val=mps_mats[0]
    for i in xrange(1,nsites):
        val=N.dot(val,mps_mats[i])

    # turn into scalar
    return N.trace(val)

def create(pdim,config):
    """
    Create dim=1 MPS
    pdim: physical dimension
    """
    nsites=len(config)
    mps=[utils.zeros([1,pdim,1]) for i in xrange(nsites)]
    for i,p in enumerate(config):
        mps[i][0,p,0]=1.
    return mps
        
def canonicalise(mps,side):
    """
    create canonical MPS
    """
    if side=='l':
        return compress(mps,'r',0)
    else:
        return compress(mps,'l',0)


#@profile
def compress(mps,side,trunc=1.e-12,check_canonical=False):
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
    assert side in ["l","r"]

    # if trunc==0, we are just doing a canonicalisation,
    # so skip check, otherwise, ensure mps is canonicalised
    if trunc != 0 and check_canonical:
        if side=="l":
            assert is_left_canonical(mps)
        else:
            assert is_right_canonical(mps)

    ret_mps=[]
    nsites=len(mps)

    if side=="l":
        res=mps[-1]
    else:
        res=mps[0]

    for i in xrange(1,nsites):
        # physical indices exclude first and last indices
        pdim=list(res.shape[1:-1])

        if side=="l":
            res=N.reshape(res,(res.shape[0],N.prod(res.shape[1:])))
        else:
            res=N.reshape(res,(N.prod(res.shape[:-1]),res.shape[-1]))

        #try:
        #u,sigma,vt=autograd.numpy.linalg.svd(res,full_matrices=False)
        try:
            u,sigma,vt=scipy.linalg.svd(res,full_matrices=False,lapack_driver='gesdd')
        except:
            print "mps compress converge failed"
            u,sigma,vt=scipy.linalg.svd(res,full_matrices=False,lapack_driver='gesvd')

            #u,sigma,vt=fast_svd(res,full_matrices=False)
        # except: # hack for scipy's too low limit on SVD iterations
        #     v,sigma,ut=svd(res.T,full_matrices=False)
        #     #v,sigma,ut=fast_svd(res.T,full_matrices=False)
        #     vt=v.T
        #     u=ut.T

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
        #sigma=N.diag(sigma[0:m_trunc])
        sigma=sigma[0:m_trunc]
        vt=vt[0:m_trunc,:]

        if side=="l":
            #u=N.dot(u,sigma)
            #res=N.dot(mps[nsites-i-1],u)
            u = N.einsum('ji, i -> ji', u, sigma)
            res=N.tensordot(mps[nsites-i-1],u,axes=([-1],[0]))
            ret_mpsi=N.reshape(vt,[m_trunc]+pdim+[vt.shape[1]/N.prod(pdim)])
        else:
            #vt=N.dot(sigma,vt)
            #res=N.tensordot(mps[i].T, vt.T, axes=([-1],[0])).T
            vt = N.einsum('i, ij -> ij', sigma, vt)
            res=N.tensordot(vt,mps[i],axes=([-1],[0]))
            ret_mpsi=N.reshape(u,[u.shape[0]/N.prod(pdim)]+pdim+[m_trunc])
                
        ret_mps.append(ret_mpsi)

    ret_mps.append(res)
    if side=="l":
        ret_mps.reverse()

    #fidelity = dot(ret_mps, mps)/dot(mps, mps)
    #print "compression fidelity:: ", fidelity
    # if N.isnan(fidelity):
    #     dddd
    return ret_mps

def mps_fci(mps):
    """
    convert MPS into a fci vector
    """
    pdim=mps[0].shape[1]
    nsites=len(mps)
    confs=fci.fci_configs(nsites,pdim)
    fvec=utils.zeros((pdim,)*nsites)
    for conf in confs:
        fvec[conf]=ceval(mps,conf)
    return fvec

def scale(mps,val):
    """
    Multiply MPS by scalar
    """
    ret=[mt.copy() for mt in mps]
    ret[-1]*=val
    return ret

def add(mpsa,mpsb):
    """
    add two mps / mpo 
    """
    if mpsa==None:
        return [mt.copy() for mt in mpsb]
    elif mpsb==None:
        return [mt.copy() for mt in mpsa]

    assert len(mpsa)==len(mpsb)
    nsites=len(mpsa)

    mpsab=[None]*nsites
    
    if mpsa[0].ndim == 3:  # MPS
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
    elif mpsa[0].ndim == 4: # MPO
        mpsab[0]=N.concatenate((mpsa[0],mpsb[0]), axis=3)
        for i in xrange(1,nsites-1):
            mta=mpsa[i]
            mtb=mpsb[i]
            pdimu = mta.shape[1]
            pdimd = mta.shape[2]
            assert pdimu==mtb.shape[1]
            assert pdimd==mtb.shape[2]

            mpsab[i]=utils.zeros([mta.shape[0]+mtb.shape[0],pdimu,pdimd,
                                  mta.shape[3]+mtb.shape[3]])
            mpsab[i][:mta.shape[0],:,:,:mta.shape[3]]=mta[:,:,:,:]
            mpsab[i][mta.shape[0]:,:,:,mta.shape[3]:]=mtb[:,:,:,:]

        mpsab[-1]=N.concatenate((mpsa[-1],mpsb[-1]), axis=0)

    return mpsab
        
def dot(mpsa,mpsb):
    """
    dot product of two mps / mpo 
    """
    assert len(mpsa)==len(mpsb)
    nsites=len(mpsa)
    e0=N.eye(1,1)
    for i in xrange(nsites):
        # sum_x e0[:,x].m[x,:,:]
        e0=N.tensordot(e0,mpsb[i],1)
        # sum_ij e0[i,p,:] mpsa[i,p,:]
        # note, need to flip a (:) index onto top,
        # therefore take transpose
        if mpsa[i].ndim == 3:
            e0=N.tensordot(e0,mpsa[i],([0,1],[0,1])).T
        if mpsa[i].ndim == 4:
            e0 = N.tensordot(e0, mpsa[i], ([0,1,2],[0,2,1])).T

    return e0[0,0]

def distance(mpsa,mpsb):
    """
    ||mpsa-mpsb||
    """
    return dot(mpsa,mpsa)-2*dot(mpsa,mpsb)+dot(mpsb,mpsb)

def liouville_to_hilbert(mpsl,basis):
    """
    convert liouville mps to hilbert mpo
    """
    pdim=len(basis)
    sdim=int(pdim**0.5)
    nsites=len(mpsl)
    mpoh=[None]*nsites
    for i, mt in enumerate(mpsl):
        tens=utils.zeros([mt.shape[0],sdim,sdim,mt.shape[2]])
        for r in xrange(mt.shape[0]):
            for s in xrange(mt.shape[2]):
                for p in xrange(pdim):
                    tens[r,:,:,s]+=mt[r,p,s]*basis[p]
        mpoh[i]=tens
    return mpoh
