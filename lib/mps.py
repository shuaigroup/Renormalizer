import numpy as N
import numpy.random
import scipy.linalg
import utils
import copy
import fci
import time
from mpompsmat import *
import opt_einsum
from tensor import *
import math

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
    """
    return [N.conj(mt) for mt in mps]

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
        return compress(mps,'r',0, QR=True)
    else:
        return compress(mps,'l',0, QR=True)

#@profile
def variational_compress(MPS,aMPS,MPO,side,nloops,trunc=1.e-12,method="1site"):
    """
    aMPS inp: canonicalise approximate MPS (or MPO)

    0<trunc<1: sigma threshold
    trunc>1: number of renormalised vectors to keep

    side='l': compress LEFT-canonicalised MPS 
              by sweeping from RIGHT to LEFT
              output MPS is right canonicalised i.e. CRRR

    side='r': reverse of 'l'
   
    returns:
         truncated or canonicalised MPS
    """
    #assert side in ["l","r"]
    assert side in ["l"]
    assert method in ["2site", "1site"]
    print "optimization method", method
    if side == "l":
        side = "L"
    else:
        side = "R"

    # construct the environment matrix
    aMPSconj = conj(aMPS)
    construct_enviro(MPS, aMPSconj, MPO, side)

    nMPS = len(MPS)
    if method == "1site":
        loop = [['R',i] for i in xrange(nMPS-1,-1,-1)] + [['L',i] for i in xrange(0,nMPS)]
    else:
        loop = [['R',i] for i in xrange(nMPS-1,0,-1)] + [['L',i] for i in xrange(1,nMPS)]
    

    ltensor = N.ones((1,1,1))
    rtensor = N.ones((1,1,1))

    for isweep in xrange(nloops):
        for system, imps in loop:
            if system == "R":
                lmethod, rmethod = "Enviro", "System"
            else:
                lmethod, rmethod = "System", "Enviro"
            
            if method == "1site":
                lsite = imps-1
            else:
                lsite= imps-2
            ltensor = GetLR('L', lsite, MPS, aMPSconj, MPO, itensor=ltensor, method=lmethod)
            rtensor = GetLR('R', imps+1, MPS, aMPSconj, MPO, itensor=rtensor, method=rmethod)
            if method == "1site":

                #S-a   l-S    #S-a   l-S
                #    d        #    d  
                #O-b-O-f-O or #O-b-O-f-O
                #    e        #    e 
                #S-c-S-k-S    #S-c-S-k-S
                #                  m

                #tmp1 = N.einsum("abc,bdef -> acdef", ltensor, MPO[imps])
                #tmp2 = N.einsum("acdef, cek -> adfk", tmp1, MPS[imps]) 
                #res = N.einsum("adfk, lfk-> adl", tmp2, rtensor)
                tmp1 = N.tensordot(ltensor,MPO[imps],axes=([1],[0]))
                tmp2 = N.tensordot(tmp1,MPS[imps],axes=([1,3],[0,1]))
                if MPS[imps].ndim == 3:
                    res = N.tensordot(tmp2,rtensor,axes=([-2,-1],[-2,-1]))
                elif MPS[imps].ndim == 4: 
                    #tmp1 = N.einsum("abc,bdef -> acdef", ltensor, MPO[imps])
                    #tmp2 = N.einsum("acdef, cemk -> adfmk", tmp1, MPSp[imps]) 
                    #res = N.einsum("adfmk, lfk-> adml", tmp2, rtensor)
                    res = N.tensordot(tmp2,rtensor,axes=([-3,-1],[-2,-1]))
                
                if system == "R":
                    ushape = [res.shape[0]]
                    vshape = list(res.shape[1:])
                    res=N.reshape(res,(res.shape[0],N.prod(res.shape[1:])))
                else:
                    vshape = [res.shape[-1]]
                    ushape = list(res.shape[:-1])
                    res=N.reshape(res,(N.prod(res.shape[:-1]),res.shape[-1]))
            else:
            
                #S-a       l-S      S-a       l-S
                #    d   g              d   g 
                #O-b-O-f-O-j-O  or  O-b-O-f-O-j-O
                #    e   h              e   h
                #S-c-S-m-S-k-S      S-c-S-m-S-k-S
                #                       n   o
                
                if MPS[imps].ndim == 3:
                    path = [([0, 3],"abc,cem -> abem")   ,\
                                ([4, 0],"abem,bdef -> amdf") ,\
                                ([1, 2],"mhk, ljk -> mhlj")  ,\
                                ([0, 2],"fghj, mhlj -> fgml"),\
                                ([0, 1],"amdf, fgml -> adgl")]
                    res = multi_tensor_contract(path,
                            ltensor, MPO[imps-1], MPO[imps], MPS[imps-1], MPS[imps], rtensor)

                elif MPS[imps].ndim == 4:
                    path = [([0, 3],"abc,cenm -> abenm")   ,\
                                ([4, 0],"abenm,bdef -> anmdf") ,\
                                ([1, 2],"mhok, ljk -> mholj")  ,\
                                ([0, 2],"fghj, mholj -> fgmol"),\
                                ([0, 1],"anmdf, fgmol -> andgol")]
                    res = multi_tensor_contract(path,
                            ltensor, MPO[imps-1], MPO[imps], MPS[imps-1], MPS[imps], rtensor)
                    res = np.moveaxis(res,2,1)

                ushape = list(res.shape[0:res.ndim/2])
                vshape = list(res.shape[res.ndim/2:])
                res=N.reshape(res,(N.prod(res.shape[0:res.ndim/2]),N.prod(res.shape[res.ndim/2:])))
            try:
                u, sigma, vt = scipy.linalg.svd(res,full_matrices=False,lapack_driver='gesdd')
            except:
                print "variational mps compress converge failed"
                u, sigma, vt = scipy.linalg.svd(res,full_matrices=False,lapack_driver='gesvd')

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
            sigma=sigma[0:m_trunc]
            vt=vt[0:m_trunc,:]
            
            if  system == "R":
                u = N.einsum('ji, i -> ji', u, sigma)
                mps = N.reshape(vt,[m_trunc]+vshape)
                compmps = N.reshape(u,ushape+[m_trunc])
            else:
                vt = N.einsum('i, ij -> ij', sigma, vt)
                compmps = N.reshape(vt,[m_trunc]+vshape)
                mps = N.reshape(u,ushape+[m_trunc])

            if method == "1site":
                aMPS[imps] = mps
                if system == "L":
                    if imps != len(aMPS)-1:
                        aMPS[imps+1] = N.tensordot(compmps, aMPS[imps+1], axes=1)
                        aMPSconj[imps+1] = N.conj(aMPS[imps+1])
                    else:
                        aMPS[imps] = N.tensordot(aMPS[imps],compmps, axes=1)
                else:
                    if imps != 0:
                        aMPS[imps-1] = N.tensordot(aMPS[imps-1],compmps,axes=1)
                        aMPSconj[imps-1] = N.conj(aMPS[imps-1])
                    else:
                        aMPS[imps] = N.tensordot(compmps, aMPS[imps],axes=1)
            else:
                if system == "L":
                    aMPS[imps-1] = mps
                    aMPS[imps] = compmps
                else:
                    aMPS[imps] = mps
                    aMPS[imps-1] = compmps
                aMPSconj[imps-1] = N.conj(aMPS[imps-1])
            
            aMPSconj[imps] = N.conj(aMPS[imps])

    ret=mapply(MPO,MPS)
    fidelity = dot(conj(aMPS), ret)/dot(conj(ret), ret)
    print "compression fidelity:: ", fidelity
    
    return aMPS

@profile
def compress(mps,side,trunc=1.e-12,check_canonical=False,QR=False):
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

        
        if QR == False:
            try:
                #RANK = 50
                print "svd shape", res.shape
                #starttime = time.time()
                #if min(res.shape) > RANK:
                #    u, sigma, vt = randomized_svd(res, n_components=RANK,
                #        n_iter=7, random_state=None)
                #    print "error", N.mean(N.abs(res-N.dot(u, N.dot(N.diag(sigma),
                #        vt))))/N.mean(N.abs(res))
                #else:
                u,sigma,vt=scipy.linalg.svd(res,full_matrices=False,lapack_driver='gesdd')
                #    print "error2", N.mean(N.abs(res-N.dot(u, N.dot(N.diag(sigma),
                #        vt))))/N.mean(N.abs(res))
                #print res.shape,"svdtime",time.time()-starttime
            except:
                print "mps compress converge failed"
                u,sigma,vt=scipy.linalg.svd(res,full_matrices=False,lapack_driver='gesvd')

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
            
            if side == "l":
                #u=N.dot(u,sigma)
                #res=N.dot(mps[nsites-i-1],u)
                u = N.einsum('ji, i -> ji', u, sigma)
            else:
                #vt=N.dot(sigma,vt)
                #res=N.tensordot(mps[i].T, vt.T, axes=([-1],[0])).T
                vt = N.einsum('i, ij -> ij', sigma, vt)
        else:
            if side == "l":
                u,vt = scipy.linalg.rq(res, mode='economic')
            else:
                u,vt = scipy.linalg.qr(res, mode='economic')
            m_trunc = u.shape[1]

        if side=="l":
            res=N.tensordot(mps[nsites-i-1],u,axes=([-1],[0]))
            ret_mpsi=N.reshape(vt,[m_trunc]+pdim+[vt.shape[1]/N.prod(pdim)])
        else:
            res=N.tensordot(vt,mps[i],axes=([-1],[0]))
            ret_mpsi=N.reshape(u,[u.shape[0]/N.prod(pdim)]+pdim+[m_trunc])
        
        ret_mps.append(ret_mpsi)

    ret_mps.append(res)
    if side=="l":
        ret_mps.reverse()

    #fidelity = dot(conj(ret_mps), mps)/dot(conj(mps), mps)
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
            e0 = N.tensordot(e0, mpsa[i], ([0,1,2],[0,1,2])).T

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


def conjtrans(mpo):
    """
    conjugated transpose of MPO
    a[lbond,upbond,downbond,rbond] -> a[lbond,downbond,upbond,rbond]*
    """

    assert mps[0].ndim == 4

    return [impo.transpose(0,2,1,3).conj() for impo in mpo]


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

#@profile
def contract(mpo,mpsb,side,thresh,mpsa=None,ncanonical=1,compress_method="svd"):
    
    assert compress_method in ["svd","variational"]
    
    if compress_method == "svd":
        """
        mapply->canonicalise->compress
        """
        ret=mapply(mpo,mpsb)
        # roundoff can cause problems, 
        # so do multiple canonicalisations
        for i in xrange(ncanonical):
            ret=canonicalise(ret,side)
        ret=compress(ret,side,thresh)
    
    elif compress_method == "variational":
        if mpsa == None:
            #mpsa = add(mpsb,None)
            mpox = canonicalise(mpo,side)
            mpsa = mapply(mpox, mpsb)
            nloops = 1
        ret=variational_compress(mpsb,mpsa,mpo,side,nloops,trunc=thresh,method="1site")
    
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
            #mt=N.einsum("apqb,cqd->acpbd",mpo[i],mps[i])
            mt=N.moveaxis(N.tensordot(mpo[i],mps[i],axes=([2],[1])),3,1)
            mt=N.reshape(mt,[mpo[i].shape[0]*mps[i].shape[0],mpo[i].shape[1],
                             mpo[i].shape[-1]*mps[i].shape[-1]])
            ret[i]=mt
    elif len(mps[0].shape)==4: 
        # mpo x mpo
        for i in xrange(nsites):
            assert mpo[i].shape[2]==mps[i].shape[1]
            #mt=N.einsum("apqb,cqrd->acprbd",mpo[i],mps[i])
            mt=N.moveaxis(N.tensordot(mpo[i],mps[i],axes=([2],[1])),[-3,-2],[1,3])
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



