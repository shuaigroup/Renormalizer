# -*- coding: utf-8 -*-
'''
This mps/mpo lib is downloaded from https://github.com/gkc1000/PrimeRib
developed by Garnet Chan <gkc1000@gmail.com>
and changed by Jiajun Ren <jiajunren0522@gmail.com>
'''

import numpy as np
import numpy.random
import copy
import scipy.linalg
import mathutils
import math
import fci
from ephMPS import mpompsmat
from ephMPS.lib import tensor
from ephMPS import svd_qn 


def check_lortho(tens):
    '''
    check L-orthogonal 
    '''
    tensm=np.reshape(tens,[np.prod(tens.shape[:-1]),tens.shape[-1]])
    s=np.dot(np.conj(tensm.T),tensm)
    return scipy.linalg.norm(s-np.eye(s.shape[0]))


def check_rortho(tens):
    '''
    check R-orthogonal 
    '''
    tensm=np.reshape(tens,[tens.shape[0],np.prod(tens.shape[1:])])
    s=np.dot(tensm,np.conj(tensm.T))
    return scipy.linalg.norm(s-np.eye(s.shape[0]))


def conj(mps, QNargs=None):
    """
    complex conjugate
    """
    if QNargs is None:
        return [np.conj(mt) for mt in mps]
    else:
        return [[np.conj(mt) for mt in mps[0]]] + copy.deepcopy(mps[1:])


def conjtrans(mpo, QNargs=None):
    """
    conjugated transpose of MPO
    a[lbond,upbond,downbond,rbond] -> a[lbond,downbond,upbond,rbond]*
    """
    if QNargs is None:
        assert mpo[0].ndim == 4
        return [impo.transpose(0,2,1,3).conj() for impo in mpo]
    else:
        assert mpo[0][0].ndim == 4
        mponew = [impo.transpose(0,2,1,3).conj() for impo in mpo[0]]
        mpoQN = []
        for mpoqn in mpo[1]:
            mpoQN.append([-i for i in mpoqn])
        return [mponew, mpoQN] + copy.deepcopy(mpo[2:])


def is_left_canonical(mps,thresh=1.e-8):
    '''
    check L-canonical
    '''
    ret=True
    for mt in mps[:-1]:
        ret*=check_lortho(mt)<thresh
    return ret


def is_right_canonical(mps,thresh=1.e-8):
    '''
    check R-canonical
    '''
    ret=True
    for mt in mps[1:]:
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
        val=np.dot(val,mps_mats[i])

    # turn into scalar
    return np.trace(val)


def create(pdim,config):
    """
    Create dim=1 MPS
    pdim: physical dimension
    """
    nsites=len(config)
    mps=[mathutils.zeros([1,pdim,1]) for i in xrange(nsites)]
    for i,p in enumerate(config):
        mps[i][0,p,0]=1.
    return mps
 

def canonicalise(mps,side,QNargs=None,direct=False):
    """
    canonicalise MPS
    """
    if side=='l':
        return compress(mps,'r',0, QR=True, QNargs=QNargs,direct=direct)
    else:
        return compress(mps,'l',0, QR=True, QNargs=QNargs,direct=direct)


def contract(mpo,mpsb,side,thresh,mpsa=None,ncanonical=1,\
        compress_method="svd",QNargs=None,direct=True):
    '''
    contract a MPO times MPS
    the "direct" method is to do the mpo/mps contraction in the compression step
    the direct = False, scaling is D^2M^2p^3 + D^3M^3p^2
    the direct = True, scaling is (D^3M^2p^2 + D^2M^3p^3) or (D^2M^3p^2 + D^3M^2p^3)
    '''

    assert compress_method in ["svd","variational"]
    
    if compress_method == "svd":
        """
        mapply->canonicalise->compress
        """
        ret=mapply(mpo,mpsb,QNargs=QNargs,direct=direct)
        # roundoff can cause problems, 
        # so do multiple canonicalisations
        for i in xrange(ncanonical):
            ret=canonicalise(ret,side, QNargs=QNargs, direct=direct)
        ret=compress(ret,side,thresh, QNargs=QNargs)
    
    elif compress_method == "variational":
        if mpsa is None:
            # this initial guess method is from
            # "The density-matrix renormalization group in the age of matrix
            # product states"
            # make sure the mpsb is side-canonical
            mpox = canonicalise(mpo,side)
            mpsa = mapply(mpox, mpsb)
            nloops = 1
        ret=variational_compress(mpsb,mpsa,mpo,side,nloops,trunc=thresh,method="1site")
    
    return ret


def mapply(mpo,mps,QNargs=None,direct=False):
    """
    apply mpo to mps, or apply mpo to mpo
    """
    if QNargs is not None:
        mpo, mpoQN, mpoQNidx, mpotot = mpo
        mps, mpsQN, mpsQNidx, mpstot = mps
    
    nsites=len(mpo)
    assert len(mps)==nsites
    
    ret=[None]*nsites
    
    if direct == False:
        if len(mps[0].shape)==3: 
            # mpo x mps
            for i in xrange(nsites):
                assert mpo[i].shape[2]==mps[i].shape[1]
                #mt=np.einsum("apqb,cqd->acpbd",mpo[i],mps[i])
                mt=np.moveaxis(np.tensordot(mpo[i],mps[i],axes=([2],[1])),3,1)
                mt=np.reshape(mt,[mpo[i].shape[0]*mps[i].shape[0],mpo[i].shape[1],
                                 mpo[i].shape[-1]*mps[i].shape[-1]])
                ret[i]=mt
        elif len(mps[0].shape)==4: 
            # mpo x mpo
            for i in xrange(nsites):
                assert mpo[i].shape[2]==mps[i].shape[1]
                #mt=np.einsum("apqb,cqrd->acprbd",mpo[i],mps[i])
                mt=np.moveaxis(np.tensordot(mpo[i],mps[i],axes=([2],[1])),[-3,-2],[1,3])
                mt=np.reshape(mt,[mpo[i].shape[0]*mps[i].shape[0],
                                 mpo[i].shape[1],mps[i].shape[2],
                                 mpo[i].shape[-1]*mps[i].shape[-1]])
                ret[i]=mt
    else:
        for i in xrange(nsites):
            ret[i] = [mpo[i], mps[i]]

    if QNargs is not None:
        mpsQNnew = svd_qn.QN_construct(mpsQN, mpsQNidx, mpoQNidx, mpstot)
        mpompsQN = []
        for i in xrange(len(mpoQN)):
            mpompsqn = np.add.outer(np.array(mpoQN[i]),np.array(mpsQNnew[i]))
            mpompsQN.append(mpompsqn.ravel().tolist())
        ret = [ret, mpompsQN, mpoQNidx, mpstot+mpotot]
    
    return ret


def variational_compress(MPS,aMPS,MPO,side,nloops,trunc=1.e-12,method="1site"):
    """
    aMPS : canonicalised approximate MPS (or MPO)

    0<trunc<1: sigma threshold
    trunc>1: number of renormalised vectors to keep

    side='l': compress LEFT-canonicalised MPS 
              by sweeping from RIGHT to LEFT and then LEFT to RIGHT
              output MPS is left canonicalised i.e. LLLLC
    side='r': reverse of 'l'
   
    returns:
         truncated MPS
    """

    #assert side in ["l","r"]
    assert side in ["l"]
    assert method in ["2site", "1site"]
    side = side.upper()

    # construct the environment matrix
    aMPSconj = conj(aMPS)
    mpompsmat.construct_enviro(MPS, aMPSconj, MPO, side)

    nMPS = len(MPS)
    if method == "1site":
        loop = [['R',i] for i in xrange(nMPS-1,-1,-1)] + [['L',i] for i in xrange(0,nMPS)]
    else:
        loop = [['R',i] for i in xrange(nMPS-1,0,-1)] + [['L',i] for i in xrange(1,nMPS)]
    

    ltensor = np.ones((1,1,1))
    rtensor = np.ones((1,1,1))

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
            ltensor = mpompsmat.GetLR('L', lsite, MPS, aMPSconj, MPO, itensor=ltensor, method=lmethod)
            rtensor = mpompsmat.GetLR('R', imps+1, MPS, aMPSconj, MPO, itensor=rtensor, method=rmethod)
            if method == "1site":

                #S-a   l-S    #S-a   l-S
                #    d        #    d  
                #O-b-O-f-O or #O-b-O-f-O
                #    e        #    e 
                #S-c-S-k-S    #S-c-S-k-S
                #                  m

                if MPS[imps].ndim == 3:
                    path = [([0, 1],"abc,cek -> abek")   ,\
                            ([2, 0],"abek,bdef -> akdf") ,\
                            ([1, 0],"akdf, lfk -> adl")] 
                    res = tensor.multi_tensor_contract(path,
                            ltensor, MPS[imps], MPO[imps], rtensor)
                elif MPS[imps].ndim == 4: 
                    path = [([0, 1],"abc,cemk -> abemk")   ,\
                            ([2, 0],"abemk,bdef -> amkdf") ,\
                            ([1, 0],"amkdf, lfk -> amdl")] 
                    res = tensor.multi_tensor_contract(path,
                            ltensor, MPS[imps], MPO[imps], rtensor)
                    res = np.moveaxis(res,2,1)
                if system == "R":
                    ushape = [res.shape[0]]
                    vshape = list(res.shape[1:])
                    res=np.reshape(res,(res.shape[0],np.prod(res.shape[1:])))
                else:
                    vshape = [res.shape[-1]]
                    ushape = list(res.shape[:-1])
                    res=np.reshape(res,(np.prod(res.shape[:-1]),res.shape[-1]))
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
                    res = tensor.multi_tensor_contract(path,
                            ltensor, MPO[imps-1], MPO[imps], MPS[imps-1], MPS[imps], rtensor)

                elif MPS[imps].ndim == 4:
                    path = [([0, 3],"abc,cenm -> abenm")   ,\
                                ([4, 0],"abenm,bdef -> anmdf") ,\
                                ([1, 2],"mhok, ljk -> mholj")  ,\
                                ([0, 2],"fghj, mholj -> fgmol"),\
                                ([0, 1],"anmdf, fgmol -> andgol")]
                    res = tensor.multi_tensor_contract(path,
                            ltensor, MPO[imps-1], MPO[imps], MPS[imps-1], MPS[imps], rtensor)
                    res = np.moveaxis(res,2,1)

                ushape = list(res.shape[0:res.ndim/2])
                vshape = list(res.shape[res.ndim/2:])
                res=np.reshape(res,(np.prod(res.shape[0:res.ndim/2]),np.prod(res.shape[res.ndim/2:])))
            try:
                u, sigma, vt = scipy.linalg.svd(res,full_matrices=False,lapack_driver='gesdd')
            except:
                print "variational mps compress converge failed"
                u, sigma, vt = scipy.linalg.svd(res,full_matrices=False,lapack_driver='gesvd')
            
            if trunc<1.:
                # count how many sing vals < trunc            
                normed_sigma=sigma/scipy.linalg.norm(sigma)
                #m_trunc=len([s for s in normed_sigma if s >trunc])
                m_trunc = np.count_nonzero(normed_sigma>trunc)
            else:
                m_trunc=int(trunc)
                m_trunc=min(m_trunc,len(sigma))
            
            u=u[:,0:m_trunc]
            sigma=sigma[0:m_trunc]

            vt=vt[0:m_trunc,:]
            
            if  system == "R":
                u = np.einsum('ji, i -> ji', u, sigma)
                mps = np.reshape(vt,[m_trunc]+vshape)
                compmps = np.reshape(u,ushape+[m_trunc])
            else:
                vt = np.einsum('i, ij -> ij', sigma, vt)
                compmps = np.reshape(vt,[m_trunc]+vshape)
                mps = np.reshape(u,ushape+[m_trunc])

            if method == "1site":
                aMPS[imps] = mps
                if system == "L":
                    if imps != len(aMPS)-1:
                        aMPS[imps+1] = np.tensordot(compmps, aMPS[imps+1], axes=1)
                        aMPSconj[imps+1] = np.conj(aMPS[imps+1])
                    else:
                        aMPS[imps] = np.tensordot(aMPS[imps],compmps, axes=1)
                else:
                    if imps != 0:
                        aMPS[imps-1] = np.tensordot(aMPS[imps-1],compmps,axes=1)
                        aMPSconj[imps-1] = np.conj(aMPS[imps-1])
                    else:
                        aMPS[imps] = np.tensordot(compmps, aMPS[imps],axes=1)
            else:
                if system == "L":
                    aMPS[imps-1] = mps
                    aMPS[imps] = compmps
                else:
                    aMPS[imps] = mps
                    aMPS[imps-1] = compmps
                aMPSconj[imps-1] = np.conj(aMPS[imps-1])
            
            aMPSconj[imps] = np.conj(aMPS[imps])

    #ret=mapply(MPO,MPS)
    #fidelity = dot(conj(aMPS), ret)/dot(conj(ret), ret)
    #print "compression fidelity:: ", fidelity
    
    return aMPS


def compress(mps,side,trunc=1.e-12,check_canonical=False, QR=False,\
        QNargs=None,normalize=None, direct=False, msite=None):
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
    assert side in ["l","r","ml","mr"]

    mpsin = mps

    if QNargs is not None:
        ephtable, ifMPO = QNargs[:]

        mps, MPSQN, MPSQNidx, MPSQNtot = mps
        if side == "l":
            MPSQNnew = svd_qn.QN_construct(MPSQN, MPSQNidx, len(mps)-1, MPSQNtot)
        elif side == "r":
            MPSQNnew = svd_qn.QN_construct(MPSQN, MPSQNidx, 0, MPSQNtot)
        else:
            MPSQNnew = MPSQN
    
    if side == "ml" and msite == 0:
        return mpsin
    elif side == "mr" and msite == len(mps)-1:
        return mpsin

    # if trunc==0, we are just doing a canonicalisation,
    # so skip check, otherwise, ensure mps is canonicalised
    if trunc != 0 and check_canonical:
        if side=="l":
            assert is_left_canonical(mps)
        elif side=="r":
            assert is_right_canonical(mps)
    
    def getmps(mps):
        if type(mps) == list:
            return mapply([mps[0]],[mps[1]])[0]
        elif type(mps) == np.ndarray:
            return mps

    ret_mps=[]
    nsites=len(mps)
    if side=="l":
        res=getmps(mps[-1])
    elif side == "r":
        res=getmps(mps[0])
    elif side == "mr" or side == "ml":
        res=getmps(mps[msite])
    
    if side == "l":
        loop = range(nsites-1,0,-1)
    elif side == "r":
        loop = range(0,nsites-1,1)
    else:
        loop = [msite]


    for idx in loop:

        # physical indices exclude first and last indices
        pdim=list(res.shape[1:-1])
        npdim = np.prod(pdim)

        if side[-1] == "l":
            res=np.reshape(res,(res.shape[0],np.prod(res.shape[1:])))
        else:
            res=np.reshape(res,(np.prod(res.shape[:-1]),res.shape[-1]))
        
        if QNargs is not None:
            if ephtable[idx] == 1:
                # e site
                if len(pdim) == 1:
                    sigmaqn = np.array([0,1])
                elif len(pdim) == 2:
                    # if MPS is a real MPO, then bra and ket are both important
                    # if MPS is a MPS or density operator MPO, then only bra is important
                    if ifMPO == False:
                        sigmaqn = np.array([0,0,1,1])
                    elif ifMPO == True:
                        sigmaqn = np.array([0,-1,1,0])
            else:
                # ph site 
                sigmaqn = np.array([0]*npdim)
            qnl = np.array(MPSQNnew[idx])
            qnr = np.array(MPSQNnew[idx+1])
            if side[-1] == "l":
                qnbigl = qnl
                qnbigr = np.add.outer(sigmaqn,qnr)
            else:
                qnbigl = np.add.outer(qnl,sigmaqn)
                qnbigr = qnr

        if QR == False:
            if QNargs is not None:
                u, sigma, qnlset, v, sigma, qnrset = svd_qn.Csvd(res, qnbigl, \
                        qnbigr, MPSQNtot, full_matrices=False, IfMPO=ifMPO)
                vt = v.T
            else:
                try:
                    u,sigma,vt=scipy.linalg.svd(res,full_matrices=False,lapack_driver='gesdd')
                except:
                    print "mps compress converge failed"
                    u,sigma,vt=scipy.linalg.svd(res,full_matrices=False,lapack_driver='gesvd')

            if trunc==0:
                m_trunc=len(sigma)
            elif trunc<1.:
                # count how many sing vals < trunc            
                normed_sigma=sigma/scipy.linalg.norm(sigma)
                #m_trunc=len([s for s in normed_sigma if s >trunc])
                m_trunc = np.count_nonzero(normed_sigma>trunc)
            else:
                m_trunc=int(trunc)
                m_trunc=min(m_trunc,len(sigma))

            u=u[:,0:m_trunc]
            sigma=sigma[0:m_trunc]
            vt=vt[0:m_trunc,:]
            
            if side[-1] == "l":
                u = np.einsum('ji, i -> ji', u, sigma)
            else:
                vt = np.einsum('i, ij -> ij', sigma, vt)
        else:
            if QNargs is not None:
                if side[-1] == "l":
                    system = "R"
                else:
                    system = "L"
                u, qnlset, v, qnrset = svd_qn.Csvd(res, qnbigl, qnbigr, MPSQNtot, \
                        QR=True, system=system, full_matrices=False, IfMPO=ifMPO)
                vt = v.T
            else:
                if side[-1] == "l":
                    u,vt = scipy.linalg.rq(res, mode='economic')
                else:
                    u,vt = scipy.linalg.qr(res, mode='economic')
            m_trunc = u.shape[1]

        if side[-1] =="l":
            if direct == False:
                res=np.tensordot(mps[idx-1],u,axes=1)
            else:
                res = u.reshape(mps[idx-1][0].shape[0],mps[idx-1][1].shape[0],-1)
                res = np.tensordot(mps[idx-1][1], res, axes=([-1],[1]))
                res = np.tensordot(mps[idx-1][0], res, axes=([-2,-1],[1,-2]))
                res = np.moveaxis(res,[2],[1]).reshape([-1]+list(mps[idx-1][1].shape[1:-1])+[u.shape[-1]])
                
            ret_mpsi=np.reshape(vt,[m_trunc]+pdim+[vt.shape[1]/npdim])
            if QNargs is not None:
                MPSQNnew[idx] = qnrset[:m_trunc]
        else:
            if direct == False:
                res=np.tensordot(vt,mps[idx+1],axes=1)
            else:
                res = vt.reshape(-1,mps[idx+1][0].shape[0],mps[idx+1][1].shape[0])
                res = np.tensordot(res, mps[idx+1][1], axes=1)
                res = np.tensordot(res, mps[idx+1][0], axes=([1,2],[0,2]))
                res = np.moveaxis(res,
                        [-2,-1],[1,-2]).reshape([vt.shape[0]]+list(mps[idx+1][1].shape[1:-1])+[-1])
            
            ret_mpsi=np.reshape(u,[u.shape[0]/npdim]+pdim+[m_trunc])
            if QNargs is not None:
                MPSQNnew[idx+1] = qnlset[:m_trunc]
        
        ret_mps.append(ret_mpsi)
        
    # normalize is the norm of the MPS
    if normalize is not None:
        res = res / np.linalg.norm(np.ravel(res)) * normalize
    
    ret_mps.append(res)
    
    if side[-1] == "l":
        ret_mps.reverse()

    #fidelity = dot(conj(ret_mps), mps)/dot(conj(mps), mps)
    #print "compression fidelity:: ", fidelity
    # if np.isnan(fidelity):
    #     dddd
    if side == "ml":
        ret_mps = mps[:msite-1] + ret_mps +  mps[msite+1:]
    elif side == "mr":
        ret_mps = mps[:msite] + ret_mps +  mps[msite+2:]

    if QNargs is not None:
        if side == "l":
            MPSQNnewidx = 0
        elif side == "r":
            MPSQNnewidx = len(mps)-1
        elif side == "ml":
            MPSQNnewidx = MPSQNidx - 1
        else:
            MPSQNnewidx = MPSQNidx + 1

        ret_mps = [ret_mps, MPSQNnew, MPSQNnewidx, MPSQNtot]
    return ret_mps


def mps_fci(mps,pbond=None,direct=False):
    """
    convert MPS into a fci vector
    direct=True is tensor product of the MPS directly
    """
    if direct == False:
        if pbond is None:
            pdim=mps[0].shape[1]
            nsites=len(mps)
            confs=fci.fci_configs(nsites,pdim)
            fvec=mathutils.zeros((pdim,)*nsites)
        else:
            confs=fci.fci_configs(None,None,pbond=pbond)
            fvec=mathutils.zeros(pbond)
        
        for conf in confs:
            fvec[conf]=ceval(mps,conf)
    else:
        fvec = np.ones([1],dtype=mps[0].dtype)
        for imps in mps:
            fvec = np.tensordot(fvec, imps, axes=1)
        fvec = np.tensordot(fvec,np.ones([1],dtype=mps[0].dtype),axes=1)

    return fvec

def scale(mps,val,QNargs=None):
    """
    Multiply MPS by scalar
    """    
    if QNargs is None:
        ret=[mt.copy() for mt in mps]
        ret[-1]*=val
        return ret
    else:
        ret=[mt.copy() for mt in mps[0]]
        ret[-1]*=val
        return [ret] + copy.deepcopy(mps[1:])


def add(mpsa,mpsb,QNargs=None):
    """
    add two mps / mpo 
    """

    if mpsa is None:
        return copy.deepcopy(mpsb)
    elif mpsb is None:
        return copy.deepcopy(mpsa)
    
    if QNargs is not None:
        mpsa, mpsaQN, mpsaQNidx, mpsaQNtot = mpsa
        mpsb, mpsbQN, mpsbQNidx, mpsbQNtot = mpsb
        assert mpsaQNtot == mpsbQNtot

    assert len(mpsa)==len(mpsb)
    nsites=len(mpsa)

    mpsab=[None]*nsites
    
    if mpsa[0].ndim == 3:  # MPS
        mpsab[0]=np.dstack([mpsa[0],mpsb[0]])
        for i in xrange(1,nsites-1):
            mta=mpsa[i]
            mtb=mpsb[i]
            pdim = mta.shape[1]
            assert pdim==mtb.shape[1]
            mpsab[i]=mathutils.zeros([mta.shape[0]+mtb.shape[0],pdim,
                                  mta.shape[2]+mtb.shape[2]])
            mpsab[i][:mta.shape[0],:,:mta.shape[2]]=mta[:,:,:]
            mpsab[i][mta.shape[0]:,:,mta.shape[2]:]=mtb[:,:,:]

        mpsab[-1]=np.vstack([mpsa[-1],mpsb[-1]])
    elif mpsa[0].ndim == 4: # MPO
        mpsab[0]=np.concatenate((mpsa[0],mpsb[0]), axis=3)
        for i in xrange(1,nsites-1):
            mta=mpsa[i]
            mtb=mpsb[i]
            pdimu = mta.shape[1]
            pdimd = mta.shape[2]
            assert pdimu==mtb.shape[1]
            assert pdimd==mtb.shape[2]

            mpsab[i]=mathutils.zeros([mta.shape[0]+mtb.shape[0],pdimu,pdimd,
                                  mta.shape[3]+mtb.shape[3]])
            mpsab[i][:mta.shape[0],:,:,:mta.shape[3]]=mta[:,:,:,:]
            mpsab[i][mta.shape[0]:,:,:,mta.shape[3]:]=mtb[:,:,:,:]

        mpsab[-1]=np.concatenate((mpsa[-1],mpsb[-1]), axis=0)
    
    if QNargs is not None:
        mpsbQNnew = svd_qn.QN_construct(mpsbQN, mpsbQNidx, mpsaQNidx, mpsbQNtot)
        mpsabQN = [mpsaQN[i]+mpsbQNnew[i] for i in range(len(mpsbQNnew))]
        mpsabQN[0], mpsabQN[-1] = [0], [0]
        mpsab = [mpsab, mpsabQN, mpsaQNidx, mpsaQNtot]

    return mpsab

 

def dot(mpsa,mpsb, QNargs=None):
    """
    dot product of two mps / mpo 
    """
    if QNargs is not None:
        mpsa = mpsa[0]
        mpsb = mpsb[0]

    assert len(mpsa)==len(mpsb)
    nsites=len(mpsa)
    e0=np.eye(1,1)
    for i in xrange(nsites):
        # sum_x e0[:,x].m[x,:,:]
        e0=np.tensordot(e0,mpsb[i],1)
        # sum_ij e0[i,p,:] mpsa[i,p,:]
        # note, need to flip a (:) index onto top,
        # therefore take transpose
        if mpsa[i].ndim == 3:
            e0=np.tensordot(e0,mpsa[i],([0,1],[0,1])).T
        elif mpsa[i].ndim == 4:
            e0 = np.tensordot(e0, mpsa[i], ([0,1,2],[0,1,2])).T

    return e0[0,0]


def exp_value(bra, O, ket, QNargs=None):
    '''
    expectation value <np.conj(bra)|O|ket>
    bra and ket could be mps and mpo
    '''
    if QNargs is not None:
        bra = bra[0]
        ket = ket[0]
        O = O[0]

    bra = conj(bra)

    assert len(bra) == len(ket)
    nsites = len(bra)
    e0 = np.ones([1,1,1])
    
    for i in xrange(nsites):
        e0 = np.tensordot(e0, bra[i], axes=(0,0))
        e0 = np.tensordot(e0, O[i], axes=([0,2],[0,1]))
        
        if bra[i].ndim == 3:
            e0 = np.tensordot(e0, ket[i], axes=([0,2],[0,1]))
        elif bra[i].ndim == 4:
            e0 = np.tensordot(e0, ket[i], axes=([0,1,3],[0,2,1]))
    
    return e0[0,0,0]


def distance(mpsa,mpsb,QNargs=None):
    """
    ||mpsa-mpsb||
    """
    return dot(conj(mpsa, QNargs=QNargs),mpsa, QNargs=QNargs) \
            -dot(conj(mpsa, QNargs=QNargs),mpsb, QNargs=QNargs) \
            -dot(conj(mpsb, QNargs=QNargs),mpsa, QNargs=QNargs) \
            +dot(conj(mpsb, QNargs=QNargs),mpsb, QNargs=QNargs)


def liouville_to_hilbert(mpsl,basis):
    """
    convert liouville mps to hilbert mpo
    """
    pdim=len(basis)
    sdim=int(pdim**0.5)
    nsites=len(mpsl)
    mpoh=[None]*nsites
    for i, mt in enumerate(mpsl):
        tens=mathutils.zeros([mt.shape[0],sdim,sdim,mt.shape[2]])
        for r in xrange(mt.shape[0]):
            for s in xrange(mt.shape[2]):
                for p in xrange(pdim):
                    tens[r,:,:,s]+=mt[r,p,s]*basis[p]
        mpoh[i]=tens
    return mpoh


def create(ops):
    """
    Create MPO operator from a
    tensor product of single site operators e.g.
    I otimes c otimes d otimes ...
    """
    pdim=ops[0].shape[0]
    assert ops[0].shape[1]==pdim

    return [np.reshape(op,[1,pdim,pdim,1]) for op in ops]

def to_mps(mpo):
    """
    flatten physical indices of MPO to MPS
    """
    pdim=mpo[0].shape[1]
    assert pdim==mpo[0].shape[2]
    
    return [np.reshape(mt,[mt.shape[0],mt.shape[1]*mt.shape[2],mt.shape[3]]) for mt in mpo]


def from_mps(mps):
    """
    squaren physical indices of MPS to MPO
    """
    MPO = []
    for imps in mps:
        pdim=int(math.sqrt(imps.shape[1]))
        assert pdim*pdim==imps.shape[1]
        MPO.append(np.reshape(imps,[imps.shape[0],pdim,pdim,imps.shape[2]]))

    return MPO


def trace(mpo):
    """
    \sum_{n1n2...} A[n1n1] A[n2n2] A[nknk]
    """
    traced_mts=[np.einsum("innj",mt) for mt in mpo]
    val=traced_mts[0]
    nsites=len(mpo)
    for i in xrange(1,nsites):
        val=np.dot(val,traced_mts[i])
    return np.trace(val)#[0,0]


def transferMat(mps, mpsconj, domain, siteidx):
    '''
    calculate the transfer matrix from the left hand or the right hand
    '''
    val = np.ones([1,1])
    if domain == "R":
        for imps in range(len(mps)-1,siteidx-1,-1):
            val = np.tensordot(mpsconj[imps], val, axes=(2,0))
            val = np.tensordot(val, mps[imps], axes=([1,2],[1,2]))
    elif domain == "L":
        for imps in range(0,siteidx+1,1):
            val = np.tensordot(mpsconj[imps], val, axes=(0,0))
            val = np.tensordot(val, mps[imps], axes=([0,2],[1,0]))
        
    return val


def MPSdtype_convert(MPS, QNargs=None):
    '''
    float64 to complex128
    '''
    if  QNargs is None:
        return [mps.astype(np.complex128) for mps in MPS]
    else:
        return [[mps.astype(np.complex128) for mps in MPS[0]]] + MPS[1:]


def truncate_MPS(MPS,trphbo):
    '''
    truncte physical bond in MPS/MPO
    trphbo is the remaining # of physical bonds in each site
    '''
    assert len(MPS) == len(trphbo)
    
    MPSnew = []
    for idx, imps in enumerate(MPS):
        impsnew = np.delete(imps, np.s_[trphbo[idx]:], 1)
        # for MPO
        if imps.ndim == 4:
            impsnew = np.delete(impsnew, np.s_[trphbo[idx]:], 2)

        MPSnew.append(impsnew)
    
    return MPSnew


def norm(MPS, QNargs=None):
    '''
    normalize the MPS(MPO) wavefunction(density matrix)
    '''
    norm2 = dot(conj(MPS, QNargs=QNargs), MPS, QNargs=QNargs).real

    return np.sqrt(norm2)
