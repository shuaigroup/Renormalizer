#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg
from elementop import *
from pyscf import lib
import mps as mpslib
from constant import *

def construct_qnmat(QN, ephtable, pbond, addlist):
    '''
        construct the qn pattern, the structure is as the coefficient
    '''
    qnl = np.array(QN[addlist[0]])
    qnr = np.array(QN[addlist[-1]+1])
    qnsigmalist = []

    qnmat = qnl.copy()

    for idx in addlist:

        if ephtable[idx] == 1:
            qnsigma = np.array([0,1])
        else:
            qnsigma = np.zeros([pbond[idx]])
        
        qnmat = np.add.outer(qnmat,qnsigma)
        qnsigmalist.append(qnsigma)

    qnmat = np.add.outer(qnmat,qnr)

    return qnmat, qnl, qnr, qnsigmalist


def Csvd(cstruct, qnbigl, qnbigr, nexciton, full_matrices=True):
    '''
        block svd the coefficient matrix (l, sigmal, sigmar, r) or (l,sigma,r)
    '''
    Gamma = cstruct.reshape(np.prod(qnbigl.shape),np.prod(qnbigr.shape))
    localqnl = qnbigl.ravel()
    localqnr = qnbigr.ravel()
    
    Uset = []
    Uset0 = []
    Vset = []
    Vset0 = []
    Sset = []
    SUset0 = []
    SVset0 = []
    qnlset = []
    qnlset0 = []
    qnrset = []
    qnrset0 = []

    # different combination
    combine = [[x, nexciton-x] for x in xrange(nexciton+1)]
    for nl, nr in combine:
        lset = [i for i, x in enumerate(localqnl) if x == nl]
        rset = [i for i, x in enumerate(localqnr) if x == nr]
        if len(lset) != 0 and len(rset) != 0:
            Gamma_block = Gamma[np.ix_(lset, rset)]
            U, S, Vt = scipy.linalg.svd(Gamma_block)
            dim = S.shape[0]
            Sset.append(S)
            
            def blockappend(vset, vset0, qnset, qnset0, svset0, v, n, dim, indice, shape):
                vset.append(blockrecover(indice, v[:,:dim], shape))
                qnset += [n] * dim
                vset0.append(blockrecover(indice, v[:,dim:],shape))
                qnset0 += [n] * (v.shape[0]-dim)
                svset0.append(np.zeros(v.shape[0]-dim))
                
                return vset, vset0, qnset, qnset0, svset0
            
            Uset, Uset0, qnlset, qnlset0, SUset0 = blockappend(Uset, Uset0, qnlset, \
                    qnlset0, SUset0, U, nl, dim, lset, Gamma.shape[0])
            Vset, Vset0, qnrset, qnrset0, SVset0 = blockappend(Vset, Vset0, qnrset, \
                    qnrset0, SVset0, Vt.T, nr, dim, rset, Gamma.shape[1])
    
    if full_matrices == True:
        Uset = np.concatenate(Uset + Uset0,axis=1)
        Vset = np.concatenate(Vset + Vset0,axis=1)
        SUset = np.concatenate(Sset + SUset0)
        SVset = np.concatenate(Sset + SVset0)
        qnlset = qnlset + qnlset0
        qnrset = qnrset + qnrset0
        
        return Uset, SUset, qnlset, Vset, SVset, qnrset
    else:
        Uset = np.concatenate(Uset,axis=1)
        Vset = np.concatenate(Vset,axis=1)
        Sset = np.concatenate(Sset)
        
        return Uset, Sset, qnlset, Vset, Sset, qnrset


def blockrecover(indices, U, dim):
    '''
        recover the block element to its original position
    '''
    resortU = np.zeros([dim, U.shape[1]],dtype=U.dtype)
    for i in xrange(len(indices)):
        resortU[indices[i],:] = np.copy(U[i,:])

    return resortU


def construct_MPS_MPO_1():
    '''
    MPO/MPS structure 1
    e1,e2,e3...en,ph11,ph12,..ph21,ph22....phn1,phn2
    '''
    MPS = []
    MPO = []
    MPSdim = []
    MPOdim = []
    
    return MPS, MPO, MPSdim, MPOdim


def construct_MPS_MPO_2(mol, J, Mmax, nexciton):
    '''
    MPO/MPS structure 2
    e1,ph11,ph12,..e2,ph21,ph22,...en,phn1,phn2...
    '''
    
    # e-ph table: e site 1, ph site 0
    ephtable = []
    # physical bond dimension
    pbond = []

    nmols = len(mol)
    for imol in xrange(nmols):
        ephtable.append(1)
        pbond.append(2)
        for iph in xrange(mol[imol].nphs):
            ephtable.append(0)
            pbond.append(mol[imol].ph[iph].nlevels)
    
    print "# of MPS,", len(pbond)
    print "physical bond,", pbond
    
    '''
    initialize MPS according to quantum number
    MPSQN: mps quantum number list
    MPSdim: mps dimension list
    MPS: mps list
    '''
    MPS, MPSdim, MPSQN = construct_MPS('L', ephtable, pbond, nexciton, Mmax)
    print "initialize left-canonical:", mpslib.is_left_canonical(MPS)
    
    '''
    initialize MPO
    MPOdim: mpo dimension list
    MPO: mpo list
    '''
    MPO, MPOdim = construct_MPO(mol, J, pbond)
    
    return MPS, MPSdim, MPSQN, MPO, MPOdim, ephtable, pbond


def construct_MPO(mol, J, pbond):
    '''
    many ways to construct MPO
    scheme 1: l to r
    scheme 2: r to l
    scheme 3: l,r to middle
    '''
    MPOdim = []
    MPO = []
    nmols = len(mol)
    
    # scheme 1: see doc
    # MPOdim  
    for imol in xrange(nmols):
        MPOdim.append((imol+1)*2)
        for iph in xrange(mol[imol].nphs):
            if imol != nmols-1:
                MPOdim.append((imol+1)*2+3)
            else:
                MPOdim.append(3)
    MPOdim.append(1)
    MPOdim[0]=1
    
    # MPO
    impo = 0
    for imol in xrange(nmols):
        # omega*coupling**2: a constant for single mol 
        e0 = 0.0
        for iph in xrange(mol[imol].nphs):
            e0 += mol[imol].ph[iph].omega * mol[imol].ph[iph].ephcoup**2
        
        # electronic part
        mpo = np.zeros([pbond[impo],pbond[impo],MPOdim[impo],MPOdim[impo+1]])
        
        for ibra in xrange(pbond[impo]):
            for iket in xrange(pbond[impo]):
                mpo[ibra,iket,-1,0]  = EElementOpera("a^\dagger a", ibra, iket) * (mol[imol].elocalex +  e0)
                mpo[ibra,iket,-1,-1] = EElementOpera("Iden", ibra, iket)
                mpo[ibra,iket,-1,1]  = EElementOpera("a^\dagger a", ibra, iket)
                
                # first column operator
                if imol != 0 :
                    mpo[ibra,iket,0,0] = EElementOpera("Iden", ibra, iket)
                    for ileft in xrange(1,2*imol+1):
                        if ileft % 2 == 1:
                            mpo[ibra,iket,ileft,0] = EElementOpera("a", ibra, iket) * J[(ileft-1)/2,imol]
                        else:
                            mpo[ibra,iket,ileft,0] = EElementOpera("a^\dagger", ibra, iket) * J[(ileft-1)/2,imol]
                # last row operator
                if imol != nmols-1 :
                    mpo[ibra,iket,-1,-2] = EElementOpera("a", ibra, iket)
                    mpo[ibra,iket,-1,-3] = EElementOpera("a^\dagger", ibra, iket)
                # mat body
                if imol != nmols-1 and imol != 0:    
                    for ileft in xrange(2,2*(imol+1)):
                        mpo[ibra,iket,ileft-1,ileft] = EElementOpera("Iden", ibra, iket)
        MPO.append(mpo)
        impo += 1
        
        # phonon part
        for iph in xrange(mol[imol].nphs):
            mpo = np.zeros([pbond[impo],pbond[impo],MPOdim[impo],MPOdim[impo+1]])
            for ibra in xrange(pbond[impo]):
                for iket in xrange(pbond[impo]):
                    # first column
                    mpo[ibra,iket,0,0] = PhElementOpera("Iden", ibra, iket)
                    mpo[ibra,iket,1,0] = PhElementOpera("b_n^\dagger + b_n",ibra, iket) * \
                                         mol[imol].ph[iph].omega * mol[imol].ph[iph].ephcoup
                    mpo[ibra,iket,-1,0] = PhElementOpera("b_n^\dagger b_n", ibra, iket) * mol[imol].ph[iph].omega
                    
                    if imol != nmols-1 or iph != mol[imol].nphs-1:
                        mpo[ibra,iket,-1,-1] = PhElementOpera("Iden", ibra, iket)
                        
                        if iph != mol[imol].nphs-1: 
                            for icol in xrange(1,MPOdim[impo+1]-1):
                                mpo[ibra,iket,icol,icol] = PhElementOpera("Iden", ibra, iket)
                        else:
                            for icol in xrange(1,MPOdim[impo+1]-1):
                                mpo[ibra,iket,icol+1,icol] = PhElementOpera("Iden", ibra, iket)

            MPO.append(mpo)
            impo += 1
                    
    print "MPOdim", MPOdim
    
    return  MPO, MPOdim 


def construct_MPS(domain, ephtable, pbond, nexciton, Mmax):
    
    '''
        construct 'domain' canonical MPS according to quantum number
    '''
    
    MPS = []
    MPSQN = [[0],]
    MPSdim = [1,]

    nmps = len(pbond)

    for imps in xrange(nmps-1):
        
        if ephtable[imps] == 1:
            # e site
            qnbig = MPSQN[imps] + [x + 1 for x in MPSQN[imps]]
        else:
            # ph site 
            qnbig = MPSQN[imps] * pbond[imps]
        
        Uset = []
        Sset = []
        qnset = []

        for iblock in xrange(min(qnbig),nexciton+1):
            # find the quantum number index
            indices = [i for i, x in enumerate(qnbig) if x == iblock]
            
            if len(indices) != 0 :
                a = np.random.random([len(indices),len(indices)])
                a = a + a.T
                S, U = scipy.linalg.eigh(a=a)
                Uset.append(blockrecover(indices, U, len(qnbig)))
                Sset.append(S)
                qnset +=  [iblock]*len(indices)

        Uset = np.concatenate(Uset,axis=1)
        Sset = np.concatenate(Sset)
        mps, mpsdim, mpsqn, nouse = updatemps(Uset, Sset, qnset, Uset, nexciton, Mmax)
        # add the next mpsdim 
        MPSdim.append(mpsdim)
        MPS.append(mps.reshape(pbond[imps], MPSdim[imps], MPSdim[imps+1]))
        MPSQN.append(mpsqn)

    # the last site, nouse in initialization
    MPSQN.append([0])
    MPSdim.append(1)
    MPS.append(np.random.random([pbond[-1],MPSdim[-2],MPSdim[-1]])-0.5)
    
    print "MPSdim", MPSdim

    return MPS, MPSdim, MPSQN 


def select_basis(qnset,Sset,qnlist):
    '''
        select basis according to qnlist requirement
    '''
    svaluefilter = [s for s,num in zip(Sset,qnset) if (num in qnlist)]  
    sidx = np.argsort(svaluefilter)[::-1]
    return sidx


def GetLR(domain, siteidx, MPS, MPO, itensor=np.ones((1,1,1)), method="Scratch"):
    
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
            itensor = addone(itensor, MPS, MPO, imps, domain)
    elif method == "Enviro" :
        itensor = Enviro_read(domain, siteidx)
    elif method == "System" :
        itensor = addone(itensor, MPS, MPO, siteidx, domain)
        Enviro_write(domain,siteidx,itensor)
    
    return itensor


def addone(intensor, MPS, MPO, isite, domain):
    '''
    add one MPO/MPS site 
    S-S-
    O-O-
    S-S-
    '''
    assert domain == "L" or domain == "R"
    
    if domain == "L":
        assert intensor.shape[0] == MPS[isite].shape[1] 
        assert intensor.shape[1] == MPO[isite].shape[2] 
        '''
        S-a-S-f
            d
        O-b-O-g
            e
        S-c-S-h
        '''
        # very slow
        #outtensor = np.einsum("abc, daf, debg, ech -> fgh", intensor, MPS[isite],
        #        MPO[isite], MPS[isite]) 
        tmp1 = np.einsum("abc, debg -> acdeg", intensor,  MPO[isite]) 
        tmp2 = np.einsum("acdeg, daf -> cegf", tmp1,  MPS[isite]) 
        outtensor = np.einsum("cegf, ech -> fgh", tmp2,  MPS[isite]) 
    else:
        assert intensor.shape[0] == MPS[isite].shape[2] 
        assert intensor.shape[1] == MPO[isite].shape[3]
        '''
        -f-S-a-S
           d
        -g-O-b-O
           e
        -h-S-c-S
        '''
        # very slow
        #outtensor = np.einsum("abc, dfa, degb, ehc -> fgh", intensor, MPS[isite],
        #        MPO[isite], MPS[isite]) 
        tmp1 = np.einsum("abc, degb -> acdeg", intensor,  MPO[isite]) 
        tmp2 = np.einsum("acdeg, dfa -> cegf", tmp1,  MPS[isite]) 
        outtensor = np.einsum("cegf, ehc -> fgh", tmp2,  MPS[isite]) 

    return outtensor


def construct_enviro(MPS, MPO):
    tensor = np.ones((1,1,1))
    for idx in xrange(len(MPS)-1):
        tensor = addone(tensor, MPS, MPO, idx, "L")
        Enviro_write("L",idx,tensor)    


def Enviro_write(domain, siteidx, tensor):
    with open(domain+str(siteidx)+".npy", 'wb') as f:
        np.save(f,tensor)


def Enviro_read(domain, siteidx):
    with open(domain + str(siteidx)+".npy", 'rb') as f:
        return np.load(f)


def optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, ephtable, pbond, nexciton, Mmax, nsweeps=1, method="2site"):
    '''
        1/2 site optimization MPS 
    '''
    
    assert method in ["2site", "1site"]
    print "optimization method", method
    
    # construct the environment matrix
    construct_enviro(MPS, MPO)

 
    nMPS = len(MPS)
    if method == "1site":
        loop = [['R',i] for i in xrange(nMPS-1,-1,-1)] + [['L',i] for i in xrange(0,nMPS)]
    else:
        loop = [['R',i] for i in xrange(nMPS-1,0,-1)] + [['L',i] for i in xrange(1,nMPS)]

    ltensor = np.ones((1,1,1))
    rtensor = np.ones((1,1,1))
    energy = []

    for isweep in xrange(nsweeps):
        for system, imps in loop:
            if system == "R":
                lmethod, rmethod = "Enviro", "System"
            else:
                lmethod, rmethod = "System", "Enviro"
            
            if method == "1site":
                lsite = imps-1
                addlist = [imps]
            else:
                lsite= imps-2
                addlist = [imps-1, imps]
            
            ltensor = GetLR('L', lsite, MPS, MPO, itensor=ltensor, method=lmethod)
            rtensor = GetLR('R', imps+1, MPS, MPO, itensor=rtensor, method=rmethod)
            
            # get the quantum number pattern
            qnmat, qnl, qnr, qnsigmalist = construct_qnmat(MPSQN, ephtable,\
                    pbond, addlist)
            cshape = qnmat.shape
            
            if method == "1site":
                # hdiag
                #   S-a c f-S
                #   O-b-O-g-O
                #   S-a c f-S
                hdiag = np.einsum("aba,ccbg,fgf -> acf",ltensor,MPO[imps],rtensor)[(qnmat==nexciton)]
                # the result is more stable to use good initial guess
                # b-S-c
                #   a  
                cguess = MPS[imps].transpose((1,0,2))[qnmat==nexciton]
            else:
                # hdiag
                #   S-a c   d f-S
                #   O-b-O-e-O-g-O
                #   S-a c   d f-S
                hdiag = np.einsum("aba,ccbe,ddeg,fgf -> acdf", ltensor, MPO[imps-1],
                        MPO[imps],rtensor)[(qnmat==nexciton)]
                # the result is more stable to use good initial guess
                # b-S-c-S-e
                #   a   d
                cguess = np.einsum("abc,dce -> bade", MPS[imps-1], MPS[imps])[qnmat==nexciton]

            nonzeros = np.sum(qnmat==nexciton)
            print "Hmat dim", nonzeros
            
            count = [0]
            
            def hop(c):
                # convert c to initial structure according to qn patter
                cstruct = c1d2cmat(cshape, c, qnmat, nexciton)
                count[0] += 1
                
                if method == "1site":
                    #S-a   l-S
                    #    d  
                    #O-b-O-f-O
                    #    e 
                    #S-c   k-S
                    
                    tmp1 = np.einsum("abc,debf -> acdef", ltensor, MPO[imps])
                    tmp2 = np.einsum("acdef, adl -> cefl", tmp1, cstruct) 
                    cout = np.einsum("cefl, lfk-> cek", tmp2, rtensor)
                else:
                    #S-a       l-S
                    #    d   g 
                    #O-b-O-f-O-j-O
                    #    e   h
                    #S-c       k-S
                    
                    tmp1 = np.einsum("abc,debf -> acdef", ltensor, MPO[imps-1])
                    tmp2 = np.einsum("acdef, adgl -> cefgl", tmp1, cstruct) 
                    tmp3 = np.einsum("cefgl, ghfj -> celhj", tmp2, MPO[imps])
                    cout = np.einsum("celhj, ljk -> cehk", tmp3, rtensor)

                # convert structure c to 1d according to qn 
                return cout[qnmat==nexciton]

            precond = lambda x, e, *args: x/(hdiag-e+1e-4)
            e, c = lib.davidson(hop, cguess, precond)
            print "HC loops:", count[0]

            print "isweep, imps, e=", isweep, imps, e*au2ev
            energy.append(e)
            
            cstruct = c1d2cmat(cshape, c, qnmat, nexciton)
            
            # update the mps
            if method == "1site":
                if system == "R":
                    qnbigl = qnl
                    qnbigr = np.add.outer(qnsigmalist[-1],qnr)
                else:
                    qnbigl = np.add.outer(qnl,qnsigmalist[0])
                    qnbigr = qnr
            else:
                qnbigl = np.add.outer(qnl,qnsigmalist[0])
                qnbigr = np.add.outer(qnsigmalist[-1],qnr)

            mps, mpsdim, mpsqn, compmps = Renormalization(cstruct, qnbigl, qnbigr,\
                    system, nexciton, Mmax)
            
            if method == "1site":
                MPS[imps] = mps
                if system == "L":
                    if imps != len(MPS)-1:
                        MPS[imps+1] = np.einsum("ab,cbd -> cad", compmps, MPS[imps+1])
                        MPSdim[imps+1] = mpsdim
                        MPSQN[imps+1] = mpsqn
                    else:
                        MPS[imps] = np.einsum("cdb,ba -> cda", MPS[imps],compmps)
                        MPSdim[imps+1] = 1
                        MPSQN[imps+1] = [0]

                else:
                    if imps != 0:
                        MPS[imps-1] = np.einsum("cdb,ba -> cda", MPS[imps-1],compmps)
                        MPSdim[imps] = mpsdim
                        MPSQN[imps] = mpsqn
                    else:
                        MPS[imps] = np.einsum("ab,cbd -> cad", compmps, MPS[imps])
                        MPSdim[imps] = 1
                        MPSQN[imps] = [0]
            else:
                if system == "L":
                    MPS[imps-1] = mps
                    MPS[imps] = compmps
                else:
                    MPS[imps] = mps
                    MPS[imps-1] = compmps

                MPSdim[imps] = mpsdim
                MPSQN[imps] = mpsqn

    lowestenergy = np.min(energy)
    print "lowest energy = ", lowestenergy * au2ev

    return energy


def Renormalization(cstruct, qnbigl, qnbigr, domain, nexciton, Mmax):
    '''
        get the new mps, mpsdim, mpdqn, complementary mps to get the next guess
    '''
    assert domain=="R" or domain=="L"
    Uset, SUset, qnlnew, Vset, SVset, qnrnew = Csvd(cstruct, qnbigl, qnbigr, nexciton)
    if domain == "R":
        mps, mpsdim, mpsqn, compmps = updatemps(Vset, SVset, qnrnew, Uset, nexciton, Mmax)
        return np.moveaxis(mps.reshape(list(qnbigr.shape)+[mpsdim]),-1,-2), mpsdim, mpsqn,\
        np.moveaxis(compmps.reshape(list(qnbigl.shape) + [mpsdim]),qnbigl.ndim-1,0)
    else:    
        mps, mpsdim, mpsqn, compmps = updatemps(Uset, SUset, qnlnew, Vset, nexciton, Mmax)
        return np.moveaxis(mps.reshape(list(qnbigl.shape) + [mpsdim]),qnbigl.ndim-1,0), mpsdim, mpsqn,\
                np.moveaxis(compmps.reshape(list(qnbigr.shape)+[mpsdim]),-1,-2)
    

def updatemps(vset, sset, qnset, compset, nexciton, Mmax):
    '''
        select basis to construct new mps, and complementary mps
    '''
    sidx = select_basis(qnset,sset,range(nexciton+1))
    mpsdim = min(len(sidx),Mmax)
    mps = np.zeros((vset.shape[0], mpsdim))
    
    compmps = np.zeros((compset.shape[0],mpsdim))

    mpsqn = []
    stot = 0.0
    for idim in xrange(mpsdim):
        mps[:, idim] = vset[:, sidx[idim]].copy()
        if sidx[idim] < compset.shape[1]:
            compmps[:,idim] = compset[:, sidx[idim]].copy() * sset[sidx[idim]]
        mpsqn.append(qnset[sidx[idim]])
        stot += sset[sidx[idim]]**2
    
    print "discard:", 1.0-stot

    return mps, mpsdim, mpsqn, compmps
    

def c1d2cmat(cshape, c, qnmat, nexciton):
    cstruct = np.zeros(cshape)
    np.place(cstruct, qnmat==nexciton, c)

    return cstruct


if __name__ == '__main__':
    import numpy as np
    from obj import *
    import scipy.constants 
    from constant import * 

    elocalex = 2.67/au2ev
    dipole_abs = 15.45
    nmols = 2
    # eV
    J = np.zeros((2,2))
    J += np.diag([-0.1],k=1)
    J += np.diag([-0.1],k=-1)
    print "J=", J
    
    # cm^-1
    omega1 = np.array([106.51, 1555.55])
    
    # a.u.
    D1 = np.array([30.1370, 8.7729])
    
    # 1
    S1 = np.array([0.2204, 0.2727])
    
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
    
    nphs = 2
    nlevels =  [7,7]
    
    phinfo = [list(a) for a in zip(omega1, nphcoup1, nlevels)]
    
    mol = []
    for imol in xrange(nmols):
        mol_local = Mol(elocalex, nphs, dipole_abs)
        mol_local.create_ph(phinfo)
        mol.append(mol_local)
    
    Mmax = 100
    nexciton = 1
    
    MPS, MPSdim, MPSQN, MPO, MPOdim, ephtable, pbond = construct_MPS_MPO_2(mol, J,\
            Mmax, nexciton)
    
    optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, ephtable, pbond, nexciton, Mmax,\
            nsweeps=4, method="1site")

    print "MPS output left canonical:", mpslib.is_left_canonical(MPS)
    
    print "MPSdim",[i.shape[1] for i in MPS]
    
