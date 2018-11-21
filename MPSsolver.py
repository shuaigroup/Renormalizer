# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
MPS/MPO structure ground state calculation solver
'''

import numpy as np
import scipy.linalg
import itertools
from pyscf import lib
from lib import mps as mpslib
from constant import *
from mpompsmat import *
from elementop import *
from lib import tensor as tensorlib
from ephMPS import svd_qn
from ephMPS import quasiboson

def construct_MPS_MPO_1():
    '''
    MPO/MPS structure 1
    e1,e2,e3...en,ph11,ph12,..ph21,ph22....phn1,phn2
    not implemented yet
    '''
    MPS = []
    MPO = []
    MPSdim = []
    MPOdim = []
    
    return MPS, MPO, MPSdim, MPOdim


def construct_MPS_MPO_2(mol, J, Mmax, nexciton, MPOscheme=2, rep="star"):
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
            if mol[imol].ph[iph].nqboson == 1:
                ephtable.append(0)
                pbond.append(mol[imol].ph[iph].nlevels)
            else:
                ephtable += [0] * mol[imol].ph[iph].nqboson
                pbond += [mol[imol].ph[iph].base] * mol[imol].ph[iph].nqboson

    
    print "# of MPS,", len(pbond)
    print "physical bond,", pbond
    print "ephtable", ephtable

    '''
    initialize MPS according to quantum number
    MPSQN: mps quantum number list
    MPSdim: mps dimension list
    MPS: mps list
    '''
    MPS, MPSdim, MPSQN = construct_MPS('L', ephtable, pbond, nexciton, Mmax, percent=1.0)
    print "initialize left-canonical:", mpslib.is_left_canonical(MPS)
    
    '''
    initialize MPO
    MPOdim: mpo dimension list
    MPO: mpo list
    '''
    MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot = construct_MPO(mol, J, pbond, \
            scheme=MPOscheme, rep=rep)
    
    return MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond


def construct_MPS(domain, ephtable, pbond, nexciton, Mmax, percent=0):
    '''
    construct 'domain' canonical MPS according to quantum number
    '''
    
    MPS = []
    MPSQN = [[0],]
    MPSdim = [1,]
 
    nmps = len(pbond)

    for imps in xrange(nmps-1):
        
        # quantum number 
        if ephtable[imps] == 1:
            # e site
            qnbig = list(itertools.chain.from_iterable([x, x+1] for x in MPSQN[imps]))
        else:
            # ph site 
            qnbig = list(itertools.chain.from_iterable([x]*pbond[imps] for x in MPSQN[imps]))
        
        Uset = []
        Sset = []
        qnset = []

        for iblock in xrange(min(qnbig),nexciton+1):
            # find the quantum number index
            indices = [i for i, x in enumerate(qnbig) if x == iblock]
            
            if len(indices) != 0 :
                a = np.random.random([len(indices),len(indices)])-0.5
                a = a + a.T
                S, U = scipy.linalg.eigh(a=a)
                Uset.append(svd_qn.blockrecover(indices, U, len(qnbig)))
                Sset.append(S)
                qnset +=  [iblock]*len(indices)

        Uset = np.concatenate(Uset,axis=1)
        Sset = np.concatenate(Sset)
        mps, mpsdim, mpsqn, nouse = updatemps(Uset, Sset, qnset, Uset, nexciton,\
                Mmax, percent=percent)
        # add the next mpsdim 
        MPSdim.append(mpsdim)
        MPS.append(mps.reshape(MPSdim[imps], pbond[imps], MPSdim[imps+1]))
        MPSQN.append(mpsqn)

    # the last site
    MPSQN.append([0])
    MPSdim.append(1)
    MPS.append(np.random.random([MPSdim[-2],pbond[-1],MPSdim[-1]])-0.5)
    
    print "MPSdim", MPSdim

    return MPS, MPSdim, MPSQN 


def updatemps(vset, sset, qnset, compset, nexciton, Mmax, percent=0):
    '''
    select basis to construct new mps, and complementary mps
    vset, compset is the column vector
    '''
    sidx = select_basis(qnset,sset,range(nexciton+1), Mmax, percent=percent)
    mpsdim = len(sidx)
    mps = np.zeros((vset.shape[0], mpsdim),dtype=vset.dtype)
    
    if compset is not None:
        compmps = np.zeros((compset.shape[0],mpsdim), dtype=compset.dtype)
    else:
        compmps = None

    mpsqn = []
    stot = 0.0
    for idim in xrange(mpsdim):
        mps[:, idim] = vset[:, sidx[idim]].copy()
        if (compset is not None) and sidx[idim]<compset.shape[1]:
            compmps[:,idim] = compset[:, sidx[idim]].copy() * sset[sidx[idim]]
        mpsqn.append(qnset[sidx[idim]])
        stot += sset[sidx[idim]]**2
    
    print "discard:", 1.0-stot

    return mps, mpsdim, mpsqn, compmps


def select_basis(qnset,Sset,qnlist,Mmax,percent=0):
    '''
    select basis according to Sset under qnlist requirement
    '''

    # convert to dict
    basdic = {}
    for i in xrange(len(qnset)):
        basdic[i] = [qnset[i],Sset[i]]
    
    # clean quantum number outside qnlist
    for ibas in basdic.iterkeys():
        if basdic[ibas][0] not in qnlist:
            del basdic[ibas]

    # each good quantum number block equally get percent/nblocks
    def block_select(basdic, qn, n):
        block_basdic = {i:basdic[i] for i in basdic if basdic[i][0]==qn}
        sort_block_basdic = sorted(block_basdic.items(), key=lambda x: x[1][1], reverse=True)
        nget = min(n, len(sort_block_basdic))
        print qn, "block # of retained basis", nget
        sidx = [i[0] for i in sort_block_basdic[0:nget]]
        for idx in sidx:
            del basdic[idx]

        return sidx

    nbasis = min(len(basdic), Mmax)
    print "# of selected basis", nbasis
    sidx = []
    
    # equally select from each quantum number block
    if percent != 0:
        nbas_block = int(nbasis * percent / len(qnlist))
        for iqn in qnlist:
            sidx += block_select(basdic, iqn, nbas_block)
    
    # others 
    nbasis = nbasis - len(sidx)
    
    sortbasdic = sorted(basdic.items(), key=lambda x: x[1][1], reverse=True)
    sidx += [i[0] for i in sortbasdic[0:nbasis]]

    assert len(sidx) == len(set(sidx))  # there must be no duplicated

    return sidx


def construct_MPO(mol, J, pbond, scheme=2, rep="star", elocal_offset=None):
    '''
    scheme 1: l to r
    scheme 2: l,r to middle, the bond dimension is smaller than scheme 1
    scheme 3: l to r, nearest neighbour exciton interaction 
    rep (representation) has "star" or "chain"
    please see doc
    '''
    assert rep in ["star", "chain"]

    MPOdim = []
    MPO = []
    nmols = len(mol)
    MPOQN = []
    
    # used in the hybrid TDDMRG/TDH algorithm
    if elocal_offset is not None:
        assert len(elocal_offset) == nmols

    # MPOdim  
    if scheme == 1:
        for imol in xrange(nmols):
            MPOdim.append((imol+1)*2)
            MPOQN.append([0]+[1,-1]*imol+[0])
            for iph in xrange(mol[imol].nphs):
                if imol != nmols-1:
                    MPOdim.append((imol+1)*2+3)
                    MPOQN.append([0,0]+[1,-1]*(imol+1)+[0])
                else:
                    MPOdim.append(3)
                    MPOQN.append([0,0,0])
    elif scheme == 2:
        # 0,1,2,3,4,5      3 is the middle 
        # dim is 1*4, 4*6, 6*8, 8*6, 6*4, 4*1 
        # 0,1,2,3,4,5,6    3 is the middle 
        # dim is 1*4, 4*6, 6*8, 8*8, 8*6, 6*4, 4*1 
        mididx = nmols/2

        def elecdim(imol):
            if imol <= mididx:
                dim = (imol+1)*2
            else:
                dim = (nmols-imol+1)*2
            return dim

        for imol in xrange(nmols):
            ldim = elecdim(imol)
            rdim = elecdim(imol+1)

            MPOdim.append(ldim)
            MPOQN.append([0]+[1,-1]*(ldim/2-1)+[0])   
            for iph in xrange(mol[imol].nphs):
                if rep == "chain":
                    if iph == 0:
                        MPOdim.append(rdim+1)
                        MPOQN.append([0,0]+[1,-1]*(rdim/2-1)+[0]) 
                    else:
                        # replace the initial a^+a to b^+ and b
                        MPOdim.append(rdim+2)  
                        MPOQN.append([0,0,0]+[1,-1]*(rdim/2-1)+[0]) 
                else:
                    MPOdim.append(rdim+1)
                    MPOQN.append([0,0]+[1,-1]*(rdim/2-1)+[0]) 
    elif scheme == 3:
        # electronic nearest neighbor hopping
        # the electronic dimension is
        # 1*4, 4*4, 4*4,...,4*1
        for imol in xrange(nmols):
            MPOdim.append(4)
            MPOQN.append([0,1,-1,0])
            for iph in xrange(mol[imol].nphs):
                if imol != nmols-1:
                    MPOdim.append(5)
                    MPOQN.append([0,0,1,-1,0])
                else:
                    MPOdim.append(3)
                    MPOQN.append([0,0,0])
    
    MPOdim[0]=1
    
    # quasi boson MPO dim
    qbopera = []   # b+b^\dagger MPO in quasi boson representation
    MPOdimnew = []   
    MPOQNnew = []
    impo = 0
    for imol in xrange(nmols):
        qbopera.append({})
        MPOdimnew.append(MPOdim[impo])
        MPOQNnew.append(MPOQN[impo]) 
        impo += 1
        for iph in xrange(mol[imol].nphs):
            nqb = mol[imol].ph[iph].nqboson
            if nqb != 1:
                if rep == "chain":
                    b = quasiboson.Quasi_Boson_MPO("b", nqb,\
                            mol[imol].ph[iph].qbtrunc, base=mol[imol].ph[iph].base)
                    bdagger = quasiboson.Quasi_Boson_MPO("b^\dagger", nqb,\
                            mol[imol].ph[iph].qbtrunc, base=mol[imol].ph[iph].base)
                    bpbdagger = quasiboson.Quasi_Boson_MPO("b + b^\dagger", nqb,\
                            mol[imol].ph[iph].qbtrunc, base=mol[imol].ph[iph].base)
                    qbopera[imol]["b"+str(iph)] = b
                    qbopera[imol]["bdagger"+str(iph)] = bdagger
                    qbopera[imol]["bpbdagger"+str(iph)] = bpbdagger

                    if iph == 0:
                        if iph != mol[imol].nphs-1:
                            addmpodim = [b[i].shape[0]+bdagger[i].shape[0]+bpbdagger[i].shape[0]-1 for i in range(nqb)]
                        else:
                            addmpodim = [bpbdagger[i].shape[0]-1 for i in range(nqb)]
                        addmpodim[0] = 0
                    else:
                        addmpodim = [(b[i].shape[0]+bdagger[i].shape[0])*2-2 for i in range((nqb))]
                        addmpodim[0] = 0

                else:
                    bpbdagger = quasiboson.Quasi_Boson_MPO("C1(b + b^\dagger) + C2(b + b^\dagger)^2", nqb,\
                            mol[imol].ph[iph].qbtrunc, \
                            base=mol[imol].ph[iph].base,\
                            C1=mol[imol].ph[iph].omega[1]**2/np.sqrt(2.*mol[imol].ph[iph].omega[0])*-mol[imol].ph[iph].dis[1], \
                            C2=0.25*(mol[imol].ph[iph].omega[1]**2-mol[imol].ph[iph].omega[0]**2)/mol[imol].ph[iph].omega[0])

                    qbopera[imol]["bpbdagger"+str(iph)] = bpbdagger
                    addmpodim = [i.shape[0] for i in bpbdagger]
                    addmpodim[0] = 0  
                    # the first quasi boson MPO the row dim is as before, while
                    # the others the a_i^\dagger a_i should exist
            else:
                addmpodim = [0]
            
            # new MPOdim
            MPOdimnew += [i + MPOdim[impo] for i in addmpodim]
            # new MPOQN
            for iqb in xrange(nqb):
                MPOQNnew.append(MPOQN[impo][0:1] + [0]* addmpodim[iqb] + MPOQN[impo][1:])
            impo += 1

    print "original MPOdim", MPOdim + [1]
    
    MPOdim = MPOdimnew
    MPOQN = MPOQNnew

    MPOdim.append(1)
    MPOQN[0] = [0]
    MPOQN.append([0])
    # the boundary site of L/R side quantum number
    MPOQNidx = len(MPOQN)-2 
    MPOQNtot = 0     # the total quantum number of each bond, for Hamiltonian it's 0              
        
    print "MPOdim", MPOdim

    # MPO
    impo = 0
    for imol in xrange(nmols):
        
        mididx = nmols/2
        
        # electronic part
        mpo = np.zeros([MPOdim[impo],pbond[impo],pbond[impo],MPOdim[impo+1]])
        for ibra in xrange(pbond[impo]):
            for iket in xrange(pbond[impo]):
                # last row operator
                elocal = mol[imol].elocalex 
                if elocal_offset is not None:
                    elocal += elocal_offset[imol]
                mpo[-1,ibra,iket,0]  = EElementOpera("a^\dagger a", ibra, iket)\
                    * (elocal + mol[imol].e0)
                mpo[-1,ibra,iket,-1] = EElementOpera("Iden", ibra, iket)
                mpo[-1,ibra,iket,1]  = EElementOpera("a^\dagger a", ibra, iket)
                
                # first column operator
                if imol != 0 :
                    mpo[0,ibra,iket,0] = EElementOpera("Iden", ibra, iket)
                    if (scheme==1) or (scheme==2 and imol<=mididx):
                        for ileft in xrange(1,MPOdim[impo]-1):
                            if ileft % 2 == 1:
                                mpo[ileft,ibra,iket,0] = EElementOpera("a", ibra, iket) * J[(ileft-1)/2,imol]
                            else:
                                mpo[ileft,ibra,iket,0] = EElementOpera("a^\dagger", ibra, iket) * J[(ileft-1)/2,imol]
                    elif (scheme == 2 and imol > mididx):
                         mpo[-3,ibra,iket,0] = EElementOpera("a", ibra, iket) 
                         mpo[-2,ibra,iket,0] = EElementOpera("a^\dagger", ibra, iket)
                    elif scheme == 3:
                         mpo[-3,ibra,iket,0] = EElementOpera("a", ibra, iket) * J[imol-1,imol]
                         mpo[-2,ibra,iket,0] = EElementOpera("a^\dagger", ibra, iket) * J[imol-1,imol]


                # last row operator
                if imol != nmols-1 :
                    if (scheme==1) or (scheme==2 and imol<mididx) or (scheme==3):
                        mpo[-1,ibra,iket,-2] = EElementOpera("a", ibra, iket)
                        mpo[-1,ibra,iket,-3] = EElementOpera("a^\dagger", ibra, iket)
                    elif scheme == 2 and imol >= mididx:
                        for jmol in xrange(imol+1,nmols):
                            mpo[-1,ibra,iket,(nmols-jmol)*2] = EElementOpera("a^\dagger", ibra, iket) * J[imol,jmol]
                            mpo[-1,ibra,iket,(nmols-jmol)*2+1] = EElementOpera("a", ibra, iket) * J[imol,jmol]

                # mat body
                if imol != nmols-1 and imol != 0:    
                    if (scheme==1) or (scheme==2 and (imol < mididx)):
                        for ileft in xrange(2,2*(imol+1)):
                            mpo[ileft-1,ibra,iket,ileft] = EElementOpera("Iden", ibra, iket)
                    elif (scheme==2 and (imol > mididx)):
                        for ileft in xrange(2,2*(nmols-imol)):
                            mpo[ileft-1,ibra,iket,ileft] = EElementOpera("Iden", ibra, iket)
                    elif (scheme==2 and imol==mididx):
                        for jmol in xrange(imol+1,nmols):
                            for ileft in xrange(imol):
                                mpo[ileft*2+1,ibra,iket,(nmols-jmol)*2] = EElementOpera("Iden", ibra, iket) * J[ileft,jmol]
                                mpo[ileft*2+2,ibra,iket,(nmols-jmol)*2+1] = EElementOpera("Iden", ibra, iket) * J[ileft,jmol]
                    # scheme 3 no body mat

        MPO.append(mpo)
        impo += 1
        
        # # of electronic operators retained in the phonon part, only used in
        # quasiboson algorithm
        if rep == "chain":
            # except E and a^\dagger a
            nIe = MPOdim[impo]-2

        # phonon part
        for iph in xrange(mol[imol].nphs):
            nqb = mol[imol].ph[iph].nqboson 
            if nqb == 1:
                mpo = np.zeros([MPOdim[impo],pbond[impo],pbond[impo],MPOdim[impo+1]])
                for ibra in xrange(pbond[impo]):
                    for iket in xrange(pbond[impo]):
                        # first column
                        mpo[0,ibra,iket,0] = PhElementOpera("Iden", ibra, iket)
                        mpo[-1,ibra,iket,0] = PhElementOpera("b^\dagger b",\
                                ibra, iket) * mol[imol].ph[iph].omega[0]  \
                                + PhElementOpera("(b^\dagger + b)^3",ibra, iket)*\
                                mol[imol].ph[iph].force3rd[0] * (0.5/mol[imol].ph[iph].omega[0])**1.5
                        if rep == "chain" and iph != 0:
                            mpo[1,ibra,iket,0] = PhElementOpera("b",ibra, iket) * \
                                             mol[imol].phhop[iph,iph-1]
                            mpo[2,ibra,iket,0] = PhElementOpera("b^\dagger",ibra, iket) * \
                                             mol[imol].phhop[iph,iph-1]
                        else:
                            mpo[1,ibra,iket,0] = PhElementOpera("b^\dagger + b",ibra, iket) * \
                                             (mol[imol].ph[iph].omega[1]**2 /\
                                                     np.sqrt(2.*mol[imol].ph[iph].omega[0]) * -mol[imol].ph[iph].dis[1] \
                                              + 3.0*mol[imol].ph[iph].dis[1]**2*mol[imol].ph[iph].force3rd[1]/\
                                              np.sqrt(2.*mol[imol].ph[iph].omega[0])) \
                                              + PhElementOpera("(b^\dagger + b)^2",ibra, iket) * \
                                             (0.25*(mol[imol].ph[iph].omega[1]**2-mol[imol].ph[iph].omega[0]**2)/mol[imol].ph[iph].omega[0]\
                                              - 1.5*mol[imol].ph[iph].dis[1]*mol[imol].ph[iph].force3rd[1]/mol[imol].ph[iph].omega[0])\
                                              + PhElementOpera("(b^\dagger + b)^3",ibra, iket) * \
                                              (mol[imol].ph[iph].force3rd[1]-mol[imol].ph[iph].force3rd[0])*(0.5/mol[imol].ph[iph].omega[0])**1.5

                        
                        if imol != nmols-1 or iph != mol[imol].nphs-1:
                            mpo[-1,ibra,iket,-1] = PhElementOpera("Iden", ibra, iket)
                            
                            if rep == "chain":
                                if iph == 0:
                                    mpo[-1,ibra,iket,1] = PhElementOpera("b^\dagger",ibra, iket) 
                                    mpo[-1,ibra,iket,2] = PhElementOpera("b",ibra, iket) 
                                    for icol in xrange(3,MPOdim[impo+1]-1):
                                        mpo[icol-1,ibra,iket,icol] = PhElementOpera("Iden", ibra, iket)
                                elif iph == mol[imol].nphs-1:
                                    for icol in xrange(1,MPOdim[impo+1]-1):
                                        mpo[icol+2,ibra,iket,icol] = PhElementOpera("Iden", ibra, iket)
                                else:
                                    mpo[-1,ibra,iket,1] = PhElementOpera("b^\dagger",ibra, iket) 
                                    mpo[-1,ibra,iket,2] = PhElementOpera("b",ibra, iket) 
                                    for icol in xrange(3,MPOdim[impo+1]-1):
                                        mpo[icol,ibra,iket,icol] = PhElementOpera("Iden", ibra, iket)

                            elif rep == "star":
                                if iph != mol[imol].nphs-1: 
                                    for icol in xrange(1,MPOdim[impo+1]-1):
                                        mpo[icol,ibra,iket,icol] = PhElementOpera("Iden", ibra, iket)
                                else:
                                    for icol in xrange(1,MPOdim[impo+1]-1):
                                        mpo[icol+1,ibra,iket,icol] = PhElementOpera("Iden", ibra, iket)
                MPO.append(mpo)
                impo += 1
            else:
                # b + b^\dagger in quasiboson representation
                for iqb in xrange(nqb):
                    mpo = np.zeros([MPOdim[impo],pbond[impo],pbond[impo],MPOdim[impo+1]])
                    
                    if rep == "star":
                        bpbdagger = qbopera[imol]["bpbdagger"+str(iph)][iqb]
                        
                        for ibra in xrange(mol[imol].ph[iph].base):
                            for iket in xrange(mol[imol].ph[iph].base):
                                mpo[0,ibra,iket,0] = PhElementOpera("Iden", ibra, iket)
                                mpo[-1,ibra,iket,0] = PhElementOpera("b^\dagger b", \
                                        ibra, iket) * mol[imol].ph[iph].omega[0] * \
                                        float(mol[imol].ph[iph].base)**(nqb-iqb-1)
                                
                                #  the # of identity operator 
                                if iqb != nqb-1:
                                    nI = MPOdim[impo+1]-bpbdagger.shape[-1]-1   
                                else:
                                    nI = MPOdim[impo+1]-1
                                
                                for iset in xrange(1,nI+1):
                                    mpo[-iset,ibra,iket,-iset] = PhElementOpera("Iden", ibra, iket)
                        
                        # b + b^\dagger 
                        if iqb != nqb-1:
                            mpo[1:bpbdagger.shape[0]+1,:,:,1:bpbdagger.shape[-1]+1] = bpbdagger
                        else:
                            mpo[1:bpbdagger.shape[0]+1,:,:,0:bpbdagger.shape[-1]] = bpbdagger 

                    elif rep == "chain":

                        b = qbopera[imol]["b"+str(iph)][iqb]
                        bdagger = qbopera[imol]["bdagger"+str(iph)][iqb]
                        bpbdagger = qbopera[imol]["bpbdagger"+str(iph)][iqb]
                        
                        for ibra in xrange(mol[imol].ph[iph].base):
                            for iket in xrange(mol[imol].ph[iph].base):
                                mpo[0,ibra,iket,0] = PhElementOpera("Iden", ibra, iket)
                                mpo[-1,ibra,iket,0] = PhElementOpera("b^\dagger b", \
                                        ibra, iket) * mol[imol].ph[iph].omega[0] * \
                                        float(mol[imol].ph[iph].base)**(nqb-iqb-1)
                                
                                #  the # of identity operator 
                                if impo == len(MPOdim)-2:
                                    nI = nIe-1
                                else:
                                    nI = nIe
                                
                                print "nI", nI
                                for iset in xrange(1,nI+1):
                                    mpo[-iset,ibra,iket,-iset] = PhElementOpera("Iden", ibra, iket)
                        
                        if iph == 0:
                            # b + b^\dagger 
                            if iqb != nqb-1:
                                mpo[1:bpbdagger.shape[0]+1,:,:,1:bpbdagger.shape[-1]+1] = bpbdagger
                            else:
                                mpo[1:bpbdagger.shape[0]+1,:,:,0:1] = \
                                    bpbdagger * \
                                    mol[imol].ph[iph].omega[1]**2/np.sqrt(2.*mol[imol].ph[iph].omega[0])\
                                    * -mol[imol].ph[iph].dis[1]
                        else:
                            # b^\dagger, b
                            if iqb != nqb-1:
                                mpo[1:b.shape[0]+1,:,:,1:b.shape[-1]+1] = b
                                mpo[b.shape[0]+1:b.shape[0]+1+bdagger.shape[0],:,:,\
                                        b.shape[-1]+1:b.shape[-1]+1+bdagger.shape[-1]] = bdagger
                            else:
                                mpo[1:b.shape[0]+1,:,:,0:1] = b*mol[imol].phhop[iph,iph-1]
                                mpo[b.shape[0]+1:b.shape[0]+1+bdagger.shape[0],:,:,0:1]\
                                        = bdagger*mol[imol].phhop[iph,iph-1]
                        
                        if iph != mol[imol].nphs-1:
                            if iph == 0:
                                loffset = bpbdagger.shape[0]
                                roffset = bpbdagger.shape[-1]
                            else:
                                loffset = b.shape[0] + bdagger.shape[0] 
                                roffset = b.shape[-1] + bdagger.shape[-1] 
                            # b^\dagger, b     
                            if iqb == 0:
                                mpo[-1:,:,:,roffset+1:roffset+1+bdagger.shape[-1]] = bdagger
                                mpo[-1:,:,:,roffset+1+bdagger.shape[-1]:roffset+1+bdagger.shape[-1]+b.shape[-1]] = b
                            elif iqb == nqb-1:
                                print "He",loffset+1,\
                                loffset+1+bdagger.shape[0],loffset+1+bdagger.shape[0]+b.shape[0],
                                mpo[loffset+1:loffset+1+bdagger.shape[0],:,:,1:2] = bdagger
                                mpo[loffset+1+bdagger.shape[0]:loffset+1+bdagger.shape[0]+b.shape[0],:,:,2:3] = b
                            else:
                                mpo[loffset+1:loffset+1+bdagger.shape[0],:,:,\
                                        roffset+1:roffset+1+bdagger.shape[-1]] = bdagger
                                mpo[loffset+1+bdagger.shape[0]:loffset+1+bdagger.shape[0]+b.shape[0],:,:,\
                                        roffset+1+bdagger.shape[-1]:roffset+1+bdagger.shape[-1]+b.shape[-1]] = b
                                
                    
                    MPO.append(mpo)
                    impo += 1
                
    return  MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot


def optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, ephtable, pbond, nexciton,\
        procedure, method="2site", nroots=1, inverse=1.0):
    '''
    1 or 2 site optimization procedure
    inverse = 1.0 / -1.0 
    -1.0 to get the largest eigenvalue
    '''
    
    assert method in ["2site", "1site"]
    print "optimization method", method
    
    # construct the environment matrix
    construct_enviro(MPS, MPS, MPO, "L")

    nMPS = len(MPS)
    # construct each sweep cycle scheme
    if method == "1site":
        loop = [['R',i] for i in xrange(nMPS-1,-1,-1)] + [['L',i] for i in xrange(0,nMPS)]
    else:
        loop = [['R',i] for i in xrange(nMPS-1,0,-1)] + [['L',i] for i in xrange(1,nMPS)]
    
    # initial matrix   
    ltensor = np.ones((1,1,1))
    rtensor = np.ones((1,1,1))
    
    energy = []
    for isweep in xrange(len(procedure)):
        print "Procedure", procedure[isweep]

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
            
            ltensor = GetLR('L', lsite, MPS, MPS, MPO, itensor=ltensor, method=lmethod)
            rtensor = GetLR('R', imps+1, MPS, MPS, MPO, itensor=rtensor, method=rmethod)
            
            # get the quantum number pattern
            qnmat, qnbigl, qnbigr = construct_qnmat(MPSQN, ephtable,
                    pbond, addlist, method, system)
            cshape = qnmat.shape
            
            # hdiag
            tmp_ltensor = np.einsum("aba -> ba",ltensor)
            tmp_MPOimps = np.einsum("abbc -> abc",MPO[imps])
            tmp_rtensor = np.einsum("aba -> ba",rtensor)
            if method == "1site":
                #   S-a c f-S
                #   O-b-O-g-O
                #   S-a c f-S
                path = [([0, 1],"ba, bcg -> acg"),\
                        ([1, 0],"acg, gf -> acf")]
                hdiag = tensorlib.multi_tensor_contract(path, tmp_ltensor,
                        tmp_MPOimps, tmp_rtensor)[(qnmat==nexciton)]
                # initial guess   b-S-c 
                #                   a    
                cguess = MPS[imps][qnmat==nexciton]
            else:
                #   S-a c   d f-S
                #   O-b-O-e-O-g-O
                #   S-a c   d f-S
                tmp_MPOimpsm1 = np.einsum("abbc -> abc",MPO[imps-1])
                path = [([0, 1],"ba, bce -> ace"),\
                        ([0, 1],"edg, gf -> edf"),\
                        ([0, 1],"ace, edf -> acdf")]
                hdiag = tensorlib.multi_tensor_contract(path, tmp_ltensor,
                        tmp_MPOimpsm1, tmp_MPOimps, tmp_rtensor)[(qnmat==nexciton)]
                # initial guess b-S-c-S-e
                #                 a   d
                cguess = np.tensordot(MPS[imps-1], MPS[imps], axes=1)[qnmat==nexciton]
            
            hdiag *= inverse
            nonzeros = np.sum(qnmat==nexciton)
            print "Hmat dim", nonzeros
            
            count = [0]
            def hop(c):
                # convert c to initial structure according to qn pattern
                cstruct = c1d2cmat(cshape, c, qnmat, nexciton)
                count[0] += 1
                
                if method == "1site":
                    #S-a   l-S
                    #    d  
                    #O-b-O-f-O
                    #    e 
                    #S-c   k-S
                    
                    path = [([0, 1],"abc, adl -> bcdl"),\
                            ([2, 0],"bcdl, bdef -> clef"),\
                            ([1, 0],"clef, lfk -> cek")]
                    cout = tensorlib.multi_tensor_contract(path, ltensor,
                            cstruct, MPO[imps], rtensor)
                else:
                    #S-a       l-S
                    #    d   g 
                    #O-b-O-f-O-j-O
                    #    e   h
                    #S-c       k-S
                    path = [([0, 1],"abc, adgl -> bcdgl"),\
                            ([3, 0],"bcdgl, bdef -> cglef"),\
                            ([2, 0],"cglef, fghj -> clehj"),\
                            ([1, 0],"clehj, ljk -> cehk")]
                    cout = tensorlib.multi_tensor_contract(path, ltensor,
                            cstruct, MPO[imps-1], MPO[imps], rtensor)
                # convert structure c to 1d according to qn 
                return inverse*cout[qnmat==nexciton]
            
            if nroots != 1:
                cguess = [cguess]
                for iroot in xrange(nroots-1):
                    cguess += [np.random.random([nonzeros])-0.5]
            
            precond = lambda x, e, *args: x/(hdiag-e+1e-4)
            e, c = lib.davidson(hop, cguess, precond, max_cycle=100,\
                    nroots=nroots) 
            # scipy arpack solver : much slower than davidson
            #A = scipy.sparse.linalg.LinearOperator((nonzeros,nonzeros), matvec=hop)
            #e, c = scipy.sparse.linalg.eigsh(A,k=1, which="SA",v0=cguess)
            print "HC loops:", count[0]
            print "isweep, imps, e=", isweep, imps, e
            
            energy.append(e)
            
            cstruct = c1d2cmat(cshape, c, qnmat, nexciton, nroots=nroots)

            if nroots == 1:
                # direct svd the coefficient matrix
                mps, mpsdim, mpsqn, compmps = Renormalization_svd(cstruct, qnbigl, qnbigr,\
                        system, nexciton, procedure[isweep][0], percent=procedure[isweep][1])
            else:
                # diagonalize the reduced density matrix
                mps, mpsdim, mpsqn, compmps = Renormalization_ddm(cstruct, qnbigl, qnbigr,\
                        system, nexciton, procedure[isweep][0], percent=procedure[isweep][1])
                
            if method == "1site":
                MPS[imps] = mps
                if system == "L":
                    if imps != len(MPS)-1:
                        MPS[imps+1] = np.tensordot(compmps, MPS[imps+1], axes=1)
                        MPSdim[imps+1] = mpsdim
                        MPSQN[imps+1] = mpsqn
                    else:
                        MPS[imps] = np.tensordot(MPS[imps],compmps, axes=1)
                        MPSdim[imps+1] = 1
                        MPSQN[imps+1] = [0]

                else:
                    if imps != 0:
                        MPS[imps-1] = np.tensordot(MPS[imps-1],compmps, axes=1)
                        MPSdim[imps] = mpsdim
                        MPSQN[imps] = mpsqn
                    else:
                        MPS[imps] = np.tensordot(compmps, MPS[imps], axes=1)
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
        

    if nroots == 1:
        lowestenergy = np.min(energy)
        print "lowest energy = ", lowestenergy

    return energy


def construct_qnmat(QN, ephtable, pbond, addlist, method, system):
    '''
    construct the quantum number pattern, the structure is as the coefficient
    QN: quantum number list at each bond
    ephtable : e-ph table 1 is electron and 0 is phonon 
    pbond : physical pbond
    addlist : the sigma orbital set
    '''
    print method
    assert method in ["1site","2site"]
    assert system in ["L","R"]
    qnl = np.array(QN[addlist[0]])
    qnr = np.array(QN[addlist[-1]+1])
    qnmat = qnl.copy()
    qnsigmalist = []

    for idx in addlist:

        if ephtable[idx] == 1:
            qnsigma = np.array([0,1])
        else:
            qnsigma = np.zeros([pbond[idx]],dtype=qnl.dtype)
        
        qnmat = np.add.outer(qnmat,qnsigma)
        qnsigmalist.append(qnsigma)

    qnmat = np.add.outer(qnmat,qnr)
    
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

    return qnmat, qnbigl, qnbigr


def c1d2cmat(cshape, c, qnmat, nexciton, nroots=1):
    # recover good quantum number vector c to matrix format
    if nroots == 1:
        cstruct = np.zeros(cshape,dtype=c.dtype)
        np.place(cstruct, qnmat==nexciton, c)
    else:
        cstruct = []
        for ic in c:
            icstruct = np.zeros(cshape,dtype=ic.dtype)
            np.place(icstruct, qnmat==nexciton, ic)
            cstruct.append(icstruct)
    
    return cstruct


def Renormalization_svd(cstruct, qnbigl, qnbigr, domain, nexciton, Mmax, percent=0):
    '''
        get the new mps, mpsdim, mpdqn, complementary mps to get the next guess
        with singular value decomposition method (1 root)
    '''
    assert domain in ["R", "L"]

    Uset, SUset, qnlnew, Vset, SVset, qnrnew = svd_qn.Csvd(cstruct, qnbigl, qnbigr, nexciton)
    if domain == "R":
        mps, mpsdim, mpsqn, compmps = updatemps(Vset, SVset, qnrnew, Uset, \
                nexciton, Mmax, percent=percent)
        return np.moveaxis(mps.reshape(list(qnbigr.shape)+[mpsdim]),-1,0), mpsdim, mpsqn,\
            compmps.reshape(list(qnbigl.shape) + [mpsdim])
    else:    
        mps, mpsdim, mpsqn, compmps = updatemps(Uset, SUset, qnlnew, Vset,\
                nexciton, Mmax, percent=percent)
        return mps.reshape(list(qnbigl.shape) + [mpsdim]), mpsdim, mpsqn,\
                np.moveaxis(compmps.reshape(list(qnbigr.shape)+[mpsdim]),-1,0)


def Renormalization_ddm(cstruct, qnbigl, qnbigr, domain, nexciton, Mmax, percent=0):
    '''
        get the new mps, mpsdim, mpdqn, complementary mps to get the next guess
        with diagonalize reduced density matrix method (> 1 root)
    '''
    nroots = len(cstruct)
    ddm = 0.0
    for iroot in xrange(nroots):
        if domain == "R":
            ddm += np.tensordot(cstruct[iroot], cstruct[iroot],\
                    axes=(range(qnbigl.ndim),range(qnbigl.ndim)))
        else:
            ddm += np.tensordot(cstruct[iroot], cstruct[iroot],\
                    axes=(range(qnbigl.ndim,cstruct[0].ndim),\
                        range(qnbigl.ndim,cstruct[0].ndim)))
    ddm /= float(nroots)
    if domain == "L":
        Uset, Sset, qnnew = svd_qn.Csvd(ddm, qnbigl, qnbigl, nexciton, ddm=True)
    else:
        Uset, Sset, qnnew = svd_qn.Csvd(ddm, qnbigr, qnbigr, nexciton, ddm=True)
    mps, mpsdim, mpsqn, compmps = updatemps(Uset, Sset, qnnew, None, \
            nexciton, Mmax, percent=percent)
    
    if domain == "R":
        return np.moveaxis(mps.reshape(list(qnbigr.shape)+[mpsdim]),-1,0),mpsdim, mpsqn,\
            np.tensordot(cstruct[0],mps.reshape(list(qnbigr.shape)+[mpsdim]),\
                axes=(range(qnbigl.ndim,cstruct[0].ndim),range(qnbigr.ndim)))
    else:    
        return mps.reshape(list(qnbigl.shape) + [mpsdim]), mpsdim, mpsqn,\
            np.tensordot(mps.reshape(list(qnbigl.shape)+[mpsdim]),cstruct[0],\
                axes=(range(qnbigl.ndim),range(qnbigl.ndim)))


def clean_MPS(system, MPS, ephtable, nexciton):
    '''
    clean MPS (or finite temperature MPO) to good quantum number(nexciton) subseciton 
    if time step is too large the quantum number would not conserve due to numerical error
    '''

    assert system in ["L","R"]
    # if a MPO convert to MPSnew   
    if MPS[0].ndim == 4:
        MPSnew = mpslib.to_mps(MPS)
    elif MPS[0].ndim == 3:
        MPSnew = mpslib.add(MPS, None)

    nMPS = len(MPSnew)
    if system == 'L':
        start = 0
        end = nMPS
        step = 1
    else:
        start = nMPS-1
        end = -1
        step = -1
    
    MPSQN = [None] * (nMPS+1)
    MPSQN[0] = [0]
    MPSQN[-1] = [0]

    for imps in xrange(start, end, step):
        
        if system == "L":
            qn = np.array(MPSQN[imps])
        else:
            qn = np.array(MPSQN[imps+1])

        if ephtable[imps] == 1:
            # e site
            if MPS[0].ndim == 3:
                sigmaqn = np.array([0,1])
            else:
                sigmaqn = np.array([0,0,1,1])
        else:
            # ph site 
            sigmaqn = np.array([0]*MPSnew[imps].shape[1])
        
        if system == "L":
            qnmat = np.add.outer(qn,sigmaqn)
            Gamma = MPSnew[imps].reshape(-1, MPSnew[imps].shape[-1])
        else:
            qnmat = np.add.outer(sigmaqn,qn)
            Gamma = MPSnew[imps].reshape(MPSnew[imps].shape[0],-1)
        
        if imps != end-step:  # last site clean at last
            qnbig = qnmat.ravel()
            qnset = []
            Uset = []
            Vset = []
            Sset = []
            for iblock in xrange(nexciton+1):
                idxset = [i for i, x in enumerate(qnbig.tolist()) if x == iblock]
                if len(idxset) != 0:
                    if system == "L":
                        Gamma_block = Gamma[np.ix_(idxset,range(Gamma.shape[1]))]
                    else:
                        Gamma_block = Gamma[np.ix_(range(Gamma.shape[0]),idxset)]
                    try:
                        U, S, Vt = scipy.linalg.svd(Gamma_block,\
                                full_matrices=False, lapack_driver='gesdd')
                    except:
                        print "clean part gesdd converge failed"
                        U, S, Vt = scipy.linalg.svd(Gamma_block,\
                                full_matrices=False, lapack_driver='gesvd')

                    dim = S.shape[0]
                    Sset.append(S)
                    
                    def blockappend(vset, qnset, v, n, dim, indice, shape):
                        vset.append(svd_qn.blockrecover(indice, v[:,:dim], shape))
                        qnset += [n] * dim
                        
                        return vset, qnset

                    if system == "L":
                        Uset, qnset = blockappend(Uset, qnset, U, iblock, dim, idxset, Gamma.shape[0])
                        Vset.append(Vt.T)
                    else:
                        Vset, qnset = blockappend(Vset, qnset, Vt.T, iblock, dim, idxset, Gamma.shape[1])
                        Uset.append(U)
                    
            Uset = np.concatenate(Uset,axis=1)
            Vset = np.concatenate(Vset,axis=1)
            Sset = np.concatenate(Sset)
            
            if system == "L":
                MPSnew[imps] = Uset.reshape([MPSnew[imps].shape[0],MPSnew[imps].shape[1],len(Sset)])
                Vset =  np.einsum('ij,j -> ij', Vset, Sset)
                MPSnew[imps+1] = np.tensordot(Vset.T, MPSnew[imps+1], axes=1)
                MPSQN[imps+1] = qnset
            else:
                MPSnew[imps] = Vset.T.reshape([len(Sset),MPSnew[imps].shape[1],MPSnew[imps].shape[-1]])
                Uset =  np.einsum('ij,j -> ij', Uset, Sset)
                MPSnew[imps-1] = np.tensordot(MPSnew[imps-1], Uset, axes=1)
                MPSQN[imps] = qnset
        
        # clean the extreme mat
        else:
            if system == "L":
                qnmat = np.add.outer(qnmat,np.array([0]))
            else:
                qnmat = np.add.outer(np.array([0]), qnmat)
            cshape = MPSnew[imps].shape
            assert cshape == qnmat.shape
            c = MPSnew[imps][qnmat==nexciton]
            MPSnew[imps] = c1d2cmat(cshape, c, qnmat, nexciton)
            
    if MPS[0].ndim == 4:
        MPSnew = mpslib.from_mps(MPSnew)
    
    return MPSnew


def construct_onsiteMPO(mol,pbond,opera,dipole=False,QNargs=None,sitelist=None):
    '''
    construct the electronic onsite operator \sum_i opera_i MPO
    '''
    assert opera in ["a", "a^\dagger", "a^\dagger a"]
    nmols = len(mol)
    if sitelist is None:
        sitelist = np.arange(nmols)

    MPOdim = []
    for imol in xrange(nmols):
        MPOdim.append(2)
        for iph in xrange(mol[imol].nphs):
            for iboson in xrange(mol[imol].ph[iph].nqboson):
                if imol != nmols-1:
                    MPOdim.append(2)
                else:
                    MPOdim.append(1)
    
    MPOdim[0] = 1
    MPOdim.append(1)
    print opera, "operator MPOdim", MPOdim

    MPO = []
    impo = 0
    for imol in xrange(nmols):
        mpo = np.zeros([MPOdim[impo],pbond[impo],pbond[impo],MPOdim[impo+1]])
        for ibra in xrange(pbond[impo]):
            for iket in xrange(pbond[impo]):
                if imol in sitelist:
                    if dipole == True:
                        factor = mol[imol].dipole
                    else:
                        factor = 1.0
                else:
                    factor = 0.0
                
                mpo[-1,ibra,iket,0] = EElementOpera(opera, ibra, iket) * factor
                if imol != 0:
                    mpo[0,ibra,iket,0] = EElementOpera("Iden",ibra,iket)
                if imol != nmols-1:
                    mpo[-1,ibra,iket,-1] = EElementOpera("Iden",ibra,iket)
        MPO.append(mpo)
        impo += 1

        for iph in xrange(mol[imol].nphs):
            for iboson in xrange(mol[imol].ph[iph].nqboson):
                mpo = np.zeros([MPOdim[impo],pbond[impo],pbond[impo],MPOdim[impo+1]])
                for ibra in xrange(pbond[impo]):
                    for idiag in xrange(MPOdim[impo]):
                        mpo[idiag,ibra,ibra,idiag] = 1.0

                MPO.append(mpo)
                impo += 1
    
    # quantum number part
    # len(MPO)-1 = len(MPOQN)-2, the L-most site is R-qn
    MPOQNidx = len(MPO)-1
    
    totnqboson = 0
    for iph in xrange(mol[-1].nphs):
        totnqboson += mol[-1].ph[iph].nqboson

    if opera == "a":
        MPOQN = [[0]] + [[-1,0]]*(len(MPO)-totnqboson-1) + [[-1]]*(totnqboson+1)
        MPOQNtot = -1
    elif opera == "a^\dagger":
        MPOQN = [[0]] + [[1,0]]*(len(MPO)-totnqboson-1) + [[1]]*(totnqboson+1)
        MPOQNtot = 1
    elif opera == "a^\dagger a":
        MPOQN = [[0]] + [[0,0]]*(len(MPO)-totnqboson-1) + [[0]]*(totnqboson+1)
        MPOQNtot = 0
    MPOQN[-1] = [0]
    
    if QNargs is None:
        return MPO, MPOdim
    else:
        return [MPO, MPOQN, MPOQNidx, MPOQNtot], MPOdim


def construct_intersiteMPO(mol,pbond,idxmol,jdxmol,QNargs=None):
    '''
    construct the electronic intersite operator \sum_i a_i^\dagger a_j
    the MPO dimension is 1
    '''
    nmols = len(mol)
    MPOdim = [1 for i in xrange(len(pbond)+1)]
    print "MPOdim", MPOdim

    MPO = []
    MPOQN = [0, ]
    impo = 0
    for imol in xrange(nmols):
        mpo = np.zeros([MPOdim[impo],pbond[impo],pbond[impo],MPOdim[impo+1]])
        if imol == idxmol:
            opera = "a^\dagger"
            MPOQN.append(1)
        elif imol == jdxmol:
            opera = "a"
            MPOQN.append(-1)
        else:
            opera = "Iden" 
            MPOQN.append(0)
        
        for ibra in xrange(pbond[impo]):
            for iket in xrange(pbond[impo]):
                mpo[0,ibra,iket,0] = EElementOpera(opera, ibra, iket)

        MPO.append(mpo)
        impo += 1

        for iph in xrange(mol[imol].nphs):
            for iboson in xrange(mol[imol].ph[iph].nqboson):
                mpo = np.zeros([MPOdim[impo],pbond[impo],pbond[impo],MPOdim[impo+1]])
                MPOQN.append(0)
                for ibra in xrange(pbond[impo]):
                    mpo[0,ibra,ibra,0] = 1.0

                MPO.append(mpo)
                impo += 1
    
    # quantum number part
    # len(MPO)-1 = len(MPOQN)-2, the L-most site is R-qn
    MPOQNidx = len(MPO)-1
    MPOQNtot = 0
    MPOQN[-1] = 0
    
    if QNargs is None:
        return MPO, MPOdim
    else:
        return [MPO, MPOQN, MPOQNidx, MPOQNtot], MPOdim
