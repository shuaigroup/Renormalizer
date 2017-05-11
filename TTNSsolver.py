#!/usr/bin/env python
# -*- coding: utf-8 -*-

# MPS/MPO structure
epb = 2
eM = 10
phM = 9
ephM = 12

def create_eMPS(epb, eM, ephM):
    # physical bond; left bond; right bond; e-p bond;
    mps = np.random.random((epb, eM, eM, ephM))
    return mps

def create_phMPS(phpb, lM, rM):
    # physical bond; left bond; right bond;
    mps = np.random.random((phpb, lM, rM))
    return mps

# MPS
# 交换一下顺序，可以使得bond不冗余
# 这个和MPO不一样, MPO是需要复制的。
eMPS = []
phMPS = []
for imol in xrange(nmols):
    # electronic MPS
    eMPS.append(create_eMPS(epb, eM, ephM))
    phMPS.append([])
    # phonon MPS
    for iph in xrange(mol[imol].nphs):
        if iph == 0:
            phMPS[imol].append(create_phMPS(mol[imol].ph[iph].nlevels, ephM, phM))
        else:
            phMPS[imol].append(create_phMPS(mol[imol].ph[iph].nlevels,  phM, phM))


limit = np.amax([ephM,eM,phM])


# the maximum bond dimension determined by the bipartion system
maxdim_eachchain = np.ones((nmols), dtype=np.int32)
for imol in xrange(nmols):
    for iph in xrange(mol[imol].nphs):
        maxdim_eachchain[imol] *= mol[imol].ph[iph].nlevels
        if maxdim_eachchain[imol] > limit:
            maxdim_eachchain[imol] = limit
            break

print maxdim_eachchain

maxdim_eachlink = 1
for imol in xrange(nmols):
    maxdim_eachlink *= epb
    if maxdim_eachlink > limit:
        maxdim_eachlink = limit
        break

print maxdim_eachlink


# MPSdim
eMPSdim = []
for imol in xrange(nmols+1):
    lmaxM = 1
    for iimol in xrange(imol):
        lmaxM *= maxdim_eachchain[iimol]*epb
        if lmaxM > limit:
            lmaxM = limit
            break

    rmaxM = 1
    for iimol in xrange(nmols-1, imol-1, -1):
        rmaxM *= maxdim_eachchain[iimol]*epb
        if rmaxM > limit:
            rmaxM = limit
            break

    eMPSdim.append(np.amin([lmaxM,rmaxM,eM]))

print "eMPSdim", eMPSdim

phMPSdim = []
for imol in xrange(nmols):
    phMPSdim.append([])

    lmaxM = maxdim_eachlink
    for iimol in xrange(imol-1):
        lmaxM *= maxdim_eachchain[iimol]
        if lmaxM > limit:
            lmaxM = limit
            break
    for iimol in xrange(nmols-1, imol, -1):
        lmaxM *= maxdim_eachchain[iimol]
        if lmaxM > limit:
            lmaxM = limit
            break

    for iph in xrange(mol[imol].nphs+1):
        llmaxM = lmaxM

        for iiph in xrange(iph):
            llmaxM *= mol[imol].ph[iiph].nlevels
            if llmaxM > limit:
                llmaxM = limit
                break
        
        rmaxM = 1
        for iiph in xrange(mol[imol].nphs-1, iph-1, -1):
            rmaxM *= mol[imol].ph[iiph].nlevels
            if rmaxM > limit:
                rmaxM = limit
                break
        if iph == 0:
            phMPSdim[imol].append(np.amin([llmaxM, rmaxM, ephM]))
        else:
            phMPSdim[imol].append(np.amin([llmaxM, rmaxM, phM]))
            
print "phMPSdim", phMPSdim


# MPO

# phMPOdim
phMPOdim = []
for imol in xrange(nmols):
    phMPOdim.append([])
    for iph in xrange(mol[imol].nphs):
        phMPOdim[imol].append(3)
    phMPOdim[imol].append(1)

print "phMPOdim",phMPOdim




def create_phMPO(ph):
    # up(bra), down(ket), l bond, r bond
    mpo = np.zeros((ph.nlevels, ph.nlevels, 3, 3))
    for ibra in xrange(ph.nlevels):
        for iket in xrange(ph.nlevels):
            mpo[ibra,iket,0,0] = PhElementOpera("Iden", ibra, iket)
            mpo[ibra,iket,1,1] = PhElementOpera("Iden", ibra, iket)
            mpo[ibra,iket,2,2] = PhElementOpera("Iden", ibra, iket)
            mpo[ibra,iket,1,0] = PhElementOpera("b_n^\dagger + b_n", ibra, iket) * ph.omega * ph.ephcoup
            mpo[ibra,iket,2,0] = PhElementOpera("b_n^\dagger b_n", ibra, iket) * ph.omega

    return mpo

phMPO = []
for imol in xrange(nmols):
    phMPO.append([])      
    for iph in xrange(mol[imol].nphs):
        phMPO[imol].append(create_phMPO(mol[imol].ph[iph]))


def create_eMPO(mollocal):
    # up(bra), down(ket), l bond, r bond, e-ph bond
    mpo = np.zeros((epb,epb,2*(mollocal+1),2*(mollocal+2),3))

    for ibra in xrange(epb):
        for iket in xrange(epb):
            # first columnn 
            mpo[ibra,iket,0,0,0] = EElementOpera("Iden", ibra, iket)
            # a, a^\dagger, a, a^\dagger
            for ileft in xrange(1,2*mollocal+1):
                if ileft % 2 == 1:
                    mpo[ibra,iket,ileft,0,0] = EElementOpera("a", ibra, iket) * \
                    J[(ileft-1)/2,mollocal]
                else:
                    mpo[ibra,iket,ileft,0,0] = EElementOpera("a^\dagger", ibra,
                            iket) * J[(ileft-1)/2,mollocal]
            mpo[ibra,iket,2*(mollocal+1)-1,0,0] = EElementOpera("a^\dagger a", \
                    ibra, iket) * mol[mollocal].elocalex

            for ileft in xrange(1,2*(mollocal+1)-1):
                    mpo[ibra,iket,ileft,ileft,0] = EElementOpera("Iden", ibra, iket)
            
            mpo[ibra,iket,2*(mollocal+1)-1,2*(mollocal+1)-1,0] = \
                EElementOpera("a^\dagger", ibra, iket)
            mpo[ibra,iket,2*(mollocal+1)-1,2*(mollocal+1),0] = \
                EElementOpera("a", ibra, iket)
            mpo[ibra,iket,2*(mollocal+1)-1,2*(mollocal+2)-1,0] = \
                EElementOpera("Iden", ibra, iket)

            # e-ph linked part
            mpo[ibra,iket,2*(mollocal+1)-1,0,1] = EElementOpera("a^\dagger a", ibra, iket)
            mpo[ibra,iket,2*(mollocal+1)-1,0,2] = EElementOpera("Iden", ibra, iket)

    return mpo

eMPO = []
for imol in xrange(nmols):
    eMPO.append(create_eMPO(imol))

#eMPOdim
eMPOdim = []
for imol in xrange(nmols):
    eMPOdim.append(2*(imol+1))
eMPOdim.append(1)
eMPOdim[0] = 1
print "eMPOdim", eMPOdim
    

'''
clean the initial MPS 
'''
def phRcanonical(phmps,pb,lb,rb):
    phmps[0:pb,0:lb,0:rb] =  \
        np.moveaxis( Rcanonical(np.moveaxis(phmps[0:pb,0:lb,0:rb],\
        [0],[2]).reshape(lb,rb*pb)).reshape(lb,rb,pb),[2],[0])

def Rcanonical(mps):
    U, S, V = np.linalg.svd(mps)
    return V[0:mps.shape[0],:]
    
for imol in range(nmols):
    for iph in range(mol[imol].nphs):
        phRcanonical(phMPS[imol][iph],mol[imol].ph[iph].nlevels, phMPSdim[imol][iph], phMPSdim[imol][iph+1])
        



