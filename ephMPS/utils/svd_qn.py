# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from __future__ import absolute_import, print_function, unicode_literals

import numpy as np
import scipy.linalg


def Csvd(cstruct, qnbigl, qnbigr, nexciton, QR=False, system=None,
        full_matrices=True, ddm=False):
    '''
    block svd the coefficient matrix (l, sigmal, sigmar, r) or (l,sigma,r)
    according to the quantum number 
    ddm is the direct diagonalization the reduced density matrix
    '''
    Gamma = cstruct.reshape(np.prod(qnbigl.shape),np.prod(qnbigr.shape))
    localqnl = qnbigl.ravel()
    localqnr = qnbigr.ravel()
    
    Uset = []     # corresponse to nonzero svd value
    Uset0 = []    # corresponse to zero svd value
    Vset = []
    Vset0 = []
    Sset = []
    SUset0 = []
    SVset0 = []
    qnlset = []
    qnlset0 = []
    qnrset = []
    qnrset0 = []
    
    if ddm == False:
        # different combination
        if not cstruct.is_virtual:
            combine = [[x, nexciton-x] for x in range(nexciton+1)]
        else:
            min0 = min(np.min(localqnl),np.min(localqnr))
            max0 = max(np.max(localqnl),np.max(localqnr))
            combine = [[x, nexciton-x] for x in range(min0, max0+1)]
    else:
        # ddm is for diagonlize the reduced density matrix for multistate
        combine = [[x, x] for x in range(nexciton+1)]

    for nl, nr in combine:
        lset = np.where(localqnl == nl)[0]
        rset = np.where(localqnr == nr)[0]
        if len(lset) != 0 and len(rset) != 0:
            #Gamma_block = Gamma[np.ix_(lset,rset)]
            Gamma_block = Gamma.ravel().take((lset * Gamma.shape[1]).reshape(-1,1) + rset)
            
            def blockappend(vset, vset0, qnset, qnset0, svset0, v, n, dim,
                    indice, shape, full_matrices=True):
                vset.append(blockrecover(indice, v[:,:dim], shape))
                qnset += [n] * dim
                if full_matrices == True:
                    vset0.append(blockrecover(indice, v[:,dim:],shape))
                    qnset0 += [n] * (v.shape[0]-dim)
                    svset0.append(np.zeros(v.shape[0]-dim))
                
                return vset, vset0, qnset, qnset0, svset0
            
            if not ddm:
                if not QR:
                    try:
                        U, S, Vt = scipy.linalg.svd(Gamma_block,
                                full_matrices=full_matrices,lapack_driver='gesdd')
                    except:
                        # print "Csvd converge failed"
                        U, S, Vt = scipy.linalg.svd(Gamma_block,
                                full_matrices=full_matrices,lapack_driver='gesvd')
                    dim = S.shape[0]
                    Sset.append(S)
                else:
                    if full_matrices:
                        mode = "full"
                    else:
                        mode = "economic"
                    if system == "R":
                        U,Vt = scipy.linalg.rq(Gamma_block, mode=mode)
                    else:
                        U,Vt = scipy.linalg.qr(Gamma_block, mode=mode)
                    dim = min(Gamma_block.shape)
                
                Uset, Uset0, qnlset, qnlset0, SUset0 = blockappend(Uset, Uset0, qnlset,
                        qnlset0, SUset0, U, nl, dim, lset, Gamma.shape[0],
                        full_matrices=full_matrices)
                Vset, Vset0, qnrset, qnrset0, SVset0 = blockappend(Vset, Vset0, qnrset,
                        qnrset0, SVset0, Vt.T, nr, dim, rset, Gamma.shape[1], full_matrices=full_matrices)
            else:
                S, U = scipy.linalg.eigh(Gamma_block)
                # numerical error for eigenvalue < 0 
                for ss in range(len(S)):
                    if S[ss] < 0:
                        S[ss] = 0.0
                S = np.sqrt(S)
                dim = S.shape[0]
                Sset.append(S)
                Uset, Uset0, qnlset, qnlset0, SUset0 = blockappend(Uset, Uset0, qnlset,
                        qnlset0, SUset0, U, nl, dim, lset, Gamma.shape[0], full_matrices=False)
                
    
    if not ddm:
        if full_matrices:
            Uset = np.concatenate(Uset + Uset0,axis=1)
            Vset = np.concatenate(Vset + Vset0,axis=1)
            qnlset = qnlset + qnlset0
            qnrset = qnrset + qnrset0
            if QR == False:
                # not sorted
                SUset = np.concatenate(Sset + SUset0)
                SVset = np.concatenate(Sset + SVset0)
                return Uset, SUset, qnlset, Vset, SVset, qnrset
            else:
                return Uset, qnlset, Vset, qnrset
        else:
            Uset = np.concatenate(Uset,axis=1)
            Vset = np.concatenate(Vset,axis=1)
            if QR == False:
                Sset = np.concatenate(Sset)
                # sort the singular value in descending order
                order = np.argsort(Sset)[::-1]
                Uset_order = Uset[:,order]
                Vset_order = Vset[:,order]
                Sset_order = Sset[order]
                qnlset_order = np.array(qnlset)[order].tolist()
                qnrset_order = np.array(qnrset)[order].tolist()
                return Uset_order, Sset_order, qnlset_order, Vset_order, Sset_order, qnrset_order
            else:
                return Uset, qnlset, Vset, qnrset
    else:
        Uset = np.concatenate(Uset,axis=1)
        Sset = np.concatenate(Sset)
        return Uset, Sset, qnlset


def blockrecover(indices, U, dim):
    '''
    recover the block element to its original position
    '''
    resortU = np.zeros([dim, U.shape[1]], dtype=U.dtype)
    resortU[indices,:] = U
    
    return resortU

