# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import copy

import numpy as np
import scipy

from ephMPS.mps.ephtable import electron, phonon
from ephMPS.mps.matrix import Matrix
from ephMPS.utils import svd_qn


class MatrixProduct(list):

    def __init__(self, enable_qn=False):
        super(MatrixProduct, self).__init__()
        self.mtype = Matrix
        self._left_domain = None
        self.enable_qn = enable_qn
        self.qn = None
        self.qnidx = None
        self.qntot = None
        self.ephtable = None


    def check_left_canonical(self):
        '''
        check L-canonical
        '''
        ret = True
        for mt in self[:-1]:
            ret *= mt.check_lortho()
        return ret


    def check_right_canonical(self):
        '''
        check R-canonical
        '''
        ret = True
        for mt in self[1:]:
            ret *= mt.check_rortho()
        return ret

    @property
    def left_domain(self):
        return self._left_domain
    
    @property
    def right_domain(self):
        return not self.left_domain
    
    def switch_domain(self):
        self._left_domain = not self._left_domain
        self.qnidx = len(self) - 1 - self.qnidx

    def conj(self):
        """
        complex conjugate
        """
        new_mp = copy.deepcopy(self)
        for idx, mt in enumerate(new_mp):
            new_mp[idx] = mt.conj()
        return new_mp

    def dot(self, other):
        """
        dot product of two mps / mpo 
        """

        assert len(self) == len(other)
        nsites = len(self)
        e0 = np.eye(1, 1)
        for mt1, mt2 in zip(self, other):
            # sum_x e0[:,x].m[x,:,:]
            e0 = np.tensordot(e0, mt2, 1)
            # sum_ij e0[i,p,:] self[i,p,:]
            # note, need to flip a (:) index onto top,
            # therefore take transpose
            if mt1.ndim == 3:
                e0 = np.tensordot(e0, mt1, ([0, 1], [0, 1])).T
            if mt2.ndim == 4:
                e0 = np.tensordot(e0, mt1, ([0, 1, 2], [0, 1, 2])).T

        return e0[0, 0]

    def distance(self, other):
        return self.conj().dot(self) - self.conj().dot(other) - other.conj().dot(self) + other.conj().dot(other)

    def move_qnidx(self, dstidx):
        '''
        Quantum number has a boundary side, left hand of the side is L system qn,
        right hand of the side is R system qn, the sum of quantum number of L system
        and R system is tot.
        '''
        # construct the L system qn
        for idx in range(self.qnidx + 1, len(self.qn) - 1):
            self.qn[idx] = [self.qntot - i for i in self.qn[idx]]

        # set boundary to fsite:
        for idx in range(len(self.qn) - 2, dstidx, -1):
            self.qn[idx] = [self.qntot - i for i in self.qn[idx]]


    def compress(self, trunc=1.e-12, check_canonical=False, QR=False, normalize=None):
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

        # if trunc==0, we are just doing a canonicalisation,
        # so skip check, otherwise, ensure mps is canonicalised
        if trunc != 0 and check_canonical:
            if self.left_domain:
                assert self.check_left_canonical()
            else:
                assert self.check_right_canonical()

        nsites = len(self)
        if self.left_domain:
            idx_list = range(nsites - 1, 0, -1)
        else:
            idx_list = range(0, nsites - 1)
        for idx in idx_list:
            mt = self[idx]
            if self.left_domain:
                mt = mt.r_combine()
            else:
                mt = mt.l_combine()
            if self.enable_qn:
                if self.ephtable.is_electron(idx):
                    # e site
                    sigmaqn = mt.elec_sigmaqn
                else:
                    # ph site 
                    sigmaqn = np.array([0] * mt.pdim_prod)
                qnl = np.array(self.qn[idx])
                qnr = np.array(self.qn[idx + 1])
                if self.left_domain:
                    qnbigl = qnl
                    qnbigr = np.add.outer(sigmaqn, qnr)
                else:
                    qnbigl = np.add.outer(qnl, sigmaqn)
                    qnbigr = qnr

            if not QR:
                if self.enable_qn:
                    u, sigma, qnlset, v, sigma, qnrset = svd_qn.Csvd(mt, qnbigl,
                                                                     qnbigr, self.qntot, full_matrices=False)
                    vt = v.T
                else:
                    try:
                        u, sigma, vt = scipy.linalg.svd(mt, full_matrices=False, lapack_driver='gesdd')
                    except:
                        # print "mps compress converge failed"
                        u, sigma, vt = scipy.linalg.svd(mt, full_matrices=False, lapack_driver='gesvd')

                if trunc == 0:
                    m_trunc = len(sigma)
                elif trunc < 1.:
                    # count how many sing vals < trunc            
                    normed_sigma = sigma / scipy.linalg.norm(sigma)
                    # m_trunc=len([s for s in normed_sigma if s >trunc])
                    m_trunc = np.count_nonzero(normed_sigma > trunc)
                else:
                    m_trunc = int(trunc)
                    m_trunc = min(m_trunc, len(sigma))

                u = u[:, 0:m_trunc]
                sigma = sigma[0:m_trunc]
                vt = vt[0:m_trunc, :]

                if self.left_domain:
                    u = np.einsum('ji, i -> ji', u, sigma)
                else:
                    vt = np.einsum('i, ij -> ij', sigma, vt)
            else:
                if self.enable_qn:
                    if self.left_domain:
                        system = "R"
                    else:
                        system = "L"
                    u, qnlset, v, qnrset = svd_qn.Csvd(mt, qnbigl, qnbigr, self.MPSQNtot,
                                                       QR=True, system=system, full_matrices=False)
                    vt = v.T
                else:
                    if self.left_domain:
                        u, vt = scipy.linalg.rq(mt, mode='economic')
                    else:
                        u, vt = scipy.linalg.qr(mt, mode='economic')
                m_trunc = u.shape[1]

            if self.left_domain:
                self[idx - 1] = np.tensordot(self[idx - 1], u, axes=1)
                ret_mpsi = np.reshape(vt, [m_trunc] + list(mt.pdim) + [vt.shape[1] / mt.pdim_prod])
                if self.enable_qn:
                    self.qn[idx] = qnrset[:m_trunc]
            else:
                self[idx + 1] = np.tensordot(vt, self[idx + 1], axes=1)
                ret_mpsi = np.reshape(u, [u.shape[0] / mt.npdim] + mt.pdim + [m_trunc])
                if self.enable_qn:
                    self.qn[idx + 1] = qnlset[:m_trunc]

            self[idx] = ret_mpsi

        # todo: normalize is the norm of the MPS

        # fidelity = dot(conj(ret_mps), mps)/dot(conj(mps), mps)
        # print "compression fidelity:: ", fidelity
        # if np.isnan(fidelity):
        #     dddd

        if self.enable_qn:
            if self.left_domain:
                self.qnidx = 0
            else:
                self.qnidx = len(self) - 1

        self.switch_domain()

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return '%s with %d sites' % (self.__class__, len(self))

    def __setitem__(self, key, array):
        super(MatrixProduct, self).__setitem__(key, self.mtype(array))

    def append(self, array):
        super(MatrixProduct, self).append(self.mtype(array))