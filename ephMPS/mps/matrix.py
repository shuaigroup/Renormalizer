# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np


class Matrix(np.ndarray):

    npdtype = np.complex128

    def __new__(cls, array):
        obj = np.array(array, dtype=cls.npdtype).view(cls)
        obj.original_shape = obj.shape
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.original_shape = getattr(obj, 'original_shape', None)

    # physical indices exclude first and last indices
    @property
    def pdim(self):
        return self.original_shape[1:-1]

    @property
    def pdim_prod(self):
        return np.prod(self.pdim)

    @property
    def elec_sigmaqn(self):
        raise NotImplementedError

    @property
    def is_virtual(self):
        return False

    @property
    def r_combine_shape(self):
        return self.original_shape[0], np.prod(self.original_shape[1:])

    @property
    def l_combine_shape(self):
        return np.prod(self.original_shape[:-1]), self.original_shape[-1]

    def r_combine(self):
        return self.reshape(self.r_combine_shape)

    def l_combine(self):
        return self.reshape(self.l_combine_shape)


    def check_lortho(self):
        """
        check L-orthogonal
        """
        tensm = np.reshape(self, [np.prod(self.shape[:-1]), self.shape[-1]])
        s = np.dot(np.conj(tensm.T), tensm)
        return np.allclose(s, np.eye(s.shape[0]))


    def check_rortho(self):
        '''
        check R-orthogonal
        '''
        tensm = np.reshape(self, [self.shape[0], np.prod(self.shape[1:])])
        s = np.dot(tensm, np.conj(tensm.T))
        return np.allclose(s, np.eye(s.shape[0]))


class MatrixState(Matrix):

    @property
    def elec_sigmaqn(self):
        return np.array([0, 1])


class MatrixOp(Matrix):

    npdtype = np.complex128

    @property
    def elec_sigmaqn(self):
        return np.array([0, -1, 1, 0])


class VirtualMatrixOp(Matrix):

    @property
    def elec_sigmaqn(self):
        return np.array([0, 0, 1, 1])

    @property
    def is_virtual(self):
        return True
