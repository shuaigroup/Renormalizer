# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np


class Matrix(np.ndarray):
    def __new__(cls, array):
        obj = np.array(array).view(cls)
        obj.original_shape = obj.shape
        # set in MatrixProduct
        obj.sigmaqn = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.original_shape = getattr(obj, "original_shape", None)

    # physical indices exclude first and last indices
    @property
    def pdim(self):
        return self.original_shape[1:-1]

    @property
    def pdim_prod(self):
        return np.prod(self.pdim)

    @property
    def bond_dim(self):
        return self.original_shape[0], self.original_shape[-1]

    @property
    def r_combine_shape(self):
        return self.original_shape[0], np.prod(self.original_shape[1:])

    @property
    def l_combine_shape(self):
        return np.prod(self.original_shape[:-1]), self.original_shape[-1]

    @property
    def is_mpo(self):
        return self.original_shape.ndim == 4

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
        return np.allclose(s, np.eye(s.shape[0]), atol=1e-3)

    def check_rortho(self):
        """
        check R-orthogonal
        """
        tensm = np.reshape(self, [self.shape[0], np.prod(self.shape[1:])])
        s = np.dot(tensm, np.conj(tensm.T))
        return np.allclose(s, np.eye(s.shape[0]))

    def __deepcopy__(self, memodict, *arg, **kwargs):
        y = super(Matrix, self).__deepcopy__(memodict, *arg, **kwargs)
        y.original_shape = self.original_shape
        y.sigmaqn = self.sigmaqn
        return y
