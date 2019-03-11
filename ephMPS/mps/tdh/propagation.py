# -*- coding: utf-8 -*-

import numpy as np
import scipy

def unitary_propagation(WFN, HAM, Etot, dt):
    """
    unitary propagation e^-iHdt * wfn(dm)
    """
    for iham, ham in enumerate(HAM):
        ndim = WFN[iham].ndim
        w, v = scipy.linalg.eigh(ham)
        if ndim == 1:
            WFN[iham] = v.dot(np.exp(-1.0j * w * dt) * v.T.dot(WFN[iham]))
        elif ndim == 2:
            WFN[iham] = v.dot(np.diag(np.exp(-1.0j * w * dt)).dot(v.T.dot(WFN[iham])))
            # print iham, "norm", scipy.linalg.norm(WFN[iham])
        else:
            assert False
    WFN[-1] *= np.exp(-1.0j * Etot * dt)
    return WFN