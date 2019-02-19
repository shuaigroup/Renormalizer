# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import scipy.linalg


def exp_value(bra, O, ket):
    """
    calculate the expectation value
    <np.conj(wfnbra) | O | wfnket>  at ZT
    trace (dmbra^\dagger | O | dmket) at FT
    """
    assert np.allclose(np.linalg.norm(bra), 1)
    assert np.allclose(np.linalg.norm(ket), 1)
    return np.vdot(bra, O.dot(ket))


def normalize(WFN, norm=None):
    """
    normalize WFN/DM and scale the prefactor
    """
    factor = 1.0
    for wfn in WFN[:-1]:
        lnorm = scipy.linalg.norm(wfn)
        wfn /= lnorm
        factor *= lnorm

    if norm is not None:
        WFN[-1] = norm
    else:
        WFN[-1] *= factor


def canonical_normalize(WFN):
    factor = 1.0
    for wfn in WFN:
        lnorm = scipy.linalg.norm(wfn)
        factor *= lnorm
    normalize(WFN, factor)
