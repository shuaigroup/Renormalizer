# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import scipy.linalg


def exp_value(bra, O, ket):
    '''
    calculate the expectation value
    <np.conj(wfnbra) | O | wfnket>  at ZT
    trace (dmbra^\dagger | O | dmket) at FT
    '''

    return np.vdot(bra, O.dot(ket))


def normalize(WFN):
    '''
    normalize WFN/DM and scale the prefactor
    '''
    norm = 1.0
    for wfn in WFN[:-1]:
        lnorm = scipy.linalg.norm(wfn)
        wfn /= lnorm
        norm *= lnorm

    WFN[-1] *= norm