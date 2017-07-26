# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
some electronic and phonon operator (second quantization) matrix element,
written in Latex format. <bra|op|ket>
'''

import numpy as np

def PhElementOpera(op, bra, ket):
    '''
    phonon operator
    '''
    assert op in ["b^\dagger b", "b^\dagger + b", "Iden"]
    assert bra >= 0
    assert ket >= 0

    if op == "b^\dagger b":
        if bra == ket:
            return float(ket)
        else:
            return 0.0

    elif op == "b^\dagger + b":
        if bra == ket + 1 : 
            return np.sqrt(float(bra))
        elif bra == ket - 1:
            return np.sqrt(float(ket))
        else:
            return 0.0

    elif op == "Iden":
        if bra == ket:
            return 1.0
        else:
            return 0.0


def EElementOpera(op, bra, ket):
    '''
    electronic operator
    '''
    assert op in ["a^\dagger", "a", "a^\dagger a", "Iden"]
    assert bra in [0, 1]
    assert ket in [0, 1]

    if op == "a^\dagger":
        if bra == ket + 1:
            return 1.0
        else:
            return 0.0

    elif op == "a":
        if bra == ket - 1:
            return 1.0
        else:
            return 0.0

    elif op == "a^\dagger a":
        if bra == 1 and ket == 1:
            return 1.0
        else:
            return 0.0

    elif op == "Iden":
        if bra == ket:
            return 1.0
        else:
            return 0.0
