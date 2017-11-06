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
    assert op in ["b", "b^\dagger", "b^\dagger b", "b^\dagger + b", "Iden",\
            "(b^\dagger + b)^2", "(b^\dagger + b)^3"]
    assert bra >= 0
    assert ket >= 0

    if op == "b^\dagger b":
        if bra == ket:
            return float(ket)
        else:
            return 0.0
    elif op == "b":
        if bra == ket - 1 : 
            return np.sqrt(float(ket))
        else:
            return 0.0
    elif op == "b^\dagger":
        if bra == ket + 1 : 
            return np.sqrt(float(bra))
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
    elif op == "(b^\dagger + b)^2":
        if bra == ket + 2:
            return np.sqrt(float(ket+1)*float(ket+2))
        elif bra == ket:
            return float(ket*2+1)
        elif bra == ket - 2:
            return np.sqrt(float(ket)*float(ket-1))
        else:
            return 0.0
    elif op == "(b^\dagger + b)^3":
        if bra == ket + 3:
            return np.sqrt((ket+1)*(ket+2)*(ket+3))
        elif bra == ket + 1:
            return np.sqrt((ket+1)**3) \
                    + np.sqrt((ket+1)*(ket+2)**2) \
                    + np.sqrt(ket**2*(ket+1))
        elif bra == ket - 1:
            return np.sqrt((ket+1)**2*ket)\
                    + np.sqrt(ket*(ket-1)**2)\
                    + np.sqrt(ket**3)
        elif bra == ket - 3:
            return np.sqrt(ket*(ket-1)*(ket-2))
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
