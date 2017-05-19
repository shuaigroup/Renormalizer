#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

# phMPO
def PhElementOpera(op, bra, ket):
    if op == "b_n^\dagger b_n":
        if bra == ket:
            return float(ket)
        else:
            return 0.0
    elif op == "b_n^\dagger + b_n":
        if bra == ket + 1 : 
            return np.sqrt(bra)
        elif bra == ket - 1:
            return np.sqrt(float(ket))
        else:
            return 0.0
    elif op == "Iden":
        if bra == ket:
            return 1.0
        else:
            return 0.0
    else:
        sys.exit("wrong op in PhElementOpera")

# eMPO
def EElementOpera(op, bra, ket):
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
        if bra == ket and ket == 1:
            return 1.0
        else:
            return 0.0
    elif op == "Iden":
        if bra == ket:
            return 1.0
        else:
            return 0.0
    else:
        sys.exit("wrong op in EElementOpera")

# dipole MPO
def dipoleOpera(op, bra, ket):
    if op == "abs":
        if bra == 1 and ket == 0:
            return 1.0
        else:
            return 0.0
    elif op == "emi":
        if bra == 0 and ket == 1:
            return 1.0
        else:
            return 0.0
    elif op == "Iden":
        if bra == ket:
            return 1.0
        else:
            return 0.0
    else:
        sys.exit("wrong op in EElementOpera")

