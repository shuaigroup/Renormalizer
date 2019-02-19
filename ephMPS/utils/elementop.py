# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

"""
some electronic and phonon operator (second quantization) matrix element,
written in Latex format. <bra|op|ket>
"""

import numpy as np


def op_matrix(op, size, type):
    assert type in ['e', 'ph']
    if type == 'e':
        element_func = e_element_op
    else:
        element_func = ph_element_op
    op_matrix = np.zeros((size, size))
    for ibra in range(size):
        for iket in range(size):
            op_matrix[ibra][iket] = element_func(op, ibra, iket)
    return op_matrix


def e_op_matrix(op, size):
    return op_matrix(op, size, 'e')


def ph_op_matrix(op, size):
    return op_matrix(op, size, 'ph')


ph_op_list = ["b", "b^\dagger", "b^\dagger b", "b^\dagger + b", "Iden", "(b^\dagger + b)^2", "(b^\dagger + b)^3"]


def ph_element_op(op, bra, ket):
    """
    phonon operator
    """
    assert op in ph_op_list
    assert bra >= 0
    assert ket >= 0

    if op == "b^\dagger b":
        if bra == ket:
            return float(ket)
        else:
            return 0.0
    elif op == "b":
        if bra == ket - 1:
            return np.sqrt(float(ket))
        else:
            return 0.0
    elif op == "b^\dagger":
        if bra == ket + 1:
            return np.sqrt(float(bra))
        else:
            return 0.0
    elif op == "b^\dagger + b":
        if bra == ket + 1:
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


e_op_list = ["a^\dagger", "a", "a^\dagger a", "Iden"]


def e_element_op(op, bra, ket):
    """
    electronic operator
    """
    assert op in e_op_list
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
    else:
        assert False


def construct_e_op_dict(pbond):
    e_op_dict = {}
    for op in e_op_list:
        e_op_dict[op] = e_op_matrix(op, pbond)
    return e_op_dict


def construct_ph_op_dict(pbond):
    ph_op_dict = {}
    for op in ph_op_list:
        ph_op_dict[op] = ph_op_matrix(op, pbond)
    return ph_op_dict