# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>


"""
automatic construct tensordot string with einsum style
"""
import numpy as np


def multi_tensor_contract(path, *operands):
    """
    ipath[0] is the index of the mat
    ipaht[1] is the contraction index
    oeprands is the arrays

    For example:  in mpompsmat.py
    path = [([0, 1],"fdla, abc -> fdlbc")   ,\
            ([2, 0],"fdlbc, gdeb -> flcge") ,\
            ([1, 0],"flcge, helc -> fgh")]
    outtensor = tensorlib.multi_tensor_contract(path, MPSconj[isite], intensor,
            MPO[isite], MPS[isite])
    """

    operands = list(operands)
    for ipath in path:

        input_str, results_str = ipath[1].split("->")
        input_str = input_str.split(",")
        input_str = [x.replace(" ", "") for x in input_str]
        results_set = set(results_str)
        inputs_set = set(input_str[0] + input_str[1])
        idx_removed = inputs_set - (inputs_set & results_set)

        tmpmat = pair_tensor_contract(
            operands[ipath[0][0]],
            input_str[0],
            operands[ipath[0][1]],
            input_str[1],
            idx_removed,
        )

        for x in sorted(ipath[0], reverse=True):
            del operands[x]

        operands.append(tmpmat)

    return operands[0]


def pair_tensor_contract(view_left, input_left, view_right, input_right, idx_removed):
    # Find indices to contract over
    left_pos, right_pos = (), ()
    for s in idx_removed:
        left_pos += (input_left.find(s),)
        right_pos += (input_right.find(s),)
    new_view = np.tensordot(view_left, view_right, axes=(left_pos, right_pos))

    return new_view
