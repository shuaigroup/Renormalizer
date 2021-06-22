# -*- coding: utf-8 -*-

import opt_einsum as oe

from renormalizer.mps.matrix import asxp


def hop_expr(ltensor, rtensor, cmo, cshape, twolayer:bool=False):

    nsite = len(cmo)
    # whether have the ancilla
    ancilla = 2 * nsite + 2 == len(cshape)
    if not ancilla:
        assert nsite + 2 == len(cshape)

    ltensor = asxp(ltensor)
    rtensor = asxp(rtensor)
    for i in range(len(cmo)):
        cmo[i] = asxp(cmo[i])

    if nsite == 0:
        # ancilla not defined
        del ancilla

    if twolayer:
        assert nsite in [1, 2]
        # Only used in ground state algorithm
        # Hopefully generalize to CV in the future
        assert not ancilla
        if nsite == 1:
            #   S-a e j-S
            #   O-b-O-g-O
            #   |   f   |
            #   O-c-O-i-O
            #   S-d h k-S
            expr = oe.contract_expression(
                "abcd, befg, cfhi, jgik, aej -> dhk",
                ltensor, cmo[0], cmo[0], rtensor, cshape,
                constants=[0, 1, 2, 3]
            )
        else:
            #   S-a e   j o-S
            #   O-b-O-g-O-l-O
            #   |   f   k   |
            #   O-c-O-i-O-n-O
            #   S-d h   m p-S
            expr = oe.contract_expression(
                "abcd, befg, cfhi, gjkl, ikmn, olnp, aejo -> dhmp",
                ltensor, cmo[0], cmo[0], cmo[1], cmo[1], rtensor, cshape,
                constants=[0, 1, 2, 3, 4, 5],
            )
        # early return
        return expr

    # Single layer, the most common case
    # Could be written in an automatic way
    # But for now probably an overkill
    if nsite == 0:
        # S-a   l-S
        #
        # O-b - b-O
        #
        # S-c   k-S
        expr = oe.contract_expression(
            "abc, lbk, ck -> al",
            ltensor, rtensor, cshape,
            constants=[0, 1],
        )
    elif nsite == 1:
        if not ancilla:
            # S-a   l-S
            #     d
            # O-b-O-f-O
            #     e
            # S-c   k-S
            expr = oe.contract_expression(
                "abc, bdef, lfk, cek -> adl",
                ltensor, cmo[0], rtensor, cshape,
                constants=[0, 1, 2],
            )
        else:
            # S-a   l-S
            #     d
            # O-b-O-f-O
            #     e
            # S-c   k-S
            #     g
            expr = oe.contract_expression(
                "abc, bdef, lfk, cegk -> adgl",
                ltensor, cmo[0], rtensor, cshape,
                constants=[0, 1, 2],
            )
    else:
        if not ancilla:
            # S-a       l-S
            #     d   g
            # O-b-O-f-O-j-O
            #     e   h
            # S-c       k-S
            expr = oe.contract_expression(
                "abc, bdef, fghj, ljk, cehk -> adgl",
                ltensor, cmo[0], cmo[1], rtensor, cshape,
                constants=[0, 1, 2, 3],
            )
        else:
            # S-a       l-S
            #     d   g
            # O-b-O-f-O-j-O
            #     e   h
            # S-c       k-S
            #     m   n
            expr = oe.contract_expression(
                "abc, bdef, fghj, ljk, cemhnk -> admgnl",
                ltensor, cmo[0], cmo[1], rtensor, cshape,
                constants=[0, 1, 2, 3],
            )

    return expr