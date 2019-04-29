# -*- coding: utf-8 -*-
from typing import List, Tuple


import numpy as np

from ephMPS.mps.matrix import EmptyMatrixError
from ephMPS.mps.mpo import Mpo
from ephMPS.mps.mpdm import MpDmFull
from ephMPS.mps.lib import compressed_sum


class SuperLiouville(Mpo):

    def __init__(self, h_mpo, dissipation=0):
        super().__init__()
        self.mol_list = h_mpo.mol_list
        self.h_mpo = h_mpo
        self.dissipation = dissipation

    def apply(self, mp: MpDmFull, canonicalise=False):
        assert mp.is_mpdm
        assert not canonicalise
        no_dissipation = self.h_mpo.contract(mp) - mp.contract(self.h_mpo)
        if self.dissipation == 0:
            return no_dissipation
        # create and destroy operators
        pm_operators: List[Tuple[Mpo, Mpo]] = mp.mol_list.get_mpos("lindblad_pm", calc_lindblad_pm)
        applied_terms = []
        for b, b_dag in pm_operators:
            try:
                res = b.apply(mp).apply(b_dag)
            except EmptyMatrixError:
                # discard vacuum states
                continue
            applied_terms.append(res)
        if len(applied_terms) == 0:
            assert mp.ph_occupations.sum() == 0
            return no_dissipation
        summed_term = compressed_sum(applied_terms, ignore_empty=True)
        bdb_operator: Mpo = mp.mol_list.get_mpos("lindblad_bdb", calc_lindblad_bdb)
        # any room for optimization? are there any simple relations between the two terms?
        try:
            lindblad = summed_term - 0.5 * (bdb_operator.contract(mp) + mp.contract(bdb_operator))
        except EmptyMatrixError:
            lindblad = summed_term
        ret = no_dissipation + 1j * self.dissipation * lindblad
        return ret

    # used when calculating energy in evolve_dmrg_prop_and_compress
    def __getitem__(self, item):
        return self.h_mpo[item]


def calc_lindblad_pm(mol_list):
    # b and b^\dagger
    ph_operators = []
    for imol, m in enumerate(mol_list):
        for jph in range(m.n_dmrg_phs):
            b = Mpo.ph_onsite(mol_list, r"b", imol, jph)
            b_dag = Mpo.ph_onsite(mol_list, r"b^\dagger", imol, jph)
            ph_operators.append((b, b_dag))
    return ph_operators


def calc_lindblad_bdb(mol_list):
    ph_operators = []
    for imol, m in enumerate(mol_list):
        for jph in range(m.n_dmrg_phs):
            bdb = Mpo.ph_onsite(mol_list, r"b^\dagger b", imol, jph)
            bdb.set_threshold(1e-5)
            ph_operators.append(bdb)
    #from functools import reduce
    #return reduce(lambda mps1, mps2: mps1.add(mps2), ph_operators)
    return compressed_sum(ph_operators)