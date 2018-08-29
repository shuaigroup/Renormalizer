# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
from ephMPS.mps import solver
from ephMPS.spectra.exact import SpectraExact
from ephMPS.spectra.finitet import SpectraAbsFiniteT, SpectraEmiFiniteT
from ephMPS.spectra.zerot import SpectraOneWayPropZeroT, SpectraTwoWayPropZeroT
from ephMPS.utils import constant


def prepare_init_mps(mol_list, j_matrix, procedure, nexciton, mpo_scheme, compress_method=None, offset=0, optimize=False):
    i_mps, h_mpo = solver.construct_mps_mpo_2(mol_list, j_matrix, procedure[0][0], nexciton, scheme=mpo_scheme)
    if compress_method is not None:
        i_mps.compress_method = compress_method
    if optimize:
        solver.optimize_mps(i_mps, h_mpo, procedure, method="2site")
    for ibra in range(h_mpo.pbond_list[0]):
        h_mpo[0][0, ibra, ibra, 0] -= offset / constant.au2ev
    return i_mps, h_mpo