# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
from ephMPS.mps import solver
from ephMPS.spectra.exact import SpectraExact
from ephMPS.spectra.finitet import SpectraAbsFiniteT, SpectraEmiFiniteT
from ephMPS.spectra.zerot import SpectraOneWayPropZeroT, SpectraTwoWayPropZeroT
from ephMPS.utils import constant, Quantity


def prepare_init_mps(mol_list, procedure, nexciton, mpo_scheme, compress_method=None, offset=Quantity(0), optimize=False):
    i_mps, h_mpo = solver.construct_mps_mpo_2(mol_list, procedure[0][0], nexciton, scheme=mpo_scheme, offset=offset)
    if compress_method is not None:
        i_mps.compress_method = compress_method
    if optimize:
        solver.optimize_mps(i_mps, h_mpo, procedure, method="2site")
    return i_mps, h_mpo