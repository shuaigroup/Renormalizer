# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
from ephMPS.mps import solver
from ephMPS.spectra.exact import SpectraExact
from ephMPS.spectra.finitet import SpectraAbsFiniteT, SpectraEmiFiniteT
from ephMPS.spectra.zerot import SpectraOneWayPropZeroT, SpectraTwoWayPropZeroT
from ephMPS.utils import constant, Quantity


def prepare_init_mps(mol_list, mmax, nexciton, mpo_scheme, compress_method=None, offset=Quantity(0), optimize=None):
    i_mps, h_mpo = solver.construct_mps_mpo_2(mol_list, mmax, nexciton, scheme=mpo_scheme, offset=offset)
    if compress_method is not None:
        i_mps.compress_method = compress_method
    if optimize is not None:
        i_mps.optimize_config = optimize
        solver.optimize_mps(i_mps, h_mpo)
    return i_mps, h_mpo