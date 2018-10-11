# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

from __future__ import division, print_function, absolute_import

import logging
from collections import OrderedDict
from functools import partial

import numpy as np

from ephMPS.mps import solver
from ephMPS.mps.mpo import Mpo
from ephMPS.mps.mps import Mps
from ephMPS.utils import TdMpsJob, constant, sizeof_fmt
from ephMPS.mps.matrix import MatrixState, DensityMatrixOp

logger = logging.getLogger(__name__)

EDGE_THRESHOLD = 1e-4


def calc_reduced_density_matrix(mp):
    if mp.mtype == DensityMatrixOp:
        density_matrix_product = mp.apply(mp.conj_trans())
        #density_matrix_product = mp
    elif mp.mtype == MatrixState:  # this step is memory consuming
        density_matrix_product = Mpo()
        density_matrix_product.mtype = DensityMatrixOp
        # todo: not elegant! figure out a better way to deal with data type
        density_matrix_product.dtype = np.complex128
        density_matrix_product.mol_list = mp.mol_list
        for mt in mp:
            bond1, phys, bond2 = mt.shape
            mt1 = mt.conj().reshape(bond1, phys, bond2, 1)
            mt2 = mt.reshape(bond1, phys, bond2, 1)
            new_mt = np.tensordot(mt1, mt2, axes=[3, 3]).transpose((0, 3, 1, 4, 2, 5)).reshape(bond1 ** 2, phys, phys, bond2 ** 2)
            density_matrix_product.append(new_mt)
        density_matrix_product.build_empty_qn()
    else:
        assert False
    mp.peak_bytes = max(mp.peak_bytes, density_matrix_product.total_bytes)
    return density_matrix_product.get_reduced_density_matrix()


class ChargeTransport(TdMpsJob):
    def __init__(self, mol_list, j_constant, temperature=0):
        self.mol_list = mol_list
        self.j_constant = j_constant
        self.temperature = temperature
        self.mpo = None
        super(ChargeTransport, self).__init__()
        self.energies = [self.tdmps_list[0].expectation(self.mpo)]
        self.reduced_density_matrices = [calc_reduced_density_matrix(self.tdmps_list[0])]
        self.custom_dump_info = OrderedDict()
        self.stop_at_edge = False
        self.memory_limit = None
        self.economic_mode = False   # if set True, only save full information of the latest mps and discard previous ones

    @property
    def mol_num(self):
        return self.mol_list.mol_num

    def create_electron(self, gs_mp):
        creation_operator = Mpo.onsite(self.mol_list, 'a^\dagger', mol_idx_set={self.mol_num // 2})
        return creation_operator.apply(gs_mp)

    def init_mps(self):
        j_matrix = construct_j_matrix(self.mol_num, self.j_constant)
        self.mpo = Mpo(self.mol_list, j_matrix, scheme=3)
        if self.temperature == 0:
            gs_mp = Mps.gs(self.mol_list, max_entangled=False)
        else:
            gs_mp = Mpo.from_mps(Mps.gs(self.mol_list, max_entangled=True))
            beta = constant.t2beta(self.temperature)
            thermal_prop = Mpo.exact_propagator(self.mol_list, - beta / 2, 'GS')
            gs_mp = thermal_prop.apply(gs_mp)
            gs_mp.normalize()
        init_mp = self.create_electron(gs_mp)
        # init_mp.invalidate_cache()
        return init_mp

    def evolve_single_step(self, evolve_dt):
        old_mps = self.latest_mps
        new_mps = old_mps.evolve(self.mpo, evolve_dt, norm=1.0)
        if self.memory_limit is not None:
            while self.memory_limit < new_mps.peak_bytes:
                old_mps.threshold *= 1.2
                logger.debug('Set threshold to {:g}'.format(old_mps.threshold))
                old_mps.peak_bytes = 0
                new_mps = old_mps.evolve(self.mpo, evolve_dt, norm=1.0)
        if self.economic_mode:
            old_mps.clear_memory()
        new_energy = new_mps.expectation(self.mpo)
        self.energies.append(new_energy)
        logger.info('Energy of the new mps: %g, %.5f%% of initial energy preserved'
                    % (new_energy, self.latest_energy_ratio * 100))
        self.reduced_density_matrices.append(calc_reduced_density_matrix(new_mps))
        return new_mps

    def stop_evolve_criteria(self):
        # electron has moved to the edge
        return self.stop_at_edge and EDGE_THRESHOLD < self.e_occupations_array[-1][0]

    def get_dump_dict(self):
        dump_dict = OrderedDict()
        dump_dict['mol list'] = self.mol_list.to_dict()
        dump_dict['J constant'] = str(self.j_constant)
        dump_dict['total steps'] = len(self.tdmps_list)
        dump_dict['total time'] = self.evolve_times[-1]
        dump_dict['diffusion'] = self.latest_mps.r_square / self.evolve_times[-1]
        dump_dict['delta energy (%)'] = (self.latest_energy_ratio - 1) * 100
        dump_dict['thresholds'] = [tdmps.threshold for tdmps in self.tdmps_list]
        dump_dict['other info'] = self.custom_dump_info
        # make np array json serializable
        dump_dict['r square array'] = list(self.r_square_array)
        dump_dict['electron occupations array'] = [list(occupations) for occupations in self.e_occupations_array]
        dump_dict['phonon occupations array'] = [list(occupations) for occupations in self.ph_occupations_array]
        dump_dict['coherent length array'] = list(self.coherent_length_array.real)
        dump_dict['final reduced density matrix real'] = [list(row.real) for row in self.reduced_density_matrices[-1]]
        dump_dict['final reduced density matrix imag'] = [list(row.imag) for row in self.reduced_density_matrices[-1]]
        dump_dict['time series'] = list(self.evolve_times)
        return dump_dict

    @property
    def initial_energy(self):
        return float(self.energies[0])

    @property
    def latest_energy(self):
        return float(self.energies[-1])

    @property
    def latest_energy_ratio(self):
        return self.latest_energy / self.initial_energy

    @property
    def r_square_array(self):
        return np.array([mps.r_square for mps in self.tdmps_list])

    @property
    def e_occupations_array(self):
        return np.array([mps.e_occupations for mps in self.tdmps_list])

    @property
    def ph_occupations_array(self):
        return np.array([mps.ph_occupations for mps in self.tdmps_list])

    @property
    def coherent_length_array(self):
        return np.array([np.abs(rdm).sum() - np.trace(rdm).real for rdm in self.reduced_density_matrices])

    def is_similar(self, other, rtol=1e-3):
        # avoid a lot of if (not) ...: return false statements
        class FalseFlag(Exception): pass
        def my_assert(condition):
            if not condition:
                raise FalseFlag
        all_close_with_tol = partial(np.allclose, rtol=rtol)
        try:
            my_assert(len(self.tdmps_list) == len(other.tdmps_list))
            my_assert(all_close_with_tol(self.evolve_times, other.evolve_times))
            my_assert(all_close_with_tol(self.r_square_array, other.r_square_array))
            my_assert(all_close_with_tol(self.energies, other.energies))
            my_assert(np.allclose(self.e_occupations_array, other.e_occupations_array, atol=1e-3))
            my_assert(np.allclose(self.ph_occupations_array, other.ph_occupations_array, atol=1e-3))
            my_assert(all_close_with_tol(self.coherent_length_array, other.coherent_length_array))
        except FalseFlag:
            return False
        else:
            return True


def construct_j_matrix(mol_num, j_constant):
    j_constant_au = j_constant.as_au()
    j_matrix = np.zeros((mol_num, mol_num))
    for i in range(mol_num):
        for j in range(mol_num):
            if i - j == 1 or i - j == -1:
                j_matrix[i][j] = j_constant_au
    return j_matrix
