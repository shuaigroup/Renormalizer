# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

from __future__ import division, print_function, absolute_import

import logging
from collections import OrderedDict

import numpy as np

from ephMPS.mps import solver
from ephMPS.mps.mpo import Mpo
from ephMPS.mps.mps import Mps
from ephMPS.utils import TdMpsJob, constant

logger = logging.getLogger(__name__)


class ChargeTransport(TdMpsJob):
    def __init__(self, mol_list, j_constant, temperature=0):
        self.mol_list = mol_list
        self.j_constant = j_constant
        self.temperature = temperature
        self.mpo = None
        super(ChargeTransport, self).__init__()
        self.energies = [self.tdmps_list[0].expectation(self.mpo)]
        self.custom_dump_info = OrderedDict()

    @property
    def mol_num(self):
        return self.mol_list.mol_num

    def create_electron(self, gs_mp):
        creation_operator = Mpo.onsite(self.mol_list, 'a^\dagger', mol_idx_set={self.mol_num // 2})
        return creation_operator.apply(gs_mp)

    def init_mps(self):
        j_matrix = construct_j_matrix(self.mol_num, self.j_constant)
        self.mpo = Mpo(self.mol_list, j_matrix, scheme=3)
        logger.debug('Energy of the Hamiltonian: %g' % solver.find_eigen_energy(self.mpo, 1, 20))
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
        new_mps = self.latest_mps.evolve(self.mpo, evolve_dt, norm=1.0)
        new_energy = new_mps.expectation(self.mpo)
        self.energies.append(new_energy)
        logger.info('Energy of the new mps: %g, %.5f%% of initial energy preserved'
                    % (new_energy, self.latest_energy_ratio * 100))
        return new_mps

    def get_dump_dict(self):
        dump_dict = OrderedDict()
        dump_dict['mol list'] = self.mol_list.to_dict()
        dump_dict['J constant'] = str(self.j_constant)
        dump_dict['total steps'] = len(self.tdmps_list)
        dump_dict['diffusion'] = self.latest_mps.r_square / self.evolve_times[-1]
        dump_dict['delta energy (%)'] = (self.latest_energy_ratio - 1) * 100
        dump_dict['other info'] = self.custom_dump_info
        # make np array json serializable
        dump_dict['r square array'] = list(self.r_square_array)
        dump_dict['electron occupations array'] = [list(occupations) for occupations in self.e_occupations_array]
        dump_dict['phonon occupations array'] = [list(occupations) for occupations in self.ph_occupations_array]
        dump_dict['time series'] = list(self.evolve_times)
        return dump_dict

    @property
    def initial_energy(self):
        return self.energies[0]

    @property
    def latest_energy(self):
        return self.energies[-1]

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


def construct_j_matrix(mol_num, j_constant):
    j_constant_au = j_constant.as_au()
    j_matrix = np.zeros((mol_num, mol_num))
    for i in range(mol_num):
        for j in range(mol_num):
            if i - j == 1 or i - j == -1:
                j_matrix[i][j] = j_constant_au
    return j_matrix