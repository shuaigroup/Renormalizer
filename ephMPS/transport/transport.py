# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

from __future__ import division, print_function, absolute_import

import logging
from collections import OrderedDict
from functools import partial

import numpy as np

from ephMPS.mps import Mpo, Mps, MpDm, solver
from ephMPS.utils import TdMpsJob, Quantity, EvolveConfig

logger = logging.getLogger(__name__)

EDGE_THRESHOLD = 1e-4


def calc_reduced_density_matrix_straight(
    mp
):  # this procedure is **very** memory consuming
    if mp.is_mpdm:
        density_matrix_product = mp.apply(mp.conj_trans())
        # density_matrix_product = mp
    elif mp.is_mps:
        density_matrix_product = MpDm()
        # todo: not elegant! figure out a better way to deal with data type
        density_matrix_product.dtype = np.complex128
        density_matrix_product.mol_list = mp.mol_list
        for mt in mp:
            bond1, phys, bond2 = mt.shape
            mt1 = mt.reshape(bond1, phys, bond2, 1)
            mt2 = mt.conj().reshape(bond1, phys, bond2, 1)
            new_mt = (
                np.tensordot(mt1, mt2, axes=[3, 3])
                .transpose((0, 3, 1, 4, 2, 5))
                .reshape(bond1 ** 2, phys, phys, bond2 ** 2)
            )
            density_matrix_product.append(new_mt)
        density_matrix_product.build_empty_qn()
    else:
        assert False
    mp.set_peak_bytes(density_matrix_product.total_bytes)
    return density_matrix_product.get_reduced_density_matrix()


# saves memory, but still time consuming, especially when the calculation starts,
# in long term the same order with expectation
def calc_reduced_density_matrix(mp):
    if mp.is_mps:
        mp1 = [mt.reshape(mt.shape[0], mt.shape[1], 1, mt.shape[2]) for mt in mp]
        mp2 = [mt.reshape(mt.shape[0], 1, mt.shape[1], mt.shape[2]).conj() for mt in mp]
    else:
        assert mp.is_mpdm
        mp1 = mp
        mp2 = mp.conj_trans()
    reduced_density_matrix = np.zeros(
        (mp.mol_list.mol_num, mp.mol_list.mol_num), dtype=np.complex128
    )
    for i in range(mp.mol_list.mol_num):
        for j in range(mp.mol_list.mol_num):
            elem = np.array([1]).reshape(1, 1)
            e_idx = -1
            for mt_idx, (mt1, mt2) in enumerate(zip(mp1, mp2)):
                if mp.ephtable.is_electron(mt_idx):
                    e_idx += 1
                    axis_idx1 = int(e_idx == i)
                    axis_idx2 = int(e_idx == j)
                    sub_mt1 = mt1[:, axis_idx1, :, :]
                    sub_mt2 = mt2[:, :, axis_idx2, :]
                    elem = np.tensordot(elem, sub_mt1, axes=(0, 0))
                    elem = np.tensordot(elem, sub_mt2, axes=[(0, 1), (0, 1)])
                else:
                    elem = np.tensordot(elem, mt1, axes=(0, 0))
                    elem = np.tensordot(elem, mt2, axes=[(0, 1, 2), (0, 2, 1)])
            reduced_density_matrix[i][j] = elem.flatten()[0]
    return reduced_density_matrix


class ChargeTransport(TdMpsJob):
    def __init__(self, mol_list, temperature=Quantity(0, "K"), evolve_config=None):
        self.mol_list = mol_list
        self.temperature = temperature
        self.mpo = None
        self.mpo_e_lbound = None  # lower bound of the energy of the hamiltonian
        self.mpo_e_ubound = None  # upper bound of the energy of the hamiltonian
        super(ChargeTransport, self).__init__(evolve_config)
        self.energies = [self.tdmps_list[0].expectation(self.mpo)]
        self.reduced_density_matrices = [
            calc_reduced_density_matrix_straight(self.tdmps_list[0])
        ]
        self.custom_dump_info = OrderedDict()
        self.stop_at_edge = False
        self.memory_limit = None
        self.economic_mode = (
            False
        )  # if set True, only save full information of the latest mps and discard previous ones

    @property
    def mol_num(self):
        return self.mol_list.mol_num

    def create_electron(self, gs_mp):
        # test code to put phonon at ground state of the electronic excited state
        """
        import math
        ph_mps = gs_mp[self.mol_num]  # suppose only one phonon mode
        mol = self.mol_list[self.mol_num // 2]
        s = mol.phs[0].coupling_constant ** 2

        for k in range(mol.phs[0].n_phys_dim):
            ph_mps[0, k, 0] = math.exp(-s) * s ** k / math.factorial(k)
        """
        # previous create electron code
        creation_operator = Mpo.onsite(
            self.mol_list, "a^\dagger", mol_idx_set={self.mol_num // 2}
        )
        return creation_operator.apply(gs_mp)

    def init_mps(self):
        # self.mpo = Mpo(self.mol_list, scheme=3)
        tentative_mpo = Mpo(self.mol_list, scheme=3)
        if self.temperature == 0:
            gs_mp = Mps.gs(self.mol_list, max_entangled=False)
        else:
            gs_mp = MpDm.max_entangled_gs(self.mol_list)
            # subtract the energy otherwise might causes numeric error because of large offset * dbeta
            energy = Quantity(gs_mp.expectation(tentative_mpo))
            mpo = Mpo(self.mol_list, scheme=3, offset=energy)
            gs_mp = gs_mp.thermal_prop_exact(
                mpo, self.temperature.to_beta() / 2, 50, "GS", True
            )
        init_mp = self.create_electron(gs_mp)
        energy = Quantity(init_mp.expectation(tentative_mpo))
        self.mpo = Mpo(self.mol_list, scheme=3, offset=energy)
        self.mpo_e_lbound = solver.find_lowest_energy(self.mpo, 1, 20)
        self.mpo_e_ubound = solver.find_highest_energy(self.mpo, 1, 20)
        init_mp.canonicalise()
        init_mp.evolve_config = self.evolve_config
        # init_mp.invalidate_cache()
        return init_mp

    def evolve_single_step(self, evolve_dt):
        old_mps = self.latest_mps
        new_mps = old_mps.evolve(self.mpo, evolve_dt)
        if self.memory_limit is not None:
            while self.memory_limit < new_mps.peak_bytes:
                old_mps.threshold *= 1.2
                logger.debug("Set threshold to {:g}".format(old_mps.threshold))
                old_mps.peak_bytes = 0
                new_mps = old_mps.evolve(self.mpo, evolve_dt)
        if self.economic_mode:
            old_mps.clear_memory()
        new_energy = new_mps.expectation(self.mpo)
        self.energies.append(new_energy)
        logger.info(
            "Energy of the new mps: %g, %.5f%% of initial energy preserved"
            % (new_energy, self.latest_energy_ratio * 100)
        )
        logger.debug("Calculating reduced density matrix")
        if self.reduced_density_matrices is not None:
            self.reduced_density_matrices.append(calc_reduced_density_matrix(new_mps))
        logger.debug("Calculate reduced density matrix finished")
        return new_mps

    def stop_evolve_criteria(self):
        # electron has moved to the edge
        return self.stop_at_edge and EDGE_THRESHOLD < self.e_occupations_array[-1][0]

    def get_dump_dict(self):
        dump_dict = OrderedDict()
        dump_dict["mol list"] = self.mol_list.to_dict()
        dump_dict["J constant"] = str(self.mol_list.j_constant)
        dump_dict["total steps"] = len(self.tdmps_list)
        dump_dict["total time"] = self.evolve_times[-1]
        dump_dict["diffusion"] = self.latest_mps.r_square / self.evolve_times[-1]
        dump_dict["delta energy (%)"] = (self.latest_energy_ratio - 1) * 100
        dump_dict["thresholds"] = [tdmps.threshold for tdmps in self.tdmps_list]
        dump_dict["other info"] = self.custom_dump_info
        # make np array json serializable
        dump_dict["r square array"] = list(self.r_square_array)
        dump_dict["electron occupations array"] = [
            list(occupations) for occupations in self.e_occupations_array
        ]
        dump_dict["phonon occupations array"] = [
            list(occupations) for occupations in self.ph_occupations_array
        ]
        dump_dict["coherent length array"] = list(self.coherent_length_array.real)
        if self.reduced_density_matrices is not None:
            dump_dict["final reduced density matrix real"] = [
                list(row.real) for row in self.reduced_density_matrices[-1]
            ]
            dump_dict["final reduced density matrix imag"] = [
                list(row.imag) for row in self.reduced_density_matrices[-1]
            ]
        dump_dict["time series"] = list(self.evolve_times)
        return dump_dict

    @property
    def initial_energy(self):
        return float(self.energies[0])

    @property
    def latest_energy(self):
        return float(self.energies[-1])

    @property
    def latest_energy_ratio(self):
        return (self.latest_energy - self.mpo_e_lbound) / (
            self.initial_energy - self.mpo_e_lbound
        )

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
        if self.reduced_density_matrices is None:
            return np.array([])
        return np.array(
            [
                np.abs(rdm).sum() - np.trace(rdm).real
                for rdm in self.reduced_density_matrices
            ]
        )

    def is_similar(self, other, rtol=1e-3):
        # avoid a lot of if (not) ...: return False statements
        class FalseFlag(Exception):
            pass

        def my_assert(condition):
            if not condition:
                raise FalseFlag

        all_close_with_tol = partial(np.allclose, rtol=rtol)
        try:
            my_assert(len(self.tdmps_list) == len(other.tdmps_list))
            my_assert(all_close_with_tol(self.evolve_times, other.evolve_times))
            my_assert(all_close_with_tol(self.r_square_array, other.r_square_array))
            if (
                not (np.array(self.energies) < 1e-5).all()
                or not (np.array(other.energies) < 1e-5).all()
            ):
                my_assert(all_close_with_tol(self.energies, other.energies))
            my_assert(
                np.allclose(
                    self.e_occupations_array, other.e_occupations_array, atol=1e-3
                )
            )
            my_assert(
                np.allclose(
                    self.ph_occupations_array, other.ph_occupations_array, atol=1e-3
                )
            )
            my_assert(
                all_close_with_tol(
                    self.coherent_length_array, other.coherent_length_array
                )
            )
        except FalseFlag:
            return False
        else:
            return True
