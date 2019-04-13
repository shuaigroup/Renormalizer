# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

from __future__ import division, print_function, absolute_import

import logging
from enum import Enum
from collections import OrderedDict
from functools import partial
from typing import Union

import numpy as np

from ephMPS.mps.backend import backend
from ephMPS.mps.matrix import tensordot, ones
from ephMPS.mps import Mpo, Mps, MpDm, solver
from ephMPS.model import MolList
from ephMPS.utils import TdMpsJob, Quantity, CompressCriteria, CompressConfig

logger = logging.getLogger(__name__)

EDGE_THRESHOLD = 1e-4


# this procedure is **very** memory consuming
def calc_reduced_density_matrix_straight(mp):
    if mp.is_mpdm:
        density_matrix_product = mp.apply(mp.conj_trans())
        # density_matrix_product = mp
    elif mp.is_mps:
        density_matrix_product = MpDm()
        # todo: not elegant! figure out a better way to deal with data type
        density_matrix_product.dtype = backend.complex_dtype
        density_matrix_product.mol_list = mp.mol_list
        for mt in mp:
            bond1, phys, bond2 = mt.shape
            mt1 = mt.reshape((bond1, phys, bond2, 1))
            mt2 = mt.conj().reshape((bond1, phys, bond2, 1))
            new_mt = (
                tensordot(mt1, mt2, axes=[3, 3])
                .transpose((0, 3, 1, 4, 2, 5))
                .reshape((bond1 ** 2, phys, phys, bond2 ** 2))
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
        (mp.mol_list.mol_num, mp.mol_list.mol_num), dtype=backend.complex_dtype
    )
    for i in range(mp.mol_list.mol_num):
        for j in range(mp.mol_list.mol_num):
            elem = ones((1, 1))
            e_idx = -1
            for mt_idx, (mt1, mt2) in enumerate(zip(mp1, mp2)):
                if mp.ephtable.is_electron(mt_idx):
                    e_idx += 1
                    axis_idx1 = int(e_idx == i)
                    axis_idx2 = int(e_idx == j)
                    sub_mt1 = mt1[:, axis_idx1, :, :]
                    sub_mt2 = mt2[:, :, axis_idx2, :]
                    elem = tensordot(elem, sub_mt1, axes=(0, 0))
                    elem = tensordot(elem, sub_mt2, axes=[(0, 1), (0, 1)])
                else:
                    elem = tensordot(elem, mt1, axes=(0, 0))
                    elem = tensordot(elem, mt2, axes=[(0, 1, 2), (0, 2, 1)])
            reduced_density_matrix[i][j] = elem.flatten()[0]
    return reduced_density_matrix


class InitElectron(Enum):
    fc = "franc-condon excitation"
    relaxed = "analytically relaxed phonon(s)"
    polaron = "optimized polaron"


class ChargeTransport(TdMpsJob):
    def __init__(
        self,
        mol_list: MolList,
        temperature=Quantity(0, "K"),
        compress_config=None,
        evolve_config=None,
        stop_at_edge=True,
        init_electron=InitElectron.relaxed,
        rdm=False,
    ):
        self.mol_list: MolList = mol_list
        self.temperature = temperature
        self.mpo = None
        self.mpo_e_lbound = None  # the ground energy of the hamiltonian
        self.init_electron=init_electron
        super(ChargeTransport, self).__init__(compress_config, evolve_config)
        self.energies = [self.tdmps_list[0].expectation(self.mpo)]
        self.reduced_density_matrices = []
        if rdm:
            self.reduced_density_matrices.append(
                calc_reduced_density_matrix_straight(self.tdmps_list[0])
            )
        self.elocalex_arrays = []
        self.j_arrays = []
        self.custom_dump_info = OrderedDict()
        self.stop_at_edge = stop_at_edge
        # if set True, only save full information of the latest mps and discard previous ones
        self.economic_mode = False

    @property
    def mol_num(self):
        return self.mol_list.mol_num

    def create_electron_fc(self, gs_mp):
        center_mol_idx = self.mol_num // 2
        creation_operator = Mpo.onsite(
            self.mol_list, r"a^\dagger", mol_idx_set={center_mol_idx}
        )
        mps = creation_operator.apply(gs_mp)
        return mps

    def create_electron_relaxed(self, gs_mp):
        assert np.allclose(gs_mp.bond_dims, np.ones_like(gs_mp.bond_dims))
        center_mol_idx = self.mol_num // 2
        center_mol = self.mol_list[center_mol_idx]
        # start from phonon
        start_idx = self.mol_list.ephtable.electron_idx(center_mol_idx) + 1
        for i, ph in enumerate(center_mol.dmrg_phs):
            mt = gs_mp[start_idx + i][0, ..., 0].asnumpy()
            evecs = ph.get_displacement_evecs()
            if gs_mp.is_mps:
                mt = evecs.dot(mt)
            elif gs_mp.is_mpdm:
                assert np.allclose(np.diag(np.diag(mt)), mt)
                mt = evecs.dot(evecs.T).dot(mt)
            else:
                assert False
            logger.debug(f"relaxed mt: {mt}")
            gs_mp[start_idx + i] = mt.reshape([1] + list(mt.shape) + [1])

        creation_operator = Mpo.onsite(
            self.mol_list, r"a^\dagger", mol_idx_set={center_mol_idx}
        )
        mps = creation_operator.apply(gs_mp)
        return mps

    def create_electron_polaron(self, gs_mp: Union[Mps, MpDm]):
        # finite temperature not implemented
        assert self.temperature == 0
        assert np.allclose(gs_mp.bond_dims, np.ones_like(gs_mp.bond_dims))
        assert gs_mp.is_left_canon
        sub_mollist, start_molidx = self.mol_list.get_sub_mollist()
        sub_mpo = Mpo(sub_mollist, scheme=3)
        mps = Mps.random(sub_mollist, 1, 10)
        energy = solver.optimize_mps(mps, sub_mpo)
        # use a more strict threshold
        mps.compress_config = CompressConfig(threshold=1e-5)
        # do the canonicalise to make sure it's still left canonicalised
        mps = mps.canonicalise().compress()
        logger.info(f"optimized sub mps: f{mps}, energy: {energy}")
        assert mps.is_left_canon
        start_idx = self.mol_list.ephtable.electron_idx(start_molidx)
        for i in range(len(mps)):
            gs_mp[start_idx + i] = mps[i]
            gs_mp.qn[start_idx + i] = mps.qn[i]
        while start_idx + i != len(gs_mp)-1:
            i += 1
            gs_mp.qn[start_idx+i] = [1]
        gs_mp.qntot += 1
        return gs_mp

    def create_electron(self, gs_mp):
        method_mapping = {InitElectron.fc: self.create_electron_fc,
                          InitElectron.relaxed: self.create_electron_relaxed,
                          InitElectron.polaron: self.create_electron_polaron
                          }
        return method_mapping[self.init_electron](gs_mp)

    def init_mps(self):
        tentative_mpo = Mpo(self.mol_list, scheme=3)
        if self.temperature == 0:
            gs_mp = Mps.gs(self.mol_list, max_entangled=False)
        else:
            gs_mp = MpDm.max_entangled_gs(self.mol_list)
            # subtract the energy otherwise might cause numeric error because of large offset * dbeta
            energy = Quantity(gs_mp.expectation(tentative_mpo))
            mpo = Mpo(self.mol_list, scheme=3, offset=energy)
            gs_mp = gs_mp.thermal_prop_exact(
                mpo, self.temperature.to_beta() / 2, 50, "GS", True
            )
        init_mp = self.create_electron(gs_mp)
        energy = Quantity(init_mp.expectation(tentative_mpo))
        self.mpo = Mpo(self.mol_list, scheme=3, offset=energy)
        logger.info(f"mpo bond dims: {self.mpo.bond_dims}")
        logger.info(f"mpo physical dims: {self.mpo.pbond_list}")
        self.mpo_e_lbound = solver.find_lowest_energy(self.mpo, 1, 20, with_hartree=False)
        init_mp.canonicalise()
        init_mp.evolve_config = self.evolve_config
        # init the compress config if not using threshold
        if self.compress_config.criteria is not CompressCriteria.threshold:
            self.compress_config.set_bondorder(length=len(init_mp) + 1)
        init_mp.compress_config = self.compress_config
        # init_mp.invalidate_cache()
        return init_mp

    def evolve_single_step(self, evolve_dt):
        old_mps = self.latest_mps
        mol_list = self.mol_list.get_fluctuation_mollist(self.latest_evolve_time)
        self.elocalex_arrays.append(mol_list.elocalex_array)
        self.j_arrays.append(mol_list.adjacent_transfer_integral)
        mpo = Mpo(mol_list, 3, offset=self.mpo.offset)
        #mpo = self.mpo
        new_mps = old_mps.evolve(mpo, evolve_dt)
        new_energy = new_mps.expectation(self.mpo)
        self.energies.append(new_energy)
        logger.info(
            "Energy of the new mps: %g, %.5f%% of initial energy preserved"
            % (new_energy, self.latest_energy_ratio * 100)
        )
        logger.info(f"r_square: {new_mps.r_square}")
        if self.economic_mode:
            old_mps.clear_memory()
        if self.reduced_density_matrices:
            logger.debug("Calculating reduced density matrix")
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
        dump_dict["tempearture"] = self.temperature.as_au()
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
        #dump_dict["elocalex arrays"] = [list(e) for e in self.elocalex_arrays]
        dump_dict["j arrays"] = [list(j) for j in self.j_arrays]
        dump_dict["coherent length array"] = list(map(float, self.coherent_length_array.real))
        if self.reduced_density_matrices:
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
        all_close_with_tol = partial(np.allclose, rtol=rtol, atol=1e-3)
        if len(self.tdmps_list) != len(other.tdmps_list):
            return False
        attrs = [
            "evolve_times",
            "r_square_array",
            "energies",
            "e_occupations_array",
            "ph_occupations_array",
            "coherent_length_array",
        ]
        for attr in attrs:
            s = getattr(self, attr)
            o = getattr(other, attr)
            if not all_close_with_tol(s, o):
                return False
        return True
