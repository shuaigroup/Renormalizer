# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

from __future__ import division, print_function, absolute_import

import logging
from enum import Enum
from collections import OrderedDict
from functools import partial

from typing import List

from renormalizer.mps import Mpo, Mps, MpDm, MpDmFull, SuperLiouville, ThermalProp
from renormalizer.model import MolList
from renormalizer.utils import TdMpsJob, Quantity, CompressConfig
from renormalizer.utils.utils import cast_float

import numpy as np

logger = logging.getLogger(__name__)

EDGE_THRESHOLD = 1e-4


class InitElectron(Enum):
    fc = "franc-condon excitation"
    relaxed = "analytically relaxed phonon(s)"


class ChargeTransport(TdMpsJob):
    def __init__(
        self,
        mol_list: MolList,
        temperature=Quantity(0, "K"),
        compress_config=None,
        evolve_config=None,
        stop_at_edge=True,
        init_electron=InitElectron.relaxed,
        logging_output: List[str] =None,
        rdm: bool =False,
        dissipation: float =0
    ):
        self.mol_list: MolList = mol_list
        self.temperature = temperature
        all_logging_output = ["r_square", "e_occupations", "ph_occupations"]
        if logging_output is None:
            self.logging_output = all_logging_output
        else:
            if not set(logging_output) < set(all_logging_output):
                raise ValueError(f"Invalid logging output option. Expected chosen from {all_logging_output}. Got {logging_output}")
            self.logging_output = logging_output
        self.mpo = None
        self.init_electron = init_electron
        self.dissipation = dissipation
        if compress_config is None:
            self.compress_config: CompressConfig = CompressConfig()
        else:
            self.compress_config: CompressConfig = compress_config
        self.energies = []
        self._r_square_array = []
        self._e_occupations_array = []
        self._ph_occupations_array = []
        self.reduced_density_matrices = [] if rdm else None
        super(ChargeTransport, self).__init__(evolve_config)
        assert self.mpo is not None

        self.elocalex_arrays = []
        self.j_arrays = []
        self.custom_dump_info = OrderedDict()
        self.stop_at_edge = stop_at_edge

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
        for i, ph in enumerate(center_mol.dmrg_phs):
            idx = self.mol_list.ph_idx(center_mol_idx, i)
            mt = gs_mp[idx][0, ..., 0].asnumpy()
            evecs = ph.get_displacement_evecs()
            if gs_mp.is_mps:
                mt = evecs.dot(mt)
            elif gs_mp.is_mpdm:
                assert np.allclose(np.diag(np.diag(mt)), mt)
                mt = evecs.dot(evecs.T).dot(mt)
            else:
                assert False
            logger.debug(f"relaxed mt: {mt}")
            gs_mp[idx] = mt.reshape([1] + list(mt.shape) + [1])

        creation_operator = Mpo.onsite(
            self.mol_list, r"a^\dagger", mol_idx_set={center_mol_idx}
        )
        mps = creation_operator.apply(gs_mp)
        return mps

    def create_electron(self, gs_mp):
        method_mapping = {InitElectron.fc: self.create_electron_fc,
                          InitElectron.relaxed: self.create_electron_relaxed,
                          }
        logger.info(f"Creating electron using {self.init_electron}")
        return method_mapping[self.init_electron](gs_mp)

    def init_mps(self):
        tentative_mpo = Mpo(self.mol_list)
        if self.temperature == 0:
            gs_mp = Mps.gs(self.mol_list, max_entangled=False)
            if self.dissipation != 0:
                gs_mp = MpDm.from_mps(gs_mp)
        else:
            gs_mp = MpDm.max_entangled_gs(self.mol_list)
            # subtract the energy otherwise might cause numeric error because of large offset * dbeta
            energy = Quantity(gs_mp.expectation(tentative_mpo))
            mpo = Mpo(self.mol_list, offset=energy)
            tp = ThermalProp(gs_mp, mpo, exact=True, space="GS")
            tp.evolve(None, len(gs_mp), self.temperature.to_beta() / 2j)
            gs_mp = tp.latest_mps
        init_mp = self.create_electron(gs_mp)
        if self.dissipation != 0:
            init_mp = MpDmFull.from_mpdm(init_mp)
        energy = Quantity(init_mp.expectation(tentative_mpo))
        self.mpo = Mpo(self.mol_list, offset=energy)
        logger.info(f"mpo bond dims: {self.mpo.bond_dims}")
        logger.info(f"mpo physical dims: {self.mpo.pbond_list}")
        init_mp.evolve_config = self.evolve_config
        init_mp.compress_config = self.compress_config
        if self.evolve_config.is_tdvp and self.temperature == 0:
            init_mp = init_mp.expand_bond_dimension(self.mpo)
        if self.dissipation != 0:
            self.mpo = SuperLiouville(self.mpo, self.dissipation)
        init_mp.canonicalise()
        return init_mp


    def process_mps(self, mps):
        new_energy = mps.expectation(self.mpo)
        self.energies.append(new_energy)
        logger.debug(f"Energy: {new_energy}")
        for attr_str in self.logging_output:
            attr = getattr(mps, attr_str)
            logger.info(f"{attr_str}: {attr}")
            self_array = getattr(self, f"_{attr_str}_array")
            self_array.append(attr)

        if self.reduced_density_matrices is not None:
            logger.debug("Calculating reduced density matrix")
            self.reduced_density_matrices.append(mps.calc_reduced_density_matrix())
            logger.debug("Calculate reduced density matrix finished")


    def evolve_single_step(self, evolve_dt):
        old_mps = self.latest_mps
        #mol_list = self.mol_list.get_fluctuation_mollist(self.latest_evolve_time)
        #self.elocalex_arrays.append(mol_list.elocalex_array)
        #self.j_arrays.append(mol_list.adjacent_transfer_integral)
        #mpo = Mpo(mol_list, 3, offset=self.mpo.offset)
        mpo = self.mpo
        new_mps = old_mps.evolve(mpo, evolve_dt)
        return new_mps

    def stop_evolve_criteria(self):
        # electron has moved to the edge
        return self.stop_at_edge and EDGE_THRESHOLD < self.e_occupations_array[-1][0]

    def get_dump_dict(self):
        dump_dict = OrderedDict()
        dump_dict["mol list"] = self.mol_list.to_dict()
        dump_dict["tempearture"] = self.temperature.as_au()
        dump_dict["total time"] = self.evolve_times[-1]
        dump_dict["diffusion"] = self.latest_mps.r_square / self.evolve_times[-1]
        dump_dict["other info"] = self.custom_dump_info
        # make np array json serializable
        dump_dict["r square array"] = cast_float(self.r_square_array)
        dump_dict["electron occupations array"] = cast_float(self.e_occupations_array)
        dump_dict["phonon occupations array"] = cast_float(self.ph_occupations_array)
        #dump_dict["elocalex arrays"] = [list(e) for e in self.elocalex_arrays]
        #dump_dict["j arrays"] = [list(j) for j in self.j_arrays]
        dump_dict["coherent length array"] = cast_float(self.coherent_length_array.real)
        if self.reduced_density_matrices:
            dump_dict["final reduced density matrix real"] = cast_float(self.reduced_density_matrices[-1].real)
            dump_dict["final reduced density matrix imag"] = cast_float(self.reduced_density_matrices[-1].imag)
        dump_dict["time series"] = list(self.evolve_times)
        return dump_dict

    @property
    def r_square_array(self):
        return np.array(self._r_square_array)

    @property
    def e_occupations_array(self):
        return np.array(self._e_occupations_array)

    @property
    def ph_occupations_array(self):
        return np.array(self._ph_occupations_array)

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

    def is_similar(self, other: "ChargeTransport", rtol=1e-3):
        all_close_with_tol = partial(np.allclose, rtol=rtol, atol=1e-3)
        if len(self.evolve_times) != len(other.evolve_times):
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
