# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
import scipy.integrate

from ephMPS.mps import MpDm, Mpo, BraKetPair
from ephMPS.mps.lib import compressed_sum
from ephMPS.utils.constant import mobility2au
from ephMPS.utils import TdMpsJob, Quantity, EvolveConfig, CompressConfig
from ephMPS.utils.utils import cast_float


logger = logging.getLogger(__name__)

# todo: zero temeprature

class TransportAutoCorr(TdMpsJob):

    def __init__(self, mol_list, temperature, insteps, ievolve_config=None, compress_config=None, evolve_config=None):
        self.mol_list = mol_list
        self.h_mpo = Mpo(mol_list)
        self.temperature = temperature
        # imaginary time evolution config
        if ievolve_config is None:
            self.ievolve_config = EvolveConfig()
        else:
            self.ievolve_config = ievolve_config
        if compress_config is None:
            self.compress_config = CompressConfig()
        else:
            self.compress_config = compress_config
        self.insteps = insteps
        super().__init__(evolve_config)

    def _construct_flux_operator(self):
        # construct flux operator
        j_list = []
        for i in range(len(self.mol_list) - 1):
            j1 = Mpo.displacement(self.mol_list, i, i + 1).scale(self.mol_list.j_matrix[i, i + 1])
            j1.compress_config.threshold = 1e-10
            j2 = j1.conj_trans().scale(-1)
            j_list.extend([j1, j2])
        j_oper = compressed_sum(j_list, batchsize=10)
        return j_oper

    def init_mps(self):
        i_mpdm = MpDm.max_entangled_ex(self.mol_list)
        i_mpdm.evolve_config = self.ievolve_config
        i_mpdm.compress_config = self.compress_config
        # only propagate half beta
        mpdm = i_mpdm.thermal_prop(
            self.h_mpo, self.insteps, self.temperature.to_beta() / 2
        )
        if self.dump_dir is not None and self.job_name is not None:
            mpdm.dump(os.path.join(self.dump_dir, self.job_name + 'impdm'))
        e = mpdm.expectation(self.h_mpo)
        self.h_mpo = Mpo(self.mol_list, offset=Quantity(e))
        mpdm.evolve_config = self.evolve_config
        # e^{\-beta H/2} \Psi
        j_oper = self._construct_flux_operator()
        ket_mpdm = j_oper.contract(mpdm).canonical_normalize()
        bra_mpdm = ket_mpdm.copy()
        return BraKetPair(bra_mpdm, ket_mpdm)

    def evolve_single_step(self, evolve_dt):
        latest_bra_mpdm, latest_ket_mpdm = self.latest_mps
        if len(self.tdmps_list) % 2 == 1:
            latest_ket_mpdm = latest_ket_mpdm.evolve(self.h_mpo, evolve_dt)
        else:
            latest_bra_mpdm = latest_bra_mpdm.evolve(self.h_mpo, -evolve_dt)
        return BraKetPair(latest_bra_mpdm, latest_ket_mpdm)

    def stop_evolve_criteria(self):
        corr = self.auto_corr
        if len(corr) < 10:
            return False
        last_corr = corr[-10:]
        return np.abs(last_corr.mean()) < 1e-3 and last_corr.std() < 1e-3

    @property
    def auto_corr(self):
        return np.array([m.ft for m in self.tdmps_list])

    def get_dump_dict(self):
        dump_dict = dict()
        dump_dict["mol list"] = self.mol_list.to_dict()
        dump_dict["tempearture"] = self.temperature.as_au()
        dump_dict["time series"] = self.evolve_times
        dump_dict["auto correlation"] = cast_float(self.auto_corr)
        dump_dict["mobility"] = self.calc_mobility()[1]
        return dump_dict

    def calc_mobility(self):
        time_series = self.evolve_times
        corr_real = self.auto_corr.real
        inte = scipy.integrate.trapz(corr_real, time_series)
        mobility_in_au = inte / self.temperature.as_au()
        mobility = mobility_in_au / mobility2au
        return mobility_in_au, mobility