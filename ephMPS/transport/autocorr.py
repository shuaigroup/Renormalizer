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

    def __init__(self, mol_list, temperature: Quantity, insteps: int=None, ievolve_config=None, compress_config=None, evolve_config=None, dump_dir: str=None, job_name: str=None):
        self.mol_list = mol_list
        self.h_mpo = Mpo(mol_list)
        self.temperature = temperature

        # imaginary time evolution config
        if ievolve_config is None:
            self.ievolve_config = EvolveConfig()
            if insteps is None:
                self.ievolve_config.adaptive = True
                # start from a small step
                self.ievolve_config.evolve_dt = temperature.to_beta() / 1e5j
                self.ievolve_config.d_energy = 1
        else:
            self.ievolve_config = ievolve_config
        self.insteps = insteps

        if compress_config is None:
            logger.debug("using default compress config")
            self.compress_config = CompressConfig()
        else:
            self.compress_config = compress_config

        super().__init__(evolve_config, dump_dir, job_name)


    def _construct_flux_operator(self):
        # construct flux operator
        logger.debug("constructing flux operator")
        j_list = []
        for i in range(len(self.mol_list) - 1):
            j1 = Mpo.displacement(self.mol_list, i, i + 1).scale(self.mol_list.j_matrix[i, i + 1])
            j1.compress_config.threshold = 1e-10
            j2 = j1.conj_trans().scale(-1)
            j_list.extend([j1, j2])
        j_oper = compressed_sum(j_list, batchsize=10)
        logger.debug(f"operator bond dim: {j_oper.bond_dims}")
        return j_oper

    def init_mps(self):
        if self._defined_output_path:
            impdm_path = os.path.join(self.dump_dir, self.job_name + '_impdm.npz')
            try:
                logger.info(f"Try load from {impdm_path}")
                mpdm = MpDm.load(self.mol_list, impdm_path)
                logger.info(f"Init mpdm loaded: {mpdm}")
                mpdm.compress_config = self.compress_config
            except FileNotFoundError:
                logger.debug(f"No file found in {impdm_path}")
                mpdm = None
        else:
            mpdm = None
        if mpdm is None:
            i_mpdm = MpDm.max_entangled_ex(self.mol_list)
            i_mpdm.evolve_config = self.ievolve_config
            i_mpdm.compress_config = self.compress_config
            # only propagate half beta
            mpdm = i_mpdm.thermal_prop(self.h_mpo, self.temperature.to_beta() / 2, self.insteps)

            if self.dump_dir is not None and self.job_name is not None:
                impdm_path = os.path.join(self.dump_dir, self.job_name + '_impdm.npz')
                mpdm.dump(impdm_path)

        e = mpdm.expectation(self.h_mpo)
        self.h_mpo = Mpo(self.mol_list, offset=Quantity(e))
        mpdm.evolve_config = self.evolve_config
        # e^{\-beta H/2} \Psi
        j_oper = self._construct_flux_operator()
        ket_mpdm = j_oper.contract(mpdm).canonical_normalize()
        bra_mpdm = ket_mpdm.copy()
        return BraKetPair(bra_mpdm, ket_mpdm)

    def evolve_single_step(self, evolve_dt):
        prev_bra_mpdm, prev_ket_mpdm = self.latest_mps
        if len(self.tdmps_list) % 2 == 1:
            latest_ket_mpdm = prev_ket_mpdm.evolve(self.h_mpo, evolve_dt)
            latest_bra_mpdm = prev_bra_mpdm.copy()
        else:
            latest_ket_mpdm = prev_ket_mpdm.copy()
            prev_bra_mpdm.evolve_config.evolve_dt = -prev_ket_mpdm.evolve_config.evolve_dt
            latest_bra_mpdm = prev_bra_mpdm.evolve(self.h_mpo, -evolve_dt)
        return BraKetPair(latest_bra_mpdm, latest_ket_mpdm)

    def stop_evolve_criteria(self):
        corr = self.auto_corr
        if len(corr) < 10:
            return False
        last_corr = corr[-10:]
        first_corr = corr[0]
        return np.abs(last_corr.mean()) < 1e-5 * np.abs(first_corr) and last_corr.std() < 1e-5 * np.abs(first_corr)

    @property
    def auto_corr(self):
        return np.array([m.ft for m in self.tdmps_list])

    def get_dump_dict(self):
        dump_dict = dict()
        dump_dict["mol list"] = self.mol_list.to_dict()
        dump_dict["tempearture"] = self.temperature.as_au()
        dump_dict["time series"] = self.evolve_times
        dump_dict["auto correlation real"] = cast_float(self.auto_corr.real)
        dump_dict["auto correlation imag"] = cast_float(self.auto_corr.imag)
        dump_dict["mobility"] = self.calc_mobility()[1]
        return dump_dict

    def calc_mobility(self):
        time_series = self.evolve_times
        corr_real = self.auto_corr.real
        inte = scipy.integrate.trapz(corr_real, time_series)
        mobility_in_au = inte / self.temperature.as_au()
        mobility = mobility_in_au / mobility2au
        return mobility_in_au, mobility