# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import numpy as np

from renormalizer.mps import Mpo, MpDm, ThermalProp
from renormalizer.spectra.base import SpectraTdMpsJobBase
from renormalizer.mps.mps import BraKetPair
from renormalizer.utils import CompressConfig, EvolveConfig

import os
import logging

logger = logging.getLogger(__name__)


class BraKetPairEmiFiniteT(BraKetPair):
    def calc_ft(self):
        return np.conj(super(BraKetPairEmiFiniteT, self).calc_ft())


class BraKetPairAbsFiniteT(BraKetPair):
    pass


class SpectraFiniteT(SpectraTdMpsJobBase):
    def __init__(
        self,
        model,
        spectratype,
        temperature,
        insteps,
        offset,
        evolve_config=None,
        icompress_config=None,
        ievolve_config=None,
        gs_shift=0,
        dump_dir: str=None,
        job_name=None,
    ):
        self.temperature = temperature
        self.insteps = insteps
        self.gs_shift = gs_shift
        self.icompress_config = icompress_config
        self.ievolve_config = ievolve_config
        if self.icompress_config is None:
            self.icompress_config = CompressConfig()
        if self.ievolve_config is None:
            self.ievolve_config = EvolveConfig()
        self.dump_dir = dump_dir
        self.job_name = job_name
        super(SpectraFiniteT, self).__init__(
            model,
            spectratype,
            temperature,
            evolve_config=evolve_config,
            offset=offset,
            dump_dir=dump_dir,
            job_name=job_name
        )

    def init_mps(self):
        if self.spectratype == "emi":
            return self.init_mps_emi()
        else:
            return self.init_mps_abs()

    def init_mps_emi(self):
        dipole_mpo = Mpo.onsite(self.model, "a", dipole=True)
        i_mpo = MpDm.max_entangled_ex(self.model)
        i_mpo.compress_config = self.icompress_config
        if self.job_name is None:
            job_name = None
        else:
            job_name = self.job_name + "_thermal_prop"
        # only propagate half beta
        tp = ThermalProp(
            i_mpo, evolve_config=self.ievolve_config,
            dump_dir=self.dump_dir, job_name=job_name
        )
        if tp._defined_output_path:
            try:
                logger.info(
                    f"load density matrix from {self._thermal_dump_path}"
                )
                ket_mpo = MpDm.load(self.model, self._thermal_dump_path)
                logger.info(f"density matrix loaded:{ket_mpo}")
            except FileNotFoundError:
                logger.debug(f"no file found in {self._thermal_dump_path}")
                tp.evolve(None, self.insteps, self.temperature.to_beta() / 2j)
                ket_mpo = tp.latest_mps
                ket_mpo.dump(self._thermal_dump_path)
        else:
            tp.evolve(None, self.insteps, self.temperature.to_beta() / 2j)
            ket_mpo = tp.latest_mps
        ket_mpo.evolve_config = self.evolve_config
        # e^{\-beta H/2} \Psi
        dipole_mpo_dagger = dipole_mpo.conj_trans()
        dipole_mpo_dagger.build_empty_qn()
        a_ket_mpo = ket_mpo.apply(dipole_mpo_dagger, canonicalise=True)
        a_ket_mpo.normalize("mps_norm_to_coeff")
        a_bra_mpo = a_ket_mpo.copy()
        return BraKetPairEmiFiniteT(a_bra_mpo, a_ket_mpo)

    @property
    def _thermal_dump_path(self):
        assert self._defined_output_path
        return os.path.join(self.dump_dir, self.job_name + "_impo.npz")

    def get_dump_dict(self):
        dump_dict = dict()
        dump_dict['temperature'] = self.temperature.as_au()
        dump_dict['time series'] = self.evolve_times
        dump_dict['autocorr'] = self.autocorr
        return dump_dict

    def stop_evolve_criteria(self):
        corr = self.autocorr
        if len(corr) < 10:
            return False
        last_corr = corr[-10:]
        first_corr = corr[0]
        return np.abs(last_corr.mean()) < 1e-5 * np.abs(first_corr) and last_corr.std() < 1e-5 * np.abs(first_corr)

    def init_mps_abs(self):
        dipole_mpo = Mpo.onsite(self.model, r"a^\dagger", dipole=True)
        i_mpo = MpDm.max_entangled_gs(self.model)
        i_mpo.compress_config = self.icompress_config
        beta = self.temperature.to_beta()
        tp = ThermalProp(i_mpo, exact=True, space="GS")
        tp.evolve(None, 1, beta / 2j)
        ket_mpo = tp.latest_mps
        ket_mpo.evolve_config = self.evolve_config
        a_ket_mpo = dipole_mpo.apply(ket_mpo, canonicalise=True)
        if self.evolve_config.is_tdvp:
            a_ket_mpo = a_ket_mpo.expand_bond_dimension(self.h_mpo)
        a_ket_mpo.normalize("mps_norm_to_coeff")
        a_bra_mpo = a_ket_mpo.copy()
        return BraKetPairAbsFiniteT(a_bra_mpo, a_ket_mpo)

    def evolve_single_step(self, evolve_dt):
        latest_bra_mpo, latest_ket_mpo = self.latest_mps
        if len(self.evolve_times) % 2 == 1:
            latest_ket_mpo = \
                latest_ket_mpo.evolve_exact(self.h_mpo, -evolve_dt, "GS")
            latest_ket_mpo = latest_ket_mpo.evolve(self.h_mpo, evolve_dt)
        else:
            latest_bra_mpo = \
                latest_bra_mpo.evolve_exact(self.h_mpo, evolve_dt, "GS")
            latest_bra_mpo = latest_bra_mpo.evolve(self.h_mpo, -evolve_dt)
        return self.latest_mps.__class__(latest_bra_mpo, latest_ket_mpo)
