# -*- coding: utf-8 -*-

import logging

import numpy as np
from ephMPS.mps import Mpo, Mps
from ephMPS.utils import TdMpsJob, Quantity, CompressConfig


logger = logging.getLogger(__name__)


class SBM(TdMpsJob):

    def __init__(self, mol_list, temperature: Quantity, compress_config=None, evolve_config=None, dump_dir=None, job_name=None):
        self.mol_list = mol_list
        self.h_mpo = Mpo(mol_list)
        self.temperature = temperature

        if compress_config is None:
            self.compress_config = CompressConfig()
        else:
            self.compress_config = compress_config

        super().__init__(evolve_config=evolve_config, dump_dir=dump_dir, job_name=job_name)


    def init_mps(self):
        logger.debug(f"mpo dimension: {self.h_mpo.bond_dims}, {self.h_mpo.pbond_list}")
        if self.temperature == 0:
            init_mps = Mps.gs(self.mol_list, False)
        else:
            raise NotImplementedError
        init_mps.compress_config = self.compress_config
        init_mps.evolve_config = self.evolve_config
        init_mps.use_dummy_qn = True
        return init_mps

    def evolve_single_step(self, evolve_dt):
        return self.latest_mps.evolve(self.h_mpo, evolve_dt)

    @property
    def spin(self):
        return np.array([1 - 2 * mps.e_occupations[0] for mps in self.tdmps_list])
