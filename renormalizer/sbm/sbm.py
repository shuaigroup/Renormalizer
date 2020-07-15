# -*- coding: utf-8 -*-

import logging
from functools import partial

from renormalizer.model import MolList
from renormalizer.mps import Mpo, Mps
from renormalizer.utils import TdMpsJob, Quantity, CompressConfig


logger = logging.getLogger(__name__)


class SpinBosonModel(TdMpsJob):

    def __init__(self, mol_list: MolList, temperature: Quantity, compress_config=None, evolve_config=None, dump_dir=None, job_name=None):
        self.mol_list = mol_list
        self.h_mpo = Mpo(mol_list)
        self.temperature = temperature

        if compress_config is None:
            self.compress_config = CompressConfig()
        else:
            self.compress_config = compress_config

        self.sigma_x = []
        self.sigma_z = []

        super().__init__(evolve_config=evolve_config, dump_dir=dump_dir, job_name=job_name)

    def init_mps(self):
        logger.debug(f"mpo bond and physical dimension: {self.h_mpo.bond_dims}, {self.h_mpo.pbond_list}")
        if self.temperature == 0:
            init_mps = Mps.ground_state(self.mol_list, False)
        else:
            raise NotImplementedError
        init_mps.compress_config = self.compress_config
        init_mps.evolve_config = self.evolve_config
        init_mps.use_dummy_qn = True
        self.h_mpo = Mpo(self.mol_list, offset=Quantity(init_mps.expectation(self.h_mpo)))
        init_mps = init_mps.expand_bond_dimension(self.h_mpo)
        return init_mps

    def process_mps(self, mps):
        sigma_z_mpo = self.mol_list.get_mpos("sigma_z", partial(Mpo.onsite, opera="sigma_z"))
        sigma_z = mps.expectation(sigma_z_mpo)
        self.sigma_z.append(sigma_z)
        sigma_x_mpo = self.mol_list.get_mpos("sigma_x", partial(Mpo.onsite, opera="sigma_x"))
        sigma_x = mps.expectation(sigma_x_mpo)
        self.sigma_x.append(sigma_x)
        logger.info(f"sigma_z: {self.sigma_z[-1]}. sigma_x: {self.sigma_x[-1]}")

    def evolve_single_step(self, evolve_dt):
        return self.latest_mps.evolve(self.h_mpo, evolve_dt)

    def get_dump_dict(self):
        dump_dict = dict()
        dump_dict["mol list"] = self.mol_list.to_dict()
        dump_dict["tempearture"] = self.temperature.as_au()
        dump_dict["time series"] = self.evolve_times
        dump_dict["sigma_x"] = self.sigma_x
        dump_dict["sigma_z"] = self.sigma_z
        return dump_dict
