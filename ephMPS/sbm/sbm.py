# -*- coding: utf-8 -*-

import logging
from functools import partial

from ephMPS.model import MolList
from ephMPS.mps import Mpo, Mps
from ephMPS.utils import TdMpsJob, Quantity, CompressConfig
from ephMPS.utils.utils import cast_float


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
        logger.debug(f"mpo dimension: {self.h_mpo.bond_dims}, {self.h_mpo.pbond_list}")
        if self.temperature == 0:
            init_mps = Mps.gs(self.mol_list, False)
        else:
            raise NotImplementedError
        init_mps.compress_config = self.compress_config
        init_mps.evolve_config = self.evolve_config
        init_mps.use_dummy_qn = True
        self._update_sigma(init_mps)
        self.h_mpo = Mpo(self.mol_list, offset=Quantity(init_mps.expectation(self.h_mpo)))
        return init_mps

    def evolve_single_step(self, evolve_dt):
        new_mps = self.latest_mps.evolve(self.h_mpo, evolve_dt)
        self._update_sigma(new_mps)
        logger.info(f"sigma_z: {self.sigma_z[-1]}. sigma_x: {self.sigma_x[-1]}")
        return new_mps

    def _update_sigma(self, mps):
        self.sigma_z.append(1 - 2 * mps.e_occupations[0])
        sigmax_mpo = self.mol_list.get_mpos("sigma_x", partial(Mpo.onsite, opera="sigmax"))
        self.sigma_x.append(mps.expectation(sigmax_mpo))


    def get_dump_dict(self):
        dump_dict = dict()
        dump_dict["mol list"] = self.mol_list.to_dict()
        dump_dict["tempearture"] = self.temperature.as_au()
        dump_dict["time series"] = self.evolve_times
        dump_dict["sigmax"] = cast_float(self.sigma_x)
        dump_dict["sigmaz"] = cast_float(self.sigma_z)
        return dump_dict
