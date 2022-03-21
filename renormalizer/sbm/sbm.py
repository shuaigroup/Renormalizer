# -*- coding: utf-8 -*-

import logging
from functools import partial

from renormalizer.model import Model
from renormalizer.mps import Mpo, Mps
from renormalizer.utils import TdMpsJob, Quantity, CompressConfig
import numpy as np

logger = logging.getLogger(__name__)


class SpinBosonDynamics(TdMpsJob):
    
    r"""
    The Spin-Boson Model

    The initial state is a Hartree product state with all vibrations at :math:`|
    0 \rangle` and spin at spin up state. 
    The class can be used at zero temperature or finite temperature with
    thermofield dynamics method. 
    
    """

    def __init__(self, 
            model: Model, 
            auto_expand: bool = True,
            compress_config=None, 
            evolve_config=None, 
            dump_dir=None,
            dump_mps=None, 
            job_name=None
            ):
        self.model = model
        self.h_mpo = Mpo(model)
        self.auto_expand = auto_expand
        
        if compress_config is None:
            self.compress_config = CompressConfig()
        else:
            self.compress_config = compress_config

        self.sigma_x = []
        self.sigma_z = []
        self.rho = []
        self.bond_entropy = []

        super().__init__(evolve_config=evolve_config, dump_dir=dump_dir,
                dump_mps=dump_mps, job_name=job_name)

    def init_mps(self):
        logger.debug(f"mpo bond and physical dimension: {self.h_mpo.bond_dims}, {self.h_mpo.pbond_list}")
        init_mps = Mps.ground_state(self.model, False)
        init_mps.compress_config = self.compress_config
        init_mps.evolve_config = self.evolve_config
        
        if self.evolve_config.is_tdvp and self.auto_expand: 
            init_mps = init_mps.expand_bond_dimension(self.h_mpo, coef=1e-16, include_ex=False)
        return init_mps

    def process_mps(self, mps):
        for idx, bas in enumerate(self.model.basis):
            if bas.is_spin:
                break
        rho = mps.calc_1site_rdm(idx=idx)[idx]
        self.rho.append(rho)
        
        #sigma_z_mpo = self.model.get_mpos("sigma_z", partial(Mpo.onsite, opera="sigma_z", dof_set={"spin"}))
        #sigma_z0 = mps.expectation(sigma_z_mpo)
        #assert np.allclose(sigma_z0, rho[0,0]-rho[1,1])
        self.sigma_z.append((rho[0,0]-rho[1,1]).real)
        
        #sigma_x_mpo = self.model.get_mpos("sigma_x", partial(Mpo.onsite, opera="sigma_x", dof_set={"spin"}))
        #sigma_x0 = mps.expectation(sigma_x_mpo)
        #assert np.allclose(sigma_x0, rho[0,1]+rho[1,0])
        self.sigma_x.append((rho[0,1]+rho[1,0]).real)
        logger.info(f"sigma_z: {self.sigma_z[-1]}. sigma_x: {self.sigma_x[-1]}")
        
        bond_entropy = mps.calc_entropy("bond")
        self.bond_entropy.append(bond_entropy)

    def evolve_single_step(self, evolve_dt):
        return self.latest_mps.evolve(self.h_mpo, evolve_dt)

    def get_dump_dict(self):
        dump_dict = dict()
        dump_dict["time series"] = self.evolve_times
        dump_dict["sigma_x"] = self.sigma_x
        dump_dict["sigma_z"] = self.sigma_z
        dump_dict["rho"] = self.rho
        dump_dict["bond_entropy"] = self.bond_entropy
        return dump_dict
