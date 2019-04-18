# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>
from __future__ import division

import os
import sys
import logging

import yaml

from ephMPS.model import Phonon, Mol, MolList
from ephMPS.transport import ChargeTransport
from ephMPS.utils import log, Quantity, EvolveConfig, EvolveMethod, RungeKutta, CompressConfig, BondDimDistri

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("No or more than one parameter file are provided, abort")
        exit(1)
    parameter_path = sys.argv[1]
    with open(parameter_path) as fin:
        param = yaml.safe_load(fin)
    log.register_file_output(
        os.path.join(param["output dir"], param["fname"] + ".log"), "w"
    )
    ph_list = [
        Phonon.simple_phonon(
            Quantity(*omega), Quantity(*displacement), param["ph phys dim"]
        )
        for omega, displacement in param["ph modes"]
    ]
    j_constant = Quantity(param["j constant"], param["j constant unit"])
    mol_list = MolList(
        [Mol(Quantity(param["elocalex"], param["elocalex unit"]), ph_list)]
        * param["mol num"],
        j_constant,
    )
    compress_config = CompressConfig(bonddim_distri=BondDimDistri.center_gauss, max_bonddim=80)
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, adaptive=True)
    # evolve_config.expected_bond_dim = 80
    #rk_config = RungeKutta("RKF45")
    #rk_config.evolve_dt = 40
    #compress_config = CompressConfig()
    #evolve_config = EvolveConfig(rk_config=rk_config)
    #evolve_config = EvolveConfig()
    ct = ChargeTransport(
        mol_list,
        temperature=Quantity(*param["temperature"]),
        compress_config=compress_config,
        evolve_config=evolve_config,
        rdm=False
    )
    # ct.stop_at_edge = True
    ct.economic_mode = True
    # ct.memory_limit = 2 ** 30  # 1 GB
    # ct.memory_limit /= 10 # 100 MB
    ct.dump_dir = param["output dir"]
    ct.job_name = param["fname"]
    ct.custom_dump_info["comment"] = param["comment"]
    ct.set_threshold(1e-4)
    # ct.latest_mps.compress_add = True
    logger.debug(f"ground energy of the Hamiltonian: {ct.mpo_e_lbound:g}")
    ct.evolve(param["evolve dt"], param.get("nsteps"), param.get("evolve time"))
    # ct.evolve(evolve_dt, 100, param.get("evolve time"))
