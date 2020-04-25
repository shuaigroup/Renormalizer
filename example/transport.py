# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import os
import sys
import logging

import yaml

from renormalizer.model import load_from_dict
from renormalizer.transport import ChargeTransport
from renormalizer.utils import log, Quantity, EvolveConfig, EvolveMethod, RungeKutta, CompressConfig

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
    mol_list, temperature = load_from_dict(param, 3, False)
    compress_config = CompressConfig(max_bonddim=16)
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, adaptive=True, guess_dt=2)
    ct = ChargeTransport(
        mol_list,
        temperature=temperature,
        compress_config=compress_config,
        evolve_config=evolve_config,
        rdm=False,
    )
    ct.dump_dir = param["output dir"]
    ct.job_name = param["fname"]
    ct.custom_dump_info["comment"] = param["comment"]
    ct.evolve(param.get("evolve dt"), param.get("nsteps"), param.get("evolve time"))
    # ct.evolve(evolve_dt, 100, param.get("evolve time"))
