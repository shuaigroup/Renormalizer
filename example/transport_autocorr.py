# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>
from __future__ import division

import os
import sys
import logging

import yaml

from ephMPS.model import load_from_dict
from ephMPS.transport import TransportAutoCorr
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
    mol_list, temperature = load_from_dict(param, 3, False)
    compress_config = CompressConfig(threshold=1e-4)
    ievolve_config = EvolveConfig(adaptive=True, evolve_dt=temperature.as_au()/1000j)
    evolve_config = EvolveConfig(adaptive=True, evolve_dt=2)
    ct = TransportAutoCorr(mol_list, temperature=temperature, evolve_config=evolve_config, dump_dir=param["output dir"], job_name=param["fname"] + "_autocorr")
    ct.economic_mode = True
    # ct.latest_mps.compress_add = True
    ct.evolve(param.get("evolve dt"), param.get("nsteps"), param.get("evolve time"))
    # ct.evolve(evolve_dt, 100, param.get("evolve time"))