# -*- coding: utf-8 -*-

import logging

from renormalizer.sbm import SpinBosonDynamics, param2mollist
from renormalizer.utils import Quantity, CompressConfig, EvolveConfig, log

log.init_log(logging.INFO)

if __name__ == "__main__":
    alpha = 0.05
    raw_delta = Quantity(1)
    raw_omega_c = Quantity(20)
    n_phonons = 300
    renormalization_p = 1
    mol_list = param2mollist(alpha, raw_delta, raw_omega_c, renormalization_p, n_phonons)

    compress_config = CompressConfig(threshold=1e-4)
    evolve_config = EvolveConfig(adaptive=True, guess_dt=0.1)
    sbm = SpinBosonDynamics(mol_list, Quantity(0), compress_config=compress_config, evolve_config=evolve_config, dump_dir="./", job_name="sbm")
    sbm.evolve(evolve_dt=0.1, evolve_time=20)