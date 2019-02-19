# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>
from __future__ import division

import os
import sys
import logging

import yaml

from ephMPS.model import Phonon, Mol, MolList
from ephMPS.mps import solver
from ephMPS.transport import ChargeTransport
from ephMPS.utils import log, Quantity, EvolveConfig, EvolveMethod

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('No or more than one parameter file are provided, abort')
        exit()
    parameter_path = sys.argv[1]
    with open(parameter_path) as fin:
        param = yaml.safe_load(fin)
    log.register_file_output(os.path.join(param['output dir'], param['fname'] + '.log'), 'w')
    ph_list = [Phonon.simple_phonon(Quantity(*omega), Quantity(*displacement), param['ph phys dim'])
               for omega, displacement in param['ph modes']]
    j_constant = Quantity(param['j constant'], param['j constant unit'])
    mol_list = MolList([Mol(Quantity(param['elocalex'], param['elocalex unit']), ph_list)] * param['mol num'], j_constant)
    evolve_config = EvolveConfig()
    evolve_config.scheme = EvolveMethod.tdvp_mctdh_new
    ct = ChargeTransport(mol_list, temperature=Quantity(*param['temperature']), evolve_config=evolve_config)
    #ct.stop_at_edge = True
    ct.economic_mode = True
    ct.memory_limit = 2 ** 30  # 1 GB
    #ct.memory_limit /= 10 # 100 MB
    ct.dump_dir = param['output dir']
    ct.job_name = param['fname']
    ct.custom_dump_info['comment'] = param['comment']
    ct.set_threshold(1e-4)
    # ct.latest_mps.compress_add = True
    evolve_dt = param['evolve dt']
    lowest_energy = ct.mpo_e_lbound
    highest_energy = ct.mpo_e_ubound
    logger.debug('Energy of the Hamiltonian: {:g} ~ {:g}'.format(lowest_energy, highest_energy))
    if evolve_dt == 'auto':
        factor = min(highest_energy * 0.1 + (ct.initial_energy - lowest_energy) / (highest_energy - lowest_energy) * highest_energy * 1.8, highest_energy)
        evolve_dt = 1 / factor
        logger.info('Auto evolve delta t: {:g}'.format(evolve_dt))
        #evolve_dt = 1 / abs(highest_energy)
    # disable calculation of reduced density matrices
    ct.reduced_density_matrices = None
    ct.evolve(evolve_dt, param.get('nsteps'), param.get('evolve time'))
