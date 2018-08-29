# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>
import os
import sys

import yaml

from ephMPS.model import Phonon, Mol, MolList
from ephMPS.transport import ChargeTransport
from ephMPS.utils import log, constant, Quantity

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
    mol_list = MolList([Mol(Quantity(param['elocalex'], param['elocalex unit']), ph_list)] * param['mol num'])
    j_constant = Quantity(param['j constant'], param['j constant unit'])
    ct = ChargeTransport(mol_list, j_constant, temperature=param['temperature'])
    ct.dump_dir = param['output dir']
    ct.job_name = param['fname']
    ct.custom_dump_info['comment'] = param['comment']
    ct.evolve(param['evolve dt'], param['nsteps'])
