# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import os

import numpy as np
import pytest

from ephMPS.tests.parameter import mol_list
from ephMPS.mps.solver import construct_mps_mpo_2, optimize_mps
from ephMPS.utils import pickle
from ephMPS.mps.tests import cur_dir

nexciton=1
procedure = [[20,0.4],[20,0.2],[20,0.1],[20,0],[20,0]]

def test_csvd():
    np.random.seed(0)
    mps1, mpo = construct_mps_mpo_2(mol_list, procedure[0][0], nexciton, scheme=2)
    mps1.threshold = 1e-6
    mps1.optimize_config.procedure = procedure
    optimize_mps(mps1, mpo)
    mps1.compress()
    with open(os.path.join(cur_dir, 'test_svd_qn.pickle'), 'rb') as fin:
        mps2 = pickle.load(fin)
    assert mps1.distance(mps2) == pytest.approx(0)
