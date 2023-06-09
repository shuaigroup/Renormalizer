# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import os

import pytest

from renormalizer.tests import parameter_multielec
from renormalizer.photophysics.tests import cur_dir
from renormalizer.photophysics.base import *
from renormalizer.utils import EvolveConfig, EvolveMethod, Quantity
import numpy as np


model = parameter_multielec.model

def test_ZTemi():
    
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    emi = ZTemi(model, 1, 0, evolve_config=evolve_config)

    nsteps = 30
    dt = 30.0
    emi.evolve(dt, nsteps)
    with open(os.path.join(cur_dir, "ZTemi.npy"), "rb") as fin:
        std = np.load(fin)
    assert np.allclose(emi.autocorr[:nsteps], std[:nsteps], rtol=1e-4)

def test_FTemi():
    
    insteps = 50
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    ievolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    emi = FTemi(model, 1, Quantity(298,"K"), insteps,
            ievolve_config=ievolve_config, evolve_config=evolve_config)
    
    nsteps = 30
    dt = 30.0
    emi.evolve(dt, nsteps)
    with open(os.path.join(cur_dir, "FTemi.npy"), "rb") as fin:
        std = np.load(fin)
    print(emi.autocorr[:nsteps], std[:nsteps])
    assert np.allclose(emi.autocorr[:nsteps], std[:nsteps], rtol=1e-3)

def test_ZTabs():
    
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    abso = ZTabs(model, 0, 0, evolve_config=evolve_config)
    nsteps = 30
    dt = 30.0
    abso.evolve(dt, nsteps)
    with open(
        os.path.join(
            cur_dir, "ZTabs.npy"
        ),
        "rb",
    ) as f:
        std = np.load(f)
    assert np.allclose(abso.autocorr[:nsteps], std[:nsteps], rtol=1e-4)

def test_FTabs():
    
    insteps = 50
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    ievolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    abso = FTabs(model, 0, Quantity(298,"K"), insteps,
            ievolve_config=ievolve_config, evolve_config=evolve_config)
    
    nsteps = 30
    dt = 30.0
    abso.evolve(dt, nsteps)
    with open(os.path.join(cur_dir, "FTabs.npy"), "rb") as fin:
        std = np.load(fin)
    print(abso.autocorr[:nsteps], std[:nsteps])
    assert np.allclose(abso.autocorr[:nsteps], std[:nsteps], rtol=1e-3)
