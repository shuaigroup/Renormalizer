# -*- coding: utf-8 -*-

import os

import numpy as np
import pytest

from renormalizer.mps import Mps, Mpo, MpDm
from renormalizer.mps.matrix import tensordot, asnumpy
from renormalizer.mps.lib import Environ
from renormalizer.tests.parameter import custom_model, holstein_model
from renormalizer.utils import CompressCriteria

def test_save_load():
    model = holstein_model
    mps = Mpo.onsite(model, r"a^\dagger", dof_set={0}) @ Mps.ground_state(model, False)
    mpo = Mpo(model)
    mps1 = mps.copy()
    for i in range(2):
        mps1 = mps1.evolve(mpo, 10)
    mps2 = mps.evolve(mpo, 10)
    fname = "test.npz"
    mps2.dump(fname)
    mps2 = Mps.load(model, fname)
    mps2 = mps2.evolve(mpo, 10)
    assert np.allclose(mps1.e_occupations, mps2.e_occupations)
    os.remove(fname)


def check_distance(a: Mps, b: Mps):
    d1 = (a - b).mp_norm
    d2 = a.distance(b)
    a_array = a.todense()
    b_array = b.todense()
    d3 = np.linalg.norm(a_array - b_array)
    assert d1 == pytest.approx(d2) == pytest.approx(d3)


def test_distance():
    model = custom_model(n_phys_dim=(2, 2))
    a = Mps.random(model, 1, 10)
    b = Mps.random(model, 1, 10)
    check_distance(a, b)
    h = Mpo(model)
    for i in range(100):
        a = a.evolve(h, 10)
        b = b.evolve(h, 10)
        check_distance(a, b)


def test_environ():
    mps = Mps.random(holstein_model, 1, 10)
    mpo = Mpo(holstein_model)
    mps = mps.evolve(mpo, 10)
    environ = Environ(mps, mpo)
    for i in range(len(mps)-1):
        l = environ.read("L", i)
        r = environ.read("R", i+1)
        e = complex(tensordot(l, r, axes=((0, 1, 2), (0, 1, 2)))).real
        assert pytest.approx(e) == mps.expectation(mpo)

# multi_mpo routine for single mpo calculation
@pytest.mark.parametrize("mpdm", (True, False))
def test_environ_multi_mpo(mpdm):
    mps = Mps.random(holstein_model, 1, 10)
    if mpdm:
        mps = MpDm.from_mps(mps)
    mpo = Mpo(holstein_model)
    mps = mps.evolve(mpo, 10)
    environ = Environ(mps, mpo)
    environ_multi_mpo = Environ(mps, [mpo])
    for i in range(len(mps)-1):
        l = environ.read("L", i)
        r = environ.read("R", i+1)
        l2 = environ_multi_mpo.read("L", i)
        r2 = environ_multi_mpo.read("R", i+1) 
        assert np.allclose(asnumpy(l), asnumpy(l2))
        assert np.allclose(asnumpy(r), asnumpy(r2))

@pytest.mark.parametrize("comp", (True, False))
@pytest.mark.parametrize("mp", (
        "mps",
        "mpdm",
        "mpo",
))
def test_svd_compress(comp, mp):
    
    if mp == "mpo":
        mps = Mpo(holstein_model)
        M = 22
    else:
        mps = Mps.random(holstein_model, 1, 10)
        if mp == "mpdm":
            mps = MpDm.from_mps(mps)
        mps.canonicalise().normalize("mps_only")
        M = 36
    if comp:
        mps = mps.to_complex(inplace=True)
    print(f"{mps}")
    
    mpo = Mpo(holstein_model)
    if comp:
        mpo = mpo.scale(-1.0j)
    print(f"{mpo.bond_dims}")
    
    std_mps = mpo.apply(mps, canonicalise=True).canonicalise()
    print(f"std_mps: {std_mps}")
    mps.compress_config.bond_dim_max_value = M
    mps.compress_config.criteria = CompressCriteria.fixed
    svd_mps = mpo.contract(mps)
    dis = svd_mps.distance(std_mps)/std_mps.mp_norm
    print(f"svd_mps: {svd_mps}, dis: {dis}")
    assert np.allclose(dis, 0.0, atol=1e-3)
    assert np.allclose(svd_mps.mp_norm, std_mps.mp_norm, atol=1e-4)
    
    
@pytest.mark.parametrize("comp", (True, False))
@pytest.mark.parametrize("mp", ("mps", "mpdm", "mpo" ))
def test_variational_compress(comp, mp):
    
    if mp == "mpo":
        mps = Mpo(holstein_model)
        M = 20
    else:
        mps = Mps.random(holstein_model, 1, 10)
        if mp == "mpdm":
            mps = MpDm.from_mps(mps)
        mps.canonicalise().normalize("mps_only")
        M = 36
    if comp:
        mps = mps.to_complex(inplace=True)
    print(f"{mps}")
    
    mpo = Mpo(holstein_model)
    if comp:
        mpo = mpo.scale(-1.0j)
    print(f"{mpo.bond_dims}")
    
    std_mps = mpo.apply(mps, canonicalise=True).canonicalise()
    print(f"std_mps: {std_mps}")
    
    # 2site algorithm
    mps.compress_config.vprocedure = [[M,1.0],[M,0.2],[M,0.1]]+[[M,0],]*10
    mps.compress_config.vmethod = "2site"
    var_mps = mps.variational_compress(mpo, guess=None)
    dis = var_mps.distance(std_mps)/std_mps.mp_norm
    print(f"var2_mps: {var_mps}, dis: {dis}")
    assert np.allclose(dis, 0.0, atol=1e-4)
    assert np.allclose(var_mps.mp_norm, std_mps.mp_norm, atol=1e-4)
    
    # 1site algorithm with 2site result as a guess
    # 1site algorithm is easy to be trapped in a local minimum
    var_mps.compress_config.vprocedure = [[M,0],]*10
    var_mps.compress_config.vmethod = "1site"
    var_mps = mps.variational_compress(mpo, guess=var_mps)
    dis = var_mps.distance(std_mps)/std_mps.mp_norm
    print(f"var1_mps: {var_mps}, dis: {dis}")
    assert np.allclose(dis, 0.0, atol=1e-4)
    assert np.allclose(var_mps.mp_norm, std_mps.mp_norm, atol=1e-4)
