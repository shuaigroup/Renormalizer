from renormalizer.model import Model, Op, basis as Ba
from renormalizer.mps import Mpo, Mps, gs

import numpy as np
import scipy
import pytest

@pytest.mark.parametrize("op", ("x","x^2","p","p^2"))
@pytest.mark.parametrize("x0", (0,10))
def test_BasisSHO(op, x0):

    sho = Ba.BasisSHO(None, 0.1, 10, x0=x0, dvr=False)
    sho_general = Ba.BasisSHO(None, 0.1, 10, x0=x0, general_xp_power=True, dvr=False)

    a = sho.op_mat(op)
    b = sho_general.op_mat(op)
    assert np.allclose(a, b)

    sho_dvr = Ba.BasisSHO(None, 0.1, 10, x0=x0, dvr=True)
    sho_dvr_general = Ba.BasisSHO(None, 0.1, 10, x0=x0, general_xp_power=True, dvr=True)
    a_dvr = sho_dvr.op_mat(op)
    b_dvr = sho_dvr_general.op_mat(op)
    a_dvr = sho_dvr.dvr_v  @ a_dvr @ sho_dvr.dvr_v.T
    b_dvr = sho_dvr_general.dvr_v  @ b_dvr @ sho_dvr_general.dvr_v.T
    if op == "x^2":
        # the last basis is not accurate in dvr
        assert np.allclose(a[:-1,:-1], a_dvr[:-1,:-1]) 
        assert np.allclose(a[:-1,:-1], b_dvr[:-1,:-1])
    else:
        assert np.allclose(a, a_dvr) 
        assert np.allclose(a, b_dvr)

def test_high_moment():
    sho = Ba.BasisSHO(None, 0.1, 10, dvr=False)
    assert np.allclose(sho.op_mat("x^2"), sho.op_mat("x x"))
    assert np.allclose(sho.op_mat("x^3"), sho.op_mat("x x x"))
    assert np.allclose(sho.op_mat("p^2"), sho.op_mat("p p"))
    assert np.allclose(sho.op_mat("p^3"), sho.op_mat("p p p"))


@pytest.mark.parametrize("basistype", ("SHO", "SHODVR", "SineDVR"))
def test_VibBasis(basistype):
    nv = 2
    pdim = 6
    hessian = np.array([[2,1],[1,3]])
    e, c = scipy.linalg.eigh(hessian)
    ham_terms = []
    basis = []
    for iv in range(nv):
        op = Op("p^2", f"v_{iv}", factor=0.5, qn=0)
        ham_terms.append(op)
        if basistype == "SineDVR":
            # sqrt(<x^2>) of the highest vibrational basis
            x_mean = np.sqrt((pdim+0.5)/np.sqrt(hessian[iv,iv]))
            bas = Ba.BasisSineDVR(f"v_{iv}", 2*pdim, -x_mean*1.5, x_mean*1.5,
                    endpoint=True)
            print("x_mean", x_mean, bas.dvr_x)
        else:
            if basistype == "SHO":
                dvr = False
            else:
                dvr = True
            bas = Ba.BasisSHO(f"v_{iv}", np.sqrt(hessian[iv,iv]), pdim, dvr=dvr)

        basis.append(bas)
    for iv in range(nv):
        for jv in range(nv):
            op = Op("x x", [f"v_{iv}", f"v_{jv}"], factor=0.5*hessian[iv, jv], qn=[0,0])
            ham_terms.append(op)
    model = Model(basis, ham_terms)
    mpo = Mpo(model)
    mps = Mps.random(model, 0, 10)
    mps.optimize_config.nroots = 2
    energy, mps = gs.optimize_mps(mps, mpo)
    w1, w2 = np.sqrt(e)
    std = [(w1+w2)*0.5, w1*1.5+w2*0.5]
    print(basistype, "calc:", energy[-1], "exact:", std)   
    assert np.allclose(energy[-1], std)



    

