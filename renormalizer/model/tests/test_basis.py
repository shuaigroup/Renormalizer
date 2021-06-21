from renormalizer.model import Model, Op, basis as Ba
from renormalizer.mps import Mpo, Mps, gs

import numpy as np
import scipy
import scipy.integrate
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


# SineDVR analytical v.s numerical results
@pytest.mark.parametrize("op", ([1,0], [2,0], [0,1],
    [0,2],[1,1],[2,1],[1,2],[2,2],[3,2]))
def test_SineDVR(op):
    moment, deri = op
    if moment == 0:
        str1 = ""
    elif moment == 1:
        str1 = "x"
    else:
        str1 = f"x^{moment}"

    if deri == 0:
        str2 = ""
    elif deri == 1:
        str2 = "partialx"
    else:
        str2 = f"partialx^{deri}"
    
    nbas = 4
    basis = Ba.BasisSineDVR("R1", nbas, 1, 7, endpoint=False)
    x0, x1 = basis.xi, basis.xf
    
    op = " ".join([str1, str2]).strip()
    mat = basis.op_mat(op)
    mat = basis.dvr_v @ mat @ basis.dvr_v.T
    def psi(x, j):
        return np.sin(j*np.pi*(x-x0)/(x1-x0)) * np.sqrt(2 / (x1-x0))
    
    def f(x,j,k):   
        if deri == 0:
            return psi(x,j) * x**moment * psi(x, k)
        else:
            return psi(x,j) * x**moment * scipy.misc.derivative(psi, x, dx=1e-3,
                    n=deri, args=(k,))  # accuracy is very sensitive to dx

    std = np.zeros((nbas,nbas))
    for j in range(1,nbas+1):
        for k in range(1,nbas+1):
            res = scipy.integrate.quad(f,x0,x1,args=(j,k))
            std[j-1,k-1] = res[0]
    
    assert np.allclose(std, mat)


