from renormalizer.utils import basis as Ba
import numpy as np
import pytest


@pytest.mark.parametrize("op", ("x","x^2","p","p^2"))
@pytest.mark.parametrize("x0", (0,10))
def test_BasisSHO(op, x0):

    sho = Ba.BasisSHO(0.1, 10, x0=x0, dvr=False)
    sho_general = Ba.BasisSHO(0.1, 10, x0=x0, general_xp_power=True, dvr=False)

    a = sho.op_mat(op)
    b = sho_general.op_mat(op)
    assert np.allclose(a, b)

    sho_dvr = Ba.BasisSHO(0.1, 10, x0=x0, dvr=True)
    sho_dvr_general = Ba.BasisSHO(0.1, 10, x0=x0, general_xp_power=True, dvr=True)
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
