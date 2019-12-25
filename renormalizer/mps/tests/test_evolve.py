# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import qutip
import pytest
import numpy as np

from renormalizer.mps import Mps, Mpo, MpDm
from renormalizer.utils import EvolveMethod, EvolveConfig, CompressConfig, CompressCriteria, Quantity
from renormalizer.tests.parameter_exact import qutip_clist, qutip_h, mol_list


# the init state
tentative_mpo = Mpo(mol_list)
init_mps = (Mpo.onsite(mol_list, r"a^\dagger", mol_idx_set={0}) @ Mps.gs(mol_list, False)).expand_bond_dimension(hint_mpo=tentative_mpo)
init_mpdm = MpDm.from_mps(init_mps).expand_bond_dimension(hint_mpo=tentative_mpo)
e = init_mps.expectation(tentative_mpo)
mpo = Mpo(mol_list, offset=Quantity(e))

# calculate exact result in ZT. FT result is exactly the same
TIME_LIMIT = 10
QUTIP_STEP = 0.01
N_POINTS = TIME_LIMIT / QUTIP_STEP + 1
qutip_time_series = np.linspace(0, TIME_LIMIT, N_POINTS)
init = qutip.Qobj(init_mps.full_wfn(), [qutip_h.dims[0], [1] * len(qutip_h.dims[0])])
# the result is not exact and the error scale is approximately 1e-5
res = qutip.sesolve(qutip_h-e, init, qutip_time_series, e_ops=[c.dag() * c for c in qutip_clist])
qutip_expectations = np.array(res.expect).T


def check_result(mps, mpo, time_step, final_time, atol=1e-4):
    # the information to be compared with the standard
    expectations = [mps.e_occupations]
    # useful in debugging
    time_series = [0]
    for i in range(round(final_time / time_step)):
        mps = mps.evolve(mpo, time_step)
        expectations.append(mps.e_occupations)
        time_series.append(time_series[-1] + time_step)
    qutip_end = round(final_time / QUTIP_STEP) + 1
    qutip_interval = round(time_step / QUTIP_STEP)
    # used for debugging
    mcd = np.abs(expectations - qutip_expectations[:qutip_end:qutip_interval]).mean()
    assert mcd < atol

# useful debugging code
# from matplotlib import pyplot as plt
# plt.plot(time_series, expectations)
# plt.plot(qutip_time_series[:qutip_end], qutip_expectations[:qutip_end])
# plt.show()


@pytest.mark.parametrize("init_state", (init_mps, init_mpdm))
def test_pc(init_state):
    mps = init_state.copy()
    mps.compress_config  = CompressConfig(CompressCriteria.fixed)
    check_result(mps, mpo, 0.2, 5)


@pytest.mark.parametrize("init_state, atol", ([init_mps, 1e-4], [init_mpdm, 1e-3]))
@pytest.mark.parametrize("with_mu", (True, False))
@pytest.mark.parametrize("force_ovlp", (True, False))
def test_tdvp_vmf(init_state, with_mu, force_ovlp, atol):
    mps = init_state.copy()
    method = EvolveMethod.tdvp_mu_vmf if with_mu else EvolveMethod.tdvp_vmf
    mps.evolve_config  = EvolveConfig(method, force_ovlp=force_ovlp)
    check_result(mps, mpo, 0.5, 2, atol)


@pytest.mark.parametrize("init_state", (init_mps, init_mpdm))
def test_tdvp_cmf(init_state):
    mps = init_state.copy()
    mps.evolve_config  = EvolveConfig(EvolveMethod.tdvp_mu_cmf)
    check_result(mps, mpo, 0.01, 0.5, 5e-4)


@pytest.mark.parametrize("init_state", (
        init_mps,
        init_mpdm,))
def test_tdvp_ps(init_state):
    mps = init_state.copy()
    mps.evolve_config  = EvolveConfig(EvolveMethod.tdvp_ps)
    check_result(mps, mpo, 0.4, 5)


# used for debugging
def compare():
    dt_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    method_list = [EvolveMethod.prop_and_compress, EvolveMethod.tdvp_mu_vmf, EvolveMethod.tdvp_mu_cmf, EvolveMethod.tdvp_ps]
    all_values = []
    for method in method_list:
        values = []
        for dt in dt_list:
            mps = init_mps.copy()
            mps.compress_config = CompressConfig(CompressCriteria.fixed)
            mps.evolve_config  = EvolveConfig(method)
            mps.evolve_config._adjust_bond_dim_counter = True
            mps.evolve_config.force_ovlp = True
            values.append(check_result(mps, mpo, dt, 2))
        print(values)
        all_values.append(values)
    print(mps.bond_dims)
    print(all_values)

