"""MPS physical gates with independent dense references and device witnesses."""
import numpy as np
import pytest

from renormalizer import Mps, Mpo
from renormalizer.mps.gs import optimize_mps
from renormalizer.utils import EvolveConfig, EvolveMethod, CompressConfig, CompressCriteria
from renormalizer.backend.tests.spin_reference import (
    spin_fixture, host_seed, selected_context, save_reference, assert_witness,
    check_ground_state, check_evolution, BONDS, TIME_STEPS, FINAL_TIME,
)


@pytest.mark.parametrize('n,field', [(4, 0.137), (3, 0.)])
def test_dmrg_dense_ground_space(n, field, tmp_path, record_property):
    model, dense = spin_fixture(n, field)
    with host_seed():
        state = Mps.random(model, qntot=0, m_max=4)
        operator = Mpo(model)
    state.optimize_config.method = '2site'
    state.optimize_config.procedure = [[4, 0]] * 4
    save_reference(tmp_path, dense, state.todense().reshape(-1), record_property)
    ctx = selected_context()
    from renormalizer.backend.testing import record_execution
    with record_execution(ctx) as ledger:
        _, result = optimize_mps(state, operator, backend_context=ctx)
    assert_witness(ledger, ctx, record_property)
    check_ground_state(dense, result.todense().reshape(-1), record_property,
                       degenerate=(n == 3))


@pytest.mark.parametrize('bond', BONDS)
@pytest.mark.parametrize('dt', TIME_STEPS)
def test_tdvp_ps_dense_reference(bond, dt, tmp_path, record_property):
    model, dense = spin_fixture()
    with host_seed():
        operator = Mpo(model)
        state = Mps.hartree_product_state(model, {1: 1})
        state.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=bond)
        state = state.expand_bond_dimension(hint_mpo=operator)
    assert max(state.bond_dims) <= bond
    state.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    psi0 = state.todense().reshape(-1).astype(np.complex128)
    save_reference(tmp_path, dense, psi0, record_property)
    ctx = selected_context()
    from renormalizer.backend.testing import record_execution
    with record_execution(ctx) as ledger:
        for _ in range(round(FINAL_TIME/dt)):
            state = state.evolve(operator, dt, normalize=False, backend_context=ctx)
            assert max(state.bond_dims) <= bond
    assert_witness(ledger, ctx, record_property)
    check_evolution(dense, psi0, state.todense().reshape(-1), bond, record_property)
