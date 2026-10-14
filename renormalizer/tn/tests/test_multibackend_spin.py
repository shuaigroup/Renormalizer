"""Independent binary-tree physical gates; MPS results are not references."""
import numpy as np
import pytest

from renormalizer.tn import BasisTree, TTNO, TTNS
from renormalizer.tn.gs import optimize_ttns
from renormalizer.mps.mps import expand_bond_dimension_general
from renormalizer.utils import EvolveConfig, EvolveMethod, CompressConfig, CompressCriteria
from renormalizer.backend.tests.spin_reference import (
    spin_fixture, host_seed, selected_context, save_reference, assert_witness,
    check_ground_state, check_evolution, BONDS, TIME_STEPS, FINAL_TIME,
)


@pytest.mark.parametrize('n,field', [(4, 0.137), (3, 0.)])
def test_optimization_dense_ground_space(n, field, tmp_path, record_property):
    model, dense = spin_fixture(n, field)
    tree = BasisTree.binary(model.basis)
    with host_seed():
        state = TTNS.random(tree, qntot=0, m_max=4)
        operator = TTNO(tree, model.ham_terms)
    # Default Davidson does not expose its residual tolerance in this legacy
    # path. ARPACK's machine-precision default exercises host callbacks while
    # matching the independently frozen 1e-8 physical residual requirement.
    state.optimize_config.algo = 'arpack'
    record_property('host_eigensolver', 'arpack')
    save_reference(tmp_path, dense, state.todense(model.basis).ravel(), record_property)
    ctx = selected_context()
    from renormalizer.backend.testing import record_execution
    with record_execution(ctx) as ledger:
        optimize_ttns(state, operator, [[4, 0]]*4, backend_context=ctx)
    assert_witness(ledger, ctx, record_property)
    check_ground_state(dense, state.todense(model.basis).ravel(), record_property,
                       degenerate=(n == 3))


@pytest.mark.parametrize('bond', BONDS)
@pytest.mark.parametrize('dt', TIME_STEPS)
def test_tdvp_vmf_dense_reference(bond, dt, tmp_path, record_property):
    model, dense = spin_fixture()
    tree = BasisTree.binary(model.basis)
    with host_seed():
        operator = TTNO(tree, model.ham_terms)
        state = TTNS(tree, {1: 1})
        state.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=bond)
        state = expand_bond_dimension_general(state, hint_mpo=operator)
    assert max(state.bond_dims) <= bond
    state.evolve_config = EvolveConfig(EvolveMethod.tdvp_vmf, ivp_rtol=1e-10,
                                      ivp_atol=1e-12, force_ovlp=False)
    psi0 = state.todense(model.basis).ravel().astype(np.complex128)
    save_reference(tmp_path, dense, psi0, record_property)
    ctx = selected_context()
    from renormalizer.backend.testing import record_execution
    with record_execution(ctx) as ledger:
        for _ in range(round(FINAL_TIME/dt)):
            state = state.evolve(operator, dt, normalize=False, backend_context=ctx)
            assert max(state.bond_dims) <= bond
    assert_witness(ledger, ctx, record_property)
    check_evolution(dense, psi0, state.todense(model.basis).ravel(), bond, record_property)
