"""Small scientific oracles under an explicitly selected backend capture.

Preparation includes selected-backend random initialization. Host dense references
are deliberate oracles; native execution requires the separate scoped witnesses.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from renormalizer.backend import execution
from renormalizer.backend.testing import record_execution
from renormalizer import BasisHalfSpin, Model, Mpo, Mps, Op
from renormalizer.model.model import heisenberg_ops
from renormalizer.mps.matrix import asnumpy, asxp
from renormalizer.mps.gs import get_ham_iterative
from renormalizer.mps.matrix import Matrix
from renormalizer.mps.mps import expand_bond_dimension_general
from renormalizer.mps.tda import TDA
from renormalizer.tn import BasisTree, TTNO, TTNS
from renormalizer.tn.node import TreeNodeTensor
from renormalizer.tn.tree import EVOLVE_METHODS
from renormalizer.utils import CompressConfig, CompressCriteria, EvolveConfig, EvolveMethod


def spin_model():
    return Model([BasisHalfSpin(i) for i in range(4)],
                 heisenberg_ops(4) + [Op('sigma_z', 0, .137)])


def dense_hamiltonian():
    pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1., -1.])]
    def kron(parts):
        result = np.ones((1, 1))
        for part in parts:
            result = np.kron(result, part)
        return result
    h = .137 * kron([pauli[2]] + [np.eye(2)] * 3)
    h = h.astype(complex)
    for i in range(3):
        for p in pauli:
            parts = [np.eye(2)] * 4
            parts[i] = parts[i+1] = p
            h += .25 * kron(parts)
    return h


def test_random_mps_initialization_selected_backend(captured_backend):
    state = Mps.random(spin_model(), qntot=0, m_max=4)
    psi = asnumpy(state.todense()).ravel()
    assert np.isfinite(psi).all()
    np.testing.assert_allclose(np.vdot(psi, psi), 1., atol=1e-12)


@pytest.mark.parametrize('family', ['mps', 'ttns'])
def test_real_scalar_expectation(captured_backend, family):
    model = spin_model()
    if family == 'mps':
        state = Mps.hartree_product_state(model, {1: 1, 3: 1})
        actual = state.expectation(Mpo(model))
    else:
        tree = BasisTree.binary(model.basis)
        state = TTNS(tree, {1: 1, 3: 1})
        actual = state.expectation1(TTNO(tree, model.ham_terms))
    psi = np.zeros(16); psi[5] = 1
    np.testing.assert_allclose(actual, np.vdot(psi, dense_hamiltonian() @ psi), atol=1e-12)


def test_complex_matrix_canonical_check(captured_backend):
    matrix = Matrix(np.array([[[1j, 0.]]]), dtype=np.complex128)
    assert matrix.check_rortho()


@pytest.mark.parametrize('kind', ['matrix', 'tree'])
def test_persistent_host_storage_is_writable(captured_backend, kind):
    source = np.ones((1, 2, 1)); source.flags.writeable = False
    obj = Matrix(source) if kind == 'matrix' else TreeNodeTensor(source)
    stored = obj.array if kind == 'matrix' else obj.tensor
    stored[...] *= 2
    np.testing.assert_array_equal(source, np.ones_like(source))
    np.testing.assert_array_equal(stored, 2 * np.ones_like(source))


MPS_METHODS = ['prop_and_compress', 'prop_and_compress_tdrk4', 'prop_and_compress_tdrk',
               'tdvp_vmf', 'tdvp_mu_vmf', 'tdvp_mu_cmf', 'tdvp_ps', 'tdvp_ps2']
METHOD_CASES = [('mps', name) for name in MPS_METHODS] + [('ttns', method.name) for method in EVOLVE_METHODS]


@pytest.mark.parametrize('family,method', METHOD_CASES)
def test_registered_evolution_dense_reference(captured_backend, family, method, monkeypatch, record_property):
    model = spin_model()
    if family == 'mps':
        state = Mps.hartree_product_state(model, {1: 1, 3: 1})
        operator = Mpo(model)
    else:
        tree = BasisTree.binary(model.basis)
        state = TTNS(tree, {1: 1, 3: 1})
        operator = TTNO(tree, model.ham_terms)
    state.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=4)
    state = expand_bond_dimension_general(state, hint_mpo=operator)
    state.evolve_config = EvolveConfig(EvolveMethod[method], ivp_rtol=1e-8, ivp_atol=1e-10, force_ovlp=False)
    def dense(value):
        return asnumpy(value.todense(model.basis) if family == 'ttns' else value.todense()).ravel()
    initial = dense(state).astype(complex)
    h = dense_hamiltonian()
    context = replace(captured_backend, host_policy='explicit')
    original_witness = execution._witness
    observed = []
    def checked_witness(result):
        # Inspect actual candidate results, not an unrelated allocation probe.
        if context.adapter.name != 'numpy':
            assert context.adapter.owns(result)
        observed.append((execution._device(result), str(result.dtype)))
        original_witness(result)
    with monkeypatch.context() as scoped:
        scoped.setattr(execution, '_witness', checked_witness)
        with record_execution(context) as ledger:
            evolved = state.evolve(operator, .001, normalize=False, backend_context=context)
            context.adapter.sync()
    for event in ledger.operations:
        assert event['adapter_id'] == id(context.adapter)
        assert event['backend'] == context.adapter.name
        assert event['device'] == str(context.device)
        assert event['dtype'].removeprefix('torch.') in ('float64', 'complex128')
    record_property('native_contraction_count', len(observed))
    record_property('execution_class', 'native-witnessed-hybrid' if observed else 'no-native-contraction-witness')
    if not method.startswith('prop_and_compress'):
        assert observed, 'TDVP should reach instrumented contractions'
    actual = dense(evolved)
    expected = expm(-1j * .001 * h) @ initial
    overlap = np.vdot(expected, actual)
    phase = overlap / abs(overlap) if abs(overlap) else 1
    assert np.linalg.norm(actual - expected * phase) < 2e-5
    assert abs(np.vdot(actual, actual) - 1) < 1e-7
    assert abs(np.vdot(actual, h @ actual) - np.vdot(initial, h @ initial)) < 1e-7


def test_two_site_rdm_dense_partial_trace(captured_backend):
    model = spin_model()
    psi = np.arange(1, 17) + 1j * np.arange(16, 0, -1)
    psi = psi / np.linalg.norm(psi)
    state = Mps.from_dense(model, psi)
    actual = state.calc_2site_rdm()
    for pair, rdm in actual.items():
        other = [i for i in range(4) if i not in pair]
        wave = psi.reshape((2,) * 4).transpose(list(pair) + other).reshape(4, 4)
        expected = wave @ wave.conj().T
        np.testing.assert_allclose(asnumpy(rdm).reshape(4, 4), expected, atol=1e-12)


@pytest.mark.parametrize('raw_left,raw_right', [(True, True), (True, False), (False, True)])
def test_dense_iterative_hamiltonian_raw_environment(captured_backend, raw_left, raw_right):
    state = SimpleNamespace(optimize_config=SimpleNamespace(method='1site', inverse=1))
    left = np.ones((1, 1, 1)); right = np.ones((1, 1, 1))
    center = np.array([[2., .3], [.3, -1.]])
    left = left if raw_left else asxp(left)
    right = right if raw_right else asxp(right)
    diag, hop = get_ham_iterative(state, np.ones((1, 2, 1), dtype=bool),
                                  left, right, [asxp(center.reshape(1, 2, 2, 1))], None)
    np.testing.assert_allclose(asnumpy(diag).ravel(), np.diag(center), atol=1e-12)
    vector = np.array([.4, -.7])
    np.testing.assert_allclose(asnumpy(hop(asxp(vector.reshape(1, 2, 1)))).ravel(), center @ vector, atol=1e-12)


def test_tda_terminal_scalar_and_excitation_energy(captured_backend):
    model = Model([BasisHalfSpin(i) for i in range(2)],
                  [Op('sigma_z', 0, -.5), Op('sigma_z', 1, -1.)])
    state = Mps.hartree_product_state(model, {})
    tda = TDA(model, Mpo(model), state, nroots=2, algo='davidson')
    energies = tda.kernel()
    np.testing.assert_allclose(energies, [-.5, .5], atol=1e-10)
    configs, _ = tda.analysis_dominant_config()
    for root, expected in enumerate(([1, 0], [0, 1])):
        np.testing.assert_array_equal(configs[root][0][0], expected)
        np.testing.assert_allclose(abs(configs[root][0][-1]), 1., atol=1e-10)


@pytest.mark.parametrize('family', ['mps', 'ttns'])
@pytest.mark.parametrize('sites', [1, 2])
def test_complex_rdm_pauli_y_expectation_consistency(captured_backend, family, sites):
    model = Model([BasisHalfSpin(i) for i in range(2)], [Op('isigma_y', 0, -1j)])
    local = np.array([1, 1j]) / np.sqrt(2)
    psi = np.kron(local, [1., 0.])
    y = np.array([[0, -1j], [1j, 0]])
    if family == 'mps':
        state = Mps.from_dense(model, psi)
        expectation = state.expectation(Mpo(model))
        rho = state.calc_1site_rdm()[0] if sites == 1 else state.calc_2site_rdm()[(0, 1)]
    else:
        tree = BasisTree.binary(model.basis)
        state = TTNS(tree, {})
        for node in state.node_list:
            if any(0 in dofs for dofs in state.tn2dofs[node]):
                node.tensor = local.reshape(node.shape)
        np.testing.assert_allclose(state.todense(model.basis).ravel(), psi)
        # TTNO supports real iY; reconstruct the Hermitian Y expectation.
        expectation = -1j * state.expectation(TTNO(tree, [Op('isigma_y', 0)]))
        rho = state.calc_1dof_rdm(0)[0] if sites == 1 else state.calc_2dof_rdm((0, 1))[(0, 1)]
    observable = y if sites == 1 else np.kron(y, np.eye(2))
    np.testing.assert_allclose(expectation, 1., atol=1e-12)
    np.testing.assert_allclose(np.trace(asnumpy(rho).reshape(observable.shape) @ observable), expectation, atol=1e-12)


def test_one_site_rdm_complex_entangled_partial_trace(captured_backend):
    model = spin_model()
    psi = np.arange(1, 17) + 1j * np.arange(16, 0, -1)
    psi /= np.linalg.norm(psi)
    state = Mps.from_dense(model, psi)
    for site, rho in state.calc_1site_rdm().items():
        wave = np.moveaxis(psi.reshape((2,) * 4), site, 0).reshape(2, -1)
        np.testing.assert_allclose(asnumpy(rho), wave @ wave.conj().T, atol=1e-12)


def test_random_mps_masks_forbidden_quantum_sector(captured_backend):
    model = Model([BasisHalfSpin(i, sigmaqn=[0, 1]) for i in range(4)], [])
    state = Mps.random(model, qntot=2, m_max=4)
    psi = asnumpy(state.todense()).ravel()
    forbidden = np.array([i.bit_count() != 2 for i in range(16)])
    np.testing.assert_array_equal(psi[forbidden], 0)
    np.testing.assert_allclose(np.vdot(psi, psi), 1., atol=1e-12)
