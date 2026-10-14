"""Independent small-spin references and frozen physical acceptance budgets."""
from contextlib import contextmanager
from functools import reduce
import hashlib
import os
import random

import numpy as np

from renormalizer import BasisHalfSpin, Model, Op
from renormalizer.backend.context import make_context, capture_backend


FINAL_TIME = 0.04
TIME_STEPS = (0.02, 0.01, 0.005)
BONDS = (2, 4)
# Frozen before running any backend: full-rank f64 ground-state and short-time
# bounds. Low bond uses a distinct truncation budget, not relaxed backend error.
ENERGY_ATOL = 1e-10
RESIDUAL_ATOL = 1e-8
BACKEND_BUDGET = 1e-9
INTEGRATION_BUDGET = 2e-6
TRUNCATION_BUDGET = {2: 1e-4, 4: 0.0}
NORM_BUDGET = 1e-8
ENERGY_DRIFT_BUDGET = 1e-8


def spin_fixture(n=4, field=0.137):
    pauli = {'sigma_x': np.array([[0., 1.], [1., 0.]]),
             'sigma_y': np.array([[0., -1j], [1j, 0.]]),
             'sigma_z': np.diag([1., -1.])}
    dense = np.zeros((2**n, 2**n), dtype=np.complex128)
    terms = []
    for i in range(n-1):
        for label, local in pauli.items():
            parts = [np.eye(2)] * n
            parts[i], parts[i+1] = local, local
            dense += 0.25 * reduce(np.kron, parts)
            # Existing symbolic MPO/TTNO construction supports real local
            # operators. YY = -(iY)(iY); the dense oracle above stays literal Y.
            symbol = 'isigma_y' if label == 'sigma_y' else label
            coefficient = -0.25 if label == 'sigma_y' else 0.25
            terms.append(Op(symbol+' '+symbol, [i, i+1], coefficient))
    if field:
        dense += field * reduce(np.kron, [pauli['sigma_z']] + [np.eye(2)]*(n-1))
        terms.append(Op('sigma_z', 0, field))
    model = Model([BasisHalfSpin(i) for i in range(n)], terms)
    # Basis convention is checked against literal Pauli matrices, never fitted
    # to a candidate wavefunction or an MPO-produced dense Hamiltonian.
    for basis in model.basis:
        for label, local in pauli.items():
            np.testing.assert_array_equal(basis.op_mat(label), local)
    return model, dense


@contextmanager
def host_seed(seed=731):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        # All stochastic preparation uses the same host implementation even
        # when an optional environment auto-selects a GPU for legacy imports.
        with capture_backend(make_context('numpy', real_dtype='float64').adapter):
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def selected_context():
    return make_context(os.environ.get('RENO_TEST_BACKEND', 'numpy'),
                        device=os.environ.get('RENO_TEST_DEVICE', 'cpu'),
                        real_dtype='float64', host_policy='explicit')


def save_reference(tmp_path, dense, initial, record_property):
    for name, value in [('hamiltonian', dense), ('initial', initial)]:
        value = np.ascontiguousarray(value)
        np.save(tmp_path / (name+'.npy'), value, allow_pickle=False)
        record_property(name+'_sha256', hashlib.sha256(value.tobytes()).hexdigest())


def assert_witness(ledger, ctx, record_property):
    contractions = [event for event in ledger.operations
                    if event['operation'] in ('contraction', 'einsum', 'matmul', 'tensordot', 'oe_contract')]
    assert contractions, 'algorithm did not witness a target-backend contraction'
    for event in contractions:
        assert event['backend'] == ctx.adapter.name
        assert event['device'] == ctx.device
        assert event['adapter_id'] == id(ctx.adapter)
        # Torch spells the same actual precision with a namespace prefix.
        # This maps metadata only; no output array is cast during acceptance.
        dtype_name = {'torch.float64': 'float64', 'torch.complex128': 'complex128'}.get(
            event['dtype'], event['dtype'])
        assert np.dtype(dtype_name) in (np.dtype('float64'), np.dtype('complex128'))
    record_property('contraction_count', len(contractions))
    record_property('backend', ctx.adapter.name)
    record_property('device', ctx.device)
    record_property('transfer_count', len(ledger.transfers))


def check_ground_state(dense, psi, record_property, degenerate=False):
    psi = np.asarray(psi).reshape(-1)
    assert np.isfinite(psi).all()
    np.testing.assert_allclose(np.vdot(psi, psi).real, 1., atol=1e-10, rtol=0)
    w, v = np.linalg.eigh(dense)
    energy = np.vdot(psi, dense @ psi)
    residual = np.linalg.norm(dense @ psi - w[0]*psi)
    cluster = np.abs(w-w[0]) <= 1e-10
    if degenerate:
        assert np.count_nonzero(cluster) > 1
    ground = v[:, cluster]
    outside = np.linalg.norm(psi - ground @ (ground.conj().T @ psi))
    assert abs(energy-w[0]) <= ENERGY_ATOL, f'energy error {abs(energy-w[0])}'
    assert residual <= RESIDUAL_ATOL, f'ground residual {residual}'
    assert outside <= RESIDUAL_ATOL, f'outside-ground-space norm {outside}'
    record_property('energy_error', float(abs(energy-w[0])))
    record_property('residual', float(residual))
    record_property('ground_space_dimension', int(cluster.sum()))


def check_evolution(dense, psi0, psi, bond, record_property):
    assert np.isfinite(psi).all()
    w, v = np.linalg.eigh(dense)
    exact = v @ (np.exp(-1j*w*FINAL_TIME) * (v.conj().T @ psi0))
    norm_error = abs(np.vdot(psi, psi).real - np.vdot(psi0, psi0).real)
    energy_error = abs(np.vdot(psi, dense@psi) - np.vdot(psi0, dense@psi0))
    assert norm_error <= NORM_BUDGET
    assert energy_error <= ENERGY_DRIFT_BUDGET
    overlap = np.vdot(exact, psi)
    phase = 1. if abs(overlap) == 0 else overlap / abs(overlap)
    state_error = np.linalg.norm(psi-phase*exact)
    assert state_error <= BACKEND_BUDGET + INTEGRATION_BUDGET + TRUNCATION_BUDGET[bond]
    for key, value in [('state_error', state_error), ('norm_error', norm_error),
                       ('energy_drift', energy_error)]:
        record_property(key, float(value))
    # These are bounded dt/bond comparisons, not an unsupported fitted order.
    return state_error
