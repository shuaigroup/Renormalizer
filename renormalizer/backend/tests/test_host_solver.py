"""Host solver policy, dtype and transfer accounting are independent of dispatch."""
from types import SimpleNamespace

import numpy as np
import pytest

from renormalizer.backend.context import make_context
from renormalizer.backend.contracts import CapabilityError, PrecisionError
from renormalizer.backend.host_solver import (
    call_host_solver, from_host, to_host, wrap_host_callback,
)


def test_forbidden_policy_precedes_conversion_and_callback_creation():
    class NeverTouched:
        def __getattr__(self, name):
            pytest.fail('conversion accessed before policy check')
    context = SimpleNamespace(host_policy='forbid', adapter=NeverTouched(), ops=NeverTouched())
    called = []
    solver = lambda x: called.append(x)
    for action in (
        lambda: call_host_solver(solver, [object()], context=context, ledger=[], reason='solver'),
        lambda: wrap_host_callback(solver, context=context, ledger=[], reason='callback'),
        lambda: to_host(object(), context=context, ledger=[], reason='boundary'),
        lambda: from_host(object(), context=context, ledger=[], reason='boundary'),
    ):
        with pytest.raises(CapabilityError, match='explicit policy'):
            action()
    assert called == []


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
def test_solver_preserves_dtype_and_nested_scalar_metadata(dtype):
    context = make_context(host_policy='explicit')
    x = np.eye(2, dtype=dtype)
    ledger = []
    marker = np.int64(3)
    def solver(a):
        assert isinstance(a, np.ndarray) and a.dtype == x.dtype
        return {'vectors': [a * 2], 'metadata': (marker, 'converged', None)}
    result = call_host_solver(solver, [x], context=context, ledger=ledger,
                              reason='reference solve', operation_id='solve-1')
    assert result['vectors'][0].dtype == x.dtype
    assert result['metadata'][0] is marker
    assert result['metadata'][1:] == ('converged', None)
    assert len(ledger) == 2
    for event in ledger:
        assert event['direction'] == 'host_to_host'
        assert event['logical_bytes'] == x.nbytes
        assert event['reason'] == 'reference solve'
        assert event['operation_id'] == 'solve-1'


@pytest.mark.parametrize('switch_to', ['torch', 'jax'])
def test_callback_uses_captured_context_despite_global_switch(switch_to):
    # The global switch must select a different backend; otherwise the captured
    # NumPy context and the global backend are indistinguishable.
    pytest.importorskip(switch_to)
    from renormalizer.cons import get_backend, set_backend
    context = make_context(host_policy='explicit')
    previous = get_backend()
    rng_state = np.random.get_state()
    ledger = []
    def callback(x):
        set_backend(switch_to)
        return context.ops.multiply(x, np.array(2., dtype=x.dtype))
    wrapped = wrap_host_callback(callback, context=context, ledger=ledger,
                                 reason='iterative matvec', operation_id='iteration')
    try:
        x = np.arange(3, dtype='float32')
        result = wrapped(x)
        assert get_backend().name == switch_to
        assert isinstance(result, np.ndarray) and result.dtype == x.dtype
        np.testing.assert_array_equal(result, 2 * x)
        assert len(ledger) == 2
        assert {item['operation_id'] for item in ledger} == {'iteration'}
        assert all(item['logical_bytes'] == x.nbytes for item in ledger)
    finally:
        import renormalizer.cons as cons
        cons._manager.current = previous
        np.random.set_state(rng_state)


def test_scipy_linear_operator_callback_is_reusable():
    from scipy.sparse.linalg import LinearOperator, eigsh
    context = make_context(host_policy='explicit')
    ledger = []
    diagonal = np.arange(1, 6, dtype='float64')
    callback = wrap_host_callback(lambda x: context.ops.multiply(diagonal, x),
                                  context=context, ledger=ledger, reason='scipy eigsh')
    operator = LinearOperator((5, 5), matvec=callback, dtype=np.float64)
    values, vectors = eigsh(operator, k=1, which='SA', v0=np.ones(5))
    np.testing.assert_allclose(values, [1.], atol=1e-12)
    np.testing.assert_allclose(diagonal * vectors[:, 0], vectors[:, 0], atol=1e-12)
    assert len(ledger) >= 2 and len(ledger) % 2 == 0
    assert len({event['operation_id'] for event in ledger}) == 1


def test_failure_is_not_retried_and_keeps_completed_transfer_record():
    context = make_context(host_policy='explicit')
    ledger, calls = [], []
    def solver(x):
        calls.append(True)
        raise RuntimeError('solver failed')
    with pytest.raises(RuntimeError, match='solver failed'):
        call_host_solver(solver, [np.ones(2)], context=context, ledger=ledger, reason='solver')
    assert calls == [True] and len(ledger) == 1


def test_device_direction_and_precision_checks_without_gpu_claim():
    class DeviceArray:
        def __init__(self, array):
            self.array, self.dtype, self.shape = array, array.dtype, array.shape
    class DeviceAdapter:
        name = 'fake_device'
        def owns(self, x):
            return isinstance(x, DeviceArray)
        def to_numpy(self, x):
            return x.array.copy()
        def array(self, x, dtype=None, copy=None):
            return DeviceArray(np.array(x, dtype=dtype, copy=True))
    context = SimpleNamespace(host_policy='explicit', device='cuda:2', adapter=DeviceAdapter())
    ledger = []
    host = np.ones((0, 2), dtype='complex64')
    device = from_host(host, context=context, ledger=ledger, reason='unit control')
    result = to_host(device, context=context, ledger=ledger, reason='unit control')
    assert result.dtype == host.dtype
    assert [item['direction'] for item in ledger] == ['H2D', 'D2H']
    assert [item['logical_bytes'] for item in ledger] == [0, 0]
    assert ledger[0]['target_device'] == ledger[1]['source_device'] == 'cuda:2'
    context.adapter.array = lambda x, dtype=None, copy=None: DeviceArray(x.astype('complex128'))
    with pytest.raises(PrecisionError):
        from_host(host, context=context, ledger=ledger, reason='bad conversion')


def test_non_numeric_solver_array_rejected_without_object_conversion():
    context = make_context(host_policy='explicit')
    with pytest.raises(CapabilityError, match='numeric'):
        call_host_solver(lambda: np.array(['answer'], dtype=object), [],
                         context=context, ledger=[], reason='invalid output')
