from dataclasses import FrozenInstanceError, replace
from importlib import metadata
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from renormalizer.backend.context import make_context
from renormalizer.backend.contracts import CapabilityError, OwnershipError
from renormalizer.backend.execution import bind_backend, contract, contract_expression
from renormalizer.backend.testing import record_execution
from renormalizer.backend.operators import (Kernel, OperatorProvider, discover_operator_providers,
                                           make_request)


def provider(operation='matmul', execute=None, supports=lambda request, *a, **k: None, name='test'):
    return OperatorProvider(name, '1', {operation: Kernel(supports, execute or (lambda r, a, b: a @ b))})


def context(p, policy='prefer'):
    return make_context(operator_policy=policy, operator_providers=[p], host_policy='explicit')


def test_defaults_do_not_discover_or_load(monkeypatch):
    monkeypatch.setattr(metadata, 'entry_points', lambda **k: pytest.fail('discovery'))
    ctx = make_context()
    np.testing.assert_array_equal(ctx.ops.matmul(np.eye(2), np.eye(2)), np.eye(2))


def test_selective_override_promotion_validation_and_metadata():
    calls = []
    def execute(request, a, b):
        calls.append(request)
        return a @ b
    ctx = context(provider(execute=execute))
    a = np.arange(6., dtype='float32').reshape(2, 3)[:, ::-1]
    b = np.ones((3, 2), dtype='float64')
    np.testing.assert_array_equal(ctx.ops.matmul(a, b), a @ b)
    assert calls[0].semantics_version == 1
    assert calls[0].arrays[0].dtype == 'float64'
    assert calls[0].input_mutation is False
    ctx.ops.einsum('ij,jk->ik', a, b)
    assert len(calls) == 1
    with pytest.raises(OwnershipError):
        ctx.ops.matmul([[1.]], np.eye(1))
    with pytest.raises(ValueError, match='finite'):
        ctx.ops.matmul(np.array([[np.nan]]), np.eye(1))
    assert len(calls) == 1


@pytest.mark.parametrize('operation', ['matmul', 'einsum', 'qr', 'svd', 'eigh', 'solve'])
def test_six_compute_contracts(operation):
    called = []
    function = getattr(np.linalg if operation in ('qr', 'svd', 'eigh', 'solve') else np, operation)
    def execute(request, *args, **kwargs):
        called.append(request.operation)
        return function(*args, **kwargs)
    ctx = context(provider(operation, execute), 'require')
    a = np.array([[3., 1.], [1., 2.]])
    args = ('ij,jk->ik', a, a) if operation == 'einsum' else (a, a) if operation in ('matmul', 'solve') else (a,)
    actual = getattr(ctx.ops, operation)(*args)
    expected = getattr(make_context().ops, operation)(*args)
    for x, y in zip(actual if isinstance(actual, tuple) else (actual,), expected if isinstance(expected, tuple) else (expected,)):
        np.testing.assert_allclose(x, y)
    assert called == [operation]


def test_fallback_require_and_explain():
    p = provider(supports=lambda r, *a, **k: 'strides unsupported', execute=lambda *a: pytest.fail('execute'))
    ctx = context(p)
    a = np.eye(2)
    req = make_request(ctx, 'matmul', (a, a))
    assert ctx.operators.explain(req)['reasons'] == (('test', 'strides unsupported'),)
    np.testing.assert_array_equal(ctx.ops.matmul(a, a), a)
    with pytest.raises(CapabilityError, match='strides unsupported'):
        context(p, 'require').ops.matmul(a, a)
    with pytest.raises(CapabilityError, match='operation not supplied'):
        context(provider(), 'require').ops.qr(a)


def test_missing_version_and_order(monkeypatch):
    monkeypatch.setattr(metadata, 'entry_points', lambda **k: ())
    ctx = context('missing')
    assert ctx.operators.unavailable == (('missing', 'not installed'),)
    with pytest.raises(CapabilityError, match='not installed'):
        context('missing', 'require')
    with pytest.raises(CapabilityError, match='protocol'):
        context(replace(provider(), protocol_version=999), 'require')
    calls = []
    first = provider(supports=lambda r, *a, **k: 'unsupported', name='first')
    second = provider(execute=lambda r, a, b: calls.append(1) or a @ b, name='second')
    ctx = make_context(operator_policy='prefer', operator_providers=[first, second])
    ctx.ops.matmul(np.eye(1), np.eye(1))
    assert calls == [1]


def test_errors_never_fallback_and_output_checks():
    def fail(*a):
        raise RuntimeError('kernel failed')
    with pytest.raises(RuntimeError, match='kernel failed'):
        context(provider(execute=fail)).ops.matmul(np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match='finite'):
        context(provider(execute=lambda *a: np.full((2, 2), np.inf))).ops.matmul(np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match='requested'):
        context(provider(execute=lambda *a: np.eye(2, dtype='float32'))).ops.matmul(np.eye(2), np.eye(2))


def test_frozen_context_and_selection():
    p = provider()
    ctx = context(p)
    with pytest.raises(FrozenInstanceError):
        ctx.operators.policy = 'builtin'
    with pytest.raises(TypeError):
        p.kernels['solve'] = p.kernels['matmul']
    with pytest.raises(FrozenInstanceError):
        ctx.operators = None


def test_execution_contractions_expression_constants_and_context_isolation():
    calls = []
    import opt_einsum as oe
    def direct(request, *args, **kwargs):
        calls.append(('contract', request))
        return oe.contract(*args, **kwargs)
    def expression(request, *args, expression, **kwargs):
        calls.append(('expression', request))
        return expression(*args, **kwargs)
    p = OperatorProvider('contractor', '1', {'contract': Kernel(lambda r, *a, **k: None, direct),
        'contract_expression': Kernel(lambda r, *a, **k: None, expression)})
    ctx = context(p, 'require')
    @bind_backend
    def run():
        a = np.eye(2)
        np.testing.assert_array_equal(contract(a, [0, 1], a, [1, 2], [0, 2], optimize='greedy'), a)
        expr = contract_expression('ij,jk->ik', a, (2, 2), constants=[0], optimize='greedy')
        np.testing.assert_array_equal(expr(a), a)
        return expr
    with record_execution(ctx) as ledger:
        expr = run(backend_context=ctx)
    assert [c[0] for c in calls] == ['contract', 'expression']
    assert len(calls[1][1].arrays) == 2  # includes the retained constant
    assert len(ledger.operations) == 2
    @bind_backend
    def wrong():
        return expr(np.eye(2))
    with pytest.raises(CapabilityError, match='another'):
        wrong(backend_context=make_context(host_policy='explicit'))


def test_concurrent_contexts_do_not_cross_select():
    def work(name):
        calls = []
        ctx = context(provider(execute=lambda r,a,b: calls.append(name) or a @ b, name=name))
        ctx.ops.matmul(np.eye(2), np.eye(2))
        return calls
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(work, ['one', 'two'])) == [['one'], ['two']]


def test_real_entry_point_discovery_is_lazy(tmp_path, monkeypatch):
    import sys
    fixture = Path(__file__).parent / 'provider_fixture'
    monkeypatch.syspath_prepend(str(fixture))
    info = tmp_path / 'reno_test_operators-0.1.0.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: reno-test-operators\nVersion: 0.1.0\n')
    (info / 'entry_points.txt').write_text('[renormalizer.operator_providers]\ntest_numpy = reno_test_operators:provider\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop('reno_test_operators', None)
    assert any(e.name == 'test_numpy' for e in discover_operator_providers())
    assert 'reno_test_operators' not in sys.modules
    ctx = context('test_numpy', 'require')
    assert 'reno_test_operators' in sys.modules
    np.testing.assert_array_equal(ctx.ops.matmul(np.eye(2), np.eye(2)), np.eye(2))
    with pytest.raises(CapabilityError, match='float64'):
        ctx.ops.matmul(np.eye(2, dtype='float32'), np.eye(2, dtype='float32'))


def test_equation_and_options_rejected_before_execute():
    calls = []
    def supports(request, equation, *arrays, **options):
        return None if equation == 'ij,jk->ik' else 'equation unsupported'
    def execute(request, equation, *arrays):
        calls.append(equation)
        return np.einsum(equation, *arrays)
    ctx = context(provider('einsum', execute, supports))
    a = np.eye(2)
    np.testing.assert_array_equal(ctx.ops.einsum('ij->ji', a), a)
    assert calls == []
    ctx.ops.einsum('ij,jk->ik', a, a)
    assert calls == ['ij,jk->ik']


def test_declared_unavailable_load_falls_back_but_factory_bugs_propagate(monkeypatch):
    from renormalizer.backend.operators import ProviderUnavailable
    class Entry:
        name = 'optional'
        def load(self):
            def factory():
                raise ProviderUnavailable('optional runtime missing')
            return factory
    monkeypatch.setattr(metadata, 'entry_points', lambda **k: (Entry(),))
    assert context('optional').operators.unavailable == (('optional', 'optional runtime missing'),)
    with pytest.raises(CapabilityError, match='runtime missing'):
        context('optional', 'require')
    monkeypatch.setattr(Entry, 'load', lambda self: (_ for _ in ()).throw(RuntimeError('bug')))
    with pytest.raises(RuntimeError, match='bug'):
        context('optional')


def test_production_output_contract_and_execution_error():
    @bind_backend
    def run():
        return contract('ij,jk->ik', np.eye(2), np.eye(2))
    with pytest.raises(OwnershipError, match='wrong backend'):
        run(backend_context=context(provider('contract', lambda *a, **k: [[1.]])))
    with pytest.raises(ValueError, match='requested'):
        run(backend_context=context(provider('contract', lambda *a, **k: np.eye(2, dtype='float32'))))
    def fail(*args, **kwargs):
        raise RuntimeError('after execute')
    with pytest.raises(RuntimeError, match='after execute'):
        run(backend_context=context(provider('contract', fail)))


def test_builtin_expression_keeps_adapter_capture_behavior():
    from renormalizer.backend.context import capture_backend
    @bind_backend
    def create():
        return contract_expression('ij,jk->ik', (2,2), (2,2))
    ctx = make_context(host_policy='explicit')
    expr = create(backend_context=ctx)
    with capture_backend(ctx.adapter):
        np.testing.assert_array_equal(expr(np.eye(2), np.eye(2)), np.eye(2))


def test_layout_request_preserves_negative_strides_without_copy():
    ctx = context(provider())
    a = np.eye(3)[:, ::-1]
    request = make_request(ctx, 'matmul', (a, a))
    assert request.arrays[0].strides == a.strides
    assert request.arrays[0].c_contiguous is False
    assert request.arrays[0].writable is True


def test_torch_cpu_override_preserves_native_arrays_and_autograd_request():
    torch = pytest.importorskip('torch')
    calls = []
    def execute(request, a, b):
        assert isinstance(a, torch.Tensor) and a.device.type == 'cpu'
        calls.append(request)
        return a @ b
    ctx = make_context('torch', operator_policy='require', operator_providers=[provider(execute=execute)])
    a = torch.eye(2, dtype=torch.float64, requires_grad=True)
    result = ctx.ops.matmul(a, a)
    result.sum().backward()
    assert calls[0].transformations == ('autograd',)
    torch.testing.assert_close(a.grad, torch.full((2, 2), 2., dtype=torch.float64))


def test_resolution_and_result_checks_once_per_signature(monkeypatch):
    from renormalizer.backend import operators
    asked, returned = [], []
    def supports(request, *a, **k):
        asked.append(request.operation)
        return None
    def execute(request, *args, **kwargs):
        returned.append(1)
        return np.eye(2) if len(returned) == 1 else np.eye(2, dtype='float32')
    ctx = context(provider('contract', execute, supports))
    @bind_backend
    def run():
        return contract('ij,jk->ik', np.eye(2), np.eye(2))
    for _ in range(3):
        run(backend_context=ctx)
    assert asked == ['contract']  # supports asked once for one signature
    # Only the first result per signature is checked unless strict.
    monkeypatch.setattr(operators, '_STRICT_PROVIDERS', True)
    with pytest.raises(ValueError, match='requested'):
        run(backend_context=ctx)
