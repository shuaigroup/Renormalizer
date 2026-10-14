import numpy as np
import opt_einsum as oe
import pytest
from renormalizer.backend.context import make_context
from renormalizer.backend.execution import bind_backend, contract
from renormalizer.backend.testing import record_execution


def test_bound_contraction_is_witnessed_and_context_restored():
    from renormalizer.backend.context import current_backend
    ctx = make_context(host_policy="explicit")
    before = current_backend()
    @bind_backend
    def f():
        return contract('ij,jk->ik', np.eye(2), np.eye(2))
    with record_execution(ctx) as ledger:
        result = f(backend_context=ctx)
    np.testing.assert_array_equal(result, np.eye(2))
    assert current_backend() is before
    assert ledger.operations[0]['adapter_id'] == id(ctx.adapter)
    assert ledger.operations[0]['operation'] == 'contraction'


def test_explicit_hybrid_policy_rejected_before_algorithm():
    ctx = make_context()
    @bind_backend
    def f():
        raise AssertionError('must reject before algorithm')
    with pytest.raises(Exception, match='explicit'):
        f(backend_context=ctx)


def test_cpu_foreign_array_records_host_boundary_without_size_method():
    from renormalizer.backend.execution import _record_transfer
    class Foreign:
        device = 'cpu'
        dtype = 'float64'
        shape = (2, 3)
        def size(self):
            raise AssertionError('not a NumPy size attribute')
    with record_execution(make_context()) as ledger:
        _record_transfer(np.zeros((2, 3)), Foreign(), 'test')
    assert ledger.transfers[0]['direction'] == 'host_to_host'
    assert ledger.transfers[0]['logical_bytes'] == 48


def test_expression_constants_captured_and_cross_instance_refused(monkeypatch):
    from renormalizer.backend.context import capture_backend
    from renormalizer.backend.execution import contract_expression
    ctx = make_context(host_policy='explicit')
    constant = np.eye(2)
    converted = []
    original = oe.contract_expression
    def traced(*args, **kwargs):
        converted.extend(args)
        return original(*args, **kwargs)
    # NumPy placement is the identity, so the captured constant itself must
    # reach opt_einsum, without a copy or a process-default conversion.
    monkeypatch.setattr(oe, 'contract_expression', traced)
    with capture_backend(ctx.adapter):
        expr = contract_expression('ij,jk->ik',constant,(2,2),constants=[0])
        np.testing.assert_array_equal(expr(np.eye(2)),constant)
    assert any(value is constant for value in converted)
    with capture_backend(make_context().adapter):
        with pytest.raises(Exception,match='another backend'):
            expr(np.eye(2))


@pytest.mark.parametrize('optimize', ['optimal', 'dp', 'auto-hq'])
@pytest.mark.parametrize('subscripts,shapes', [
    ('ij,jk,kl->il', [(6, 5), (5, 7), (7, 4)]),
    ('abc,bd,dce,ef->af', [(3, 4, 5), (4, 6), (6, 5, 2), (2, 3)]),
    ('ij->ji', [(3, 4)]),
])
def test_host_contract_matches_opt_einsum_and_searches_the_path_once(subscripts, shapes, optimize):
    from renormalizer.backend import execution
    rng = np.random.default_rng(11)
    operands = [rng.standard_normal(s) + 1j * rng.standard_normal(s) for s in shapes]
    expected = oe.contract(subscripts, *operands, optimize=optimize)
    execution._cached_expression.cache_clear()
    execution._cached_path.cache_clear()
    for _ in range(3):
        assert np.array_equal(contract(subscripts, *operands, optimize=optimize), expected)
    # One path search and one expression for three calls.
    assert execution._cached_path.cache_info().misses == 1
    assert execution._cached_expression.cache_info().misses == 1
    assert execution._cached_expression.cache_info().hits == 2


@pytest.mark.parametrize('extra', [{'memory_limit': 10 ** 6}, {'optimize': True}])
def test_contract_with_other_options_falls_back_to_opt_einsum(extra):
    from renormalizer.backend import execution
    rng = np.random.default_rng(13)
    a, b = rng.standard_normal((4, 5)), rng.standard_normal((5, 3))
    kwargs = dict({'optimize': 'optimal'}, **extra)
    execution._cached_expression.cache_clear()
    assert np.array_equal(contract('ij,jk->ik', a, b, **kwargs), oe.contract('ij,jk->ik', a, b, **kwargs))
    assert execution._cached_expression.cache_info().currsize == 0


def test_expression_with_constants_matches_numpy_backend():
    from renormalizer.backend import execution
    rng = np.random.default_rng(14)
    a, c = rng.standard_normal((4, 5)), rng.standard_normal((3, 6))
    b = rng.standard_normal((5, 3))
    expected = oe.contract_expression('ij,jk,kl->il', a, b.shape, c, constants=[0, 2],
                                      optimize='optimal')(b, backend='numpy')
    expr = execution.contract_expression('ij,jk,kl->il', a, b.shape, c, constants=[0, 2], optimize='optimal')
    assert np.array_equal(expr(b), expected)


def test_host_contract_interleaved_labels_match_opt_einsum():
    # Tree networks pass (operand, labels, ..., output labels) with tuple labels.
    rng = np.random.default_rng(12)
    a, b, c = (rng.standard_normal(s) for s in [(3, 4, 5), (4, 6), (5, 6, 2)])
    x, p, y, q, r = ('x', 0), ('p', 0), ('y', 1), ('q', 0), ('r', 2)
    args = [a, [x, p, y], b, [p, q], c, [y, q, r], [x, r]]
    expected = oe.contract(*args, optimize='optimal')
    for _ in range(2):
        assert np.array_equal(contract(*args, optimize='optimal'), expected)
    # Implicit output.
    assert np.array_equal(contract(a, [0, 1, 2], b, [1, 3], optimize='optimal'),
                          oe.contract(a, [0, 1, 2], b, [1, 3], optimize='optimal'))


def test_contract_falls_back_for_mixed_labels_and_converts_array_subclasses():
    from renormalizer.backend import execution
    rng = np.random.default_rng(15)
    a, b = rng.standard_normal((3, 4)), rng.standard_normal((4, 2))
    # Labels opt_einsum cannot sort: both paths raise the same error type.
    with pytest.raises(TypeError):
        oe.contract(a, [('x', 0), 'p'], b, ['p', 'q'], optimize='optimal')
    with pytest.raises(TypeError):
        contract(a, [('x', 0), 'p'], b, ['p', 'q'], optimize='optimal')

    class Sub(np.ndarray):
        pass
    # to_backend turns subclasses into plain ndarrays, so they take the cached path.
    execution._cached_expression.cache_clear()
    result = contract('ij,jk->ik', a.view(Sub), b.view(Sub), optimize='optimal')
    assert type(result) is np.ndarray
    assert np.array_equal(result, oe.contract('ij,jk->ik', a, b, optimize='optimal'))
    assert execution._cached_expression.cache_info().currsize == 1


def test_plain_runs_carry_no_run_state_and_recording_survives_binding():
    from renormalizer.backend import execution
    from renormalizer.backend.testing import record_execution
    seen = []
    @bind_backend
    def probe():
        seen.append(execution._run.get())
        return contract('ij,jk->ik', np.eye(2), np.eye(2))
    probe()
    assert seen[-1] is None
    ctx = make_context(host_policy='explicit')
    with record_execution(ctx) as ledger:
        probe(backend_context=ctx)
    assert seen[-1].context is ctx and seen[-1].ledger is ledger
    assert ledger.operations
    assert execution._run.get() is None
