"""Backend API edge cases and disabled-diagnostics/reusable-expression costs."""
import numpy as np
import pytest

from renormalizer.backend.context import make_context, capture_backend


@pytest.fixture
def torch_ctx():
    pytest.importorskip('torch')
    return make_context('torch', device='cpu', host_policy='explicit')


@pytest.mark.parametrize('operation', ['sum', 'max', 'min'])
@pytest.mark.parametrize('complex_input', [False, True])
@pytest.mark.parametrize('keepdims', [False, True])
def test_empty_reduction_axes(torch_ctx, operation, complex_input, keepdims):
    host = np.arange(6.).reshape(2, 3)
    if complex_input:
        host = host + 1j * host
    ops = torch_ctx.ops
    actual = getattr(ops, operation)(ops.array(host), axis=(), keepdims=keepdims)
    np.testing.assert_array_equal(ops.to_numpy(actual), host)


@pytest.mark.parametrize('operation', ['sum', 'max', 'min', 'all', 'any'])
def test_positional_reduction_axis(torch_ctx, operation):
    host = np.arange(6.).reshape(2, 3)
    b = torch_ctx.adapter
    np.testing.assert_array_equal(b.numpy(getattr(b, operation)(b.asarray(host), 0)),
                                  getattr(np, operation)(host, 0))
    with pytest.raises(TypeError):
        getattr(b, operation)(b.asarray(host), 0, axis=1)


def test_eye_positional_parameters(torch_ctx):
    b = torch_ctx.adapter
    for k in (-3, -1, 0, 1, 4):
        np.testing.assert_array_equal(b.numpy(b.eye(2, 3, k, np.float32)),
                                      np.eye(2, 3, k, np.float32))
    with pytest.raises(TypeError):
        b.eye(2, 3, M=4)


def test_sum_dtype_out_and_keepdims(torch_ctx):
    b = torch_ctx.adapter
    x = b.asarray(np.arange(6., dtype=np.float32).reshape(2, 3))
    out = b.zeros((1, 3), dtype=np.float64)
    assert b.sum(x, 0, np.float64, out, True) is out
    np.testing.assert_array_equal(b.numpy(out), [[3., 5., 7.]])
    unchanged = torch_ctx.ops.sum(x, axis=(), dtype=np.float64)
    assert b.dtype_of(unchanged) == np.float64
    np.testing.assert_array_equal(b.numpy(unchanged), b.numpy(x))
    with pytest.raises(TypeError):
        b.max(x, mystery=True)


def test_negative_slices_broadcast_and_clip(torch_ctx):
    host = np.arange(20.).reshape(4, 5)
    ops = torch_ctx.ops
    x = ops.array(host)
    for start in (None, -9, -1, 0, 3, 9):
        for stop in (None, -9, -1, 0, 3, 9):
            index = (slice(None, None, -2), slice(start, stop, -2))
            expected = host.astype(complex)
            expected[index] += 2j
            actual = ops.at_add(x, index, 2j)
            np.testing.assert_array_equal(ops.to_numpy(actual), expected)
    np.testing.assert_array_equal(ops.to_numpy(x), host)


@pytest.mark.parametrize('operation', ['set', 'add', 'sub', 'mul'])
@pytest.mark.parametrize('index', [
    (slice(None), slice(None, None, -1)),
    (slice(None, None, -1), slice(None, None, -2)),
    (1, slice(2, 0, -1)),
    (slice(None), slice(0, 2, -1)),
])
def test_negative_slice_update(torch_ctx, operation, index):
    host = np.arange(12.).reshape(3, 4)
    value = np.arange(host[index].size, dtype=float).reshape(host[index].shape) + 2
    expected = host.copy()
    if operation == 'set':
        expected[index] = value
    elif operation == 'add':
        expected[index] += value
    elif operation == 'sub':
        expected[index] -= value
    else:
        expected[index] *= value
    ops = torch_ctx.ops
    x = ops.array(host)
    result = getattr(ops, 'at_' + operation)(x, index, ops.array(value))
    np.testing.assert_array_equal(ops.to_numpy(result), expected)
    np.testing.assert_array_equal(ops.to_numpy(x), host)
    with pytest.raises(ValueError):
        getattr(ops, 'at_' + operation)(x, slice(None, None, 0), 1.)


@pytest.mark.parametrize('operation', ['max', 'min'])
def test_complex_numpy_integer_axis(torch_ctx, operation):
    host = np.array([[1+2j, 1+3j], [4+2j, 2+1j]])
    ops = torch_ctx.ops
    np.testing.assert_array_equal(ops.to_numpy(getattr(ops, operation)(
        ops.array(host), axis=np.int64(1))), getattr(np, operation)(host, axis=1))


@pytest.mark.parametrize('method', ['normal', 'uniform', 'random', 'randint'])
def test_jax_numpy_integer_size(method):
    pytest.importorskip('jax')
    b = make_context('jax', device='cpu').adapter
    fn = getattr(b.random, method)
    args = (0, 10) if method == 'randint' else ()
    b.random.seed(7)
    actual = fn(*args, size=np.int64(3))
    b.random.seed(7)
    expected = fn(*args, size=3)
    np.testing.assert_array_equal(b.numpy(actual), b.numpy(expected))


def test_execution_recording_is_scoped_to_record_execution():
    from renormalizer.backend import execution
    from renormalizer.backend.testing import record_execution
    ctx = make_context(host_policy='explicit')
    @execution.bind_backend
    def run():
        return execution.contract('ij,jk->ik', np.eye(2), np.eye(2))
    np.testing.assert_array_equal(run(backend_context=ctx), np.eye(2))
    with record_execution(ctx) as ledger:
        run(backend_context=ctx)
    assert ledger.transfers and ledger.operations


def test_torch_expression_reuses_unchanged_precision(torch_ctx, monkeypatch):
    from renormalizer.backend import execution
    original = execution.oe.contract_expression
    builds = []
    def counted(*args, **kwargs):
        builds.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(execution.oe, 'contract_expression', counted)
    with capture_backend(torch_ctx.adapter):
        expression = execution.contract_expression('ij,jk->ik', np.eye(2), (2, 2), constants=[0])
        for dtype in (np.float64, np.complex128, np.float64):
            x = np.eye(2, dtype=dtype) * (1j if dtype == np.complex128 else 2)
            np.testing.assert_array_equal(torch_ctx.adapter.numpy(expression(x)), x)
            before = len(builds)
            np.testing.assert_array_equal(torch_ctx.adapter.numpy(expression(x * 3)), x * 3)
            assert len(builds) == before
