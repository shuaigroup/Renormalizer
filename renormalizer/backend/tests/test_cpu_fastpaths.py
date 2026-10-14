"""CPU overhead reductions must preserve mutation, copy, precision and lifetime."""
from dataclasses import replace
import os
import weakref

import numpy as np
import pytest

from renormalizer.backend.context import make_context, capture_backend


@pytest.fixture
def ctx():
    return make_context(os.environ.get('RENO_TEST_BACKEND', 'numpy'), device='cpu')


def test_replaced_context_has_its_own_operations():
    first = make_context()
    first.ops
    second = replace(first, host_policy='explicit')
    assert second.ops.context is second
    assert first.ops.context is first


def test_operation_wrapper_keeps_context_alive_without_an_owner_cycle():
    ctx = make_context()
    ops = ctx.ops
    reference = weakref.ref(ctx)
    del ctx
    assert reference() is not None
    np.testing.assert_array_equal(ops.ones((2,)), [1., 1.])
    del ops
    assert reference() is None


@pytest.mark.parametrize('operation', ['matmul', 'einsum', 'solve'])
@pytest.mark.parametrize('bad_input', [0, 1])
def test_numeric_inputs_are_checked_again_after_host_mutation(ctx, operation, bad_input):
    # A warmed call must never make later values implicitly trusted.
    arrays = [np.eye(2), np.eye(2)]
    def run():
        a, b = [ctx.ops.from_numpy(x) for x in arrays]
        if operation == 'einsum':
            return ctx.ops.einsum('ij,jk->ik', a, b)
        return getattr(ctx.ops, operation)(a, b)
    run()
    arrays[bad_input][0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        run()


@pytest.mark.parametrize('operation', ['matmul', 'einsum'])
def test_output_finiteness_still_checked(ctx, operation):
    x = ctx.ops.array([[1e200]])
    with pytest.raises((ValueError, FloatingPointError), match='finite|overflow'):
        if operation == 'einsum':
            ctx.ops.einsum('ij,jk->ik', x, x)
        else:
            ctx.ops.matmul(x, x)
    np.testing.assert_array_equal(ctx.ops.to_numpy(x), [[1e200]])


def test_native_no_copy_and_explicit_copy_keep_their_meanings(ctx):
    adapter = ctx.adapter
    x = adapter.from_numpy(np.arange(6.).reshape(2, 3))
    assert adapter.asarray(x) is x
    assert adapter.asarray(x, dtype=np.float64) is x
    copied = adapter.array(x, copy=True)
    assert copied is not x
    changed_dtype = adapter.asarray(x, dtype=np.complex128)
    np.testing.assert_array_equal(adapter.numpy(changed_dtype), adapter.numpy(x))
    if adapter.name == 'numpy':
        assert not np.shares_memory(x, copied)
    elif adapter.name == 'torch':
        assert x.data_ptr() != copied.data_ptr()
        assert x.data_ptr() != changed_dtype.data_ptr()
    elif adapter.name == 'jax':
        assert x.unsafe_buffer_pointer() != copied.unsafe_buffer_pointer()


def test_torch_native_fastpath_preserves_autograd():
    pytest.importorskip('torch')
    ctx = make_context('torch')
    x = ctx.adapter.from_numpy(np.ones(3)).requires_grad_()
    y = ctx.adapter.asarray(x)
    (y*y).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [2., 2., 2.])


def test_torch_expression_promotes_constants_once_and_releases_them(monkeypatch):
    pytest.importorskip('torch')
    from renormalizer.backend import execution
    ctx = make_context('torch')
    conversions, refs, builds = [], [], []
    original_array = ctx.adapter.asarray
    original_expression = execution.oe.contract_expression
    def convert(x, dtype=None):
        result = original_array(x, dtype=dtype)
        if dtype is not None and result is not x:
            conversions.append(result.numel()*result.element_size())
            refs.append(weakref.ref(result))
        return result
    def expression(*args, **kwargs):
        builds.append(1)
        return original_expression(*args, **kwargs)
    monkeypatch.setattr(ctx.adapter, 'asarray', convert)
    monkeypatch.setattr(execution.oe, 'contract_expression', expression)
    with capture_backend(ctx.adapter):
        call = execution.contract_expression('ij,jk->ik', np.eye(8), (8,8), constants=[0])
        real = ctx.adapter.from_numpy(np.eye(8))
        complex_value = ctx.adapter.from_numpy(np.eye(8)*1j)
        call(real)
        assert len(builds) == 1
        for _ in range(3):
            np.testing.assert_array_equal(ctx.adapter.numpy(call(complex_value)), np.eye(8)*1j)
        # One promoted constant belongs to this expression, not to each call.
        assert len(conversions) == 1
        assert len(builds) == 2
        assert refs[0]() is not None
        call(real)
        assert refs[0]() is None
        assert len(builds) == 2
        del call


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('equation,shapes', [
    ('ij,jk->ik', ((3,4),(4,5))),
    (' ji, jk -> ik ', ((4,3),(4,5))),
    ('ij,kj->ki', ((3,4),(5,4))),
    ('ji,kj->ik', ((4,3),(5,4))),
    ('ab,bc->ca', ((3,4),(4,5))),
    ('ij,jk->ik', ((0,4),(4,5))),
    ('ij,jk->ik', ((3,0),(0,5))),
    # Full einsum fallbacks: named-axis broadcast, diagonals and ellipses.
    ('ij,jk->ik', ((3,1),(4,5))),
    ('ii,ik->ik', ((3,3),(3,5))),
    ('...j,jk->...k', ((3,4),(4,5))),
])
def test_numpy_matrix_einsum_dispatch_preserves_labels_and_layout(dtype, equation, shapes):
    ctx = make_context()
    rng = np.random.default_rng(2026)
    values = [rng.normal(size=s).astype(dtype) for s in shapes]
    if np.dtype(dtype).kind == 'c':
        for x in values:
            x += 1j*rng.normal(size=x.shape)
    values = [x[::-1] for x in values]
    for x in values:
        x.flags.writeable = False
    expected = np.einsum(equation, *values)
    actual = ctx.ops.einsum(equation, *values)
    assert actual.dtype == expected.dtype
    tolerance = 2e-5 if actual.dtype in (np.dtype('float32'), np.dtype('complex64')) else 1e-12
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('equation', ['ij,jk->ii', 'ij,jk->iz', 'ij,jk->ik->ik'])
def test_numpy_matrix_einsum_invalid_labels_still_fail(equation):
    with pytest.raises(ValueError):
        make_context().ops.einsum(equation, np.eye(3), np.eye(3))


def test_jax_native_fastpath_preserves_weak_types_and_tracing():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    ctx = make_context('jax')
    weak = jnp.array(1.)
    assert weak.weak_type
    assert ctx.adapter.asarray(weak).weak_type
    assert not ctx.adapter.asarray(weak, dtype=np.float64).weak_type
    assert not ctx.ops.add(weak, weak).weak_type
    function = jax.jit(lambda x: ctx.adapter.asarray(x)*2)
    np.testing.assert_array_equal(function(jnp.ones(3)), [2., 2., 2.])


def test_jax_tracking_does_not_retain_borrowed_arrays():
    pytest.importorskip('jax')
    ctx = make_context('jax')
    value = ctx.adapter.from_numpy(np.arange(6.))
    reference = weakref.ref(value)
    for _ in range(5):
        assert ctx.adapter.asarray(value) is value
    ctx.adapter.sync()
    del value
    assert reference() is None
