"""Consumed legacy-array and aggregate-IVP semantics; explicit backend fixture.

Host NumPy analytic/oracle expressions are deliberate and outside candidate
operations. Optional runtime availability is owned by backend_fixtures.
"""
import numpy as np
import pytest


def host(ctx, value):
    if ctx.adapter.name == 'numpy':
        assert isinstance(value, (np.ndarray, np.generic))
    else:
        assert ctx.adapter.owns(value), 'candidate result must retain selected native ownership'
    return np.asarray(ctx.adapter.numpy(value))


def test_finfo_is_host_precision_metadata(captured_backend):
    b = captured_backend.adapter
    assert b.finfo(b.real_dtype).eps == np.finfo(np.float64).eps


def test_empty_like_owned_workspace_preserves_shape_dtype(captured_backend):
    ctx = captured_backend
    x = ctx.ops.array([[1+2j, 3], [4, 5-1j]])
    y = ctx.adapter.empty_like(x)
    if ctx.adapter.name == 'numpy':
        assert isinstance(y, np.ndarray)
        dtype = y.dtype
    else:
        assert ctx.adapter.owns(y)
        dtype = ctx.adapter.dtype_of(y)
    assert y.shape == (2, 2) and dtype == np.dtype('complex128')
    # Do not read/assert uninitialized values; test only owned scratch semantics.
    y = ctx.adapter.write_owned(y, (0, 1), 7+3j)
    assert host(ctx, y[0, 1]) == 7+3j
    np.testing.assert_array_equal(host(ctx, x), [[1+2j, 3], [4, 5-1j]])


def test_mixed_real_complex_allclose_for_canonical_identity(captured_backend):
    ctx = captured_backend
    complex_identity = ctx.ops.eye(2, dtype='complex128')
    real_identity = ctx.ops.eye(2, dtype='float64')
    assert bool(ctx.adapter.allclose(complex_identity, real_identity, atol=1e-12, rtol=0))
    shifted = complex_identity + ctx.ops.eye(2, dtype='complex128') * 1j
    assert not bool(ctx.adapter.allclose(shifted, real_identity, atol=1e-12, rtol=0))


def test_vdot_flattens_conjugates_and_promotes(captured_backend):
    ctx = captured_backend
    a = np.array([[1+2j, 3-1j], [2j, 4]], dtype=np.complex128)
    b = np.array([[2, 1], [3, 5]], dtype=np.float64)
    actual = ctx.adapter.vdot(ctx.ops.from_numpy(a), ctx.ops.from_numpy(b))
    np.testing.assert_allclose(host(ctx, actual), np.vdot(a, b), atol=1e-12)


def test_allclose_scalar_rhs_consumed_by_tda(captured_backend):
    ctx = captured_backend
    assert bool(ctx.adapter.allclose(ctx.ops.array([1.]), 1))
    assert not bool(ctx.adapter.allclose(ctx.ops.array([2.]), 1))


def test_exposed_dot_high_rank_retains_numpy_contraction(captured_backend):
    ctx = captured_backend
    a = np.arange(24.).reshape(2, 3, 4)
    b = np.arange(40.).reshape(2, 4, 5)
    actual = ctx.adapter.dot(ctx.ops.from_numpy(a), ctx.ops.from_numpy(b))
    np.testing.assert_allclose(host(ctx, actual), np.dot(a, b))


def test_exposed_sort_returns_array(captured_backend):
    ctx = captured_backend
    source = np.array([[3., 1.], [2., -1.]])
    actual = ctx.adapter.sort(ctx.ops.from_numpy(source), axis=0)
    np.testing.assert_array_equal(host(ctx, actual), np.sort(source, axis=0))


@pytest.mark.parametrize('values', [[1., -2., 0.], [1+0j, 2+3j, -1j]])
def test_real_imag_iscomplex_consumed_array_semantics(captured_backend, values):
    ctx = captured_backend
    reference = np.asarray(values)
    candidate = ctx.ops.from_numpy(reference)
    for operation in ('real', 'imag', 'iscomplex'):
        actual = getattr(ctx.adapter, operation)(candidate)
        np.testing.assert_array_equal(host(ctx, actual), getattr(np, operation)(reference))


@pytest.mark.parametrize('axes', [[-1, 0], (-1, 0), [[2], [0]]])
def test_tensordot_axes_forms_consumed_by_tree_updates(captured_backend, axes):
    ctx = captured_backend
    a = np.arange(24., dtype=np.float64).reshape(2, 3, 4)
    b = np.arange(20., dtype=np.float64).reshape(4, 5).astype(np.complex128) * (1+1j)
    result = ctx.adapter.tensordot(ctx.ops.from_numpy(a), ctx.ops.from_numpy(b), axes=axes)
    np.testing.assert_allclose(host(ctx, result), np.tensordot(a, b, axes=axes), rtol=1e-12)


def test_diff_and_searchsorted_match_numpy(captured_backend):
    ctx = captured_backend
    t = ctx.ops.array([0., .1, .4, .9])
    np.testing.assert_allclose(host(ctx, ctx.adapter.diff(t)), [.1, .3, .5])
    query = ctx.ops.array([.9, .1, -.1, .3])
    for side in ('left', 'right'):
        result = ctx.adapter.searchsorted(t, query, side=side)
        np.testing.assert_array_equal(host(ctx, result), np.searchsorted([0., .1, .4, .9], [.9, .1, -.1, .3], side=side))


@pytest.mark.parametrize('size,shape', [(None, ()), (3, (3,)), ((2, 3), (2, 3))])
def test_legacy_rng_random_size_shape_seed(captured_backend, size, shape):
    ctx = captured_backend
    ctx.adapter.random.seed(19)
    first = ctx.adapter.random.random(size=size)
    ctx.adapter.random.seed(19)
    second = ctx.adapter.random.random(size=size)
    # NumPy scalar random() is legitimate legacy behavior; other adapters may
    # return native zero-dimensional arrays. Shape/values are the contract.
    if size is None and np.isscalar(first):
        assert ctx.adapter.name == 'numpy'
        left, right = np.asarray(first), np.asarray(second)
    else:
        left, right = host(ctx, first), host(ctx, second)
    assert left.shape == shape
    assert np.all((left >= 0) & (left < 1))
    np.testing.assert_array_equal(left, right)


def integrate(ctx, *, dense_output=False, t_eval=None, events=None):
    from renormalizer.lib.integrate.integrate import solve_ivp
    return solve_ivp(lambda t, y: -y, (0., .1), ctx.ops.array([1., 2.]),
                     method='RK45', rtol=1e-9, atol=1e-11,
                     dense_output=dense_output, t_eval=t_eval, events=events)


def assert_analytic(ctx, value, times):
    times = np.asarray(times)
    reference = np.array([1., 2.]) * np.exp(-times) if times.ndim == 0 else np.array([1., 2.])[:, None] * np.exp(-times)
    np.testing.assert_allclose(host(ctx, value), reference, rtol=1e-8, atol=1e-9)


def test_internal_ivp_endpoint(captured_backend):
    result = integrate(captured_backend)
    assert result.success
    assert_analytic(captured_backend, result.y[:, -1], .1)


def test_internal_ivp_t_eval(captured_backend):
    times = np.array([0., .025, .05, .1])
    result = integrate(captured_backend, t_eval=times)
    assert result.success
    assert_analytic(captured_backend, result.y, times)


@pytest.mark.parametrize('times', [.05, np.array([.08, .01, .06, .1, .01])], ids=['scalar', 'unsorted_repeated_vector'])
def test_internal_ivp_aggregate_dense_output(captured_backend, times):
    result = integrate(captured_backend, dense_output=True)
    assert result.success
    assert_analytic(captured_backend, result.sol(times), times)


def test_internal_ivp_terminal_event(captured_backend):
    # Event time is host metadata; no host conversion of a native state is
    # hidden inside the callback used as an oracle.
    def event(t, y):
        return t-.05
    event.terminal = True
    event.direction = 1.
    result = integrate(captured_backend, events=event)
    assert result.success and result.status == 1
    np.testing.assert_allclose(np.asarray(captured_backend.adapter.numpy(result.t_events[0])), [.05], atol=1e-9)
    assert_analytic(captured_backend, result.y[:, -1], .05)


def test_internal_ivp_aggregate_retains_creating_context(numerical_context):
    from renormalizer.backend.context import capture_backend, make_context, current_backend
    ctx = numerical_context
    previous = current_backend()
    with capture_backend(ctx.adapter):
        result = integrate(ctx, dense_output=True)
    assert current_backend() is previous
    alternate = make_context('numpy').adapter
    times = np.array([.08, .01, .1])
    with capture_backend(alternate):
        assert_analytic(ctx, result.sol(times), times)
        assert current_backend() is alternate
    assert current_backend() is previous


@pytest.mark.parametrize('mode', ['endpoint', 't_eval', 'dense'])
@pytest.mark.parametrize('scenario', ['backward', 'complex'])
def test_internal_ivp_backward_and_complex(captured_backend, scenario, mode):
    from renormalizer.lib.integrate.integrate import solve_ivp
    from renormalizer.backend.context import capture_backend, make_context, current_backend
    ctx = captured_backend
    start, stop = (.1, 0.) if scenario == 'backward' else (0., .1)
    initial = np.array([1., 2.]) if scenario == 'backward' else np.array([1+2j, 2-1j])
    sample = np.linspace(start, stop, 5)
    result = solve_ivp(lambda t, y: -y, (start, stop), ctx.ops.from_numpy(initial),
                       method='RK45', rtol=1e-9, atol=1e-11,
                       t_eval=sample if mode == 't_eval' else None,
                       dense_output=mode == 'dense')
    assert result.success
    if mode == 'endpoint':
        np.testing.assert_allclose(host(ctx, result.y[:, -1]),
                                   initial * np.exp(-(stop-start)), rtol=1e-8, atol=1e-9)
    elif mode == 't_eval':
        np.testing.assert_allclose(host(ctx, result.y),
                                   initial[:, None] * np.exp(-(sample-start)), rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(host(ctx, result.t), sample, atol=1e-14)
    else:
        queries = np.array([.08, .01, .06, .1, .01, 0.])
        alternate = make_context('numpy').adapter
        previous = current_backend()
        with capture_backend(alternate):
            values = result.sol(queries)
            np.testing.assert_allclose(host(ctx, values),
                                       initial[:, None] * np.exp(-(queries-start)), rtol=1e-8, atol=1e-9)
            assert current_backend() is alternate
        assert current_backend() is previous
