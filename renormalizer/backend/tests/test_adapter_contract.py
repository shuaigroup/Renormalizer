import importlib

import numpy as np
import pytest

from renormalizer.backend.tests.numerical_checks import check_qr, check_svd, check_eigh, check_solve


@pytest.fixture(params=['float32', 'float64'])
def ops(request):
    module = importlib.import_module('renormalizer.backend.context')
    return module.make_context('numpy', device='cpu', real_dtype=request.param).ops


def test_creation_default_precision_and_auxiliary_domain(ops):
    dtype = ops.context.real_dtype
    complex_dtype = ops.context.complex_dtype
    assert ops.array([1., 2.]).dtype == dtype
    assert ops.array([1+2j]).dtype == complex_dtype
    assert ops.zeros((2, 3)).dtype == dtype
    assert ops.ones((2,)).dtype == dtype
    np.testing.assert_array_equal(ops.eye(2, 3), np.eye(2, 3))
    assert ops.array([1, 2], dtype='int64').dtype == np.int64
    assert ops.array([True], dtype=bool).dtype == bool
    with pytest.raises(NotImplementedError, match='dtype'):
        ops.array(['text'])


def test_explicit_conversion_and_copy_contract(ops):
    x = np.arange(4., dtype=ops.context.real_dtype)
    assert ops.from_numpy(x, copy=False) is x
    assert ops.to_numpy(x, copy=False) is x
    y = ops.from_numpy(x, copy=True)
    assert not np.shares_memory(x, y)
    with pytest.raises(ValueError, match='copy=False'):
        ops.array([1., 2.], copy=False)
    with pytest.raises(TypeError, match='NumPy'):
        ops.from_numpy([1., 2.])


def test_shape_conjugation_and_cast(ops):
    x = ops.array([[1+2j, 3-4j], [5+0j, 6+1j]])
    before = x.copy()
    np.testing.assert_array_equal(ops.transpose(x), x.T)
    np.testing.assert_array_equal(ops.reshape(x, (4,)), x.reshape(4))
    np.testing.assert_array_equal(ops.conj(x), x.conj())
    np.testing.assert_array_equal(ops.real(x), x.real)
    np.testing.assert_array_equal(ops.imag(x), x.imag)
    assert ops.astype(x, 'complex128').dtype == np.complex128
    np.testing.assert_array_equal(x, before)


def test_binary_broadcast_promotion_and_scalar_results(ops):
    a = ops.array([[1], [2]], dtype='float64')
    b = ops.array([2+1j, 4-1j], dtype='complex64')
    for name, reference in [('add', np.add), ('subtract', np.subtract),
                            ('multiply', np.multiply), ('divide', np.divide)]:
        out = getattr(ops, name)(a, b)
        assert out.dtype == np.complex128 and out.shape == (2, 2)
        np.testing.assert_allclose(out, reference(a, b.astype('complex128')))
    x = ops.array([1., 2.])
    out = ops.einsum('i,i->', x, x)
    assert isinstance(out, np.ndarray) and out.shape == ()
    assert out.dtype == ops.context.real_dtype
    assert ops.scalar(out) == 5
    with pytest.raises(ValueError, match='scalar'):
        ops.scalar(x)


def test_reductions_and_empty_results(ops):
    x = ops.array([[1., 2.], [3., 4.]])
    np.testing.assert_array_equal(ops.sum(x, axis=0, keepdims=True), [[4., 6.]])
    assert ops.sum(x, dtype='float64').dtype == np.float64
    assert ops.max(x).shape == () and ops.max(x).item() == 4
    assert ops.min(x).shape == () and ops.min(x).item() == 1
    empty = ops.zeros((0, 3))
    assert ops.sum(empty).item() == 0
    with pytest.raises(ValueError):
        ops.max(empty)
    assert ops.norm(empty).item() == 0
    assert ops.norm(x).shape == ()
    np.testing.assert_allclose(ops.norm(x), np.linalg.norm(x), rtol=1e-6)
    with pytest.raises(NotImplementedError, match='norm'):
        ops.norm(x, ord=2)


def test_matmul_layouts_and_ownership(ops):
    a = ops.array(np.arange(24.).reshape(2, 3, 4))
    b = ops.array(np.arange(8.).reshape(4, 2))
    np.testing.assert_allclose(ops.matmul(a, b), a @ b)
    negative = a[:, :, ::-1]
    np.testing.assert_allclose(ops.matmul(negative, b), negative @ b)
    np.testing.assert_array_equal(ops.matmul(ops.zeros((2, 0)), ops.zeros((0, 3))), np.zeros((2,3)))
    with pytest.raises(TypeError, match='array'):
        ops.matmul([[1.]], np.ones((1,1)))
    with pytest.raises(NotImplementedError, match='dtype'):
        ops.matmul(np.ones((2,2), dtype=int), np.ones((2,2), dtype=int))
    with pytest.raises(ValueError, match='explicit output'):
        ops.einsum('ij,jk', ops.eye(2), ops.eye(2))


def test_decompositions_reconstruct_complex_inputs(ops):
    a = ops.array([[1+1j, 2], [3, 5-1j], [2j, 7]])
    tol = 1e-5 if a.dtype == np.complex64 else 1e-12
    q, r = ops.qr(a)
    check_qr(a, q, r)
    assert q.shape == (3, 2) and r.shape == (2, 2)
    assert q.dtype == a.dtype and r.dtype == a.dtype
    np.testing.assert_allclose(q@r, a, atol=tol, rtol=tol)
    np.testing.assert_allclose(q.conj().T@q, np.eye(2), atol=tol, rtol=tol)
    u, s, vh = ops.svd(a)
    check_svd(a, u, s, vh)
    assert u.shape == (3,2) and s.shape == (2,) and vh.shape == (2,2)
    assert s.dtype == ops.context.real_dtype
    np.testing.assert_allclose((u*s)@vh, a, atol=tol, rtol=tol)
    np.testing.assert_allclose(s, np.linalg.svd(a, compute_uv=False), atol=tol, rtol=tol)
    with pytest.raises(NotImplementedError):
        ops.qr(a, mode='complete')
    with pytest.raises(NotImplementedError):
        ops.svd(a, full_matrices=True)


@pytest.mark.parametrize('triangle', ['L', 'U'])
def test_eigh_triangle_semantics(ops, triangle):
    a = ops.array([[2+3j, 20+2j], [1+1j, 4+5j]])
    w, v = ops.eigh(a, UPLO=triangle)
    check_eigh(a, w, v, UPLO=triangle)
    tri = np.tril(a, -1) if triangle == 'L' else np.triu(a, 1)
    h = tri + tri.conj().T + np.diag(a.diagonal().real)
    tol = 1e-5 if a.dtype == np.complex64 else 1e-12
    assert w.dtype == ops.context.real_dtype
    np.testing.assert_allclose(h@v, v*w, atol=tol, rtol=tol)
    np.testing.assert_allclose(v.conj().T@v, np.eye(2), atol=tol, rtol=tol)
    np.testing.assert_allclose(w, np.linalg.eigvalsh(h), atol=tol, rtol=tol)


@pytest.mark.parametrize('rhs_shape', [(2,), (2, 3)])
def test_solve_vector_and_multirhs(ops, rhs_shape):
    a = ops.array([[3., 1.], [1., 2.]])
    b = ops.array(np.arange(np.prod(rhs_shape)).reshape(rhs_shape)+1.)
    x = ops.solve(a, b)
    check_solve(a, b, x)
    assert x.shape == b.shape and x.dtype == b.dtype
    np.testing.assert_allclose(a@x, b, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(x, np.linalg.solve(a,b), atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match='RHS'):
        ops.solve(a, ops.ones((3,)))
    with pytest.raises(NotImplementedError, match='square'):
        ops.solve(ops.ones((2, 2, 2)), b)
    with pytest.raises(np.linalg.LinAlgError):
        ops.solve(ops.zeros((2, 2)), b)


def test_functional_updates_repeat_and_do_not_mutate(ops):
    x = ops.array([1., 2., 3.])
    idx = np.array([1, 1])
    np.testing.assert_array_equal(ops.at_add(x, idx, ops.array([2., 3.])), [1,7,3])
    np.testing.assert_array_equal(ops.at_sub(x, idx, ops.array([2., 3.])), [1,-3,3])
    np.testing.assert_array_equal(ops.at_mul(x, idx, ops.array([2., 3.])), [1,12,3])
    np.testing.assert_array_equal(ops.at_set(x, 1, 5.), [1,5,3])
    np.testing.assert_array_equal(x, [1,2,3])
    with pytest.raises(NotImplementedError, match='duplicate'):
        ops.at_set(x, idx, ops.array([2., 3.]))
    with pytest.raises(NotImplementedError, match='index'):
        ops.at_add(x, np.array([True, False, True]), ops.array([1.,1.]))


def test_nonfinite_computation_rejected_without_mutation(ops):
    bad = ops.array([[np.nan]])
    with pytest.raises(ValueError, match='finite'):
        ops.solve(bad, ops.ones((1,)))
    with pytest.raises(ValueError, match='finite'):
        ops.matmul(bad, ops.ones((1,1)))
    assert np.isnan(bad[0,0])


@pytest.mark.parametrize('left', ['float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('right', ['float32', 'float64', 'complex64', 'complex128'])
def test_fixed_promotion_table(ops, left, right):
    expected = {
        'float32': ['float32', 'float64', 'complex64', 'complex128'],
        'float64': ['float64', 'float64', 'complex128', 'complex128'],
        'complex64': ['complex64', 'complex128', 'complex64', 'complex128'],
        'complex128': ['complex128', 'complex128', 'complex128', 'complex128'],
    }[left][['float32', 'float64', 'complex64', 'complex128'].index(right)]
    out = ops.add(ops.ones((1,), dtype=left), ops.ones((1,), dtype=right))
    assert out.dtype == np.dtype(expected)


def test_overlap_and_readonly_inputs_not_mutated(ops):
    base = ops.array([1.,2.,3.])
    overlap = np.lib.stride_tricks.as_strided(base, shape=(2,2), strides=(base.itemsize,base.itemsize))
    before = base.copy()
    result = ops.at_set(overlap, (0, 1), 99.)
    np.testing.assert_array_equal(base, before)
    assert result[0,1] == 99 and result[1,0] == 2
    base.flags.writeable = False
    result = ops.at_add(base, slice(None), 1.)
    np.testing.assert_array_equal(base, before)
    np.testing.assert_array_equal(result, before+1)


def test_scalar_promotion_uses_context(ops):
    x = ops.ones((1,))
    assert ops.add(x, 1.).dtype == ops.context.real_dtype
    assert ops.add(x, 1j).dtype == ops.context.complex_dtype


def test_capability_surface_does_not_proxy_namespace(ops):
    assert not hasattr(ops, 'fft')
    assert ops.capability('matmul', 'float64')['status'] == 'supported'
    assert ops.capability('fft', 'float64')['status'] == 'unsupported'
    assert ops.capability('matmul', 'int64')['status'] == 'unsupported'
    assert ops.capability('matmul', 'float64')['device'] == 'cpu'
    ops.sync()


def test_binary_errors_do_not_silently_produce_nonfinite(ops):
    with pytest.raises((ValueError, FloatingPointError), match='finite|zero|invalid'):
        ops.divide(ops.ones((1,)), ops.zeros((1,)))


def test_reduced_empty_decompositions(ops):
    a = ops.zeros((0, 3))
    q, r = ops.qr(a)
    check_qr(a, q, r)
    assert q.shape == (0, 0) and r.shape == (0, 3)
    u, s, vh = ops.svd(a)
    check_svd(a, u, s, vh)
    assert u.shape == (0, 0) and s.shape == (0,) and vh.shape == (0, 3)
    w, v = ops.eigh(ops.zeros((0,0)))
    assert w.shape == (0,) and v.shape == (0,0)


def test_auxiliary_capabilities_distinguish_creation_from_algebra(ops):
    assert ops.capability('array', 'int64')['status'] == 'supported'
    assert ops.capability('from_numpy', 'bool')['status'] == 'supported'
    assert ops.capability('matmul', 'int64')['status'] == 'unsupported'


@pytest.mark.parametrize('dtype', ['object', 'U2', 'int16'])
@pytest.mark.parametrize('operation', ['reshape', 'transpose', 'to_numpy', 'astype'])
def test_structural_operations_reject_unsupported_native_dtypes(ops, dtype, operation):
    source = np.array([[1, 2]], dtype=dtype)
    with pytest.raises(NotImplementedError, match='dtype'):
        if operation == 'reshape':
            ops.reshape(source, (2,))
        elif operation == 'astype':
            ops.astype(source, 'float64')
        else:
            getattr(ops, operation)(source)


@pytest.mark.parametrize('dtype', ['bool', 'int32', 'int64'])
def test_structural_operations_allow_auxiliary_dtypes(ops, dtype):
    source = np.array([[1, 0]], dtype=dtype)
    assert ops.reshape(source, (2,)).dtype == np.dtype(dtype)
    assert ops.transpose(source).dtype == np.dtype(dtype)
    assert ops.to_numpy(source, copy=False) is source
    np.testing.assert_array_equal(ops.astype(source, 'float64'), [[1., 0.]])


def test_structural_cast_rejects_object_before_user_conversion(ops):
    calls = []
    class Convertible:
        def __float__(self):
            calls.append(True)
            return 1.
    with pytest.raises(NotImplementedError, match='dtype'):
        ops.astype(np.array([Convertible()], dtype=object), 'float64')
    assert calls == []


def test_explicit_creation_can_convert_int16_to_supported_dtype(ops):
    source = np.array([1, 2], dtype='int16')
    converted = ops.array(source, dtype='float64')
    assert converted.dtype == np.float64
    np.testing.assert_array_equal(converted, [1., 2.])
