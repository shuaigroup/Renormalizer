import importlib

import numpy as np
import pytest


def checks():
    return importlib.import_module('renormalizer.backend.tests.numerical_checks')


def test_dual_error_empty_zero_and_dtype_rules():
    c = checks()
    c.check_error(np.zeros((0,)), np.zeros((0,)), atol=0, rtol=0)
    c.check_error(np.array([1e-14]), np.array([0.]), atol=1e-12, rtol=0)
    with pytest.raises(AssertionError):
        c.check_error(np.array([np.nan]), np.array([0.]))
    with pytest.raises(AssertionError):
        c.check_error(np.array([1.], dtype='float32'), np.array([1.], dtype='float64'))
    with pytest.raises(AssertionError):
        c.check_error(np.array([1e-4, 0.]), np.zeros(2), atol=1e-5, rtol=0)


def test_decomposition_checks_reject_corrupt_outputs():
    c = checks()
    a = np.array([[1+1j, 2], [3, 4j], [5, 6]], dtype='complex128')
    q, r = np.linalg.qr(a)
    c.check_qr(a, q, r)
    with pytest.raises(AssertionError):
        c.check_qr(a, q, r+1)
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    c.check_svd(a, u, s, vh)
    with pytest.raises(AssertionError):
        c.check_svd(a, u, s, vh.conj())
    h = a.conj().T @ a
    w, v = np.linalg.eigh(h)
    c.check_eigh(h, w, v)
    with pytest.raises(AssertionError):
        c.check_eigh(h, w+1, v)


def test_degenerate_projectors_allow_basis_rotation():
    c = checks()
    a = np.diag([1., 1., 4.])
    w, v = np.linalg.eigh(a)
    v[:, :2] = v[:, :2] @ np.array([[0., -1.], [1., 0.]])
    c.check_eigh(a, w, v)
    u, s, vh = np.linalg.svd(a)
    c.check_svd(a, u, s, vh)


def test_each_rhs_is_checked_independently():
    c = checks()
    a = np.eye(2)
    b = np.array([[1e10, 1.], [1e10, 1.]])
    c.check_solve(a, b, b.copy())
    bad = b.copy()
    bad[:, 1] += 1e-3
    with pytest.raises(AssertionError):
        c.check_solve(a, b, bad)


@pytest.mark.parametrize('a', [np.array([[1., np.nan], [0., 2.]]),
    np.array([[complex(1., float('inf')), 0.], [0., 2.]])])
def test_eigh_rejects_nonfinite_unused_triangle_and_imaginary_diagonal(a):
    with pytest.raises(AssertionError, match='nonfinite'):
        checks().check_eigh(a, np.array([1.,2.]), np.eye(2, dtype=a.dtype))


@pytest.mark.parametrize('shape', [(0,), (0,3)])
def test_empty_solve_check_accepts_legal_empty_rhs(shape):
    result = checks().check_solve(np.zeros((0,0)), np.zeros(shape), np.zeros(shape))
    assert result['per_rhs_backward_error'] == ([0.] if len(shape) == 1 else [0.,0.,0.])
    assert result['condition_number'] is None


def test_eigh_relative_residual_uses_effective_matrix_norm():
    a = np.diag([2.,4.])
    w = np.array([2.,4.])+1e-7
    v = np.eye(2)*(1+1e-7)
    result = checks().check_eigh(a, w, v, atol=1e-5, rtol=1e-5)
    expected = np.linalg.norm(a@v-v*w)/np.linalg.norm(a)
    assert result['residual']['relative_frobenius'] == pytest.approx(expected, rel=1e-14, abs=0)


@pytest.mark.parametrize('dtype',['int64','uint64','float16','float32','float64','complex64','complex128'])
def test_shared_arithmetic_preserves_legacy_numeric_domain(dtype):
    value=np.array([1,2],dtype=dtype)
    checks().check_error(value,value,atol=np.float32(0),rtol=np.float64(0))


def test_test_helper_keeps_extended_tolerance_bounds():
    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip('platform has no extended exponent range')
    # Original helper accepts finite longdouble bounds beyond binary64.
    large=np.longdouble('1e400')
    checks().check_error(np.ones(1),np.zeros(1),atol=large,rtol=0)
