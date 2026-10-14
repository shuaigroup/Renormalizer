"""Trusted, CPU NumPy acceptance for backend tests, independent of M3 candidates.

These small-fixture defaults are fixed before execution. Larger, ill-conditioned
or differently scaled fixtures must supply separately frozen tolerances. Neither
this module nor its references dispatch through Renormalizer's selected backend.
"""
import numpy as np


def _finite(x):
    assert np.isfinite(x).all(), 'nonfinite validation data'


def _norm(x):
    values = np.abs(np.asarray(x)).ravel()
    _finite(values)
    if not values.size:
        return 0.0
    scale = values.max()
    if scale == 0:
        return 0.0
    result = float(scale * np.sqrt(np.sum((values / scale) ** 2)))
    assert np.isfinite(result), 'validation norm overflow'
    return result


def _maximum(x):
    return float(np.max(np.abs(x))) if x.size else 0.0


def check_error(actual, reference, *, atol=1e-12, rtol=1e-10,
                atol_max=None, rtol_max=None):
    """Check shape/dtype, then both Frobenius and maximum absolute error."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape, 'shape mismatch'
    assert actual.dtype == reference.dtype, 'dtype mismatch'
    _finite(actual)
    _finite(reference)
    assert np.isfinite([atol, rtol]).all() and atol >= 0 and rtol >= 0
    atol_max = atol if atol_max is None else atol_max
    rtol_max = rtol if rtol_max is None else rtol_max
    assert np.isfinite([atol_max, rtol_max]).all() and atol_max >= 0 and rtol_max >= 0
    try:
        # Widen before subtracting so the comparison itself adds no rounding.
        precision = np.clongdouble if actual.dtype.kind == 'c' else np.longdouble
        delta = actual.astype(precision) - reference.astype(precision)
        error, scale = _norm(delta), _norm(reference.astype(precision))
        max_error, max_scale = _maximum(delta), _maximum(reference.astype(precision))
        bound, max_bound = atol + rtol * scale, atol_max + rtol_max * max_scale
    except (FloatingPointError, OverflowError) as exc:
        raise AssertionError('validation arithmetic overflow') from exc
    assert np.isfinite([bound, max_bound, max_error, max_scale]).all(), 'validation arithmetic overflow'
    assert error <= bound, f'Frobenius error {error} > {bound}'
    assert max_error <= max_bound, f'maximum error {max_error} > {max_bound}'
    relative = error / scale if scale else (0.0 if error == 0 else float('inf'))
    return {'absolute_frobenius': error, 'relative_frobenius': relative,
            'maximum_error': max_error}


def _tolerance(a, atol, rtol):
    single = a.dtype in (np.dtype('float32'), np.dtype('complex64'))
    return (2e-5 if single else 1e-12) if atol is None else atol, (2e-5 if single else 1e-10) if rtol is None else rtol


def _orthonormal(columns, *, atol, rtol):
    k = columns.shape[1]
    gram = columns.conj().T @ columns
    reference = np.eye(k, dtype=columns.dtype)
    check_error(gram, reference, atol=atol, rtol=rtol)
    return _norm(gram - reference) / np.sqrt(k) if k else 0.0


def _clusters(spectrum, *, cluster_atol, cluster_rtol):
    start = 0
    for end in range(1, len(spectrum)):
        threshold = cluster_atol + cluster_rtol * max(abs(spectrum[end]), abs(spectrum[end-1]))
        if abs(spectrum[end] - spectrum[end-1]) > threshold:
            yield slice(start, end)
            start = end
    if len(spectrum):
        yield slice(start, len(spectrum))


def _subspaces(actual, reference, spectrum, *, atol, rtol, cluster_atol,
               cluster_rtol, zero_atol=None):
    metrics = []
    for cluster in _clusters(spectrum, cluster_atol=cluster_atol, cluster_rtol=cluster_rtol):
        # A reduced rectangular SVD need not span a unique partial zero-space.
        if zero_atol is not None and np.max(np.abs(spectrum[cluster])) <= zero_atol:
            continue
        a, b = actual[:, cluster], reference[:, cluster]
        d = a.shape[1]
        delta = a @ a.conj().T - b @ b.conj().T
        error = _norm(delta) / np.sqrt(d)
        assert error <= atol + rtol, 'spectral cluster projector mismatch'
        metrics.append({'dimension': d, 'projector_error': error})
    return metrics


def check_qr(a, q, r, *, atol=None, rtol=None):
    atol, rtol = _tolerance(a, atol, rtol)
    m, n = a.shape
    k = min(m, n)
    assert q.shape == (m, k) and r.shape == (k, n)
    assert q.dtype == a.dtype and r.dtype == a.dtype
    reconstruction = check_error(q @ r, a, atol=atol, rtol=rtol)
    orthogonality = _orthonormal(q, atol=atol, rtol=rtol)
    check_error(np.tril(r, -1), np.zeros_like(r), atol=atol, rtol=rtol)
    return {'reconstruction': reconstruction, 'orthogonality': orthogonality}


def check_svd(a, u, s, vh, *, atol=None, rtol=None,
              cluster_atol=1e-12, cluster_rtol=1e-8, zero_atol=1e-12):
    atol, rtol = _tolerance(a, atol, rtol)
    m, n = a.shape
    k = min(m, n)
    assert u.shape == (m,k) and s.shape == (k,) and vh.shape == (k,n)
    assert u.dtype == a.dtype and vh.dtype == a.dtype and s.dtype == a.real.dtype
    _finite(s)
    assert np.all(s >= 0) and np.all(s[:-1] >= s[1:]), 'singular values must be nonnegative descending'
    reference_u, reference_s, reference_vh = np.linalg.svd(a, full_matrices=False)
    reconstruction = check_error((u*s) @ vh, a, atol=atol, rtol=rtol)
    _orthonormal(u, atol=atol, rtol=rtol)
    _orthonormal(vh.conj().T, atol=atol, rtol=rtol)
    check_error(s, reference_s, atol=atol, rtol=rtol)
    left = _subspaces(u, reference_u, reference_s, atol=atol, rtol=rtol,
                      cluster_atol=cluster_atol, cluster_rtol=cluster_rtol, zero_atol=zero_atol)
    right = _subspaces(vh.conj().T, reference_vh.conj().T, reference_s,
                       atol=atol, rtol=rtol, cluster_atol=cluster_atol,
                       cluster_rtol=cluster_rtol, zero_atol=zero_atol)
    return {'reconstruction': reconstruction, 'left_clusters': left, 'right_clusters': right}


def check_eigh(a, w, v, *, UPLO='L', atol=None, rtol=None,
               cluster_atol=1e-12, cluster_rtol=1e-8):
    atol, rtol = _tolerance(a, atol, rtol)
    n = a.shape[0]
    assert a.shape == (n,n) and w.shape == (n,) and v.shape == (n,n)
    assert w.dtype == a.real.dtype and v.dtype == a.dtype
    assert UPLO in ('L', 'U')
    _finite(a)
    triangle = np.tril(a, -1) if UPLO == 'L' else np.triu(a, 1)
    hermitian = triangle + triangle.conj().T + np.diag(a.diagonal().real).astype(a.dtype)
    reference_w, reference_v = np.linalg.eigh(hermitian)
    _finite(w)
    assert np.all(w[:-1] <= w[1:]), 'eigenvalues must be ascending'
    check_error(w, reference_w, atol=atol, rtol=rtol)
    residual = check_error(hermitian @ v, v*w, atol=atol, rtol=rtol)
    matrix_norm = _norm(hermitian)
    error = residual['absolute_frobenius']
    residual['relative_frobenius'] = error / matrix_norm if matrix_norm else (0.0 if error == 0 else float('inf'))
    residual['reference_norm'] = matrix_norm
    orthogonality = _orthonormal(v, atol=atol, rtol=rtol)
    clusters = _subspaces(v, reference_v, reference_w, atol=atol, rtol=rtol,
                          cluster_atol=cluster_atol, cluster_rtol=cluster_rtol)
    return {'residual': residual, 'orthogonality': orthogonality, 'clusters': clusters}


def check_solve(a, b, x, *, atol=None, rtol=None, backward_tol=None):
    """For these well-conditioned fixtures, require forward and every-RHS error."""
    atol, rtol = _tolerance(a, atol, rtol)
    backward_tol = (2e-5 if a.dtype in (np.dtype('float32'), np.dtype('complex64')) else 1e-12) if backward_tol is None else backward_tol
    assert x.shape == b.shape and x.dtype == np.result_type(a.dtype, b.dtype)
    _finite(a)
    _finite(b)
    _finite(x)
    reference = np.linalg.solve(a, b)
    forward = check_error(x, reference, atol=atol, rtol=rtol)
    # Widen residual arithmetic independently of backend result dtype.
    dtype = np.complex128 if a.dtype.kind == 'c' or b.dtype.kind == 'c' else np.float64
    aa = a.astype(dtype)
    bb = b.astype(dtype)
    xx = x.astype(dtype)
    if b.ndim == 1:
        bb, xx = bb[:, None], xx[:, None]
    a_norm = float(np.linalg.norm(aa, ord=2)) if aa.size else 0.0
    errors = []
    for j in range(bb.shape[1]):
        numerator = _norm(aa @ xx[:,j] - bb[:,j])
        denominator = a_norm * _norm(xx[:,j]) + _norm(bb[:,j])
        assert np.isfinite([numerator, denominator]).all(), 'backward-error overflow'
        error = numerator / denominator if denominator else (0.0 if numerator == 0 else float('inf'))
        assert error <= backward_tol, f'RHS {j} backward error {error}'
        errors.append(error)
    return {'forward': forward, 'per_rhs_backward_error': errors,
            'condition_number': float(np.linalg.cond(aa)) if aa.size else None}
