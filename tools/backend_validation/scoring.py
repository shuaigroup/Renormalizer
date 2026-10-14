"""Trusted array scoring; this module never imports candidate implementations."""
import math
import numpy as np

TOLERANCE_KEYS = frozenset(('atol_F', 'rtol_F', 'atol_max', 'rtol_max'))


def validate_tolerances(tolerances):
    if not isinstance(tolerances, dict) or set(tolerances) != TOLERANCE_KEYS:
        raise ValueError('exact four frozen tolerances required')
    try:
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
               for x in tolerances.values()):
            raise ValueError('tolerances must be finite nonnegative numbers')
    except OverflowError as error:
        raise ValueError('tolerances must be finite nonnegative numbers') from error


def relative_diagnostic(error, scale):
    return error / scale if scale else (0.0 if error == 0 else 'infinity')


def _norm(array):
    values = np.abs(array).ravel()
    if not values.size:
        return 0.0
    scale = values.max()
    result = 0.0 if scale == 0 else float(scale * np.sqrt(np.sum((values / scale) ** 2)))
    if not math.isfinite(result):
        raise FloatingPointError('verification norm overflow')
    return result


def _maximum(array):
    result = float(np.max(np.abs(array))) if array.size else 0.0
    if not math.isfinite(result):
        raise FloatingPointError('verification maximum overflow')
    return result


def error_metrics(candidate, reference, *, atol_F, rtol_F, atol_max, rtol_max):
    """Pure CPU arithmetic after the caller validates its input domain.

    Preserve integer bits by widening directly to longdouble, never through
    float64. Bounds retain the caller's scalar arithmetic: the candidate
    scorer requires finite binary64 bounds, whereas legacy test helpers also
    permit finite extended-precision tolerance scalars. Policy stays outside
    this helper so sharing arithmetic does not narrow either public contract.
    """
    precision = np.clongdouble if candidate.dtype.kind == 'c' else np.longdouble
    a, b = candidate.astype(precision), reference.astype(precision)
    delta = a - b
    error, scale = _norm(delta), _norm(b)
    maximum, max_scale = _maximum(delta), _maximum(b)
    return {'absolute_frobenius': error, 'reference_frobenius': scale,
            'maximum_error': maximum, 'reference_maximum': max_scale,
            'frobenius_bound': atol_F + rtol_F * scale,
            'maximum_bound': atol_max + rtol_max * max_scale,
            'verification_precision': np.dtype(precision).name}


def score_array(candidate, reference, *, tolerances):
    """Use float64/complex128 output domain and widened acceptance arithmetic.

    Float32/complex64 are also accepted without casting their original metadata.
    Reductions use platform longdouble; final metrics and acceptance bounds must
    fit finite binary64. This fixed verifier policy rejects metric overflow.
    """
    validate_tolerances(tolerances)
    def fail(reason):
        return {'status': 'fail', 'reason': reason, 'metrics': {}}
    if not isinstance(candidate, np.ndarray) or not isinstance(reference, np.ndarray):
        return fail('array_required')
    if candidate.shape != reference.shape:
        return fail('shape_mismatch')
    if candidate.dtype != reference.dtype:
        return fail('dtype_mismatch')
    if candidate.dtype not in tuple(np.dtype(x) for x in ('float32','float64','complex64','complex128')):
        return fail('unsupported_dtype')
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        return fail('nonfinite')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            metrics = error_metrics(candidate, reference, **tolerances)
            error, scale = metrics['absolute_frobenius'], metrics['reference_frobenius']
            maximum = metrics['maximum_error']
            bound, max_bound = metrics['frobenius_bound'], metrics['maximum_bound']
            if not math.isfinite(bound) or not math.isfinite(max_bound):
                raise FloatingPointError('verification bound overflow')
    except (FloatingPointError, OverflowError):
        return fail('verification_overflow')
    relative = relative_diagnostic(error, scale)
    if isinstance(relative, float) and not math.isfinite(relative):
        relative = 'infinity'
    return {'status': 'pass' if error <= bound and maximum <= max_bound else 'fail',
            'reason': 'accepted' if error <= bound and maximum <= max_bound else 'numerical_error',
            'metrics': {key: value for key, value in
                        dict(metrics, relative_frobenius=relative).items()
                        if key != 'reference_maximum'}}
