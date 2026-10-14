"""Shared backend vocabulary: error types, dtype domains and checks.

The strict array API built on it is a development and test tool and lives in
``renormalizer.backend.testing.strict``.
"""
import numpy as np


class CapabilityError(NotImplementedError):
    pass


class PrecisionError(ValueError):
    pass


class OwnershipError(TypeError):
    pass


NUMERIC_DTYPES = tuple(np.dtype(x) for x in ('float32', 'float64', 'complex64', 'complex128'))
AUXILIARY_DTYPES = tuple(np.dtype(x) for x in ('bool', 'int32', 'int64'))
_PROMOTION_ROWS = (
    ('float32', 'float64', 'complex64', 'complex128'),
    ('float64', 'float64', 'complex128', 'complex128'),
    ('complex64', 'complex128', 'complex64', 'complex128'),
    ('complex128', 'complex128', 'complex128', 'complex128'),
)
PROMOTION = {
    (a, b): np.dtype(_PROMOTION_ROWS[i][j])
    for i, a in enumerate(NUMERIC_DTYPES) for j, b in enumerate(NUMERIC_DTYPES)
}


def check_actual_dtype(actual, requested):
    if np.dtype(actual) != np.dtype(requested):
        raise PrecisionError(f'requested {requested}, received {actual}')


def check_solve_shapes(a, b):
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise CapabilityError('solve requires a two-dimensional square matrix')
    if b.ndim not in (1, 2) or b.shape[0] != a.shape[0]:
        raise ValueError('solve RHS must have shape (n,) or (n,k)')
