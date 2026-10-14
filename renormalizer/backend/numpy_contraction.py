# -*- coding: utf-8 -*-

"""NumPy contraction functions with ``tensordot`` bookkeeping cached by shape.

``numpy.tensordot`` re-derives its transpose order and matrix shapes in Python
on every call: microseconds that dominate the small contractions of MPS
sweeps, which repeat a handful of shapes per call site millions of times.
:func:`tensordot` caches that derivation per (shapes, axes) and then performs
exactly NumPy's operations (transpose, reshape, ``dot``, reshape), so results
are bitwise identical. Anything it does not recognise goes to NumPy unchanged,
errors included, with one exception: axes are validated when a plan is first
built, so non-integer axes equal to integers (``1.0``), which NumPy rejects,
reuse a plan already cached for the integer axes. Set ``_STRICT_AXES`` below to
True during development to check every axis on every call, at a small cost.

The module also serves as an opt_einsum backend, which resolves ``tensordot``,
``transpose`` and ``einsum`` by name from it.
"""

import math
import operator
from functools import lru_cache

import numpy as np

einsum = np.einsum
transpose = np.transpose


def tensordot(a, b, axes=2):
    if type(a) is not np.ndarray or type(b) is not np.ndarray:
        return np.tensordot(a, b, axes)
    try:
        plan = _plan(a.shape, b.shape, _axes_key(axes))
    except (TypeError, ValueError, IndexError):
        # Unhashable, unsupported or invalid axes: NumPy decides, and raises.
        return np.tensordot(a, b, axes)
    axes_a, shape_a, axes_b, shape_b, shape_out = plan
    at = a.transpose(axes_a).reshape(shape_a)
    bt = b.transpose(axes_b).reshape(shape_b)
    return np.dot(at, bt).reshape(shape_out)


_SEQUENCES = (list, tuple, range)


def _fast_axes_key(axes):
    # Hashable stand-in for axes, built without checking each axis (_plan checks on
    # a cache miss). As in numpy, a sized part is a list of axes, anything else one
    # axis; only sequences that iteration does not consume are taken apart.
    if type(axes) is int:
        return axes
    if not isinstance(axes, _SEQUENCES):
        return operator.index(axes)
    axes_a, axes_b = axes
    return (tuple(axes_a) if isinstance(axes_a, _SEQUENCES) else (axes_a,),
            tuple(axes_b) if isinstance(axes_b, _SEQUENCES) else (axes_b,))


def _strict_axes_key(axes):
    # As _fast_axes_key, but every axis passes operator.index, so axes NumPy
    # rejects raise TypeError whatever the cache holds.
    if type(axes) is int:
        return axes
    if not isinstance(axes, _SEQUENCES):
        return operator.index(axes)
    axes_a, axes_b = axes
    index = operator.index
    return (tuple([index(x) for x in axes_a]) if isinstance(axes_a, _SEQUENCES) else (index(axes_a),),
            tuple([index(x) for x in axes_b]) if isinstance(axes_b, _SEQUENCES) else (index(axes_b),))


# Development switch: check every axis on every call.
_STRICT_AXES = False
_axes_key = _strict_axes_key if _STRICT_AXES else _fast_axes_key


@lru_cache(maxsize=8192)
def _plan(as_, bs, axes):
    # numpy.tensordot's bookkeeping, step for step.
    if isinstance(axes, int):
        axes_a = list(range(-axes, 0))
        axes_b = list(range(axes))
    else:
        axes_a = [operator.index(axis) for axis in axes[0]]
        axes_b = [operator.index(axis) for axis in axes[1]]
    if len(set(axes_a)) != len(axes_a) or len(set(axes_b)) != len(axes_b):
        raise ValueError("duplicate axes")
    nda, ndb = len(as_), len(bs)
    if len(axes_a) != len(axes_b):
        raise ValueError("shape-mismatch for sum")
    for k in range(len(axes_a)):
        if as_[axes_a[k]] != bs[axes_b[k]]:
            raise ValueError("shape-mismatch for sum")
        if axes_a[k] < 0:
            axes_a[k] += nda
        if axes_b[k] < 0:
            axes_b[k] += ndb

    notin_a = [k for k in range(nda) if k not in axes_a]
    notin_b = [k for k in range(ndb) if k not in axes_b]
    shape_a = (math.prod(as_[k] for k in notin_a), math.prod(as_[k] for k in axes_a))
    shape_b = (math.prod(bs[k] for k in axes_b), math.prod(bs[k] for k in notin_b))
    shape_out = tuple(as_[k] for k in notin_a) + tuple(bs[k] for k in notin_b)
    return tuple(notin_a + axes_a), shape_a, tuple(axes_b + notin_b), shape_b, shape_out
