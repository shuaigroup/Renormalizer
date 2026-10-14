"""Explicit CPU solver boundaries with a logical transfer ledger.

Conversions go through the context's adapter and check ownership, the dtype
domain, and that dtype and shape survive the transfer.
Records count logical array bytes crossing each boundary, not physical transfers,
allocator peaks, or extra copies inside third-party solvers. NumPy boundaries are
host_to_host, including zero-copy views. Scalar solver metadata remains on host.
"""
from math import prod
from uuid import uuid4

import numpy as np

from renormalizer.backend.contracts import (CapabilityError, OwnershipError, NUMERIC_DTYPES,
                                            AUXILIARY_DTYPES, check_actual_dtype)

_DTYPES = NUMERIC_DTYPES + AUXILIARY_DTYPES


def _authorize(context):
    if context.host_policy != 'explicit':
        raise CapabilityError('host solver requires explicit policy')


def _dtype(array):
    # Torch exposes torch.dtype rather than a NumPy dtype object.
    dtype = array.dtype
    return dtype if isinstance(dtype, np.dtype) else np.dtype(str(dtype).removeprefix('torch.'))


def _check_owned(context, array):
    adapter = context.adapter
    owned = isinstance(array, np.ndarray) if adapter.name == 'numpy' else adapter.owns(array)
    if not owned:
        raise OwnershipError('array ownership/device mismatch; use explicit conversion')


def _identifier(operation_id):
    return uuid4().hex if operation_id is None else operation_id


def _record(ledger, *, source, target, shape, dtype, reason, operation_id):
    if ledger is None:
        return
    source_host = str(source).split(':')[0].lower() == 'cpu'
    target_host = str(target).split(':')[0].lower() == 'cpu'
    if source_host and target_host:
        direction = 'host_to_host'
    elif source_host:
        direction = 'H2D'
    elif target_host:
        direction = 'D2H'
    else:
        direction = 'D2D'
    ledger.append(dict(direction=direction, logical_bytes=prod(shape) * dtype.itemsize,
                       source_device=str(source), target_device=str(target),
                       reason=reason, operation_id=operation_id))


def to_host(array, *, context, ledger, reason, operation_id=None):
    """Download one backend array under explicit policy; preserve its dtype.

    All four public helpers take the same keyword-only context/ledger/reason;
    operation_id is optional, and callers can share it across related boundaries.
    ledger must provide append(dict), or be None to disable recording.
    Conversion failures are never retried.
    """
    _authorize(context)
    operation_id = _identifier(operation_id) if ledger is not None else operation_id
    _check_owned(context, array)
    dtype = _dtype(array)
    if dtype not in _DTYPES:
        raise CapabilityError(f'unsupported input dtype {dtype}')
    host = context.adapter.to_numpy(array)
    if not isinstance(host, np.ndarray):
        raise TypeError('to_numpy must return a NumPy array')
    if ledger is not None:
        _record(ledger, source=context.device, target='cpu', shape=array.shape,
                dtype=dtype, reason=reason, operation_id=operation_id)
    check_actual_dtype(host.dtype, dtype)
    if host.shape != tuple(array.shape):
        raise ValueError('host conversion changed array shape')
    return host


def from_host(array, *, context, ledger, reason, operation_id=None):
    """Upload one host numeric array, without applying context default dtype."""
    _authorize(context)
    if not isinstance(array, np.ndarray) or array.dtype.kind not in 'biufc':
        raise CapabilityError('host solver arrays must have numeric dtypes')
    if array.dtype not in _DTYPES:
        raise CapabilityError(f'unsupported dtype {array.dtype}')
    operation_id = _identifier(operation_id) if ledger is not None else operation_id
    result = context.adapter.array(array, dtype=array.dtype, copy=None)
    if context.adapter.name != 'numpy' and not context.adapter.owns(result):
        raise OwnershipError('conversion returned wrong device')
    if ledger is not None:
        _record(ledger, source='cpu', target=context.device, shape=array.shape,
                dtype=array.dtype, reason=reason, operation_id=operation_id)
    check_actual_dtype(_dtype(result), array.dtype)
    if tuple(result.shape) != array.shape:
        raise ValueError('device conversion changed array shape')
    return result


def _upload_result(value, *, context, ledger, reason, operation_id):
    if isinstance(value, np.ndarray):
        return from_host(value, context=context, ledger=ledger, reason=reason,
                         operation_id=operation_id)
    if isinstance(value, (tuple, list, dict)):
        def convert(item):
            return _upload_result(item, context=context, ledger=ledger, reason=reason,
                                  operation_id=operation_id)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        items = [convert(item) for item in value]
        if isinstance(value, tuple):
            return type(value)(*items) if hasattr(value, '_fields') else tuple(items)
        return items
    # Python and NumPy scalars are metadata; do not manufacture device scalars.
    return value


def call_host_solver(solver, arrays, *, context, ledger, reason, operation_id=None):
    """Call a CPU solver once; upload array leaves of its returned containers.

    arrays is an iterable of backend arrays passed as positional arguments.
    Bind solver configuration with functools.partial or a closure. Array-valued
    metadata should be extracted by that closure if it must remain on host.
    """
    _authorize(context)
    operation_id = _identifier(operation_id) if ledger is not None else operation_id
    host_arrays = [to_host(array, context=context, ledger=ledger, reason=reason,
                           operation_id=operation_id) for array in arrays]
    result = solver(*host_arrays)
    return _upload_result(result, context=context, ledger=ledger, reason=reason,
                          operation_id=operation_id)


def wrap_host_callback(callback, *, context, ledger, reason, operation_id=None):
    """Adapt backend matvec/matmat to SciPy/primme's host-array callback.

    A single operation ID groups all iterations. Both construction and invocation
    check policy; a forbidden callback is rejected before any solver is launched.
    The callback receives an array on the captured context's device and must
    return one backend array. Matrix inputs support block iterative solvers.
    """
    _authorize(context)
    operation_id = _identifier(operation_id) if ledger is not None else operation_id

    def host_callback(array):
        device_array = from_host(array, context=context, ledger=ledger, reason=reason,
                                 operation_id=operation_id)
        result = callback(device_array)
        return to_host(result, context=context, ledger=ledger, reason=reason,
                       operation_id=operation_id)

    return host_callback
