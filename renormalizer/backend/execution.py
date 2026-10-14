"""Captured hybrid algorithm calls and observable contraction boundaries.

Records describe logical array transfers, not allocator peaks or all library
workspace. Existing Matrix/tree storage and scientific solvers remain on host.
Recording itself lives in ``renormalizer.backend.testing``; this module only
keeps the hooks it calls, and only while a recording is active.
"""
from contextvars import ContextVar
from functools import lru_cache, wraps
from typing import NamedTuple
from math import prod
import numpy as np
import opt_einsum as oe
from .context import capture_backend, current_backend
from .contracts import CapabilityError, NUMERIC_DTYPES, AUXILIARY_DTYPES

_HOST_DTYPES = frozenset(NUMERIC_DTYPES + AUXILIARY_DTYPES)

class _Run(NamedTuple):
    """Explicit numerical context and/or execution ledger of the current call."""
    context: object
    ledger: object


# None in plain runs: hot paths read this once and skip every context and
# recording branch.
_run = ContextVar('renormalizer_execution_run', default=None)


def _set_run(context, ledger):
    return _run.set(None if context is None and ledger is None else _Run(context, ledger))


def bind_backend(function):
    @wraps(function)
    def wrapped(*args, backend_context=None, **kwargs):
        run = _run.get()
        context = backend_context or (None if run is None else run.context)
        if context is not None and context.host_policy != 'explicit':
            raise CapabilityError('legacy hybrid algorithm requires explicit host policy')
        selected = context.adapter if context is not None else current_backend()
        token = _set_run(context, None if run is None else run.ledger)
        try:
            # The device scope matters for CuPy, which launches every kernel on
            # the process-wide current device: an explicitly selected device
            # must govern the whole algorithm, not only adapter calls.
            with capture_backend(selected), selected.device_scope():
                return function(*args, **kwargs)
        finally:
            _run.reset(token)
    return wrapped


def _device(array):
    device = getattr(array, 'device', 'cpu')
    if callable(device):
        device = device()
    platform = getattr(device, 'platform', None)
    if platform == 'cpu':
        return 'cpu'
    if platform in ('gpu', 'cuda'):
        return f'cuda:{device.id}'
    if hasattr(device, 'id') and type(array).__module__.startswith('cupy'):
        return f'cuda:{device.id}'
    return str(device)


def _record_transfer(source, result, reason):
    # Recording hook; a no-op unless a ledger is active.
    run = _run.get()
    if run is not None and run.ledger is not None and source is not result:
        run.ledger.record_transfer(source, result, reason)


def to_backend(array):
    selected = current_backend()
    run = _run.get()
    # NumPy hot path: uploading a host ndarray to the NumPy adapter is the
    # identity, and without a ledger there is nothing to record. DMRG/TDVP issue
    # tens of thousands of small contractions, so skip the conversion chain.
    # Context uploads keep from_host's dtype domain; others take the slow path.
    if type(array) is np.ndarray and selected.name == 'numpy' and (run is None or (
            run.ledger is None and (run.context is None or array.dtype in _HOST_DTYPES))):
        return array
    context, ledger = (None, None) if run is None else run
    if context is not None and isinstance(array, np.ndarray):
        from .host_solver import from_host
        return from_host(array, context=context, ledger=None if ledger is None else ledger.transfers,
                         reason='algorithm operand', operation_id=0 if ledger is None else len(ledger.operations))
    result = selected.from_numpy(array) if isinstance(array,np.ndarray) else selected.asarray(array)
    _record_transfer(array,result,'algorithm operand')
    return result


def to_host(array):
    # Every adapter's numpy() returns host ndarrays unchanged, and an unchanged
    # array is never recorded as a transfer.
    if type(array) is np.ndarray:
        return array
    run = _run.get()
    context, ledger = (None, None) if run is None else run
    if context is not None and not isinstance(array, np.ndarray):
        from .host_solver import to_host as download
        return download(array, context=context, ledger=None if ledger is None else ledger.transfers,
                        reason='host storage or solver boundary', operation_id=0 if ledger is None else len(ledger.operations))
    result = current_backend().numpy(array)
    _record_transfer(array,result,'host storage or solver boundary')
    return result


def _witness(result):
    # Recording hook, called only while a ledger is active.
    run = _run.get()
    if run is not None and run.ledger is not None:
        run.ledger.record_contraction(current_backend(), result)


def _contraction_dtype(values, adapter):
    from .contracts import PROMOTION
    arrays = [a for a in values if hasattr(a, 'shape') and hasattr(a, 'dtype')]
    if not arrays:
        return None
    dtype = adapter.dtype_of(arrays[0])
    for a in arrays[1:]:
        other = adapter.dtype_of(a)
        promoted = PROMOTION.get((dtype, other))
        dtype = np.result_type(dtype, other) if promoted is None else promoted
    return dtype


def _promote_operands(values, adapter, dtype=None):
    if adapter.name != 'torch':
        return tuple(values)
    dtype = _contraction_dtype(values, adapter) if dtype is None else dtype
    if dtype is None:
        return tuple(values)
    result = []
    for a in values:
        if hasattr(a, 'shape') and hasattr(a, 'dtype'):
            b = a if adapter.dtype_of(a) == dtype else adapter.asarray(a, dtype=dtype)
            _record_transfer(a, b, 'explicit contraction dtype promotion')
            result.append(b)
        else:
            result.append(a)
    return tuple(result)


def contract(*args, **kwargs):
    selected = current_backend()
    # Interleaved labels and shapes are metadata, never converted to tensors.
    # to_backend resolves the backend again for each operand; that repetition is
    # negligible at any size. For small tensors (bond dimension up to about 8) a
    # contraction is cheap enough that the dispatch in this function as a whole,
    # and opt_einsum's per-step overhead, become a visible share of each call.
    args = tuple(to_backend(a) if hasattr(a,'shape') and hasattr(a,'dtype') else a for a in args)
    args = _promote_operands(args, selected)
    kwargs['backend'] = selected.opt_einsum_module
    run = _run.get()
    context = None if run is None else run.context
    if (selected.name == 'jax' and hasattr(selected, 'fused_einsum')
            and args and isinstance(args[0], str) and all(hasattr(a, 'shape') for a in args[1:])
            and set(kwargs) <= {'backend', 'optimize'} and isinstance(kwargs.get('optimize', 'auto'), str)
            and (context is None or context.operators.policy == 'builtin')):
        # Same single-computation evaluation as _fused_expression, path cached by shape.
        shapes = tuple(tuple(map(int, a.shape)) for a in args[1:])
        path = _cached_path(args[0], shapes, (('optimize', kwargs.get('optimize', 'auto')),))
        result = selected.fused_einsum(args[0], path)(*args[1:])
    elif context is None or context.operators.policy == 'builtin':
        cached = _host_expression(selected, args, kwargs)
        if cached is None:
            result = oe.contract(*args, **kwargs)
        else:
            expression, operands = cached
            result = expression(*operands, backend=kwargs['backend'])
    else:
        from .operators import dispatch
        result = dispatch(context, 'contract', oe.contract, args, kwargs)
    if run is not None and run.ledger is not None:
        _witness(result)
    return result


def _host_expression(selected, args, kwargs):
    # oe.contract searches the contraction path on every call; sweeps repeat the
    # same subscripts and shapes many times, and the search depends only on those
    # and the options. A cached expression holds the same contraction list, so the
    # same pairwise operations run. Returns None for calls of any other form.
    #
    # A ValueError raised inside the contraction arrives reworded by
    # ContractExpression ("Internal error while evaluating ..."), same type.
    if not (selected.name == 'numpy' and args and set(kwargs) == {'backend', 'optimize'}
            and isinstance(kwargs['optimize'], str)):
        return None
    if isinstance(args[0], str):
        subscripts, operands = args[0], args[1:]
    else:
        # Interleaved (operand, labels, ..., [output labels]), as tree networks
        # use: converted with opt_einsum's own mapping on every call (see
        # _interleaved_subscripts for why the conversion is not cached).
        try:
            subscripts, operands = oe.parser.convert_interleaved_input(args)
        except TypeError:  # labels opt_einsum cannot map either
            return None
    if not all(type(a) is np.ndarray for a in operands):
        return None
    return _cached_expression(subscripts, tuple(a.shape for a in operands), kwargs['optimize']), operands


@lru_cache(maxsize=4096)
def _interleaved_subscripts(labels, output):
    """Cached interleaved-label conversion. Kept as a record; not called.

    Tried as a cache in front of opt_einsum's conversion (labels mapped in
    sorted order), keyed on the raw labels. Conclusions from trying it on the
    tree-network examples:

    * A hit is not free: building and hashing the key from large labels costs
      about half a conversion. The cache pays off only when well over half of
      the calls hit; below that it is slower than converting every call.
    * Tree networks put object ids in their labels (``str(id(ttns))``) and
      create new objects every time step, so the same contraction rarely
      repeats its labels. Hits stay low and the cache is a net loss.
    * With the ids taken out, most examples hit almost always, but some still
      have more distinct label structures than the cache holds. Stripping ids
      at call time costs more than it saves; it would have to happen where the
      labels are made (``tn/tree.py``).
    * Even at best the gain is small, because the conversion itself is a small
      part of the run. The cached expression keyed on the converted subscripts
      already removes the expensive part, the path search.
    """
    interleaved = [x for sub in labels for x in (None, list(sub))]
    if output is not None:
        interleaved.append(list(output))
    return oe.parser.convert_interleaved_input(interleaved)[0]


# Built from the cached path; holds no operands, so one expression serves every call.
@lru_cache(maxsize=4096)
def _cached_expression(subscripts, shapes, optimize):
    path = _cached_path(subscripts, shapes, (('optimize', optimize),))
    return oe.contract_expression(subscripts, *shapes, optimize=path)


@lru_cache(maxsize=4096)
def _cached_path(subscripts, shapes, options):
    path, _ = oe.contract_path(subscripts, *shapes, shapes=True, **dict(options))
    return path


def _fused_expression(selected, converted, kwargs):
    # Eager JAX pays one dispatch per pairwise step of the expression, every
    # Krylov iteration. The adapter evaluates the same path as one jitted
    # computation; constants are passed as arguments rather than baked in, so
    # one compilation per shape serves every site and time step.
    subscripts, specs = converted[0], converted[1:]
    constants = {i: specs[i] for i in kwargs.get('constants', ())}
    fused = selected.fused_einsum(subscripts, kwargs['optimize'])
    def call(*operands, **options):
        if current_backend() is not selected:
            raise CapabilityError('contraction expression belongs to another backend instance')
        options.pop('backend', None)
        if options:
            raise TypeError(f'unsupported contraction options: {sorted(options)}')
        runtime = iter(to_backend(a) for a in operands)
        result = fused(*(constants[i] if i in constants else next(runtime) for i in range(len(specs))))
        run = _run.get()
        if run is not None and run.ledger is not None:
            _witness(result)
        return result
    return call


def contract_expression(*args, **kwargs):
    # Constants must be placed by the captured adapter, not opt_einsum's
    # process-default device conversion. Expressions belong to one instance.
    selected = current_backend()
    converted = tuple(to_backend(a) if hasattr(a, 'dtype') and hasattr(a, 'shape') else a for a in args)
    run = _run.get()
    context = None if run is None else run.context
    if isinstance(kwargs.get('optimize', 'auto'), str) and converted and isinstance(converted[0], str):
        # Sweeps rebuild the same expression at every site; a named path search
        # depends only on subscripts, shapes and options, so reuse its result.
        shapes = tuple(tuple(map(int, a.shape)) if hasattr(a, 'shape') else tuple(map(int, a))
                       for a in converted[1:])
        options = tuple(sorted((k, v) for k, v in kwargs.items() if k != 'constants'))
        try:
            kwargs['optimize'] = _cached_path(converted[0], shapes, options)
        except TypeError:  # unhashable option: search the path as before
            pass
    if (selected.name == 'jax' and hasattr(selected, 'fused_einsum')
            and isinstance(converted[0], str) and isinstance(kwargs.get('optimize'), list)
            and set(kwargs) <= {'constants', 'optimize'}
            and (context is None or context.operators.policy == 'builtin')):
        return _fused_expression(selected, converted, kwargs)
    expr = oe.contract_expression(*converted, **kwargs)
    # Retain only the latest precision specialization and its constants, never
    # runtime operand contents. Metadata comparisons need no content hashes.
    specialization = None
    def call(*operands, **options):
        nonlocal specialization
        if current_backend() is not selected:
            raise CapabilityError('contraction expression belongs to another backend instance')
        options['backend'] = selected.opt_einsum_module
        native = tuple(to_backend(a) for a in operands)
        active_expr = expr
        if selected.name == 'torch':
            dtype = _contraction_dtype(converted + native, selected)
            cached = specialization
            if cached is None or cached[0] != dtype:
                constants = _promote_operands(converted, selected, dtype)
                if any(a is not b for a, b in zip(constants, converted)):
                    active_expr = oe.contract_expression(*constants, **kwargs)
                specialization = (dtype, active_expr)
            else:
                active_expr = cached[1]
            native = _promote_operands(native, selected, dtype)
        run = _run.get()
        if (context is not None and context.operators.policy != 'builtin'
                and (None if run is None else run.context) is not context):
            raise CapabilityError('contraction expression belongs to another numerical context')
        if context is None or context.operators.policy == 'builtin':
            result = active_expr(*native, **options)
        else:
            from .operators import dispatch
            def evaluate(*values, expression, **opts):
                return expression(*values, **opts)
            result = dispatch(context, 'contract_expression', evaluate, native,
                              dict(options, expression=active_expr),
                              metadata_values=converted + native)
        if run is not None and run.ledger is not None:
            _witness(result)
        return result
    return call
