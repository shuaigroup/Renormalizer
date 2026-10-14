"""Explicit numerical policy and per-call dispatch, separate from legacy globals."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np

from renormalizer.backend.factory import create_backend, normalize_backend_name
from renormalizer.backend.proxy import BackendProxy
# Read the legacy selection directly: the two calls through runtime_backend()
# add up over the millions of ``xp.<op>`` lookups in a sweep.
from renormalizer.cons import _manager as _legacy

_active = ContextVar('renormalizer_algorithm_backend', default=None)


def current_backend():
    selected = _active.get()
    return _legacy.current if selected is None else selected


@contextmanager
def capture_backend(selected=None):
    """Keep a real instance for this call; nesting and exceptions restore it.

    Unlike the public compatibility proxy, this dispatch is local to the current
    thread/async context. A raw legacy adapter retains its legacy precision rules;
    use make_context for an instance whose precision is locked.
    """
    captured = current_backend() if selected is None else selected
    if isinstance(captured, (BackendProxy, InternalBackendProxy)):
        captured = captured.current
    token = _active.set(captured)
    try:
        yield captured
    finally:
        _active.reset(token)


class InternalBackendProxy:
    @property
    def current(self):
        return current_backend()

    def __getattr__(self, name):
        # Inlined current_backend(): every ``xp.<op>`` in the solvers lands here.
        selected = _active.get()
        return getattr(_legacy.current if selected is None else selected, name)

    def __setattr__(self, name, value):
        setattr(self.current, name, value)


internal_backend = InternalBackendProxy()


@dataclass(frozen=True)
class NumericalContext:
    """Factory-owned adapter plus immutable device, precision and host policy."""
    adapter: object
    device: str
    real_dtype: np.dtype
    complex_dtype: np.dtype
    host_policy: str
    operators: object = None

    def __post_init__(self):
        if self.operators is None:
            from .operators import OperatorSelection
            object.__setattr__(self, "operators", OperatorSelection())

    @property
    def ops(self):
        # The strict array API is a development and test tool; imported on use
        # so that production imports never load it. A new object per access
        # keeps no ops -> context -> ops cycle.
        from renormalizer.backend.testing.strict import StrictOperations, DeviceOperations
        return StrictOperations(self) if self.adapter.name == "numpy" else DeviceOperations(self)


def make_context(name='numpy', *, device='cpu', real_dtype='float64', host_policy='forbid',
                 operator_policy='builtin', operator_providers=()):
    """Create a private instance without selecting it or seeding global RNGs.

    Device selection and requested precision are verified before publishing the
    context. Optional libraries are imported only when explicitly selected.
    Selecting JAX float64 enables JAX's process-wide x64 support; existing
    arrays retain their dtype. Explicit float32 never disables x64 support.
    """
    from renormalizer.backend.contracts import CapabilityError, PrecisionError
    from .operators import select_operators
    operators = select_operators(operator_policy, operator_providers)
    if host_policy not in ('forbid', 'explicit'):
        raise ValueError("host_policy must be 'forbid' or 'explicit'")
    dtype = np.dtype(real_dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('real_dtype must be float32 or float64')
    normalized = normalize_backend_name(name)
    if normalized == 'numpy':
        if device != 'cpu':
            raise ValueError('NumPy context device must be cpu')
        adapter = create_backend(normalized, explicit=True)
    elif normalized == 'jax':
        adapter = create_backend(normalized, explicit=True, device=device, real_dtype=dtype)
    else:
        adapter = create_backend(normalized, explicit=True, device=device)
    complex_dtype = np.dtype('complex64' if dtype == np.dtype('float32') else 'complex128')
    # Explicit policy wins over the old presence-based RENO_FP32 default.
    adapter.dtypes = (dtype.type, complex_dtype.type)
    if normalized == "cupy":
        adapter.isolate_rng()
    probe = adapter.zeros((0,), dtype=adapter.real_dtype)
    actual = probe.dtype if normalized == "numpy" else adapter.dtype_of(probe)
    if actual != dtype:
        raise PrecisionError(f'requested {dtype}, received {actual}')
    adapter.first_mp = True
    return NumericalContext(adapter, device, dtype, complex_dtype, host_policy, operators)
