"""Opt-in, stateless operator providers. See OPERATOR_PROVIDERS.md for v1."""
from dataclasses import dataclass, field
from importlib import metadata
from types import MappingProxyType

from .contracts import CapabilityError

PROTOCOL_VERSION = 1
ENTRY_POINT_GROUP = 'renormalizer.operator_providers'
class ProviderUnavailable(CapabilityError):
    """Factory explicitly reports a missing optional runtime, before execution."""


OPERATIONS = frozenset(('matmul', 'einsum', 'qr', 'svd', 'eigh', 'solve',
                        'contract', 'contract_expression'))


@dataclass(frozen=True)
class ArraySpec:
    shape: tuple
    dtype: str
    device: str
    strides: tuple | None
    # None means unknown, never contiguous or writable by implication.
    c_contiguous: bool | None
    f_contiguous: bool | None
    writable: bool | None


@dataclass(frozen=True)
class OperatorRequest:
    operation: str
    semantics_version: int
    backend: str
    device: str
    arrays: tuple
    transformations: tuple
    input_mutation: bool = False
    output_aliasing: str = 'unspecified'


@dataclass(frozen=True)
class Kernel:
    """supports(request, *args, **kwargs) returns None when supported, otherwise a reason string.

    execute(request, *args, **kwargs) must honor the operation's v1 semantics.
    Both callables must be stateless with respect to operand contents/lifetimes.
    supports decides from request metadata and plain non-array arguments only,
    never from array contents or other objects: its answer is cached per call signature.
    """
    supports: object
    execute: object


@dataclass(frozen=True)
class OperatorProvider:
    name: str
    version: str
    kernels: object
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self):
        if not self.name or self.name == 'builtin':
            raise ValueError('provider needs a non-builtin name')
        kernels = dict(self.kernels)
        if set(kernels) - OPERATIONS:
            raise ValueError('unknown operator semantics')
        for kernel in kernels.values():
            if not isinstance(kernel, Kernel) or not callable(kernel.supports) or not callable(kernel.execute):
                raise TypeError('kernels must contain Kernel callables')
        object.__setattr__(self, 'kernels', MappingProxyType(kernels))


def discover_operator_providers():
    """Return entry-point metadata without importing any provider code."""
    return tuple(metadata.entry_points(group=ENTRY_POINT_GROUP))


# Development switch: check every provider result, not only the first one per
# call signature.
_STRICT_PROVIDERS = False


def _signature(request, args, kwargs):
    """Cache request metadata and plain values without retaining other objects."""
    def is_plain(value):
        return (type(value) in (str, int, float, complex, bool, type(None)) or
                (type(value) is tuple and all(is_plain(item) for item in value)))

    key = (request, tuple(a for a in args if is_plain(a)),
           tuple(sorted((k, v) for k, v in kwargs.items() if is_plain(v))))
    try:
        hash(key)
    except TypeError:
        return None
    return key


@dataclass(frozen=True)
class OperatorSelection:
    policy: str = 'builtin'
    providers: tuple = ()
    unavailable: tuple = ()
    # Per call signature: the chosen kernel (resolve) and signatures whose
    # results were already checked (dispatch). Never part of equality.
    _resolved: dict = field(default_factory=dict, compare=False, repr=False)
    _checked: set = field(default_factory=set, compare=False, repr=False)

    def resolve(self, request, *args, **kwargs):
        key = _signature(request, args, kwargs)
        if key is not None:
            hit = self._resolved.get(key)
            if hit is not None:
                return hit
        found = self._resolve(request, *args, **kwargs)
        if key is not None:
            self._resolved[key] = found
        return found

    def _resolve(self, request, *args, **kwargs):
        reasons = list(self.unavailable)
        for provider in self.providers:
            kernel = provider.kernels.get(request.operation)
            reason = 'operation not supplied' if kernel is None else kernel.supports(request, *args, **kwargs)
            if reason is None:
                return kernel, provider, tuple(reasons)
            if not isinstance(reason, str):
                raise TypeError('supports must return None or an unsupported reason string')
            reasons.append((provider.name, reason))
        if self.policy == 'require':
            raise CapabilityError(f'no required provider supports {request.operation}: {reasons}')
        return None, None, tuple(reasons)

    def explain(self, request, *args, **kwargs):
        _, provider, reasons = self.resolve(request, *args, **kwargs)
        return {'operation': request.operation, 'implementation': 'builtin' if provider is None else provider.name,
                'version': None if provider is None else provider.version, 'reasons': reasons}

    def call(self, request, builtin, *args, **kwargs):
        kernel, _, _ = self.resolve(request, *args, **kwargs)
        if kernel is None:
            return builtin(*args, **kwargs)
        # No exception handler here: execution is never retried on another kernel.
        return kernel.execute(request, *args, **kwargs)


def select_operators(policy='builtin', providers=()):
    if policy not in ('builtin', 'prefer', 'require'):
        raise ValueError('operator_policy must be builtin, prefer, or require')
    providers = tuple(providers)
    if policy == 'builtin':
        if providers:
            raise ValueError('providers require an explicit prefer or require policy')
        return OperatorSelection()
    if not providers:
        raise ValueError('prefer/require policy needs operator_providers')
    entries = None
    selected, unavailable, names = [], [], set()
    for candidate in providers:
        requested_name = candidate if isinstance(candidate, str) else candidate.name
        if requested_name in names:
            raise ValueError(f'duplicate provider {requested_name}')
        names.add(requested_name)
        if isinstance(candidate, str):
            if entries is None:
                entries = discover_operator_providers()
            matches = [entry for entry in entries if entry.name == candidate]
            if len(matches) > 1:
                raise CapabilityError(f'ambiguous provider entry point {candidate}')
            if not matches:
                unavailable.append((candidate, 'not installed'))
                continue
            # Explicit opt-in authorizes this import. Import/factory errors are real errors.
            try:
                candidate = matches[0].load()()
            except ProviderUnavailable as exc:
                unavailable.append((requested_name, str(exc)))
                continue
            if not isinstance(candidate, OperatorProvider) or candidate.name != requested_name:
                raise TypeError('entry-point factory must return a matching OperatorProvider')
        if not isinstance(candidate, OperatorProvider):
            raise TypeError('expected provider object or entry-point name')
        if candidate.protocol_version != PROTOCOL_VERSION:
            unavailable.append((candidate.name, 'unsupported protocol version'))
        else:
            selected.append(candidate)
    if policy == 'require' and unavailable:
        raise CapabilityError(f'required providers unavailable: {unavailable}')
    return OperatorSelection(policy, tuple(selected), tuple(unavailable))


def make_request(context, operation, values):
    """Metadata only; never copy operands, synchronize, or scan array contents."""
    arrays, transforms = [], set()
    from .execution import _device
    for value in values:
        if not hasattr(value, 'dtype') or not hasattr(value, 'shape'):
            continue
        flags = getattr(value, 'flags', None)
        strides = getattr(value, 'strides', None)
        if strides is None and callable(getattr(value, 'stride', None)):
            strides = tuple(s * value.element_size() for s in value.stride())
        arrays.append(ArraySpec(tuple(value.shape), str(value.dtype).removeprefix('torch.'),
                                _device(value), None if strides is None else tuple(strides),
                                getattr(flags, 'c_contiguous', None), getattr(flags, 'f_contiguous', None),
                                getattr(flags, 'writeable', None)))
        if getattr(value, 'requires_grad', False):
            transforms.add('autograd')
        if context.adapter.name == 'jax':
            # Conservative: v1 does not certify eager JAX separately from tracing.
            transforms.add('jax')
    return OperatorRequest(operation, 1, context.adapter.name, context.device,
                           tuple(arrays), tuple(sorted(transforms)))


def dispatch(context, operation, builtin, args, kwargs=None, *, metadata_values=None):
    selection = context.operators
    if selection.policy == 'builtin':
        return builtin(*args, **(kwargs or {}))
    request = make_request(context, operation, args if metadata_values is None else metadata_values)
    result = selection.call(request, builtin, *args, **(kwargs or {}))
    # Result checks run once per call signature (every call with
    # _STRICT_PROVIDERS): a kernel is chosen per signature, so its first result
    # stands for the rest.
    key = _signature(request, args, kwargs or {})
    if not _STRICT_PROVIDERS and key is not None:
        if key in selection._checked:
            return result
    # Providers remain inside the selected array backend. Do not invoke implicit
    # host conversions on foreign results, including inside ctx.ops._result.
    import numpy as np
    from .contracts import OwnershipError, PrecisionError
    results = result if isinstance(result, tuple) else (result,)
    for value in results:
        owned = (isinstance(value, (np.ndarray, np.generic)) if context.adapter.name == 'numpy'
                 else context.adapter.owns(value))
        if not owned:
            raise OwnershipError('operator returned an array on the wrong backend/device')
    if operation in ('contract', 'contract_expression') and request.arrays:
        dtype = np.dtype((kwargs or {}).get('dtype') or np.result_type(*(a.dtype for a in request.arrays)))
        actual = np.dtype(str(result.dtype).removeprefix('torch.'))
        if actual != dtype:
            raise PrecisionError(f'requested {dtype}, received {actual}')
    if key is not None:
        selection._checked.add(key)
    return result
