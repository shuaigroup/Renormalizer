# Operator providers, protocol v1

An operator provider supplies its own implementation (a kernel) for some of
Renormalizer's numerical operations, for example a faster contraction on a given
device. Providers are opt-in: nothing changes unless a context selects them.

## Selecting providers

```python
from renormalizer.backend.context import make_context

ctx = make_context('numpy', host_policy='explicit',
                   operator_policy='prefer',          # or 'require'
                   operator_providers=['my_provider'])  # entry-point names or objects
state.evolve(mpo, dt, backend_context=ctx)
```

* `builtin` (default): no provider is consulted and nothing is loaded.
* `prefer`: the first provider whose kernel supports a call runs it; otherwise
  the builtin implementation runs.
* `require`: a call no selected provider supports raises `CapabilityError`.

Providers are given as `OperatorProvider` objects or as names of entry points in
the group `renormalizer.operator_providers`. Entry points are loaded only when
named; a factory may raise `ProviderUnavailable` to report a missing runtime.

## Writing a provider

```python
from renormalizer.backend.operators import Kernel, OperatorProvider

def supports(request, *args, **kwargs):
    if request.backend != 'numpy' or any(a.dtype != 'float64' for a in request.arrays):
        return 'CPU float64 only'          # a reason string: not supported
    return None                            # supported

def execute(request, *args, **kwargs):
    ...                                    # same semantics as the builtin operation

def provider():
    return OperatorProvider('my_provider', '0.1.0', {'contract': Kernel(supports, execute)})
```

Operations (`OPERATIONS`): `matmul`, `einsum`, `qr`, `svd`, `eigh`, `solve`,
`contract`, `contract_expression`.

The request (`OperatorRequest`) carries the operation, semantics version,
backend, device, one `ArraySpec` per array argument (shape, dtype, device,
strides, contiguity, writability) and active transformations (`autograd`, `jax`).

## Rules

* `supports` decides from request metadata and plain non-array arguments only,
  never from array contents or other objects. Plain values are `str`, `int`,
  `float`, `complex`, `bool`, `None`, and tuples recursively containing only
  these values. Its answer is cached per call signature (request plus plain
  arguments); other objects are omitted from the key without preventing caching.
* `execute` must keep the builtin operation's semantics, return arrays of the
  selected backend and device, and keep the requested dtype. The first result
  per call signature is checked for this (`operators._STRICT_PROVIDERS = True`
  checks every result, for development).
* Kernels are stateless with respect to operand contents and lifetimes. A
  failing kernel is never retried on another implementation.
* Results need not be bitwise identical to the builtin implementation (another
  BLAS, another device). The bitwise guarantee applies to the default path only.

## Where providers take effect

* `contract` and `contract_expression` dispatch in `backend/execution.py`; every
  MPS and TTNS contraction passes there.
* `matmul`, `einsum`, `qr`, `svd`, `eigh` and `solve` dispatch in the strict
  array API (`NumericalContext.ops`, `backend/testing/strict.py`), which
  algorithms do not call. Hooks at the algorithms' own decompositions (the
  SVD/QR in `mps/svd_qn.py`, eigensolvers) are not wired yet: each adds a check
  per call on the default path and needs a decision first.

## Certifying a provider

The strict array API and the execution evidence in `renormalizer.backend.testing`
check semantics, that results stay on the requested device, and which arrays
cross the host/device boundary (`record_execution`).
