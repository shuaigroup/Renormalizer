# -*- coding: utf-8 -*-

from __future__ import annotations

import contextlib
from typing import Any

import numpy as _np

from renormalizer.backend.mpi import SingleProcessDistributedMixin
from renormalizer.backend.transforms import UnavailableTransforms


class AbstractBackend(SingleProcessDistributedMixin):
    name = "abstract"
    array_namespace = None
    ndarray = ()
    memory_errors = (MemoryError,)
    opt_einsum_name = "numpy"
    supports_autodiff = False
    supports_jit = False
    supports_functional_update = True
    # Whether workspaces an algorithm keeps across steps (e.g. MPS environments)
    # should stay as native arrays rather than host storage. Host storage costs
    # nothing for NumPy; adapters that copy on every upload opt in.
    native_workspace_storage = False
    # Unset tolerances follow the precision. Class-level defaults keep the
    # property off the failed-lookup path: adapters forward unknown attributes
    # to their array library, so a missing one costs an exception per read.
    _canonical_atol = None
    _canonical_rtol = None

    def __init__(self):
        self.first_mp = False
        self._real_dtype = None
        self._complex_dtype = None
        self.transforms = UnavailableTransforms(self.name)
        self.use_64bits()

    def use_32bits(self):
        self.dtypes = (_np.float32, _np.complex64)

    def use_64bits(self):
        self.dtypes = (_np.float64, _np.complex128)

    @property
    def is_32bits(self) -> bool:
        return self._real_dtype == _np.float32

    @property
    def real_dtype(self):
        return self._real_dtype

    @real_dtype.setter
    def real_dtype(self, tp):
        if self.first_mp:
            raise RuntimeError("Can't alter backend data type")
        self._real_dtype = tp

    @property
    def complex_dtype(self):
        return self._complex_dtype

    @complex_dtype.setter
    def complex_dtype(self, tp):
        if self.first_mp:
            raise RuntimeError("Can't alter backend data type")
        self._complex_dtype = tp

    @property
    def dtypes(self):
        return self.real_dtype, self.complex_dtype

    @dtypes.setter
    def dtypes(self, target):
        self.real_dtype, self.complex_dtype = target

    @property
    def canonical_atol(self):
        if self._canonical_atol is None:
            return 1e-4 if self.is_32bits else 1e-8
        return self._canonical_atol

    @canonical_atol.setter
    def canonical_atol(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f'canonical_atol must be a non-negative number, got {value!r}')
        self._canonical_atol = value

    @property
    def canonical_rtol(self):
        if self._canonical_rtol is None:
            return 1e-2 if self.is_32bits else 1e-5
        return self._canonical_rtol

    @canonical_rtol.setter
    def canonical_rtol(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f'canonical_rtol must be a non-negative number, got {value!r}')
        self._canonical_rtol = value

    @property
    def opt_einsum_module(self):
        """The opt_einsum backend Renormalizer's own contractions run on.

        ``opt_einsum_name`` by default. An adapter may name a module providing
        the same ``tensordot``, ``transpose`` and ``einsum`` tuned for it.
        """
        return self.opt_einsum_name

    def numpy(self, x: Any):
        raise NotImplementedError

    def from_numpy(self, x: _np.ndarray):
        raise NotImplementedError

    def is_array(self, x: Any) -> bool:
        return isinstance(x, self.ndarray)

    def device_scope(self):
        """Context making this adapter's device current for array operations.

        Adapters whose libraries launch work on a process-wide current device
        (CuPy) return that device; others need nothing.
        """
        return contextlib.nullcontext()

    def sync(self):
        return None

    def free_all_blocks(self):
        return None

    def log_memory_usage(self, header=""):
        return None

    def at_set(self, x, idx, value):
        y = self.array(x, copy=True)
        y[idx] = value
        return y

    def write_owned(self, x, idx, value):
        """Internal write into caller-owned scratch storage; retain the result.

        Unlike public at_set, this may mutate x. Callers must own the workspace
        and must not pass user inputs or storage aliased by another live value.
        Immutable adapters override this method and return updated storage.
        """
        # Krylov/RK workspaces are already private: copying their full backing
        # array for every row update would introduce quadratic memory traffic.
        x[idx] = value
        return x

    # Lanczos recurrence on the private Krylov workspace V (lib/krylov). The
    # defaults are the exact statements of the recurrence, so eager adapters keep
    # identical arithmetic; adapters with per-call dispatch cost (JAX) override
    # them to evaluate each group as one computation. The host-side Lanczos
    # coefficients, tridiagonal eigensolve and convergence decisions stay in
    # krylov.py.
    def lanczos_alpha(self, V, j, w):
        """alpha_j = Re <w, V[j]>, still a backend scalar."""
        return self.vdot(w, V[j]).real

    def lanczos_orthogonalize(self, V, j, w, alpha_j, beta_prev):
        """w -= alpha_j V[j] + beta_prev V[j-1] (no beta term at j == 0); returns (w, ||w||)."""
        w -= alpha_j*V[j] + (beta_prev*V[j-1] if j > 0 else 0)
        return w, self.linalg.norm(w)

    def krylov_combine(self, V, coefficients):
        """Krylov result V @ coefficients, from double-precision host coefficients.

        Cast the coefficients to the basis precision at the explicit upload
        boundary, promoting a real basis when complex time requires it. CuPy
        overrides this with its original promotion.
        """
        basis_dtype = V.dtype if isinstance(V.dtype, _np.dtype) else _np.dtype(str(V.dtype).removeprefix("torch."))
        result_dtype = _np.result_type(basis_dtype, _np.complex64) if _np.iscomplexobj(coefficients) else basis_dtype
        if basis_dtype.kind != 'c' and _np.iscomplexobj(coefficients):
            # A complex time step needs a complex result, not a complex copy of
            # the entire real basis on each convergence check. Two same-dtype
            # real matvecs trade an extra GEMV for avoiding that basis allocation;
            # only the resulting vectors are combined into complex storage.
            real = self.dot(V, self.asarray(coefficients.real, dtype=basis_dtype))
            imag = self.dot(V, self.asarray(coefficients.imag, dtype=basis_dtype))
            return self.asarray(real + 1j * imag, dtype=result_dtype)
        return self.dot(V, self.asarray(coefficients, dtype=result_dtype))

    def lanczos_append(self, V, j, w, beta_j):
        """Store the next Krylov vector V[j] = w / beta_j; retain the returned workspace."""
        return self.write_owned(V, j, w / beta_j)

    def at_add(self, x, idx, value):
        y = self.array(x, copy=True)
        y[idx] += value
        return y

    def at_sub(self, x, idx, value):
        y = self.array(x, copy=True)
        y[idx] -= value
        return y

    def at_mul(self, x, idx, value):
        y = self.array(x, copy=True)
        y[idx] *= value
        return y
