# -*- coding: utf-8 -*-

import logging
import os

import numpy as np

from renormalizer.backend.numpy_backend import NumpyBackend

logger = logging.getLogger(__name__)


SUPPORTED_BACKENDS = ("numpy", "cupy", "jax", "torch")


def normalize_backend_name(name):
    if name is None:
        return "numpy"
    normalized = str(name).lower().strip()
    if normalized in {"np", "numpy"}:
        return "numpy"
    if normalized in {"cupy", "cp"}:
        return "cupy"
    if normalized in {"jax", "jnp"}:
        return "jax"
    if normalized in {"torch", "pytorch"}:
        return "torch"
    raise ValueError(
        f"Unknown backend '{name}'. Supported backends: {', '.join(SUPPORTED_BACKENDS)}"
    )


def probe_legacy_cupy(device_id=None):
    """Old startup probe: return (enabled, namespace, effective GPU_ID).

    RENO_GPU values keep their old string representation. CuPy absence or its
    device-initialization runtime error is the historical NumPy fallback, not a
    fallback from a failed explicit backend request.
    """
    try:
        import cupy as cp
    except ImportError:
        if device_id is not None:
            logger.warning("CuPy is not installed; RENO_GPU=%s has no effect", device_id)
        return False, np, device_id
    effective_id = 0 if device_id is None else device_id
    try:
        cp.cuda.Device(effective_id).use()
    except cp.cuda.runtime.CUDARuntimeError:
        logger.warning("Failed to initialize CuPy; using NumPy", exc_info=True)
        return False, np, effective_id
    return True, cp, effective_id


def create_backend(name=None, *, explicit=True, device=None, real_dtype=None):
    if name is None and not explicit:
        enabled, _, device_id = probe_legacy_cupy(os.environ.get("RENO_GPU"))
        if enabled:
            from renormalizer.backend.cupy_backend import CupyBackend
            candidate = CupyBackend()
        else:
            candidate = NumpyBackend()
        candidate.legacy_gpu_id = device_id
        return candidate
    normalized = normalize_backend_name(name)
    if normalized == "numpy":
        return NumpyBackend()
    if normalized == "cupy":
        from renormalizer.backend.cupy_backend import CupyBackend
        return CupyBackend(device=device)
    if normalized == "jax":
        from renormalizer.backend.jax_backend import JaxBackend
        return JaxBackend(device=device, real_dtype=real_dtype)
    if normalized == "torch":
        from renormalizer.backend.torch_backend import TorchBackend
        return TorchBackend(device=device or "cpu")
    raise ValueError(
        f"Unknown backend '{name}'. Supported backends: {', '.join(SUPPORTED_BACKENDS)}"
    )
