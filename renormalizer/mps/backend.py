# -*- coding: utf-8 -*-

import logging
import os

import numpy as np

from renormalizer.cons import backend, get_backend, runtime_backend, set_backend, xp
from renormalizer.backend.abstract import AbstractBackend
from renormalizer.backend.factory import probe_legacy_cupy
# Preserve the historical facade import as the exact utility object, not a
# wrapper; callers may have imported this formatter from mps.backend directly.
from renormalizer.utils.utils import sizeof_fmt

try:
    import primme
    IMPORT_PRIMME_EXCEPTION = None
except Exception as e:
    primme = None
    IMPORT_PRIMME_EXCEPTION = e


logger = logging.getLogger(__name__)


def get_git_commit_hash():
    from renormalizer.cons import get_git_commit_hash as _get_git_commit_hash

    return _get_git_commit_hash()


GPU_KEY = "RENO_GPU"
GPU_ID = getattr(get_backend(), "legacy_gpu_id", os.environ.get(GPU_KEY))
xpseed, npseed, randomseed = 2019, 9012, 1092


def try_import_cupy():
    global GPU_ID
    enabled, namespace, GPU_ID = probe_legacy_cupy(GPU_ID)
    return enabled, namespace


class Backend(AbstractBackend):
    """Compatibility name: the legacy singleton already exists at import."""
    _init_once_flag = True

    def __new__(cls):
        raise RuntimeError("Backend should only be initialized once")


def backend_snapshot():
    """Current configuration; exported scalar constants are import snapshots."""
    selected = get_backend()
    array_types = getattr(selected, "array_types", selected.ndarray)
    if not isinstance(array_types, tuple):
        array_types = (array_types,)
    return {
        "USE_GPU": selected.name == "cupy",
        "OE_BACKEND": selected.opt_einsum_name,
        "MEMORY_ERRORS": selected.memory_errors,
        "ARRAY_TYPES": array_types,
    }


# Import-time values kept for old imports; backend_snapshot() gives current ones.
_snapshot = backend_snapshot()
USE_GPU = _snapshot["USE_GPU"]
OE_BACKEND = _snapshot["OE_BACKEND"]
MEMORY_ERRORS = _snapshot["MEMORY_ERRORS"]
ARRAY_TYPES = _snapshot["ARRAY_TYPES"]

__all__ = [
    "GPU_KEY", "GPU_ID", "try_import_cupy", "xpseed", "npseed",
    "randomseed", "Backend", "backend_snapshot",
    "np",
    "xp",
    "backend",
    "set_backend",
    "get_backend",
    "runtime_backend",
    "USE_GPU",
    "OE_BACKEND",
    "MEMORY_ERRORS",
    "ARRAY_TYPES",
    "primme",
    "IMPORT_PRIMME_EXCEPTION",
    "get_git_commit_hash",
    "sizeof_fmt",
]
