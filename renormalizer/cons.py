# -*- coding: utf-8 -*-

# multibackend: the global (legacy) backend selection, moved here from
# mps/backend.py; set_backend / get_backend / backend keep their old meaning.

import logging
import os
from pathlib import Path
import random
import subprocess

import numpy as np

from renormalizer.backend.proxy import BackendManager, BackendProxy

logger = logging.getLogger(__name__)


def get_git_commit_hash():
    # Identify this source checkout, not the application that happens to import
    # it. Installed packages without their own Git metadata have no source hash.
    root = Path(__file__).resolve().parents[1]
    if not (root / '.git').exists():
        return "Unknown"
    # Inherited Git routing/configuration can redirect even `git -C`; remove it
    # only in the child environment, preserving the importing process's state.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith('GIT_')}
    try:
        actual_root = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            stderr=subprocess.PIPE, env=environment, timeout=5).decode("utf-8").strip()
        # Do not borrow provenance from an enclosing unrelated checkout.
        if Path(actual_root).resolve() != root:
            return "Unknown"
        commit_hash = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.PIPE, env=environment, timeout=5)
        return commit_hash.strip().decode("utf-8")
    # Provenance is diagnostic: absent/inaccessible Git or broken metadata
    # must not break imports. Interrupts and other BaseExceptions still escape.
    except (subprocess.SubprocessError, OSError, UnicodeError):
        return "Unknown"


_manager = BackendManager()
backend = BackendProxy(_manager)
xp = backend


def set_backend(name):
    # Legacy selection resets its RNG. Restore host RNGs if candidate setup
    # fails; do not expose a half-initialized backend via the public proxy.
    with _manager._selection_lock:
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        try:
            return _manager.set_backend(
                name, explicit=True,
                initialize=lambda candidate: candidate.random.seed(2019),
            )
        except BaseException:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            raise


def get_backend():
    return _manager.get_backend()


def runtime_backend():
    return _manager.get_backend()


backend.random.seed(2019)
np.random.seed(9012)
random.seed(1092)

logger.info("Use %s as backend", backend.name)
logger.info("numpy random seed is 9012")
logger.info("backend random seed is 2019")
logger.info("random seed is 1092")
logger.info("Git Commit Hash: %s", get_git_commit_hash())
