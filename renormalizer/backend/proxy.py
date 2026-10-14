# -*- coding: utf-8 -*-

import importlib
from threading import RLock

from renormalizer.backend.factory import create_backend


class BackendManager:
    def __init__(self, initial_backend=None):
        self._selection_lock = RLock()
        self.current = create_backend(initial_backend, explicit=False)

    def set_backend(self, name, *, explicit=True, initialize=None):
        # Publish only a fully initialized candidate. Previously returned real
        # instances and cached public proxy references keep their own semantics.
        with self._selection_lock:
            candidate = create_backend(name, explicit=explicit)
            if initialize is not None:
                initialize(candidate)
            self.current = candidate
            return candidate

    def get_backend(self):
        return self.current


class BackendProxy:
    def __init__(self, manager: BackendManager):
        object.__setattr__(self, "_manager", manager)

    @property
    def current(self):
        return self._manager.current

    def __getattr__(self, name):
        try:
            return getattr(self.current, name)
        except AttributeError:
            if name.startswith("__"):
                raise
        # ``renormalizer.backend`` is both this public proxy and the backend
        # subpackage, so dotted module paths (``import renormalizer.backend.x as
        # y``, ``mock.patch("renormalizer.backend.x.f")``) arrive here. Backend
        # attributes take precedence; otherwise resolve the submodule.
        module_name = f"renormalizer.backend.{name}"
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise  # the submodule exists but one of its imports is missing
            raise AttributeError(f"backend has no attribute or submodule {name!r}") from None

    def __setattr__(self, name, value):
        if name == "_manager":
            object.__setattr__(self, name, value)
            return
        setattr(self.current, name, value)

    def __array_namespace__(self):
        return self.current.array_namespace
