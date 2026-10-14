# -*- coding: utf-8 -*-


class UnavailableTransforms:
    """Autodiff transform namespace for backends that do not support autodiff."""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    def _raise(self, transform_name: str):
        raise NotImplementedError(
            f"Backend '{self.backend_name}' does not provide autodiff transform "
            f"'{transform_name}'. Select an autodiff-capable backend before using it."
        )

    def grad(self, *args, **kwargs):
        self._raise("grad")

    def value_and_grad(self, *args, **kwargs):
        self._raise("value_and_grad")

    def jit(self, *args, **kwargs):
        self._raise("jit")

    def vmap(self, *args, **kwargs):
        self._raise("vmap")

    def stop_gradient(self, x):
        return x


# ---------------------------------------------------------------------------
# Gradient transform stubs — reserved for future autodiff implementation.
# These classes define the intended API surface but do not implement real
# gradient computation. See follow-on gradient plans for implementation.
# ---------------------------------------------------------------------------

class GradientTransforms(UnavailableTransforms):
    """Placeholder for backends that WILL support autodiff transforms.

    Currently raises NotImplementedError like UnavailableTransforms.
    Future: override grad, value_and_grad, jit, vmap with real implementations.
    """

    def __init__(self, backend_name: str):
        super().__init__(backend_name)


class ParameterResponseGradient:
    """TODO: Implement parameter-response gradients with Hellmann-Feynman checks.

    Intended API:
        grad = backend.transforms.param_grad(loss_fn)
        grads = grad(params, state)
    """

    pass


class ArrayExpectationGradient:
    """TODO: Implement pure-array MPS expectation gradients with finite-difference checks.

    Intended API:
        grad = backend.transforms.array_grad(expectation_fn)
        grads = grad(tensors, hamiltonian)
    """

    pass
