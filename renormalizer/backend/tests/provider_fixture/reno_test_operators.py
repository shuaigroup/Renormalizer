"""Pure Python protocol proof, not an optimized implementation."""
import numpy as np
from renormalizer.backend.operators import Kernel, OperatorProvider


def supports(request, *args, **kwargs):
    if request.backend != 'numpy' or request.transformations:
        return 'eager NumPy only'
    if any(a.device != 'cpu' or a.dtype != 'float64' for a in request.arrays):
        return 'CPU float64 only'
    return None


def matmul(request, a, b):
    return np.matmul(a, b)


def provider():
    return OperatorProvider('test_numpy', '0.1.0', {'matmul': Kernel(supports, matmul)})
