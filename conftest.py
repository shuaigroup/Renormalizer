"""Explicit backend selection shared by application-semantic regression tests.

Selecting a backend is a requirement: missing optional
packages or devices must fail the selected matrix job, never turn it green via
skip. Default developer runs exercise NumPy without importing optional runtimes.
"""
import os
import numpy as np
import pytest


def pytest_addoption(parser):
    group = parser.getgroup('renormalizer numerical backends')
    group.addoption('--reno-backend', choices=('numpy', 'torch', 'jax', 'cupy'), default=None)
    group.addoption('--reno-device', default=None)


def pytest_configure(config):
    # Share CLI choices with helpers that read the environment during collection.
    for option, variable in (('--reno-backend', 'RENO_TEST_BACKEND'),
                             ('--reno-device', 'RENO_TEST_DEVICE')):
        value = config.getoption(option)
        if value is not None:
            os.environ[variable] = value


@pytest.fixture
def numerical_context(request):
    from renormalizer.backend.context import make_context
    name = request.config.getoption('--reno-backend') or os.environ.get('RENO_TEST_BACKEND', 'numpy')
    device = request.config.getoption('--reno-device') or os.environ.get('RENO_TEST_DEVICE', 'cpu')
    ctx = make_context(name, device=device, real_dtype='float64')
    probe = ctx.ops.zeros((1,))
    if name != 'numpy':
        assert ctx.adapter.owns(probe), 'selected backend must own the allocation'
        assert ctx.adapter.current_device() == device, 'selected device must be honored'
    assert ctx.adapter.name == name
    assert ctx.real_dtype == np.dtype('float64')
    return ctx


@pytest.fixture
def captured_backend(numerical_context):
    from renormalizer.backend.context import capture_backend, current_backend
    previous = current_backend()
    state = np.random.get_state()
    try:
        with capture_backend(numerical_context.adapter):
            assert current_backend() is numerical_context.adapter
            yield numerical_context
    finally:
        np.random.set_state(state)
        assert current_backend() is previous


def pytest_ignore_collect(collection_path, config):
    # --doctest-modules imports every source module. JAX alone has eager imports
    # in its optional adapter; absence is not an error in a NumPy installation.
    # Do not suppress errors from an installed-but-broken optional runtime.
    import importlib.util
    if tuple(collection_path.parts[-3:]) == ('renormalizer', 'backend', 'jax_backend.py'):
        return importlib.util.find_spec('jax') is None
    return None
