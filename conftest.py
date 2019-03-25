# -*- coding: utf-8 -*-

import pytest

from ephMPS.mps import backend

@pytest.fixture
def switch_to_32backend():
    if backend.is_32bits:
        pytest.skip("already testing in 32 bits")
    # a hack for tests, shouldn't be used in production code
    backend.running = False
    backend.use_32bits()
    yield
    backend.running = False
    backend.use_64bits()


@pytest.fixture
def switch_to_64backend():
    if backend.is_32bits:
        # a hack for tests, shouldn't be used in production code
        backend.running = False
        backend.use_64bits()
        yield
        backend.running = False
        backend.use_32bits()