import importlib
from dataclasses import FrozenInstanceError
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest


def context_module():
    return importlib.import_module('renormalizer.backend.context')


def test_capture_stable_nested_and_exception_safe():
    context = context_module()
    from renormalizer.cons import backend, set_backend
    first = set_backend('numpy')
    with pytest.raises(ValueError, match='failure'):
        with context.capture_backend(first):
            second = set_backend('numpy')
            assert backend.current is second
            assert context.internal_backend.current is first
            with context.capture_backend(second):
                assert context.internal_backend.current is second
            assert context.internal_backend.current is first
            raise ValueError('failure')
    assert context.internal_backend.current is second


def test_thread_contexts_are_independent():
    context = context_module()
    from renormalizer.backend.numpy_backend import NumpyBackend
    barrier = Barrier(2)
    def worker():
        adapter = NumpyBackend()
        with context.capture_backend(adapter):
            barrier.wait(timeout=5)
            assert context.internal_backend.current is adapter
            return adapter
    with ThreadPoolExecutor(2) as pool:
        a, b = [future.result() for future in [pool.submit(worker), pool.submit(worker)]]
    assert a is not b


def test_explicit_context_pins_precision_and_preserves_rng(monkeypatch):
    context = context_module()
    monkeypatch.setenv('RENO_FP32', '')
    state = np.random.get_state()
    ctx = context.make_context('numpy', device='cpu', real_dtype='float64')
    assert ctx.real_dtype == np.dtype('float64')
    assert ctx.complex_dtype == np.dtype('complex128')
    with pytest.raises(FrozenInstanceError):
        ctx.real_dtype = np.float32
    with pytest.raises(RuntimeError):
        ctx.adapter.use_32bits()
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_context_rejects_invalid_policy_and_device():
    context = context_module()
    with pytest.raises(ValueError, match='host_policy'):
        context.make_context('numpy', device='cpu', host_policy='silent')
    with pytest.raises(ValueError, match='device'):
        context.make_context('numpy', device='cuda:0')
    with pytest.raises(ValueError, match='real_dtype'):
        context.make_context('numpy', device='cpu', real_dtype='int64')


def test_async_contexts_are_independent():
    import asyncio
    context = context_module()
    from renormalizer.backend.numpy_backend import NumpyBackend
    async def task():
        adapter = NumpyBackend()
        with context.capture_backend(adapter):
            await asyncio.sleep(0)
            assert context.current_backend() is adapter
            return adapter
    async def main():
        first, second = await asyncio.gather(task(), task())
        assert first is not second
    asyncio.run(main())


def test_actual_precision_is_checked_at_construction(monkeypatch):
    context = context_module()
    from renormalizer.backend.numpy_backend import NumpyBackend
    adapter = NumpyBackend()
    adapter.zeros = lambda *args, **kwargs: np.zeros((0,), dtype='float32')
    monkeypatch.setattr(context, 'create_backend', lambda *args, **kwargs: adapter)
    with pytest.raises(ValueError, match='requested float64, received float32'):
        context.make_context('numpy', real_dtype='float64')


def test_public_proxy_is_unwrapped_once_on_capture():
    context = context_module()
    from renormalizer.cons import backend as public_proxy, set_backend
    first = set_backend('numpy')
    with context.capture_backend(public_proxy) as captured:
        second = set_backend('numpy')
        assert captured is first
        assert context.current_backend() is first
        with context.capture_backend(context.internal_backend) as nested:
            assert nested is first
        assert public_proxy.current is second
