import os
import numpy as np
from renormalizer.backend.context import make_context, capture_backend
from renormalizer.lib.integrate._ivp.rk import RK45


def test_dense_output_retains_creating_adapter():
    ctx=make_context(os.environ.get('RENO_TEST_BACKEND','numpy'),device=os.environ.get('RENO_TEST_DEVICE','cpu'),host_policy='explicit')
    with capture_backend(ctx.adapter):
        solver=RK45(lambda t,y:-y,0,ctx.ops.array([1+1j,2-1j]),0.1,rtol=1e-10,atol=1e-12)
        while solver.status=='running':solver.step()
        output=solver.dense_output()
    with capture_backend(make_context().adapter):
        value=output(np.array([0.099,0.1]))
    assert ctx.adapter.owns(value) if hasattr(ctx.adapter, "owns") else isinstance(value, np.ndarray)
    assert str(value.dtype).removeprefix("torch.")=="complex128"
    host=ctx.adapter.numpy(value)
    np.testing.assert_allclose(host,np.array([1+1j,2-1j])[:,None]*np.exp(-np.array([0.099,0.1])),rtol=1e-8,atol=1e-10)
