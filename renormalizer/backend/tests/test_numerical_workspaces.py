"""Solver contracts for host metadata and exclusively owned native workspaces."""
import os
import numpy as np
import pytest
from renormalizer.backend.context import make_context, capture_backend
from renormalizer.lib.integrate._ivp.base import ConstantDenseOutput
from renormalizer.lib.integrate._ivp.rk import RK23, RK45
from renormalizer.lib.krylov.krylov import expm_krylov


@pytest.fixture
def context():
    return make_context(os.environ.get('RENO_TEST_BACKEND','numpy'),
                        device=os.environ.get('RENO_TEST_DEVICE','cpu'),
                        host_policy='explicit')


@pytest.mark.parametrize('solver_type',[RK23,RK45])
def test_host_stage_times_and_explicit_first_step(context,solver_type):
    with capture_backend(context.adapter):
        solver=solver_type(lambda t,y:-y,0.,context.ops.ones((2,)),.1,first_step=.01)
        assert isinstance(solver.C,np.ndarray)
        while solver.status=='running':solver.step()
        np.testing.assert_allclose(context.adapter.numpy(solver.y),np.exp(-.1),rtol=1e-5)


def test_vectorized_wrapper_accepts_immutable_arrays(context):
    with capture_backend(context.adapter):
        solver=RK45(lambda t,y:-y,0.,context.ops.ones((2,)),.1)
        y=context.ops.array([[1.,2.,3.],[4.,5.,6.]])
        actual=solver.fun_vectorized(0.,y)
        np.testing.assert_array_equal(context.adapter.numpy(actual),-context.adapter.numpy(y))
        empty=context.ops.zeros((2,0))
        assert solver.fun_vectorized(0.,empty).shape==(2,0)


def test_constant_dense_output_is_a_native_copy_without_multiplication(context,monkeypatch):
    # Complex infinity detects arithmetic masquerading as broadcasting/copying.
    host=np.array([complex(np.inf,2.),complex(-0.,-0.)])
    with capture_backend(context.adapter):
        value=context.adapter.asarray(host)
        output=ConstantDenseOutput(0.,1.,value)
        def forbidden(*args,**kwargs):raise AssertionError('unnecessary ones allocation')
        monkeypatch.setattr(context.adapter,'ones',forbidden,raising=False)
        result=output(np.array([.2,.8]))
        actual=context.adapter.numpy(result)
        np.testing.assert_array_equal(actual,np.broadcast_to(host[:,None],(2,2)))
        assert np.array_equal(np.signbit(actual.imag),np.signbit(np.broadcast_to(host[:,None],(2,2)).imag))
        assert result is not value


@pytest.mark.parametrize('dtype',['float32','float64','complex64','complex128'])
@pytest.mark.parametrize('dt',[.1,.1j])
def test_krylov_preserves_basis_precision_and_complex_time(context,dtype,dt):
    adapter=context.adapter
    with capture_backend(adapter):
        vector=adapter.asarray(np.array([1.,2.,3.],dtype=dtype))
        diagonal=adapter.asarray(np.array([1.,2.,4.],dtype=dtype))
        result,_=expm_krylov(lambda x:diagonal*x,dt,vector,block_size=2)
        actual=adapter.numpy(result)
        expected_dtype=np.dtype(dtype)
        if np.iscomplexobj(dt):expected_dtype=np.dtype('complex64' if expected_dtype.itemsize<=4 or dtype=='complex64' else 'complex128')
        assert actual.dtype==expected_dtype
        np.testing.assert_allclose(actual,np.array([1.,2.,3.])*np.exp(dt*np.array([1.,2.,4.])),rtol=3e-5,atol=3e-6)


def test_solver_workspace_writes_do_not_use_public_pure_updates(context,monkeypatch):
    with capture_backend(context.adapter):
        def forbidden(*args,**kwargs):raise AssertionError('public at_set copies owned workspace')
        if context.adapter.name != 'jax':
            monkeypatch.setattr(context.adapter,'at_set',forbidden)
        solver=RK45(lambda t,y:-y,0.,context.ops.ones((2,)),.1)
        solver.step()
        expm_krylov(lambda x:2*x,.1,context.ops.ones((2,)))


def test_complex_krylov_does_not_materialize_a_complex_basis(context,monkeypatch):
    adapter=context.adapter
    original=adapter.asarray
    real_matvecs=[]
    original_dot=adapter.dot
    def no_complex_basis(value,dtype=None):
        if getattr(value,'ndim',0)==2 and dtype is not None and np.dtype(str(dtype).removeprefix('torch.')).kind=='c':
            raise AssertionError('full basis complex conversion')
        return original(value,dtype=dtype)
    def record_dot(left,right):
        if left.ndim==2:
            assert str(left.dtype)==str(right.dtype)
            assert 'complex' not in str(left.dtype)
            real_matvecs.append(tuple(left.shape))
        return original_dot(left,right)
    monkeypatch.setattr(adapter,'asarray',no_complex_basis)
    monkeypatch.setattr(adapter,'dot',record_dot,raising=False)
    with capture_backend(adapter):
        vector=adapter.asarray(np.ones(8,dtype='float32'))
        diagonal=adapter.asarray(np.arange(8,dtype='float32'))
        result,_=expm_krylov(lambda x:diagonal*x,.1j,vector,block_size=3)
    assert len(real_matvecs)>=2 and len(real_matvecs)%2==0
    np.testing.assert_allclose(adapter.numpy(result),np.exp(.1j*np.arange(8)),atol=3e-6,rtol=3e-5)
