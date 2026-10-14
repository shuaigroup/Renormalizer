"""PyTorch adapter with explicit dtype/device conversion and NumPy-style calls.

Narrow migration reference: ef971a3 torch_backend.py's dtype/device and RNG
mapping. No BackendConfig, distributed, execution-IR or resident dependencies.
"""
import os
import sys

import numpy as np

from renormalizer.backend.abstract import AbstractBackend


class TorchBackend(AbstractBackend):
    name = 'torch'
    opt_einsum_name = 'torch'

    def __init__(self, device='cpu'):
        if 'torch' not in sys.modules:
            # Reno alternates Torch kernels with NumPy/SciPy BLAS calls
            # (Davidson, SVD, eigh). Under the OpenMP default, Torch's OpenMP
            # and MKL workers spin after each parallel region and starve the
            # OpenBLAS threads on the same cores: a pinned 2-thread ground state
            # took 57 s instead of 17 s. The policy is read once when Torch's
            # OpenMP runtime loads, so it is only defaulted before that and
            # never overrides a policy the user chose.
            os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
        try:
            import torch
        except (ImportError, OSError) as error:
            raise ImportError('PyTorch is not installed or cannot be loaded') from error
        self._torch = torch
        if device != 'cpu' and not (isinstance(device,str) and device.startswith('cuda:') and device[5:].isdigit()):
            raise ValueError(f'unsupported Torch device {device}')
        self._device = torch.device(device)
        if self._device.type == 'cuda' and (not torch.cuda.is_available() or self._device.index >= torch.cuda.device_count()):
            raise ValueError(f'Torch device {device} unavailable')
        self.ndarray = torch.Tensor
        self.device_array_types = (torch.Tensor,)
        self.array_types = (np.ndarray, torch.Tensor)
        self._dtype_map = {np.dtype(name): getattr(torch,name) for name in (
            'bool','int8','uint8','int16','int32','int64','float16','float32','float64','complex64','complex128')}
        self._reverse_dtypes = {value:key for key,value in self._dtype_map.items()}
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(2019)
        self.random = _TorchRandom(self)
        self.linalg = _TorchLinalg(self)
        self.array_namespace = self
        self.memory_errors = (MemoryError, torch.OutOfMemoryError)
        super().__init__()
        if os.environ.get('RENO_FP32') is not None:
            self.use_32bits()

    def current_device(self):
        return str(self._device)

    def owns(self,x):
        return isinstance(x,self._torch.Tensor) and x.device == self._device

    def is_array(self,x):
        return isinstance(x,self.array_types)

    def dtype_of(self,x):
        try:
            return self._reverse_dtypes[x.dtype]
        except KeyError as error:
            raise TypeError(f'unsupported Torch dtype {x.dtype}') from error

    def _dtype(self,dtype):
        if isinstance(dtype,self._torch.dtype):
            return dtype
        return self._dtype_map[np.dtype(dtype)]

    def array(self,data,dtype=None,*,copy=True):
        if copy is not None and type(copy) is not bool:
            raise TypeError('copy must be None, True, or False')
        if (copy is None and self.owns(data)
                and (dtype is None or data.dtype == self._dtype(dtype))):
            return data
        if dtype is None:
            dtype = self.dtype_of(data) if isinstance(data,self._torch.Tensor) else np.asarray(data).dtype
        dtype = self._dtype(dtype)
        if copy is False:
            if isinstance(data,self._torch.Tensor):
                if not self.owns(data) or data.dtype != dtype:
                    raise ValueError('copy=False cannot transfer or change dtype')
                return data
            if (not isinstance(data,np.ndarray) or self._device.type != 'cpu'
                    or self._dtype(data.dtype) != dtype
                    or any(stride < 0 for stride in data.strides)
                    or not data.flags.writeable):
                raise ValueError('copy=False requires representable shared CPU storage')
        if isinstance(data,np.ndarray) and (any(stride<0 for stride in data.strides) or not data.flags.writeable):
            data=np.array(data,copy=True,order='C')
        result=self._torch.as_tensor(data,dtype=dtype,device=self._device)
        return result.clone() if copy is True else result

    def asarray(self,data,dtype=None):
        return self.array(data,dtype=dtype,copy=None)

    def from_numpy(self,x,*,copy=None):
        return self.array(x,dtype=x.dtype,copy=copy)

    def to_numpy(self,x,*,copy=None):
        if copy is not None and type(copy) is not bool:
            raise TypeError('copy must be None, True, or False')
        if copy is False and (x.device.type!='cpu' or x.is_conj() or x.is_neg()):
            raise ValueError('copy=False cannot transfer or materialize a tensor')
        result=x.detach().resolve_conj().resolve_neg().cpu().numpy()
        return result.copy() if copy is True else result

    def numpy(self,x):
        if x is None:
            return None
        if isinstance(x,np.ndarray):
            return x
        return self.to_numpy(x)

    @staticmethod
    def _arguments(args, kwargs, names, defaults):
        """Bind the supported NumPy signature without discarding arguments."""
        if len(args) > len(names):
            raise TypeError('too many positional arguments')
        values = dict(defaults)
        for name, value in zip(names, args):
            if name in kwargs:
                raise TypeError(f'multiple values for {name}')
            values[name] = value
        for name, value in kwargs.items():
            if name not in names:
                raise TypeError(f'unsupported argument {name}')
            values[name] = value
        if names[0] not in values:
            raise TypeError(f'missing required argument {names[0]}')
        return values

    def strict_call(self,name,*args,**kwargs):
        t=self._torch
        if name=='astype':
            return args[0].to(dtype=self._dtype(args[1]),copy=kwargs.get('copy',True))
        if name in ('zeros','ones','empty','full','arange'):
            if name == 'full' and isinstance(args[0], (int, np.integer)):
                args = ((int(args[0]),), *args[1:])
            if kwargs.get('dtype') is not None:
                kwargs['dtype']=self._dtype(kwargs['dtype'])
            elif name != 'arange':
                kwargs['dtype']=self._dtype(self.real_dtype)
            return getattr(t,name)(*args,device=self._device,**kwargs)
        if name=='eye':
            values=self._arguments(args,kwargs,('N','M','k','dtype','order'),
                dict(M=None,k=0,dtype=self.real_dtype,order='C'))
            if values['order'] != 'C':
                raise NotImplementedError('Torch eye supports C order only')
            n,m=values['N'],values['M']
            m=n if m is None else m
            dtype=self._dtype(self.real_dtype if values['dtype'] is None else values['dtype'])
            if values['k'] == 0:
                return t.eye(n,m,dtype=dtype,device=self._device)
            result=t.zeros((n,m),dtype=dtype,device=self._device)
            result.diagonal(offset=values['k']).fill_(1)
            return result
        if name=='transpose':
            x,axes=args
            return x.permute(tuple(reversed(range(x.ndim))) if axes is None else tuple(axes))
        if name=='reshape':
            return args[0].reshape(args[1])
        if name=='moveaxis':
            return t.movedim(*args,**kwargs)
        if name=='imag' and not args[0].is_complex():
            return t.zeros_like(args[0])
        if name in ('sum','max','min','all','any'):
            names=('a','axis','dtype','out','keepdims') if name=='sum' else ('a','axis','out','keepdims')
            values=self._arguments(args,kwargs,names,dict(axis=None,out=None,keepdims=False))
            x=self.asarray(values['a'])
            axis=values['axis']
            if isinstance(axis,np.integer):
                axis=int(axis)
            elif isinstance(axis,tuple):
                axis=tuple(int(a) if isinstance(a,np.integer) else a for a in axis)
            keepdims=values['keepdims']
            def finish(result):
                out=values['out']
                if out is not None:
                    out.copy_(result)
                    return out
                return result
            if isinstance(axis,tuple) and not axis:
                if name=='sum':
                    dtype=values.get('dtype')
                    dtype=self._dtype(dtype) if dtype is not None else (x.dtype if x.is_floating_point() or x.is_complex() else t.int64)
                    return finish(x.to(dtype=dtype).clone())
                return finish(x.bool() if name in ('all','any') else x.clone())
            if name=='sum':
                dtype=values.get('dtype')
                return finish(t.sum(x,dim=axis,keepdim=keepdims,dtype=None if dtype is None else self._dtype(dtype)))
            if name in ('max','min'):
                reduce=t.amax if name=='max' else t.amin
                if x.is_complex():
                    real=reduce(x.real,dim=axis,keepdim=True)
                    infinity=-float('inf') if name=='max' else float('inf')
                    imag=reduce(t.where(x.real==real,x.imag,infinity),dim=axis,keepdim=True)
                    out=t.complex(real,imag)
                    if not keepdims:
                        if axis is None:
                            out=out.reshape(())
                        else:
                            axes=(axis,) if isinstance(axis,int) else axis
                            for index in sorted((a%x.ndim for a in axes),reverse=True):
                                out=out.squeeze(index)
                    return finish(out)
                return finish(reduce(x,dim=axis,keepdim=keepdims))
            return finish(getattr(t,name)(x,dim=axis,keepdim=keepdims))
        if name=='linalg.norm':
            kwargs['dim']=kwargs.pop('axis',None)
            kwargs['keepdim']=kwargs.pop('keepdims',False)
        if name.startswith('linalg.'):
            return getattr(t.linalg,name.split('.')[1])(*args,**kwargs)
        return getattr(t,name)(*args,**kwargs)

    def strict_update(self,name,x,idx,value):
        y=x.clone()
        flipped=[]
        if not isinstance(idx,np.ndarray):
            parts=list(idx if isinstance(idx,tuple) else (idx,))
            for axis,part in enumerate(parts):
                if isinstance(part,slice) and part.step is not None and part.step < 0:
                    start,stop,step=part.indices(x.shape[axis])
                    count=len(range(start,stop,step))
                    first=x.shape[axis]-1-start
                    parts[axis]=slice(first,first+count*(-step),-step) if count else slice(0,0)
                    flipped.append(axis)
            if flipped:
                y=self._torch.flip(y,flipped)
                idx=tuple(parts)
        if isinstance(idx,np.ndarray):
            values=self._torch.broadcast_to(value,idx.shape)
            for position,index in enumerate(idx):
                if name=='set':
                    y[int(index)]=values[position]
                elif name=='add':
                    y[int(index)]+=values[position]
                elif name=='sub':
                    y[int(index)]-=values[position]
                else:
                    y[int(index)]*=values[position]
        elif name=='set':
            y[idx]=value
        elif name=='add':
            y[idx]+=value
        elif name=='sub':
            y[idx]-=value
        else:
            y[idx]*=value
        return self._torch.flip(y,flipped) if flipped else y

    def transpose(self,x,axes=None):
        return self.strict_call('transpose',x,axes)

    def identity(self, n, dtype=None):
        # NumPy-style solver identity must stay on this captured device; Torch
        # has eye rather than identity and its global default dtype can differ.
        return self._torch.eye(n, dtype=self._dtype(self.real_dtype if dtype is None else dtype),
                               device=self._device)

    def argmax(self,x,axis=None):
        return self._torch.argmax(self.asarray(x),dim=axis)

    def repeat(self,x,repeats,axis=None):
        if isinstance(repeats,np.ndarray):
            repeats=self.asarray(repeats)
        return self._torch.repeat_interleave(self.asarray(x),repeats,dim=axis)

    def nonzero(self,x):
        return self._torch.nonzero(self.asarray(x),as_tuple=True)

    def equal(self,a,b):
        return self._torch.eq(self.asarray(a),self.asarray(b))

    def unique(self,x):
        return self._torch.unique(self.asarray(x),sorted=True)

    def absolute(self,x):
        return self._torch.abs(self.asarray(x))

    def tensordot(self,a,b,axes=2):
        a,b=self.asarray(a),self.asarray(b)
        dtype=self._torch.promote_types(a.dtype,b.dtype)
        if isinstance(axes,np.integer):
            axes=int(axes)
        if isinstance(axes,(tuple,list)):
            axes=tuple([int(v)] if isinstance(v,(int,np.integer)) else list(v) for v in axes)
        return self._torch.tensordot(a.to(dtype),b.to(dtype),dims=axes)

    def dot(self,a,b):
        a,b=self.asarray(a),self.asarray(b)
        dtype=self._torch.promote_types(a.dtype,b.dtype)
        a,b=a.to(dtype),b.to(dtype)
        if a.ndim == 0 or b.ndim == 0:
            return a*b
        return self._torch.tensordot(a,b,dims=([-1],[-1 if b.ndim == 1 else -2]))

    def finfo(self,dtype):
        # Precision constants are host metadata, not state arrays.
        return np.finfo(self._reverse_dtypes.get(dtype,dtype))

    def empty_like(self,x,dtype=None):
        return self._torch.empty_like(x,dtype=x.dtype if dtype is None else self._dtype(dtype),device=self._device)

    def allclose(self,a,b,rtol=1e-5,atol=1e-8,equal_nan=False):
        a,b=self.asarray(a),self.asarray(b)
        dtype=self._torch.promote_types(a.dtype,b.dtype)
        return self._torch.allclose(a.to(dtype),b.to(dtype),rtol=rtol,atol=atol,equal_nan=equal_nan)

    def iscomplex(self,x):
        x=self.asarray(x)
        return x.imag != 0 if x.is_complex() else self._torch.zeros_like(x,dtype=self._torch.bool)

    def diff(self,x,n=1,axis=-1):
        return self._torch.diff(self.asarray(x),n=n,dim=axis)

    def searchsorted(self,a,v,side='left',sorter=None):
        if side not in ('left','right'):
            raise ValueError("side must be 'left' or 'right'")
        return self._torch.searchsorted(self.asarray(a),self.asarray(v),right=side=='right',sorter=sorter)

    def sort(self,a,axis=-1,kind=None,order=None):
        if order is not None or kind not in (None,'stable'):
            raise NotImplementedError('Torch sort supports default or stable ordering without fields')
        a=self.asarray(a)
        if axis is None:
            a,axis=a.reshape(-1),-1
        return self._torch.sort(a,dim=axis,stable=kind=='stable').values

    def vdot(self,a,b):
        a,b=self.asarray(a),self.asarray(b)
        dtype=self._torch.promote_types(a.dtype,b.dtype)
        return self._torch.vdot(a.to(dtype).reshape(-1),b.to(dtype).reshape(-1))

    def maximum(self,a,b):
        a,b=self.asarray(a),self.asarray(b)
        dtype=self._torch.promote_types(a.dtype,b.dtype)
        return self._torch.maximum(a.to(dtype),b.to(dtype))

    def minimum(self,a,b):
        a,b=self.asarray(a),self.asarray(b)
        dtype=self._torch.promote_types(a.dtype,b.dtype)
        return self._torch.minimum(a.to(dtype),b.to(dtype))

    def concatenate(self,arrays,axis=0):
        return self._torch.cat(tuple(arrays),dim=axis)

    def ascontiguousarray(self,x,dtype=None):
        return self.asarray(x,dtype=dtype).contiguous()

    def iscomplexobj(self,x):
        return x.is_complex() if isinstance(x,self._torch.Tensor) else np.iscomplexobj(x)

    def __getattr__(self,name):
        # A bounded NumPy-style surface. Namespace availability does not imply
        # strict support; contexts advertise only tested operation contracts.
        if name in ('float32','float64','complex64','complex128','int32','int64','bool_'):
            return getattr(np,name)
        if name in ('zeros','ones','empty','full','eye','arange','sum','max','min','all','any',
                    # Dense output broadcasts natively, then explicitly copies
                    # before any private workspace write to expanded storage.
                    'reshape','moveaxis','broadcast_to','conj','real','imag','abs','sqrt','exp','log','sin','cos',
                    'isfinite','matmul','einsum','diag','diagonal','stack','hstack','vstack',
                    'allclose','isclose','where','sign','argsort','sort','clip'):
            return lambda *args,**kwargs:self.strict_call(name,*args,**kwargs)
        raise AttributeError(name)

    def sync(self):
        if self._device.type=='cuda':
            self._torch.cuda.synchronize(self._device)


class _TorchLinalg:
    def __init__(self,backend):
        self.backend=backend

    def __getattr__(self,name):
        if name not in ('qr','svd','eigh','eigvalsh','solve','norm','inv'):
            raise AttributeError(name)
        return lambda *args,**kwargs:self.backend.strict_call('linalg.'+name,*args,**kwargs)


class _TorchRandom:
    def __init__(self,backend):
        self.backend=backend

    def seed(self,value):
        self.backend._generator.manual_seed(value)

    @staticmethod
    def _shape(size):
        return () if size is None else ((int(size),) if isinstance(size,(int,np.integer)) else tuple(size))

    def uniform(self,low=0.,high=1.,size=None,dtype=None):
        b=self.backend
        return b._torch.rand(self._shape(size),dtype=b._dtype(dtype or b.real_dtype),device=b._device,generator=b._generator)*(high-low)+low

    def normal(self,loc=0.,scale=1.,size=None,dtype=None):
        b=self.backend
        return b._torch.randn(self._shape(size),dtype=b._dtype(dtype or b.real_dtype),device=b._device,generator=b._generator)*scale+loc

    def rand(self,*shape):
        return self.uniform(size=shape)

    def randn(self,*shape):
        return self.normal(size=shape)

    def random(self,size=None):
        return self.uniform(size=size)

    def randint(self,low,high=None,size=None,dtype=np.int64):
        if high is None:
            low,high=0,low
        b=self.backend
        return b._torch.randint(low,high,self._shape(size),dtype=b._dtype(dtype),device=b._device,generator=b._generator)
