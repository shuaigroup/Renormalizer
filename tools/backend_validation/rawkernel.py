"""Reviewable contiguous float64 matmul control, one thread per output."""
import hashlib
import numpy as np

SOURCE = r'''
extern "C" __global__ void matmul64(const double* a, const double* b,
 double* c, const long long m, const long long k, const long long n) {
 const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
 if (i >= m*n) return;
 const long long row = i/n, col = i%n;
 double sum = 0;
 for (long long j=0; j<k; ++j) sum += a[row*k+j]*b[j*n+col];
 c[i] = sum;
}
'''
OPTIONS = ('--std=c++11',)
SOURCE_SHA256 = hashlib.sha256(SOURCE.encode()).hexdigest()


def matmul(a, b):
    import cupy as cp
    if not isinstance(a, cp.ndarray) or not isinstance(b, cp.ndarray):
        raise TypeError('CuPy operands required')
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('rank or contraction shape mismatch')
    if a.dtype != np.dtype('float64') or b.dtype != np.dtype('float64'):
        raise TypeError('float64 only')
    if not a.flags.c_contiguous or not b.flags.c_contiguous:
        raise ValueError('C contiguous operands required')
    if a.device.id != b.device.id:
        raise ValueError('operands must share device')
    m, k = a.shape; n = b.shape[1]
    if max(m*k, k*n, m*n) >= 2**63:
        raise ValueError('index domain exceeded')
    with a.device:
        out = cp.empty((m,n), dtype=cp.float64)
        if out.size:
            kernel = cp.RawKernel(SOURCE, 'matmul64', options=OPTIONS)
            kernel(((out.size+127)//128,), (128,), (a,b,out,np.int64(m),np.int64(k),np.int64(n)))
    return out
