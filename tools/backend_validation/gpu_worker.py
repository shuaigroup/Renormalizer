"""Trusted fixed control worker; never loads arbitrary candidate code."""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
import cupy as cp
import types
from .timing import measure_reviewed

REVIEWED_RAWKERNEL_SHA256='f4507c9da0cce2ea487086ab4edaf1f579ab791f25a82f445cc83d9cb37ece44'

def main():
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--timing',action='store_true');a=p.parse_args()
    source_path=Path(__file__).with_name('rawkernel.py')
    source_bytes=source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest()!=REVIEWED_RAWKERNEL_SHA256:
        raise ValueError('reviewed kernel source changed')
    rawkernel=types.ModuleType('reviewed_rawkernel')
    exec(compile(source_bytes,str(source_path),'exec'),rawkernel.__dict__)
    d=Path(a.directory)
    if any((d/name).stat().st_size>2*1024*1024 for name in ('a.npy','b.npy')):
        raise ValueError('input file budget exceeded')
    x=np.load(d/'a.npy',allow_pickle=False,mmap_mode='r',max_header_size=4096);y=np.load(d/'b.npy',allow_pickle=False,mmap_mode='r',max_header_size=4096)
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[0] or max(*x.shape,*y.shape)>256:
        raise ValueError('fixed control shape/work budget exceeded')
    if not x.flags.c_contiguous or not y.flags.c_contiguous:
        raise ValueError('noncontiguous control input')
    if x.nbytes+y.nbytes>16*1024*1024 or x.dtype!=np.float64 or y.dtype!=np.float64:
        raise ValueError('control input budget/dtype exceeded')
    with cp.cuda.Device(0):
        if a.timing:
            out,candidate=measure_reviewed(cp,rawkernel.matmul,[x,y],repeats=5,warmup=1)
            baseline_out,baseline=measure_reviewed(cp,lambda u,v:u@v,[x,y],repeats=5,warmup=1)
            np.save(d/'baseline_result.npy',baseline_out,allow_pickle=False)
            if any(s['t_kernel']<=0 or s['t_e2e']<s['t_kernel'] for data in (candidate,baseline) for s in data['samples']):
                raise ValueError('timing calibration failed')
            evidence={'candidate':candidate,'baseline':baseline,'known_matmul_flops':2*x.shape[0]*x.shape[1]*y.shape[1], 'calibration':'same-shape CuPy matmul; positive device clock and enclosing wall clock'}
        else:
            result=rawkernel.matmul(cp.asarray(x),cp.asarray(y));out=result.get()
            evidence={'device':f'cuda:{result.device.id}','dtype':str(result.dtype),'shape':list(result.shape),'cupy_version':cp.__version__}
        np.save(d/'result.npy',out,allow_pickle=False)
        (d/'evidence.json').write_text(json.dumps(evidence,allow_nan=False))
if __name__=='__main__':main()
