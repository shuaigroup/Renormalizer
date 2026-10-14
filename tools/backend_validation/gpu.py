"""Parent-scored, fixed reviewed GPU control; arbitrary GPU candidates refused."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
from .scoring import score_array


def run_reviewed_gpu(a,b,reference,*,tolerances):
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0] or a.dtype!=np.float64 or b.dtype!=np.float64:
        raise ValueError('reviewed control requires compatible float64 matrices')
    if max(*a.shape,*b.shape)>256:
        raise ValueError('fixed control shape/work budget exceeded')
    if not a.flags.c_contiguous or not b.flags.c_contiguous or a.nbytes+b.nbytes>16*1024*1024:
        raise ValueError('reviewed control layout/budget exceeded')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite inputs')
    with tempfile.TemporaryDirectory(prefix='reno-reviewed-gpu-') as directory:
        path=Path(directory);np.save(path/'a.npy',a,allow_pickle=False);np.save(path/'b.npy',b,allow_pickle=False)
        env=dict(os.environ)
        if not env.get('CUDA_VISIBLE_DEVICES'):
            raise ValueError('explicit allocated CUDA visibility required')
        def launch(timing=False):
            start=time.perf_counter()
            cmd=[sys.executable,'-m','tools.backend_validation.gpu_worker',directory]+(['--timing'] if timing else [])
            done=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=90,check=True)
            elapsed=time.perf_counter()-start
            result=np.load(path/'result.npy',allow_pickle=False)
            return result,json.loads((path/'evidence.json').read_text()),elapsed
        value,witness,cold=launch()
        verdict=score_array(value,reference,tolerances=tolerances)
        if verdict['status']!='pass':return {'status':'fail','correctness':verdict,'device_witness':witness}
        # Wrong-index and omitted-tail controls are host mutations of the same
        # computed result: no additional unreviewed GPU kernel is launched.
        wrong=np.roll(value.ravel(),1).reshape(value.shape)
        tail=value.copy()
        if tail.size:tail.ravel()[-1]+=1
        controls=[score_array(x,reference,tolerances=tolerances) for x in (wrong,tail)]
        if any(x['status']!='fail' for x in controls):
            raise ValueError('fixture does not distinguish both negative controls')
        value,measurements,total=launch(True)
        final=score_array(value,reference,tolerances=tolerances)
        baseline_value=np.load(path/'baseline_result.npy',allow_pickle=False)
        baseline_score=score_array(baseline_value,reference,tolerances=tolerances)
        for name, scored in (('candidate',value),('baseline',baseline_value)):
            data=measurements[name]
            expected=hashlib.sha256(scored.tobytes(order='C')).hexdigest()
            if data.get('output_hashes') != [expected]*(1+data['warmup']+data['repeats']):
                raise ValueError('timed outputs differ from scored output')
        if baseline_score['status'] != 'pass':
            raise ValueError('baseline correctness failed')
        return dict(status=final['status'],correctness=final,negative_controls=controls,
                    device_witness=witness,t_cold=cold,t_cold_boundary='process/import/compile/transfer/result completion',
                    timing_process_seconds=total,
                    timings={**measurements, 't_cold': cold,
                             't_kernel': measurements['candidate']['summary']['t_kernel']['median'],
                             't_e2e': measurements['candidate']['summary']['t_e2e']['median']},
                    resource_coverage='reviewed owned subprocess; pool endpoint samples; no physical GPU cap')
