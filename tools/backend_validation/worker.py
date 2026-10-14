"""Trusted reviewed CPU entry harness, copied into each run without scorer code.

It receives only execution inputs. It does not compute a verdict or benchmark.
The reviewed subprocess is not a security sandbox.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys


def main():
    root=Path.cwd()
    limits=json.loads((root/'limits.json').read_text())
    resource.setrlimit(resource.RLIMIT_AS,(limits['max_memory_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(limits['max_file_bytes'],)*2)
    cpu_seconds=max(1,math.ceil(limits['timeout_seconds'])+1)
    resource.setrlimit(resource.RLIMIT_CPU,(cpu_seconds,)*2)
    import numpy as np
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['device']!='cpu':
        raise ValueError('CPU worker requires cpu device')
    backing=[]; inputs=[]; fingerprints=[]
    for desc in manifest['inputs']:
        path=root/'inputs'/desc['file']
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=desc['sha256']:
            raise ValueError('input_hash_mismatch')
        array=np.load(path,allow_pickle=False,max_header_size=4096)
        backing.append(array)
        fingerprints.append(hashlib.sha256(array.tobytes(order='C')).digest())
        view=np.ndarray(tuple(desc['shape']),dtype=array.dtype,buffer=array,
                        offset=desc['offset'],strides=tuple(desc['strides']))
        view.flags.writeable=False
        inputs.append(view)
    sys.path.insert(0,str(root/'code'))
    spec=importlib.util.spec_from_file_location('reviewed_candidate',root/'code'/'candidate.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output=module.candidate(*inputs)
    for array,fingerprint in zip(backing,fingerprints):
        if hashlib.sha256(array.tobytes(order='C')).digest()!=fingerprint:
            raise ValueError('input_mutation')
    if not isinstance(output,np.ndarray) or output.dtype.kind not in 'fc':
        raise TypeError('candidate must return one numeric NumPy array')
    single_output='--single-output' in sys.argv
    with (root/'output'/'result.npy').open('wb' if single_output else 'xb') as stream:
        np.save(stream,output,allow_pickle=False)
    if single_output:
        return
    witness={'device':'cpu','dtype':str(output.dtype),'shape':list(output.shape),
             'input_strides':[list(a.strides) for a in inputs],
             'input_offsets':[d['offset'] for d in manifest['inputs']]}
    with (root/'output'/'witness.json').open('x') as stream:
        json.dump(witness,stream,sort_keys=True,allow_nan=False)


if __name__=='__main__':
    main()
