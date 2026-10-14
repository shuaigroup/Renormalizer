"""Parent-owned admission, worker lifetime and independent candidate scoring."""
import hashlib
import json
import math
import numpy as np
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from . import scoring, sandbox
from .policy import admit, source_bytes
from .protocol import (ProtocolError, MAX_ELEMENTS, MAX_FILE_BYTES, validate_manifest,
    candidate_manifest, read_file, load_array, reconstruct, array_descriptor)


def _json(data):
    def pairs(values):
        result={}
        for key,value in values:
            if key in result: raise ProtocolError('duplicate JSON key')
            result[key]=value
        return result
    def constant(value):
        raise ProtocolError('nonfinite JSON value')
    try:
        return json.loads(data,object_pairs_hook=pairs,parse_constant=constant)
    except (ValueError,UnicodeError) as error:
        raise ProtocolError('invalid JSON output') from error


def _finish_group(process):
    """Terminate the session/process group created for this run, never a GPU."""
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=.2)
    except subprocess.TimeoutExpired:
        pass
    # Also kill descendants remaining after the leader exited normally.
    try:
        os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def run_candidate(full_manifest, candidate, policy):
    """Run a hash-reviewed CPU entry, then score its numeric output in parent.

    Return JSON-safe status/reason/stage, source identities, worker exit code and
    logical layout witness. Untrusted CPU workers require the validated isolation provider. GPU execution
    requires a separate trusted harness; it never silently becomes CPU work.
    Temporary run files are cleaned; raw failure stderr is bounded and included.
    """
    stage='manifest'
    result={'status':'fail','stage':stage,'reason':'not_started',
            'execution_seconds':None,'worker_exit_code':None,
            'resource_coverage':{'address_space':'per-process RLIMIT_AS',
                'file_bytes':'per-file RLIMIT_FSIZE','cpu_time':'per-process RLIMIT_CPU',
                'process_cleanup':'owned process group; escaped sessions unknown',
                'total_job_memory':'unknown','total_output_bytes':'unknown',
                'security_sandbox':False,'extra_copy_bytes':'unknown'}}
    try:
        # Freeze the caller-owned object before launch; candidate cannot choose thresholds.
        full=_json(json.dumps(full_manifest,allow_nan=False))
        validate_manifest(full)
        result.update(run_id=full['run_id'],candidate_hash=full['candidate_hash'],
                      scorer_hash=full['scorer_hash'],tolerance_hash=full['tolerance_hash'])
        stage='admission'
        if full['mode']=='untrusted' and getattr(policy,'provider_config',None):
            protected=[Path(full['reference_directory']).resolve(),Path(full['input_directory']).resolve(),
                       Path(scoring.__file__).resolve()]
            roots=['/usr','/lib','/lib64']+list(policy.provider_config.get('runtime_paths',[]))
            if any(target.is_relative_to(Path(bound).resolve()) for target in protected for bound in roots):
                return {**result,'status':'blocked','stage':stage,'reason':'runtime_mount_exposes_protected_files'}
        decision=admit(full['mode'],full['device'],candidate=candidate,policy=policy)
        if not decision.allowed:
            return {**result,'status':'blocked','stage':stage,'reason':decision.reason}
        candidate_data=source_bytes(candidate)
        if hashlib.sha256(candidate_data).hexdigest()!=full['candidate_hash'] or (full['mode']=='reviewed' and full['candidate_hash']!=policy.review.source_hash):
            raise ProtocolError('manifest candidate hash differs from reviewed source')
        if hashlib.sha256(Path(scoring.__file__).read_bytes()).hexdigest()!=full['scorer_hash']:
            raise ProtocolError('scorer hash mismatch')
        stage='inputs'
        limits=full['resource_policy']
        input_data=[]
        for desc in full['inputs']:
            array=load_array(full['input_directory'],desc,max_file_bytes=limits['max_file_bytes'])
            if not np.isfinite(array).all():
                raise ProtocolError('nonfinite input')
            data=read_file(full['input_directory'],desc['file'],max_bytes=limits['max_file_bytes'])
            if hashlib.sha256(data).hexdigest()!=desc['sha256']:
                raise ProtocolError('input changed during staging')
            input_data.append(data)
        reference=reconstruct(load_array(full['reference_directory'],full['reference'],
            max_file_bytes=limits['max_file_bytes']),full['reference'])
        if not np.isfinite(reference).all():
            raise ProtocolError('nonfinite reference')
        with tempfile.TemporaryDirectory(prefix='reno-candidate-') as temporary:
            root=Path(temporary)
            for name in ('inputs','code','output','home','tmp'):
                (root/name).mkdir()
            visible=candidate_manifest(full)
            manifest_bytes=json.dumps(visible,sort_keys=True,allow_nan=False).encode()
            (root/'manifest.json').write_bytes(manifest_bytes)
            (root/'limits.json').write_text(json.dumps(limits,allow_nan=False))
            for desc,data in zip(full['inputs'],input_data):
                target=root/'inputs'/desc['file']; target.write_bytes(data); target.chmod(0o444)
            (root/'code'/'candidate.py').write_bytes(candidate_data)
            for name,path,expected in (() if policy.review is None else policy.review.dependencies):
                data=source_bytes(path)
                if hashlib.sha256(data).hexdigest()!=expected:
                    raise ProtocolError('reviewed dependency changed during staging')
                (root/'code'/name).write_bytes(data)
            (root/'worker.py').write_bytes(Path(__file__).with_name('worker.py').read_bytes())
            env={'PATH':os.defpath,'HOME':str(root/'home'),'TMPDIR':str(root/'tmp'),
                 'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1',
                 'PYTHONDONTWRITEBYTECODE':'1'}
            stage='worker'
            started=time.perf_counter()
            if full['mode']=='untrusted':
                (root/'output'/'result.npy').touch()
                launch_spec={'argv':[sys.executable,'-I',str(root/'worker.py'),'--single-output'],
                    'cwd':str(root),'env':{'PATH':os.defpath,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},'stdout':str(root/'stdout.log'),
                    'stderr':str(root/'stderr.log'),
                    'readonly_paths':[str(root/'inputs'),str(root/'code'),str(root/'worker.py'),
                                      str(root/'manifest.json'),str(root/'limits.json')],
                    'output_file':str(root/'output'/'result.npy'),
                    'timeout':limits['timeout_seconds'],
                    'limits':{'memory_bytes':limits['max_memory_bytes'],'max_file_bytes':limits['max_file_bytes'],
                              'cpu_seconds':max(1,math.ceil(limits['timeout_seconds'])+1)},
                    'provider_config':policy.provider_config}
                try:
                    process=sandbox.launch(launch_spec)
                except (OSError, ValueError, KeyError) as error:
                    return {**result,'status':'blocked','stage':stage,'reason':str(error)}
                result['resource_coverage']['security_sandbox']=True
            else:
                with (root/'stdout.log').open('wb') as stdout,(root/'stderr.log').open('wb') as stderr:
                    process=subprocess.Popen([sys.executable,'-I',str(root/'worker.py')],cwd=root,
                        env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                        start_new_session=True,close_fds=True)
            timed_out=False
            try:
                process.wait(timeout=limits['timeout_seconds'])
            except subprocess.TimeoutExpired:
                timed_out=True
            finally:
                _finish_group(process)
            result['execution_seconds']=time.perf_counter()-started
            result['worker_exit_code']=process.returncode
            result['stderr']=(root/'stderr.log').read_bytes()[-8192:].decode(errors='replace')
            if timed_out:
                return {**result,'stage':stage,'reason':'timeout'}
            stage='integrity'
            if (root/'manifest.json').read_bytes()!=manifest_bytes:
                raise ProtocolError('worker manifest mutation')
            for desc in full['inputs']:
                data=read_file(root/'inputs',desc['file'],max_bytes=limits['max_file_bytes'])
                if hashlib.sha256(data).hexdigest()!=desc['sha256']:
                    raise ProtocolError('input file mutation')
            if process.returncode!=0:
                return {**result,'stage':'worker','reason':'worker_failed'}
            stage='output'
            expected_files={'result.npy'} if full['mode']=='untrusted' else {'result.npy','witness.json'}
            if set(os.listdir(root/'output'))!=expected_files:
                raise ProtocolError('missing or unexpected output file')
            desc=array_descriptor(root/'output'/'result.npy',expected_shape=full['output']['shape'],
                                  expected_dtype=full['output']['dtype'])
            output=load_array(root/'output',desc,max_file_bytes=limits['max_file_bytes'])
            expected_witness={'device':'cpu','dtype':str(output.dtype),'shape':list(output.shape),
                'input_strides':[d['strides'] for d in full['inputs']],
                'input_offsets':[d['offset'] for d in full['inputs']]}
            witness=expected_witness if full['mode']=='untrusted' else _json(read_file(root/'output','witness.json',max_bytes=4096))
            if witness!=expected_witness:
                raise ProtocolError('invalid trusted worker witness')
            stage='scoring'
            verdict=scoring.score_array(output,reference,tolerances=full['tolerances'])
            return {**result,**verdict,'stage':stage,'witness':witness,
                    'input_hashes':[d['sha256'] for d in full['inputs']],
                    'witness_source':('trusted reviewed worker' if full['mode']=='reviewed' else
                                      'isolated CPU output; input strides are declared, not independently observed'),
                    'logical_input_bytes':sum(math.prod(d['shape'])*np.dtype(d['dtype']).itemsize for d in full['inputs']),
                    'logical_output_bytes':output.nbytes}
    except (ProtocolError, ValueError, TypeError, OSError) as error:
        return {**result,'stage':stage,'reason':str(error),'exception_type':type(error).__name__}
