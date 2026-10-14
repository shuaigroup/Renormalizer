import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import pytest
from tools.backend_validation.protocol import freeze_manifest, array_descriptor
from tools.backend_validation.policy import ExecutionPolicy, review_source
from tools.backend_validation.runner import run_candidate


def setup_run(tmp_path, source):
    inputs=tmp_path/'input'; inputs.mkdir()
    reference=tmp_path/'secret'; reference.mkdir()
    a=np.arange(15,dtype='float64').reshape(3,5)
    b=np.arange(10,dtype='float64').reshape(5,2)
    np.save(inputs/'a.npy',a); np.save(inputs/'b.npy',b)
    np.save(reference/'reference.npy',a@b)
    candidate=tmp_path/'candidate.py'; candidate.write_text(source)
    review=review_source(candidate)
    manifest=freeze_manifest(inputs,reference,candidate_hash=review.source_hash,
                            source_hash='a'*64,environment_hash='b'*64)
    return manifest,candidate,ExecutionPolicy(review=review)


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS worker limits are only supported on Linux")
@pytest.mark.parametrize('source,status,reason',[
    ('def candidate(a,b): return a @ b\n','pass','accepted'),
    ('import numpy as np\ndef candidate(a,b): return np.zeros((a.shape[0],b.shape[1]))\n','fail','numerical_error'),
    ('def candidate(a,b): return (a @ b).astype("float32")\n','fail','dtype_mismatch'),
    ('def candidate(a,b): return (a @ b).ravel()\n','fail','shape_mismatch'),
    ('def candidate(a,b): return (a @ b) * float("nan")\n','fail','nonfinite'),
    ('def candidate(a,b): raise RuntimeError("crash")\n','fail','worker_failed'),
    ('import os\ndef candidate(a,b): os._exit(0)\n','fail','missing or unexpected output file'),
    ('def candidate(a,b): return {"passed":True,"timing":0}\n','fail','worker_failed'),
])
def test_runner_adversarial_controls(tmp_path, source, status, reason):
    manifest,candidate,policy=setup_run(tmp_path,source)
    result=run_candidate(manifest,candidate,policy)
    assert result['status']==status, result
    assert result['reason']==reason, result
    json.dumps(result,allow_nan=False)


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS worker limits are only supported on Linux")
def test_timeout_and_input_mutation(tmp_path):
    manifest,candidate,policy=setup_run(tmp_path,'import time\ndef candidate(a,b): time.sleep(30)\n')
    manifest['resource_policy']['timeout_seconds']=.25
    assert run_candidate(manifest,candidate,policy)['reason']=='timeout'
    candidate.write_text('def candidate(a,b):\n a.flags.writeable=True\n a[0,0]=999\n return a @ b\n')
    review=review_source(candidate); manifest['candidate_hash']=review.source_hash
    # Restore a normal deadline so failure is the mutation detector, not timeout.
    manifest['resource_policy']['timeout_seconds']=10
    result=run_candidate(manifest,candidate,ExecutionPolicy(review=review))
    assert result['reason']=='worker_failed' and 'input_mutation' in result['stderr']


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS worker limits are only supported on Linux")
def test_projection_omits_reference_and_extra_output_is_rejected(tmp_path):
    source='''import json
from pathlib import Path
def candidate(a,b):
 manifest=json.loads(Path('manifest.json').read_text())
 assert set(manifest)=={'schema_version','run_id','op','inputs','output','device','layout','mutation_policy'}
 assert not Path('reference.npy').exists()
 Path('output/forged.json').write_text('{"passed":true}')
 return a@b
'''
    manifest,candidate,policy=setup_run(tmp_path,source)
    result=run_candidate(manifest,candidate,policy)
    assert result['reason']=='missing or unexpected output file'


def test_untrusted_source_never_runs(tmp_path):
    sentinel=tmp_path/'sentinel'
    manifest,candidate,policy=setup_run(tmp_path,f'from pathlib import Path\nPath({str(sentinel)!r}).touch()\n')
    manifest['mode']='untrusted'
    assert run_candidate(manifest,candidate,policy)['status']=='blocked'
    assert not sentinel.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS worker limits are only supported on Linux")
def test_oversized_output_is_rejected(tmp_path):
    # Symlinked outputs are rejected by protocol.read_file; see
    # test_protocol.py::test_symlink_header_and_budget_rejection.
    manifest,candidate,_=setup_run(tmp_path,'def candidate(a,b): return a @ b\n')
    candidate.write_text('import numpy as np\ndef candidate(a,b): return np.ones((100,100))\n')
    review=review_source(candidate); manifest['candidate_hash']=review.source_hash
    manifest['resource_policy']['max_file_bytes']=1024
    result=run_candidate(manifest,candidate,ExecutionPolicy(review=review))
    assert result['status']=='fail'


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS worker limits are only supported on Linux")
def test_layout_reconstruction_witness(tmp_path):
    manifest,candidate,policy=setup_run(tmp_path,'def candidate(a,b):\n assert not a.flags.c_contiguous\n return a @ b\n')
    input_dir=Path(manifest['input_directory']); ref_dir=Path(manifest['reference_directory'])
    backing=np.arange(30,dtype='float64')
    np.save(input_dir/'a.npy',backing)
    desc=array_descriptor(input_dir/'a.npy',shape=(3,5),strides=(80,16))
    manifest['inputs'][0]=desc
    b=np.load(input_dir/'b.npy')
    np.save(ref_dir/'reference.npy',backing.reshape(3,10)[:,::2]@b)
    manifest['reference']=array_descriptor(ref_dir/'reference.npy')
    result=run_candidate(manifest,candidate,policy)
    assert result['status']=='pass',result
    assert result['witness']['input_strides'][0]==[80,16]


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS worker limits are only supported on Linux")
def test_parent_environment_not_inherited(tmp_path,monkeypatch):
    monkeypatch.setenv('RENOVALIDATOR_TEST_SECRET','private-test-value')
    manifest,candidate,policy=setup_run(tmp_path,'''import os
def candidate(a,b):
 assert 'RENOVALIDATOR_TEST_SECRET' not in os.environ
 return a@b
''')
    assert run_candidate(manifest,candidate,policy)['status']=='pass'


@pytest.mark.skipif(not Path('/proc/self/stat').exists(), reason='process state is read from Linux /proc')
@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS worker limits are only supported on Linux")
def test_owned_child_is_terminated_after_leader_exits(tmp_path):
    pidfile=tmp_path/'owned-child-pid'
    source=f'''import subprocess,sys
from pathlib import Path
def candidate(a,b):
 child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
 Path({str(pidfile)!r}).write_text(str(child.pid))
 return a@b
'''
    manifest,candidate,policy=setup_run(tmp_path,source)
    result=run_candidate(manifest,candidate,policy)
    assert result['status']=='pass',result
    pid=int(pidfile.read_text())
    # An orphan zombie can await the host init reaper, but must not be running.
    import time
    for _ in range(30):
        path=Path(f'/proc/{pid}/stat')
        if not path.exists() or path.read_text().split()[2]=='Z':
            break
        time.sleep(.01)
    else:
        pytest.fail('owned child still running after worker cleanup')


def test_runtime_mount_cannot_expose_reference_or_scorer(tmp_path):
    manifest,candidate,_=setup_run(tmp_path,'def candidate(a,b): return a@b\n')
    manifest['mode']='untrusted'
    policy=ExecutionPolicy(provider_config={'runtime_paths':[str(tmp_path)]})
    result=run_candidate(manifest,candidate,policy)
    assert result['status']=='blocked'
    assert result['reason']=='runtime_mount_exposes_protected_files'


def test_out_of_range_manifest_tolerance_fails_before_launch(tmp_path, monkeypatch):
    import subprocess
    manifest, candidate, policy = setup_run(tmp_path, 'def candidate(a,b): return a@b\n')
    manifest['tolerances']['atol_F'] = 10**400
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('must not launch'))
    result = run_candidate(manifest, candidate, policy)
    assert result['status'] == 'fail'
    assert result['stage'] == 'manifest'
    assert 'finite nonnegative numbers' in result['reason']
