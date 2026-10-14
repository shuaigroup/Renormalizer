"""Actual provider/NumPy integration; absent pinned config is a coverage gap."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import pytest
from tools.backend_validation.policy import ExecutionPolicy
from tools.backend_validation.protocol import freeze_manifest
from tools.backend_validation.runner import run_candidate
from tools.backend_validation import scoring


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS worker limits are only supported on Linux")


@pytest.fixture
def isolated(tmp_path):
    path=os.environ.get('RENO_TEST_SANDBOX_CONFIG')
    if not path:
        pytest.skip('actual isolation provider configuration absent; untrusted coverage remains unverified')
    config=json.loads(Path(path).read_text())
    inputs=tmp_path/'inputs';inputs.mkdir()
    refs=tmp_path/'private-reference';refs.mkdir()
    np.save(inputs/'a.npy',np.eye(2));np.save(inputs/'b.npy',np.eye(2))
    np.save(refs/'reference.npy',np.eye(2))
    def run(source, timeout=10, max_file_bytes=4096):
        candidate=tmp_path/'candidate.py';candidate.write_text(source)
        manifest=freeze_manifest(inputs,refs,mode='untrusted',
            candidate_hash=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            source_hash='a'*64,environment_hash='b'*64)
        manifest['resource_policy'].update(timeout_seconds=timeout,max_file_bytes=max_file_bytes)
        return run_candidate(manifest,candidate,ExecutionPolicy(provider_config=config))
    return run,refs


def test_actual_isolated_numeric_and_hidden_reference(isolated):
    run,refs=isolated
    result=run(f'''from pathlib import Path
def candidate(a,b):
 assert not Path({str(refs)!r}).exists()
 assert not Path({str(Path(scoring.__file__).resolve())!r}).exists()
 for name in ['/tmp/forbidden-file','/forbidden-file','output/extra.npy']:
  try:
   Path(name).write_text('forbidden')
  except OSError:
   pass
  else:
   raise AssertionError('outside output write succeeded')
 return a@b
''')
    assert result['status']=='pass',result
    assert result['resource_coverage']['security_sandbox'] is True


def test_actual_file_limit_and_timeout(isolated):
    run,_=isolated
    result=run('''def candidate(a,b):
 with open('output/result.npy','wb') as stream:
  stream.write(b'x'*8192)
 return a@b
''',max_file_bytes=1024)
    assert result['status']=='fail' and result['reason']=='worker_failed',result
    result=run('import time\ndef candidate(a,b): time.sleep(30)\n',timeout=.25)
    assert result['reason']=='timeout',result


def test_actual_provider_denies_child_creation(isolated):
    import uuid
    import time
    marker='reno-isolation-child-'+uuid.uuid4().hex
    run,_=isolated
    result=run(f'''import subprocess,sys
from pathlib import Path
def candidate(a,b):
 child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',{marker!r}],start_new_session=True)
 return a@b
''')
    assert result['status']=='fail' and result['reason']=='worker_failed',result
    # Inspect only the unique child argv marker created by this test.
    def own_live_pids():
        found=[]
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                argv=(entry/'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError,PermissionError,ProcessLookupError):
                continue
            if marker.encode() in argv:
                found.append(int(entry.name))
        return found
    for _ in range(30):
        alive=own_live_pids()
        if not alive:
            break
        time.sleep(.01)
    assert not alive, f'owned detached children remain alive: {alive}'
