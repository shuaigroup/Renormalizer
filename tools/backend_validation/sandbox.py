"""Pinned bubblewrap provider. No ordinary subprocess fallback for untrusted code."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import ctypes
import errno
import json
import os
import resource
import signal
import subprocess
import tempfile

@dataclass(frozen=True)
class IsolationProbe:
    verified: bool
    reason: str
    discovered_bwrap: str | None = None
    discovered_unshare: str | None = None
    config: dict | None = None


def _check_config(config):
    if not isinstance(config, dict):
        raise ValueError('pinned provider configuration required')
    exe = Path(config['executable']).resolve(strict=True)
    if hashlib.sha256(exe.read_bytes()).hexdigest() != config['sha256']:
        raise ValueError('provider executable hash mismatch')
    return str(exe)


def _seccomp_fd():
    """Prevent nested user namespaces from restoring mount capabilities."""
    lib=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    lib.seccomp_rule_add_array.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint,ctypes.c_void_p]
    lib.seccomp_export_bpf.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.seccomp_release.argtypes=[ctypes.c_void_p]
    class Compare(ctypes.Structure):
        _fields_=[('arg',ctypes.c_uint),('op',ctypes.c_int),('a',ctypes.c_uint64),('b',ctypes.c_uint64)]
    context=lib.seccomp_init(0x7fff0000)
    if not context:raise ValueError('seccomp initialization failed')
    with tempfile.TemporaryFile() as temporary_filter:
        fd=os.dup(temporary_filter.fileno())
    try:
        for name in ('unshare','setns','mount','umount2','pivot_root','fsopen','fsconfig','fsmount','open_tree','move_mount','mount_setattr','clone3','fork','vfork','clone'):
            number=lib.seccomp_syscall_resolve_name(name.encode())
            if number<0:continue
            action=0x50000 | (errno.ENOSYS if name=='clone3' else errno.EPERM)
            if lib.seccomp_rule_add_array(context,action,number,0,None)!=0:raise ValueError('seccomp rule failed')
        # v1 is a single-process NumPy worker: deny child/thread creation so
        # RLIMIT_AS bounds its address space rather than each fork independently.
        if lib.seccomp_export_bpf(context,fd)!=0:raise ValueError('seccomp export failed')
        os.lseek(fd,0,os.SEEK_SET)
        return fd
    except BaseException:
        os.close(fd);raise
    finally:lib.seccomp_release(context)


def launch(spec):
    config = spec['provider_config']
    exe = _check_config(config)
    original = Path(spec['output_file'])
    if original.is_symlink():
        raise ValueError('symlink output forbidden')
    output = original.resolve(strict=True)
    if not output.is_file():
        raise ValueError('precreated regular output required')
    roots = ['/usr', '/lib', '/lib64'] + list(config.get('runtime_paths', [])) + list(spec.get('readonly_paths', []))
    argv = [exe, '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL', '--clearenv', '--tmpfs', '/tmp']
    for path in dict.fromkeys(roots):
        path = str(Path(path).absolute())
        if not Path(path).exists():
            raise ValueError('missing runtime/input path')
        # Never bind a broad host home/root or special device/proc namespace.
        if path in ('/', '/home', '/home/interns', '/dev', '/proc', '/run', '/tmp'):
            raise ValueError('broad or special mount forbidden')
        argv += ['--ro-bind', path, path]
    argv += ['--symlink', 'usr/bin', '/bin', '--proc', '/proc', '--dev-bind', '/dev/urandom', '/dev/urandom', '--dev-bind', '/dev/null', '/dev/null']
    # Only a single output inode is writable. The directory and all other
    # paths remain read-only; RLIMIT_FSIZE is a hard per-output byte ceiling.
    argv += ['--ro-bind', str(output.parent), str(output.parent), '--bind', str(output), str(output)]
    for key,value in spec.get('env', {}).items():
        if key not in ('PATH','PYTHONPATH','PYTHONNOUSERSITE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):
            raise ValueError('environment variable not admitted')
        argv += ['--setenv',key,str(value)]
    seccomp_fd=_seccomp_fd()
    argv += ['--seccomp',str(seccomp_fd)]
    argv += ['--remount-ro', '/tmp', '--remount-ro', '/', '--chdir', str(spec['cwd']), '--', *map(str,spec['argv'])]
    limits=spec.get('limits', {})
    def restrict():
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        for name,key,default in [(resource.RLIMIT_FSIZE,'max_file_bytes',16777216),
                                  (resource.RLIMIT_AS,'memory_bytes',2147483648),
                                  (resource.RLIMIT_CPU,'cpu_seconds',20),
                                  (resource.RLIMIT_NPROC,'processes',65536)]:
            value=int(limits.get(key,default));resource.setrlimit(name,(value,value))
    handles=[]
    try:
        for key in ('stdout','stderr'):
            handles.append(open(spec[key],'wb') if isinstance(spec.get(key),(str,Path)) else spec.get(key,subprocess.PIPE))
        return subprocess.Popen(argv, cwd='/', env={}, stdin=subprocess.DEVNULL,
                                stdout=handles[0],stderr=handles[1],start_new_session=True,preexec_fn=restrict,pass_fds=(seccomp_fd,))
    finally:
        os.close(seccomp_fd)
        for handle in handles:
            if hasattr(handle,'close'):handle.close()


def probe(provider_config=None):
    try:
        exe=_check_config(provider_config)
        with tempfile.TemporaryDirectory(prefix='reno-isolation-probe-') as temp:
            root=Path(temp);code=root/'code';out=root/'out';code.mkdir();out.mkdir()
            secret=root/'secret';secret.write_text('not exposed')
            result=out/'result.npy';result.touch()
            script=code/'probe.py'
            script.write_text('''import os,socket,json
from pathlib import Path
secret,output,hostpid=__import__('sys').argv[1:]
checks={}
checks['secret_hidden']=not Path(secret).exists()
checks['host_process_hidden']=not Path('/proc/'+hostpid).exists()
checks['gpu_hidden']=not Path('/dev/nvidia0').exists()
checks['entropy_device']=len(open('/dev/urandom','rb').read(8))==8
import subprocess
try:
 checks['nested_namespace_denied']=subprocess.run(['/usr/bin/unshare','--user','--map-root-user','--mount','/bin/true'],capture_output=True).returncode!=0
except OSError:checks['nested_namespace_denied']=True
import ctypes
checks['nested_namespace_denied'] = checks['nested_namespace_denied'] and ctypes.CDLL(None).unshare(0x10020000) != 0
for forbidden in ['/tmp/forbidden','/forbidden']:
 try:
  Path(forbidden).write_text('bad');checks[forbidden]=False
 except OSError:checks[forbidden]=True
try:
 Path('/usr/reno-forbidden-write').write_text('bad');checks['outside_write_denied']=False
except OSError: checks['outside_write_denied']=True
try:
 Path(output).with_name('extra').write_text('bad');checks['extra_output_denied']=False
except OSError: checks['extra_output_denied']=True
s=socket.socket();s.settimeout(1)
try:
 s.connect(('1.1.1.1',53));checks['network_denied']=False
except OSError: checks['network_denied']=True
finally:s.close()
Path(output).write_text(json.dumps(checks))
''')
            p=launch(dict(provider_config=provider_config,argv=['/usr/bin/python3',str(script),str(secret),str(result),str(os.getpid())],
                          cwd=str(code),readonly_paths=[str(code)],output_file=str(result),env={},
                          stdout=str(root/'stdout'),stderr=str(root/'stderr'),limits={}))
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait();raise ValueError('probe timeout')
            if p.returncode != 0:raise ValueError('probe process failed: '+(root/'stderr').read_text()[-300:])
            checks=json.loads(result.read_text())
            if len(checks)!=10 or not all(v is True for v in checks.values()):raise ValueError('negative probe failed')
        return IsolationProbe(True,'negative probes and permitted output passed',exe,None,dict(provider_config))
    except (OSError,ValueError,KeyError,TypeError) as error:
        return IsolationProbe(False,str(error))
