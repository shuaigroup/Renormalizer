"""Shared subprocess, provenance and timing helpers for the benchmark drivers.

Imported by the entry points; ``python _common.py --help`` describes this module.
Workers validate the imported tree and backend before collecting measurements.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import random
import timeit

ROOT = Path(__file__).resolve().parents[2]
THREAD_KEYS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'RENO_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
               'BLIS_NUM_THREADS')


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def source_tree(value):
    path = Path(value).expanduser().resolve()
    if not (path / 'renormalizer' / '__init__.py').is_file():
        raise argparse.ArgumentTypeError('source must contain renormalizer/__init__.py')
    return path


def interpreter(value):
    found = shutil.which(value)
    if found:
        return str(Path(found).absolute())
    path = Path(value).expanduser().absolute()
    if not path.is_file():
        raise ValueError('Python interpreter not found: ' + value)
    return str(path)


def environment(source, threads=1, gpu=None, precision=64):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('RENO_', 'OMP_', 'OPENBLAS_', 'MKL_', 'NUMEXPR_',
                                'VECLIB_', 'BLIS_', 'PYTHONPATH', 'PYTHONHOME'))}
    env.update({k: str(threads) for k in THREAD_KEYS})
    env.update(PYTHONHASHSEED='0', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(source), LC_ALL='C', CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
    if gpu is not None:
        env['RENO_GPU'] = '0'  # index inside CUDA_VISIBLE_DEVICES
    if precision == 32:
        env['RENO_FP32'] = '1'
    return env


def core_list(spec=None):
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    if spec is None:
        return allowed
    result = set()
    for part in spec.split(','):
        lo, sep, hi = part.partition('-')
        lo, hi = int(lo), int(hi) if sep else int(lo)
        if lo < 0 or hi < lo:
            raise ValueError('invalid CPU range: ' + part)
        result.update(range(lo, hi + 1))
    if not result or not result <= set(allowed):
        raise ValueError('CPU pool is empty or outside process affinity')
    return sorted(result)


def pin_prefix(cores, unpinned=False):
    if unpinned:
        return []
    exe = shutil.which('taskset')
    if not exe:
        raise RuntimeError('CPU pinning requires taskset; use --unpinned to explicitly opt out')
    return [exe, '-c', ','.join(map(str, cores))]


def stop_process(proc, first=signal.SIGTERM, grace=5):
    for sig in (first, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue
    proc.wait()


def run_process(command, cwd, env, log, cap):
    start = time.perf_counter()
    timed_out = False
    with open(log, 'w') as stream:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_process(proc)
        except BaseException:
            stop_process(proc)
            raise
    return dict(command=command, returncode=proc.returncode, timed_out=timed_out,
                wall_seconds=time.perf_counter() - start)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def provenance(source, gpu=False):
    import renormalizer
    import numpy as np
    from renormalizer.mps.backend import USE_GPU
    actual = Path(renormalizer.__file__).resolve()
    if Path(source).resolve() not in actual.parents:
        raise RuntimeError('Imported unexpected source: ' + str(actual))
    if bool(USE_GPU) != gpu:
        raise RuntimeError('Unexpected backend: USE_GPU=' + str(USE_GPU))
    np.random.seed(9012)
    random.seed(1092)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    git_env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    git = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True, env=git_env, timeout=10) if shutil.which('git') else None
    dirty = subprocess.run(['git', '-C', str(source), 'status', '--porcelain'],
                           capture_output=True, text=True, env=git_env, timeout=10) if git and git.returncode == 0 else None
    return dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                package=str(actual), commit=git.stdout.strip() if git and git.returncode == 0 else None,
                source_status=dirty.stdout.splitlines() if dirty and dirty.returncode == 0 else None,
                affinity=core_list(), numpy_config=buf.getvalue(), gpu=gpu,
                environment={k: os.environ.get(k) for k in THREAD_KEYS +
                             ('PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES', 'RENO_FP32')})


def measure(fn, number, repeat, warmup):
    for _ in range(warmup):
        fn()
    samples = [v / number for v in timeit.repeat(fn, number=number, repeat=repeat)]
    return dict(median_seconds=statistics.median(samples), seconds=samples,
                number=number, warmup=warmup)


def bitwise(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and not a.dtype.hasobject
            and a.tobytes(order='C') == b.tobytes(order='C'))


def worker_parser(doc):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--current', type=source_tree, default=ROOT)
    ap.add_argument('--baseline', type=source_tree)
    ap.add_argument('--python', default=sys.executable, help='current worker interpreter')
    ap.add_argument('--baseline-python', help='defaults to --python')
    ap.add_argument('--out', type=Path, required=True, help='new output directory')
    ap.add_argument('--cores', help='CPU pool; first N cores used by sequential workers')
    ap.add_argument('--threads', type=positive, default=1)
    ap.add_argument('--unpinned', action='store_true')
    ap.add_argument('--cap', type=float, default=900, help='seconds per worker')
    ap.add_argument('--number', type=positive, default=10000)
    ap.add_argument('--repeat', type=positive, default=5)
    ap.add_argument('--warmup', type=positive, default=10, help='untimed calls')
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--source', type=source_tree, help=argparse.SUPPRESS)
    return ap


def dispatch(opt, script):
    """Launch isolated interpreters. Return True only in the worker process."""
    if opt.worker:
        return True
    if opt.cap <= 0:
        raise ValueError('--cap must be positive')
    out = opt.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cores = core_list(opt.cores)
    if len(cores) < opt.threads:
        raise ValueError('not enough cores for --threads')
    prefix = pin_prefix(cores[:opt.threads], opt.unpinned)
    versions = [('current', opt.current, opt.python)]
    if opt.baseline:
        versions.insert(0, ('baseline', opt.baseline, opt.baseline_python or opt.python))
    rows = []
    for label, source, python in versions:
        work = out / label
        work.mkdir()
        arguments = []
        omit = {'current', 'baseline', 'python', 'baseline_python', 'out', 'worker', 'source'}
        for key, value in vars(opt).items():
            if key in omit or value is None or value is False:
                continue
            arguments.append('--' + key.replace('_', '-'))
            if value is not True:
                arguments.append(str(value))
        command = prefix + [interpreter(python), str(Path(script).resolve()), *arguments,
                            '--worker', '--source', str(source), '--out', str(work)]
        row = run_process(command, work, environment(source, opt.threads), work / 'output.log', opt.cap)
        row.update(version=label, source=str(source))
        rows.append(row)
        write_json(out / 'summary.json', rows)
    if any(r['returncode'] or r['timed_out'] for r in rows):
        raise SystemExit('Worker failed or timed out; see summary.json and output.log')
    print(out / 'summary.json')
    return False


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
