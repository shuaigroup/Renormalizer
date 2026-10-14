"""Instrument one CPU example: H-v call timings, cache hits or NumPy tensordot sites.

Example: python trace_example.py caches --current . --out trace --example ttns/sbm_zt.py -- 050 001 050
Copies CURRENT/example (or --example-dir), runs the selected interpreter and
writes trace.json even after a graceful timeout. Instrumented timings include
observer overhead and must not be compared to uninstrumented benchmark times.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import runpy
import shutil
import signal
import sys
import time
from _common import (ROOT, core_list, environment, interpreter, pin_prefix, positive,
                     provenance, run_process, source_tree, write_json)


def install(mode):
    import numpy as np
    start = time.perf_counter()
    if mode == 'caches':
        from renormalizer.backend import execution, numpy_contraction
        seen = Counter()
        optimize, operands = Counter(), Counter()
        original = execution._host_expression
        def host(selected, args, kwargs):
            result = original(selected, args, kwargs)
            seen['cached' if result is not None else 'bypass'] += 1
            optimize[str(kwargs.get('optimize'))] += 1
            if result is not None:
                operands[len(result[1])] += 1
            return result
        execution._host_expression = host
        def dump():
            caches = {}
            for name, module, attr in [('path', execution, '_cached_path'),
                    ('expression', execution, '_cached_expression'),
                    ('interleaved_subscripts', execution, '_interleaved_subscripts'),
                    ('tensordot_plan', numpy_contraction, '_plan')]:
                fn = getattr(module, attr, None)
                caches[name] = fn.cache_info()._asdict() if hasattr(fn, 'cache_info') else {'unavailable': True}
            return dict(wall_seconds=time.perf_counter() - start, contract=dict(seen),
                        optimize=dict(optimize), operands=dict(operands), caches=caches)
        return dump
    if mode == 'hv':
        from renormalizer.backend import execution
        original = execution.contract_expression
        durations = defaultdict(list)
        def expression(*args, **kwargs):
            frame = sys._getframe(1)
            while frame is not None and 'oe_contract_wrap' in frame.f_code.co_filename:
                frame = frame.f_back
            site = '?' if frame is None else f'{frame.f_code.co_filename}:{frame.f_lineno}'
            expr = original(*args, **kwargs)
            def timed(*a, **k):
                before = time.perf_counter()
                result = expr(*a, **k)
                durations[site].append(time.perf_counter() - before)
                return result
            return timed
        execution.contract_expression = expression
        def dump():
            sites = {}
            for site, values in durations.items():
                a = np.asarray(values) * 1000
                sites[site] = dict(calls=len(values), total_seconds=float(a.sum() / 1000),
                    median_ms=float(np.median(a)), p10_ms=float(np.percentile(a, 10)),
                    p90_ms=float(np.percentile(a, 90)), max_ms=float(a.max()),
                    fraction_over_0p5ms=float((a > .5).mean()), fraction_over_1ms=float((a > 1).mean()),
                    time_fraction_over_1ms=float(a[a > 1].sum() / a.sum()) if a.sum() else 0)
            return dict(wall_seconds=time.perf_counter() - start, sites=sites)
        return dump
    original = np.tensordot
    stats = defaultdict(lambda: [0, 0.0, set(), 0])
    skip = ('/numpy/', '/opt_einsum/', 'renormalizer/mps/matrix.py', 'renormalizer/backend/')
    def tensordot(a, b, axes=2):
        frame = sys._getframe(1)
        via_oe = False
        while frame is not None and any(s in frame.f_code.co_filename for s in skip):
            via_oe |= '/opt_einsum/' in frame.f_code.co_filename
            frame = frame.f_back
        site = '?' if frame is None else f'{frame.f_code.co_filename}:{frame.f_lineno}'
        key = (site, 'oe' if via_oe else 'direct', str(axes))
        before = time.perf_counter()
        result = original(a, b, axes)
        row = stats[key]
        row[0] += 1
        row[1] += time.perf_counter() - before
        if len(row[2]) < 100000:
            row[2].add((np.shape(a), np.shape(b)))
        row[3] = max(row[3], result.size)
        return result
    np.tensordot = tensordot
    def dump():
        return dict(wall_seconds=time.perf_counter() - start, sites=[
            dict(site=k[0], path=k[1], axes=k[2], calls=v[0], seconds=v[1],
                 distinct_shapes=len(v[2]), shape_count_capped=len(v[2]) == 100000, max_output_size=v[3])
            for k, v in sorted(stats.items(), key=lambda item: -item[1][0])])
    return dump


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['hv', 'caches', 'tensordot'])
    ap.add_argument('--current', type=source_tree, default=ROOT)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--example-dir', type=Path)
    ap.add_argument('--example', required=True, help='relative script inside example-dir; sbm_short.py also supported')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cores')
    ap.add_argument('--threads', type=positive, default=1)
    ap.add_argument('--unpinned', action='store_true')
    ap.add_argument('--cap', type=float, default=240)
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    # Split the explicit separator so example options cannot consume driver options.
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    opt = ap.parse_args(argv[:split])
    example_args = argv[split + 1:]
    if opt.worker:
        metadata = provenance(opt.current)
        dump = install(opt.mode)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(124))
        sys.argv = [opt.example] + example_args
        sys.path.insert(0, str(Path(opt.example).resolve().parent))
        completed = False
        try:
            runpy.run_path(opt.example, run_name='__main__')
            completed = True
        finally:
            write_json(opt.out / 'trace.json', dict(metadata=metadata, completed=completed, mode=opt.mode, **dump()))
        return
    if opt.cap <= 0 or Path(opt.example).is_absolute() or '..' in Path(opt.example).parts:
        ap.error('positive cap and a relative example path are required')
    out = opt.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(opt.example_dir or opt.current / 'example', out / 'run')
    shutil.copy2(Path(__file__).with_name('sbm_short.py'), out / 'run' / 'sbm_short.py')
    cores = core_list(opt.cores)
    if len(cores) < opt.threads:
        ap.error('not enough cores')
    cmd = pin_prefix(cores[:opt.threads], opt.unpinned) + [interpreter(opt.python), str(Path(__file__).resolve()),
        opt.mode, '--worker', '--current', str(opt.current), '--out', str(out), '--example', opt.example, '--', *example_args]
    row = run_process(cmd, out / 'run', environment(opt.current, opt.threads), out / 'output.log', opt.cap)
    write_json(out / 'meta.json', row)
    if row['returncode'] or row['timed_out']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
