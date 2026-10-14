"""Schedule CPU examples on disjoint physical cores in one comparison batch.

Example: python run_examples.py --baseline ../reno-baseline --current . --out results
summary.json/csv contain per-job status, GNU time measurements and placement;
medians.json excludes warmup and profiling runs. Capped jobs are incomplete.
Linux taskset is required; hwloc is optional with /sys topology fallback.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from _common import (ROOT, core_list, environment, interpreter, positive,
                     source_tree, stop_process, write_json)
from topo_alloc import Allocator

EXAMPLES = [
    ("fmo", ["fmo.py"]),
    ("sbm", ["sbm.py"]),
    ("h2o_qc", ["h2o_qc.py"]),
    ("dynamics", ["dynamics.py", "std.yaml"]),
    ("transport_kubo", ["transport_kubo.py", "std.yaml"]),
    ("ttns_junction_zt", ["ttns/junction_zt.py"]),
    ("ttns_junction_ft", ["ttns/junction_ft.py", "32", "1", "100"]),
    ("ttns_sbm_zt", ["ttns/sbm_zt.py", "050", "001", "050"]),
    ("ttns_sbm_ft", ["ttns/sbm_ft.py"]),
    ("ssh", ["ssh.py"]),
    ("hubbard", ["hubbard.py"]),
]

EXAMPLES.append(('sbm_short', ['sbm_short.py']))


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--baseline', type=source_tree)
    ap.add_argument('--current', type=source_tree, default=ROOT)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--baseline-python')
    ap.add_argument('--version', action='append', nargs=3, metavar=('LABEL', 'TREE', 'PYTHON'),
                    help='additional configuration; repeat for environment comparisons')
    ap.add_argument('--example-dir', type=Path, help='shared input copy; default CURRENT/example')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--run-id', default=datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    ap.add_argument('--threads', default='1', help='comma-separated thread counts')
    ap.add_argument('--repeat', type=positive, default=3)
    ap.add_argument('--warmup', type=int, default=1, help='untimed process runs per configuration')
    ap.add_argument('--profile-threads', type=int, default=0, help='0 disables separate cProfile runs')
    ap.add_argument('--cores', help='allowed pool; defaults to process affinity')
    ap.add_argument('--jobs-per-l3', default='auto')
    ap.add_argument('--hwloc-bin', type=Path)
    ap.add_argument('--no-membind', action='store_true')
    ap.add_argument('--gnu-time', help='GNU time executable; auto-detect time or gtime')
    ap.add_argument('--cap', type=float, default=900)
    ap.add_argument('--example-cap', action='append', default=[], metavar='NAME=SECONDS')
    ap.add_argument('--examples', default='all', help='comma-separated names or all')
    ap.add_argument('--dry-run', action='store_true', help='print placement/configuration without launching jobs')
    return ap


def gnu_time(executable):
    choices = [executable] if executable else ['gtime', '/usr/bin/time']
    for name in choices:
        path = shutil.which(name)
        if path:
            check = subprocess.run([path, '--version'], capture_output=True, text=True)
            if check.returncode == 0 and 'GNU' in check.stdout + check.stderr:
                return path
    raise RuntimeError('GNU time is required; install it or specify --gnu-time')


def loadavg():
    return list(os.getloadavg()) if hasattr(os, 'getloadavg') else None


def time_report(path):
    result = {}
    for line in path.read_text().splitlines():
        key, _, value = line.strip().rpartition(': ')
        if key.startswith('Elapsed (wall clock) time'):
            seconds = 0.0
            for part in value.split(':'):
                seconds = seconds * 60 + float(part)
            result['wall_seconds'] = seconds
        elif key == 'Maximum resident set size (kbytes)':
            result['max_rss_mb'] = int(value) / 1024
        elif key in ('User time (seconds)', 'System time (seconds)'):
            result['user_seconds' if key.startswith('User') else 'system_seconds'] = float(value)
        elif key == 'Percent of CPU this job got':
            result['cpu_percent'] = float(value.rstrip('%'))
    return result


def summaries(root, rows):
    write_json(root / 'summary.json', rows)
    fields = ['example', 'version', 'threads', 'kind', 'replica', 'returncode', 'timed_out',
              'wall_seconds', 'scheduler_wall_seconds', 'max_rss_mb', 'user_seconds',
              'system_seconds', 'cpu_percent', 'cores', 'domains', 'loadavg_start', 'loadavg_end']
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        if row['kind'] == 'time' and row['returncode'] == 0 and not row['timed_out']:
            groups[(row['example'], row['version'], row['threads'])].append(row['wall_seconds'])
    write_json(root / 'medians.json', [dict(example=k[0], version=k[1], threads=k[2],
               completed_repeats=len(v), seconds=v, median_seconds=statistics.median(v))
               for k, v in groups.items()])


def run_batch(opt):
    versions = {'current': dict(code=str(opt.current), python=interpreter(opt.python))}
    if opt.baseline:
        versions = {'baseline': dict(code=str(opt.baseline), python=interpreter(opt.baseline_python or opt.python)), **versions}
    for label, tree, python in opt.version or []:
        if not label.replace('_', '').replace('-', '').isalnum() or label in versions:
            raise ValueError('version labels must be unique simple names')
        versions[label] = dict(code=str(source_tree(tree)), python=interpreter(python))
    example_dir = (opt.example_dir or opt.current / 'example').resolve()
    threads = sorted({positive(t) for t in opt.threads.split(',')}, reverse=True)
    names = dict(EXAMPLES)
    wanted = list(names) if opt.examples == 'all' else opt.examples.split(',')
    if set(wanted) - names.keys():
        raise ValueError('unknown examples: ' + str(set(wanted) - names.keys()))
    caps = {name: opt.cap for name in wanted}
    for item in opt.example_cap:
        name, value = item.split('=', 1)
        if name not in wanted:
            raise ValueError('cap for unselected example: ' + name)
        caps[name] = float(value)
    if opt.warmup < 0 or opt.profile_threads < 0 or any(v <= 0 for v in caps.values()):
        raise ValueError('invalid warmup, profile thread count or cap')
    if Path(opt.run_id).name != opt.run_id or opt.run_id in ('.', '..'):
        raise ValueError('run-id must be a single directory name')
    alloc = Allocator(core_list(opt.cores), opt.jobs_per_l3, not opt.no_membind, opt.hwloc_bin)
    by_node = defaultdict(int)
    for (node, _), size in alloc.size.items():
        by_node[node] += size
    if max(threads + [opt.profile_threads]) > max(by_node.values()):
        raise ValueError('thread count exceeds available physical cores on one NUMA node')
    queue = []
    for name in wanted:
        for kind, count in [('warmup', opt.warmup), ('time', opt.repeat), ('profile', int(opt.profile_threads > 0))]:
            for rep in range(count):
                for n in ([opt.profile_threads] if kind == 'profile' else threads):
                    order = list(versions) if rep % 2 == 0 else list(reversed(versions))
                    for version in order:
                        queue.append(dict(example=name, version=version, threads=n, kind=kind, replica=rep))
    root = opt.out.resolve() / opt.run_id
    print(alloc.describe(), flush=True)
    print(f'{len(queue)} jobs -> {root}', flush=True)
    if opt.dry_run:
        for job in queue:
            print(job)
        return root, []
    timer = gnu_time(opt.gnu_time)
    for name in wanted:
        if name != 'sbm_short' and not (example_dir / names[name][0]).is_file():
            raise FileNotFoundError(example_dir / names[name][0])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'run_config.json', dict(options={k: str(v) if isinstance(v, Path) else v for k, v in vars(opt).items()},
               versions=versions, example_dir=str(example_dir), topology=alloc.describe()))
    running, rows = [], []
    try:
        while queue or running:
            for job in list(queue):
                placement = alloc.take(job['threads'])
                if placement is None:
                    continue
                domains, cores = placement
                rel = Path(job['example']) / job['version'] / f"{job['kind']}_t{job['threads']}_r{job['replica']}"
                work = root / rel
                shutil.copytree(example_dir, work / 'run')
                shutil.copy2(Path(__file__).with_name('sbm_short.py'), work / 'run' / 'sbm_short.py')
                input_artifacts = {
                    str(path.relative_to(work)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (work / 'run').rglob('*')
                    if path.is_file() and path.suffix in ('.npz', '.log')
                }
                version = versions[job['version']]
                cmd = [timer, '-v', '-o', str(work / 'gnu_time.txt')] + alloc.launcher(domains, cores)
                cmd += [version['python'], str(Path(__file__).with_name('_example_worker.py').resolve()),
                        '--source', version['code'], '--metadata', str(work / 'environment.json')]
                if job['kind'] == 'profile':
                    cmd += ['--profile', str(work / 'profile.prof')]
                cmd += names[job['example']]
                stream = (work / 'output.log').open('w')
                try:
                    proc = subprocess.Popen(cmd, cwd=work / 'run', env=environment(version['code'], job['threads']),
                                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    stream.close()
                    raise
                meta = dict(job, directory=str(rel), command=cmd, cores=cores, domains=domains,
                            input_artifacts=input_artifacts,
                            loadavg_start=loadavg(), started=datetime.datetime.now().isoformat())
                running.append((proc, stream, time.monotonic(), work, meta))
                queue.remove(job)
                print('[start]', rel, cores, flush=True)
            time.sleep(0.2)
            for active in list(running):
                proc, stream, start, work, meta = active
                timeout = proc.poll() is None and time.monotonic() - start >= caps[meta['example']]
                if timeout:
                    stop_process(proc, signal.SIGINT if meta['kind'] == 'profile' else signal.SIGTERM)
                if proc.poll() is None:
                    continue
                stream.close()
                wall = time.monotonic() - start
                meta.update(returncode=proc.returncode, timed_out=timeout, scheduler_wall_seconds=wall,
                            wall_seconds=wall, loadavg_end=loadavg())
                try:
                    meta.update(time_report(work / 'gnu_time.txt'))
                except (OSError, ValueError):
                    meta['time_report_missing_or_invalid'] = True
                if (work / 'profile.prof').exists():
                    with (work / 'profile.txt').open('w') as report:
                        subprocess.run([versions[meta['version']]['python'], '-c',
                            "import pstats,sys; s=pstats.Stats(sys.argv[1]); s.sort_stats('tottime').print_stats(60); s.sort_stats('cumulative').print_stats(60); s.sort_stats('tottime').print_stats('renormalizer/backend/|ContextVar')",
                            str(work / 'profile.prof')], stdout=report, stderr=subprocess.STDOUT, timeout=60)
                write_json(work / 'meta.json', meta)
                rows.append(meta)
                running.remove(active)
                alloc.release(meta['domains'], meta['cores'])
                summaries(root, rows)
                print('[done]', meta['directory'], 'rc=', proc.returncode, 'timeout=', timeout, flush=True)
    finally:
        for proc, stream, _, _, _ in running:
            stop_process(proc)
            stream.close()
    summaries(root, rows)
    return root, rows


def main():
    root, rows = run_batch(parser().parse_args())
    print(root)
    if any(r['returncode'] or r['timed_out'] for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
