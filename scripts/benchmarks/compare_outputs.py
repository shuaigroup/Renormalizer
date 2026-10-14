"""Run baseline/current examples in one batch, then compare NPZ bytes and logs.

Example: python compare_outputs.py --baseline ../reno-baseline --current . --out parity --examples sbm_short,fmo
Writes comparison.json, normalized log files and TTN same-step elapsed times.
A nonzero exit means mismatch, failed/capped jobs, unsupported object arrays,
or missing comparable output. Capped prefixes never count as full parity.
"""
import datetime
import ast
import difflib
import hashlib
import re
import numpy as np
from _common import bitwise, write_json
from run_examples import parser, run_batch

STAMP = re.compile(r'^\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d[,\.]\d+')
ADDRESS = re.compile(r'0x[0-9a-fA-F]+')
NOISE = re.compile(r'wall.?time|elapsed|time (?:cost|used|taken)|(?:cost|took).*\b(?:s|sec|seconds)\b|'
                   r'\b(?:[KMG]iB|bytes|memory|pid)\b|Git Commit|(?:numpy|cupy|python) version|'
                   r'use (?:numpy|cupy) as backend|using (?:numpy|cupy)|backend.*(?:dtype|device)|'
                   r'\[INFO\]\s+use (?:32|64) bits\s*$|random seed', re.I)


def normalized(path):
    result = []
    for line in path.read_text(errors='replace').splitlines():
        line = ADDRESS.sub('0x?', STAMP.sub('', line)).rstrip()
        if line.strip() and not NOISE.search(line):
            result.append(line)
    return result


def npz_compare(left, right):
    rows = []
    try:
        with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
            for key in sorted(set(a.files) | set(b.files)):
                if key not in a.files or key not in b.files:
                    rows.append(dict(key=key, bitwise=False, reason='missing key'))
                    continue
                try:
                    x, y = a[key], b[key]
                    rows.append(dict(key=key, bitwise=bitwise(x, y),
                                     baseline_shape=list(x.shape), current_shape=list(y.shape),
                                     baseline_dtype=str(x.dtype), current_dtype=str(y.dtype)))
                except ValueError as error:
                    rows.append(dict(key=key, bitwise=False, reason=str(error)))
    except (ValueError, OSError) as error:
        rows.append(dict(bitwise=False, reason=str(error)))
    return dict(equal=bool(rows) and all(r['bitwise'] for r in rows), arrays=rows)


def steps(path):
    values = []
    for line in path.read_text(errors='replace').splitlines():
        match = STAMP.match(line)
        if match and '[INFO] (' in line:
            result = line[line.index('[INFO] ('):]
            # Junction also logs parameter tuples and (step, bond_dims).
            # Count only flat numeric observable tuples as evolution results.
            try:
                fields = ast.literal_eval(result[len('[INFO] '):])
            except (ValueError, SyntaxError):
                continue
            if (not isinstance(fields, tuple) or len(fields) < 2
                    or not all(type(value) in (int, float, complex) for value in fields)):
                continue
            stamp = datetime.datetime.fromisoformat(match.group().replace(',', '.'))
            values.append((stamp, result))
    return values


def throughput(left, right):
    a, b = steps(left), steps(right)
    common = min(len(a), len(b))
    prefix = 0
    for x, y in zip(a, b):
        if x[1] != y[1]:
            break
        prefix += 1
    result = dict(baseline_steps=len(a), current_steps=len(b), common_steps=common,
                  identical_result_prefix=prefix)
    if prefix >= 2:
        ta = (a[prefix - 1][0] - a[0][0]).total_seconds()
        tb = (b[prefix - 1][0] - b[0][0]).total_seconds()
        result.update(baseline_seconds=ta, current_seconds=tb,
                      current_over_baseline=tb / ta if ta > 0 else None,
                      measured_intervals=prefix - 1)
    return result


def compare_job(root, a, b):
    left, right = root / a['directory'], root / b['directory']
    complete = all(r['returncode'] == 0 and not r['timed_out'] for r in (a, b))
    report = dict(example=a['example'], threads=a['threads'], replica=a['replica'],
                  complete=complete, files=[], unchanged_inputs=[], baseline_seconds=a['wall_seconds'],
                  current_seconds=b['wall_seconds'])
    def files(folder):
        return {str(p.relative_to(folder)): p for p in folder.rglob('*')
                if p.suffix in ('.log', '.npz') and p.is_file()}
    af, bf = files(left), files(right)
    for name in sorted(set(af) | set(bf)):
        if (name in af and name in bf and name in a.get('input_artifacts', {})
                and name in b.get('input_artifacts', {})
                and hashlib.sha256(af[name].read_bytes()).hexdigest() == a['input_artifacts'][name]
                and hashlib.sha256(bf[name].read_bytes()).hexdigest() == b['input_artifacts'][name]):
            report['unchanged_inputs'].append(name)
            continue
        row = dict(file=name)
        if name not in af or name not in bf:
            row.update(equal=False, reason='missing file')
        elif name.endswith('.npz'):
            row.update(npz_compare(af[name], bf[name]))
        else:
            x, y = normalized(af[name]), normalized(bf[name])
            row.update(equal=x == y, baseline_lines=len(x), current_lines=len(y))
            row['first_difference'] = next((i + 1 for i, (u, v) in enumerate(zip(x, y)) if u != v),
                                           min(len(x), len(y)) + 1 if len(x) != len(y) else None)
            for path, lines in ((af[name], x), (bf[name], y)):
                path.with_suffix('.normalized.txt').write_text('\n'.join(lines) + '\n')
            if x != y:
                row['diff_preview'] = list(difflib.unified_diff(x, y, n=2))[:60]
            row['ttn_throughput'] = throughput(af[name], bf[name])
        report['files'].append(row)
    evidence = any(row.get('arrays') or row.get('baseline_lines', 0) for row in report['files'])
    report['has_comparable_output'] = evidence
    report['parity'] = complete and evidence and all(r['equal'] for r in report['files'])
    return report


def main():
    ap = parser()
    ap.description = __doc__
    ap.set_defaults(examples='sbm_short,fmo,dynamics,hubbard,h2o_qc,ttns_sbm_zt,ttns_junction_zt,ttns_sbm_ft')
    opt = ap.parse_args()
    if not opt.baseline:
        ap.error('--baseline is required')
    if opt.version:
        ap.error('use run_examples.py for additional versions')
    root, rows = run_batch(opt)
    if opt.dry_run:
        return
    index = {(r['example'], r['threads'], r['replica'], r['version']): r
             for r in rows if r['kind'] == 'time'}
    reports = []
    for key, row in index.items():
        if key[-1] == 'baseline':
            reports.append(compare_job(root, row, index[key[:-1] + ('current',)]))
    write_json(root / 'comparison.json', reports)
    print(root / 'comparison.json')
    if (not reports or not all(r['parity'] for r in reports)
            or any(r['returncode'] or r['timed_out'] for r in rows)):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
