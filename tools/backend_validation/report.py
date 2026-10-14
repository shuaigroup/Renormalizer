"""Strict JSON evidence publication; never infer completion from missing cells."""
import json
from pathlib import Path
import re
import math

HASHES=('source_hash','environment_hash','fixture_hash','reference_hash','scorer_hash','tolerance_hash','candidate_hash')

STATUSES = frozenset(('pass', 'fail', 'blocked', 'unverified'))


def _gpu_witness(record):
    device = record.get('device')
    witness = record.get('device_witness')
    if not (str(device).startswith('cuda') or witness is not None):
        return
    if not isinstance(witness, dict):
        raise ValueError('GPU pass needs structured device witness')
    if not isinstance(witness.get('device'), str) or not re.fullmatch(r'cuda:[0-9]+', witness['device']):
        raise ValueError('invalid witnessed GPU device')
    if device is not None and device != witness['device']:
        raise ValueError('GPU witness device mismatch')
    if witness.get('dtype') not in ('float32', 'float64', 'complex64', 'complex128'):
        raise ValueError('invalid witnessed dtype')
    shape = witness.get('shape')
    if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError('invalid witnessed shape')
    if not isinstance(witness.get('cupy_version'), str) or not witness['cupy_version'].strip():
        raise ValueError('missing witnessed CuPy version')
    for key in ('dtype', 'shape'):
        if key in record and record[key] != witness[key]:
            raise ValueError('GPU witness ' + key + ' mismatch')


def _result_cells(record):
    results = record.get('results')
    if not isinstance(results, dict) or not results:
        raise ValueError('pass requires named result cells')
    expected = record.get('expected_statuses', {key: 'pass' for key in results})
    if (not isinstance(expected, dict) or set(expected) != set(results)
            or any(value not in ('pass', 'fail') for value in expected.values())):
        raise ValueError('expected statuses must exactly declare pass/fail result keys')
    for key, value in results.items():
        _result(value, expected[key])


def _result(value, expected='pass'):
    status = value.get('status') if isinstance(value, dict) else value
    if not isinstance(status, str) or status not in STATUSES or status != expected:
        raise ValueError('result status contradicts expectation')
    if isinstance(value, dict):
        if value.get('skipped', 0):
            raise ValueError('skipped result cannot complete a check')
        if expected == 'pass':
            _gpu_witness(value)
        _nested(value)


def _nested(value):
    if isinstance(value, dict):
        if 'results' in value:
            _result_cells(value)
        for key, child in value.items():
            if key in ('results', 'expected_statuses'):
                continue
            if key == 'negative_controls':
                if not isinstance(child, list) or not child:
                    raise ValueError('negative controls require nonempty failed score records')
                for control in child:
                    if not isinstance(control, dict):
                        raise ValueError('negative control must be a score record')
                    _result(control, 'fail')
            elif isinstance(child, dict) and 'status' in child:
                _result(child)
            elif isinstance(child, (dict, list)):
                _nested(child)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, dict) and 'status' in child:
                _result(child)
            else:
                _nested(child)


def validate_report(report):
    if report.get('schema_version') != 1:
        raise ValueError('report schema missing')
    for key in HASHES:
        if not isinstance(report.get(key),str) or not re.fullmatch('[0-9a-f]{64}',report[key]):
            raise ValueError('missing identity: '+key)
    if report.get('status') not in ('pass','fail','blocked','unverified'):
        raise ValueError('invalid status')
    if report.get('status') == 'pass' and report.get('skipped',0):
        raise ValueError('skipped checks cannot be a complete pass')
    if str(report.get('device','')).startswith('cuda') and report.get('status')=='pass' and not report.get('device_witness'):
        raise ValueError('GPU pass needs device witness')
    if report.get('status') == 'pass':
        _gpu_witness(report)
        _result_cells(report)
    if report.get('claim') == 'performance':
        _gpu_witness(report)
        timings=report.get('timings',{})
        if not isinstance(timings,dict) or not report.get('device_witness'):
            raise ValueError('performance needs timings and device witness')
        for key in ('t_cold','t_kernel','t_e2e'):
            value=timings.get(key)
            if type(value) not in (float,int) or not math.isfinite(value) or value<=0:
                raise ValueError('missing valid timing: '+key)
    for key in ('domain','mode','resource_coverage','unsupported','results'):
        if key not in report: raise ValueError('missing field: '+key)
    json.dumps(report,allow_nan=False)
    return report


def write_report(path, report):
    validate_report(report)
    # Exclusive creation cannot overwrite a frozen reference or older report.
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
