import numpy as np
import pytest
from tools.backend_validation.gpu import run_reviewed_gpu


def test_output_work_budget_checked_before_launch(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, 'run', lambda *a,**k: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='budget'):
        run_reviewed_gpu(np.empty((257,0)),np.empty((0,257)),np.empty((0,0)),tolerances={})


def test_reviewed_gpu_timings_can_populate_performance_report(monkeypatch):
    """Exercise producer-to-report wiring without requiring a CUDA runtime."""
    import hashlib
    import json
    from pathlib import Path
    from tools.backend_validation import gpu
    from tools.backend_validation.report import HASHES, validate_report

    a = np.array([[1., 2.], [3., 4.]])
    b = np.eye(2)
    output_hash = hashlib.sha256(a.tobytes(order='C')).hexdigest()
    witness = dict(device='cuda:0', dtype='float64', shape=[2, 2], cupy_version='13.6.0')
    measurements = {
        name: dict(warmup=1, repeats=2, output_hashes=[output_hash]*4,
                   summary={'t_kernel': {'median': kernel}, 't_e2e': {'median': e2e}})
        for name, kernel, e2e in [('candidate', .2, .4), ('baseline', .1, .3)]
    }
    launches = []

    def fake_worker(cmd, **kwargs):
        directory = Path(cmd[3])
        timing = cmd[-1] == '--timing'
        launches.append(timing)
        np.save(directory/'result.npy', a, allow_pickle=False)
        if timing:
            np.save(directory/'baseline_result.npy', a, allow_pickle=False)
        (directory/'evidence.json').write_text(json.dumps(measurements if timing else witness))

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(gpu.subprocess, 'run', fake_worker)
    ticks = iter([10., 12., 20., 23.])
    monkeypatch.setattr(gpu.time, 'perf_counter', lambda: next(ticks))
    result = run_reviewed_gpu(a, b, a, tolerances=dict(
        atol_F=1e-12, rtol_F=1e-10, atol_max=1e-12, rtol_max=1e-10))
    assert launches == [False, True]
    assert result['timings']['t_cold'] == 2.
    assert result['timings']['t_kernel'] == .2
    assert result['timings']['t_e2e'] == .4
    assert result['timings']['candidate'] == measurements['candidate']
    assert result['timings']['baseline'] == measurements['baseline']
    assert result['timing_process_seconds'] == 3.
    report = dict(schema_version=1, status='pass', claim='performance',
                  domain='fixed control', mode='reviewed', unsupported=[],
                  resource_coverage=result['resource_coverage'], results={'gpu': result},
                  timings=result['timings'], device_witness=result['device_witness'],
                  **{key: 'a'*64 for key in HASHES})
    assert validate_report(report) is report
