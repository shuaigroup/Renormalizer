"""Same-batch CuPy baseline/current numerical parity and synchronized timing.

Example: python cupy_parity.py --baseline ../reno-baseline --current . --python CUPY_PYTHON --gpu 0 --out gpu-parity
Shared float64 baseline fixtures feed both precisions and all process repeats.
Tests Krylov/RK45 TDVP, complex Mps.expectations, Davidson ground state and
RK45 vector/scalar dense output. NPZ files, worker metadata, comparison.json
and summary.json expose exact bytes, reference errors, repeats and medians.
A mismatch, fallback, timeout or slowdown above --slowdown-tolerance fails.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
from _common import (bitwise, core_list, environment, interpreter, pin_prefix, positive,
                     provenance, run_process, worker_parser, write_json)


def worker(opt):
    import logging
    import numpy as np
    import scipy
    import scipy.linalg
    import cupy as cp
    from renormalizer.model import Phonon, Mol, HolsteinModel
    from renormalizer.utils import Quantity, EvolveConfig, EvolveMethod, CompressConfig, CompressCriteria
    from renormalizer.mps import Mps, Mpo
    from renormalizer.mps.backend import backend
    from renormalizer.mps.matrix import asxp, asnumpy
    import renormalizer.mps.gs as gs
    metadata = provenance(opt.source, gpu=True)
    cp.random.seed(2019)
    logging.getLogger('renormalizer').setLevel(logging.WARNING)
    expected = np.dtype('float32' if opt.precision == 32 else 'float64')
    if np.dtype(backend.real_dtype) != expected:
        raise RuntimeError('Requested precision was not selected')
    metadata.update(cupy=cp.__version__, scipy=scipy.__version__,
                    cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                    gpu_properties=str(cp.cuda.runtime.getDeviceProperties(0)),
                    real_dtype=str(backend.real_dtype), precision=opt.precision)
    model = HolsteinModel([Mol(Quantity(0), [Phonon.simple_phonon(Quantity(1), Quantity(1), 6)])] * 3,
                          Quantity(0.2), 3)
    mpo = Mpo(model)
    operators = [Mpo.onsite(model, r'a^\dagger a', dof_set={i}) for i in range(3)]
    sync = cp.cuda.Stream.null.synchronize
    def dense(mps):
        return np.asarray(asnumpy(mps.todense())).reshape(-1)
    def confirm(mps):
        array = asxp(mps[0])
        if not isinstance(array, cp.ndarray):
            raise RuntimeError('MPS silently fell back to a host array')
        if array.dtype not in (expected, np.dtype('complex64' if opt.precision == 32 else 'complex128')):
            raise RuntimeError('MPS precision does not match requested precision')
        metadata['array_type'] = str(type(array))
    def configure(mps, solver):
        mps.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=16)
        mps.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, ivp_solver=solver)
        return mps
    fixture = opt.fixture
    if opt.role == 'setup':
        initial = Mps.random(model, 1, 16)
        confirm(initial)
        initial.dump(str(fixture / 'initial.npz'))
        h = np.asarray(asnumpy(mpo.todense())).astype(np.float64)
        shape = tuple(initial.pbond_list)
        indices = np.indices(shape).reshape(len(shape), -1)
        mask = indices[[0, 2, 4]].sum(axis=0) == 1
        h = h[np.ix_(mask, mask)]
        w, u = scipy.linalg.eigh(h)
        psi = dense(initial)[mask].astype(np.complex128)
        times = np.arange(1, opt.steps + 1) * opt.dt
        reference = np.array([u @ (np.exp(-1j * w * t) * (u.T @ psi)) for t in times])
        evolved = configure(initial.copy(), 'krylov')
        for _ in range(opt.steps):
            evolved = evolved.evolve(mpo, opt.dt)
        evolved.dump(str(fixture / 'evolved.npz'))
        vector = dense(evolved)[mask].astype(np.complex128)
        np.savez(fixture / 'reference.npz', mask=mask, states=reference, energy=w[0],
                 expectations=indices[[0, 2, 4]][:, mask] @ abs(vector)**2,
                 initial=psi, ham=h)
        write_json(opt.out / 'environment.json', metadata)
        return
    initial = Mps.load(model, str(fixture / 'initial.npz'))
    confirm(initial)
    outputs, timings = {}, {}
    calls = {'davidson': 0}
    original = gs.davidson
    def counted(*a, **kw):
        calls['davidson'] += 1
        return original(*a, **kw)
    gs.davidson = counted
    for solver in ('krylov', 'RK45'):
        def evolve():
            mps = configure(initial.copy(), solver)
            sync()
            start = time.perf_counter()
            states = []
            for _ in range(opt.steps):
                mps = mps.evolve(mpo, opt.dt)
                states.append(mps)
            sync()
            elapsed = time.perf_counter() - start
            confirm(mps)
            return elapsed, np.array([dense(m) for m in states])
        for _ in range(opt.warmup):
            evolve()
        values = [evolve() for _ in range(opt.repeat)]
        timings[solver] = [v[0] for v in values]
        outputs[solver] = values[0][1]
        outputs[solver + '_replicas'] = np.array([v[1] for v in values])
    evolved = Mps.load(model, str(fixture / 'evolved.npz'))
    confirm(evolved)
    if not np.iscomplexobj(asnumpy(evolved[0])):
        raise RuntimeError('expectations probe requires a complex MPS')
    def expectations():
        sync()
        start = time.perf_counter()
        for _ in range(opt.number):
            value = evolved.expectations(operators)
        sync()
        elapsed = (time.perf_counter() - start) / opt.number
        return elapsed, np.asarray(asnumpy(value))
    for _ in range(opt.warmup):
        expectations()
    values = [expectations() for _ in range(opt.repeat)]
    timings['expectations'] = [v[0] for v in values]
    outputs['expectations'] = values[0][1]
    outputs['expectations_replicas'] = np.array([v[1] for v in values])
    def ground():
        mps = initial.copy()
        mps.optimize_config.procedure = [[16, 0.0]] * opt.sweeps
        mps.optimize_config.method = '2site'
        if mps.optimize_config.algo != 'davidson':
            raise RuntimeError('ground-state algorithm must be Davidson')
        sync()
        start = time.perf_counter()
        energies, mps = gs.optimize_mps(mps, mpo)
        sync()
        elapsed = time.perf_counter() - start
        confirm(mps)
        return elapsed, np.asarray(energies), dense(mps)
    for _ in range(opt.warmup):
        ground()
    values = [ground() for _ in range(opt.repeat)]
    timings['ground'] = [v[0] for v in values]
    outputs['ground'] = values[0][1]
    outputs['ground_replicas'] = np.array([v[1] for v in values])
    outputs['ground_state'] = values[0][2]
    outputs['ground_state_replicas'] = np.array([v[2] for v in values])
    if calls['davidson'] == 0:
        raise RuntimeError('Davidson was not exercised')
    from renormalizer.lib.integrate._ivp.rk import RK45
    with np.load(fixture / 'reference.npz') as reference:
        h = cp.asarray(reference['ham'], dtype=backend.real_dtype)
        y = cp.asarray(reference['initial'], dtype=backend.complex_dtype)
        w, u = scipy.linalg.eigh(reference['ham'])
        psi = reference['initial']
    def dense_probe():
        sync()
        start = time.perf_counter()
        solver = RK45(lambda t, state: -1j * (h @ state), 0, y, opt.dt,
                      first_step=opt.dt, max_step=opt.dt, rtol=1e-5, atol=1e-8)
        solver.step()
        if solver.status == 'failed':
            raise RuntimeError('dense-output integrator failed')
        interpolation = solver.dense_output()
        query = solver.t * np.array([.25, .5, .75])
        values = interpolation(query)
        scalar = interpolation(float(query[1]))
        if not isinstance(values, cp.ndarray) or not isinstance(scalar, cp.ndarray):
            raise RuntimeError('dense output silently fell back to NumPy')
        sync()
        elapsed = time.perf_counter() - start
        return elapsed, cp.asnumpy(values), cp.asnumpy(scalar), query
    for _ in range(opt.warmup):
        dense_probe()
    values = [dense_probe() for _ in range(opt.repeat)]
    timings['dense'] = [v[0] for v in values]
    outputs['dense'] = values[0][1]
    outputs['dense_scalar'] = values[0][2]
    outputs['dense_times'] = values[0][3]
    outputs['dense_replicas'] = np.array([v[1] for v in values])
    outputs['dense_scalar_replicas'] = np.array([v[2] for v in values])
    outputs['dense_times_replicas'] = np.array([v[3] for v in values])
    outputs['dense_reference'] = np.array([u @ (np.exp(-1j * w * t) * (u.T @ psi))
                                           for t in outputs['dense_times']]).T
    metadata['calls'] = calls
    np.savez(opt.out / 'outputs.npz', **outputs)
    write_json(opt.out / 'environment.json', metadata)
    write_json(opt.out / 'timings.json', timings)


def compare(out, opt):
    import numpy as np
    reports, summaries = [], []
    all_equal = True
    for precision in map(int, opt.precisions.split(',')):
        accumulated = {version: {} for version in ('baseline', 'current')}
        first = {}
        for process in range(opt.process_repeats):
            folders = {v: out / f'{v}{precision}_r{process}' for v in accumulated}
            with np.load(folders['baseline'] / 'outputs.npz') as a, np.load(folders['current'] / 'outputs.npz') as b:
                for key in sorted(set(a.files) | set(b.files)):
                    equal = key in a.files and key in b.files and bitwise(a[key], b[key])
                    row = dict(precision=precision, process_repeat=process, item=key, bitwise=equal)
                    if key in a.files and key in b.files and a[key].shape == b[key].shape:
                        x, y = a[key], b[key]
                        finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
                        all_equal &= finite
                        row.update(finite=finite, baseline_dtype=str(x.dtype), current_dtype=str(y.dtype),
                                   max_abs=float(np.max(abs(x - y))),
                                   max_rel=float(np.max(abs(x.astype(np.complex128) - y.astype(np.complex128))
                                       / np.maximum(abs(x).astype(np.float64), np.finfo(float).tiny))))
                    all_equal &= equal
                    reports.append(row)
                for version, data in [('baseline', a), ('current', b)]:
                    with np.load(out / 'fixtures' / 'reference.npz') as ref:
                        errors = dict(krylov=float(abs(data['krylov'][:, ref['mask']] - ref['states']).max()),
                                      RK45=float(abs(data['RK45'][:, ref['mask']] - ref['states']).max()),
                                      expectations=float(abs(data['expectations'] - ref['expectations']).max()),
                                      ground=float(abs(data['ground'][-1] - ref['energy'])),
                                      dense=float(abs(data['dense'] - data['dense_reference']).max()),
                                      dense_scalar=float(abs(data['dense_scalar'] - data['dense_reference'][:, 1]).max()))
                    if precision == 32 and (out / 'baseline64_r0' / 'outputs.npz').is_file():
                        with np.load(out / 'baseline64_r0' / 'outputs.npz') as fp64:
                            errors.update({key + '_vs_baseline_fp64': float(abs(data[key] - fp64[key]).max())
                                           for key in ('krylov', 'RK45')})
                    repeats = {}
                    for key in data.files:
                        if key.endswith('_replicas'):
                            base = key[:-9]
                            repeats[base] = all(bitwise(value, data[base]) for value in data[key])
                    process_equal = True
                    if version not in first:
                        first[version] = {k: data[k].copy() for k in data.files}
                    else:
                        process_equal = (set(data.files) == set(first[version])
                                         and all(bitwise(data[k], first[version][k]) for k in data.files))
                    all_equal &= all(repeats.values()) and process_equal
                    reports.append(dict(precision=precision, process_repeat=process, version=version,
                                        reference_max_abs=errors, replicas_bitwise=repeats,
                                        process_repeat_bitwise=process_equal))
            for version, folder in folders.items():
                for item, values in json.loads((folder / 'timings.json').read_text()).items():
                    accumulated[version].setdefault(item, []).extend(values)
        for item, baseline in accumulated['baseline'].items():
            current = accumulated['current'][item]
            ratio = statistics.median(current) / statistics.median(baseline)
            summaries.append(dict(precision=precision, item=item, baseline_seconds=baseline, current_seconds=current,
                                  baseline_median=statistics.median(baseline), current_median=statistics.median(current),
                                  current_over_baseline=ratio, no_slowdown=ratio <= 1 + opt.slowdown_tolerance))
    write_json(out / 'comparison.json', reports)
    write_json(out / 'summary.json', dict(parity=bool(all_equal), slowdown_tolerance=opt.slowdown_tolerance,
                                        timings=summaries))
    return all_equal and all(row['no_slowdown'] for row in summaries)


def main():
    ap = worker_parser(__doc__)
    ap.set_defaults(number=20, repeat=3, warmup=1)
    ap.add_argument('--gpu', default='0', help='physical GPU index or CUDA UUID')
    ap.add_argument('--precisions', default='64,32')
    ap.add_argument('--process-repeats', type=positive, default=2)
    ap.add_argument('--steps', type=positive, default=3)
    ap.add_argument('--dt', type=float, default=.05)
    ap.add_argument('--sweeps', type=positive, default=4)
    ap.add_argument('--slowdown-tolerance', type=float, default=.10, help='allowed fractional median increase')
    ap.add_argument('--role', choices=['setup', 'measure'], default='measure', help=argparse.SUPPRESS)
    ap.add_argument('--precision', type=int, default=64, help=argparse.SUPPRESS)
    ap.add_argument('--fixture', type=Path, help=argparse.SUPPRESS)
    opt = ap.parse_args()
    if opt.worker:
        worker(opt)
        return
    if not opt.baseline:
        ap.error('--baseline is required')
    if opt.cap <= 0 or opt.dt <= 0 or opt.slowdown_tolerance < 0:
        ap.error('cap/dt must be positive and tolerance nonnegative')
    precisions = list(dict.fromkeys(map(int, opt.precisions.split(','))))
    if not precisions or set(precisions) - {32, 64}:
        ap.error('--precisions must contain 64 and/or 32')
    out = opt.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture = out / 'fixtures'
    fixture.mkdir()
    cores = core_list(opt.cores)
    if len(cores) < opt.threads:
        ap.error('not enough cores for requested thread count')
    prefix = pin_prefix(cores[:opt.threads], opt.unpinned)
    jobs = [('setup', 'baseline', 64, 0)]
    for process in range(opt.process_repeats):
        for precision in precisions:
            order = ('baseline', 'current') if process % 2 == 0 else ('current', 'baseline')
            jobs.extend(('measure', version, precision, process) for version in order)
    statuses = []
    write_json(out / 'run_config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(opt).items()})
    for role, version, precision, process in jobs:
        source = opt.baseline if version == 'baseline' else opt.current
        python = (opt.baseline_python or opt.python) if version == 'baseline' else opt.python
        folder = fixture if role == 'setup' else out / f'{version}{precision}_r{process}'
        if folder != fixture:
            folder.mkdir()
        command = prefix + [interpreter(python), str(Path(__file__).resolve()), '--worker',
            '--source', str(source), '--out', str(folder), '--fixture', str(fixture), '--role', role,
            '--precision', str(precision), '--steps', str(opt.steps), '--dt', str(opt.dt),
            '--sweeps', str(opt.sweeps), '--repeat', str(opt.repeat), '--warmup', str(opt.warmup),
            '--number', str(opt.number)]
        env = environment(source, opt.threads, opt.gpu, precision)
        env.update(CUPY_CACHE_DIR=str(out / 'cupy_cache'), XDG_CACHE_HOME=str(out / 'cache'), TMPDIR=str(folder))
        row = run_process(command, folder, env, folder / 'output.log', opt.cap)
        row.update(role=role, version=version, precision=precision, process_repeat=process)
        statuses.append(row)
        write_json(out / 'status.json', statuses)
        print(folder.name, 'rc=', row['returncode'], flush=True)
        if row['returncode'] or row['timed_out']:
            raise SystemExit('GPU worker failed or timed out; see status.json and output.log')
    if not compare(out, opt):
        raise SystemExit('Parity, reproducibility or configured speed threshold failed; see reports')
    print(out / 'summary.json')


if __name__ == '__main__':
    main()
