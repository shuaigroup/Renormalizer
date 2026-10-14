"""Time H-v on each backend exactly as TDVP-PS builds and applies it.

Example: python hop_backends.py --current . --python JAX_PYTHON --backends numpy,jax --out hop
Each backend runs in its own pinned interpreter, selected with set_backend as a
user would. Operands are handed over as Mps._evolve_tdvp_ps does: environments
through asworkspace then asxp (Environ.write/read), the MPO site through asxp,
the Krylov vector as a backend array. The operator is renormalizer's hop_expr,
applied as hop(y.reshape(shape)).ravel(); nothing is re-implemented here.
results.json gives build (first and warm) and apply medians per bond/dtype/kind,
the JAX/NumPy apply ratio and the max difference between backends.
"""
import argparse
import json
from pathlib import Path
import zlib
from _common import (core_list, environment, interpreter, measure, pin_prefix,
                     provenance, run_process, worker_parser, write_json)

# kind -> number of MPO sites passed to hop_expr: single-site H-v and the
# zero-site (bond) step of the projector-splitting sweep.
KINDS = {'single': 1, 'zero': 0}


def operands(bond, pbond, mpo_bond, dtype, nsite, seed):
    """Seeded host arrays shaped as in TDVP-PS: L, R environments, MPO site, vector shape."""
    import numpy as np
    rng = np.random.default_rng(seed)
    def draw(shape, complex_):
        x = rng.standard_normal(shape)
        return x + 1j * rng.standard_normal(shape) if complex_ else x
    complex_ = dtype == 'complex128'
    left = draw((bond, mpo_bond, bond), complex_)
    right = draw((bond, mpo_bond, bond), complex_)
    mo = draw((mpo_bond, pbond, pbond, mpo_bond), False)  # Hamiltonian MPO stays real
    shape = [bond, pbond, bond] if nsite else [bond, bond]
    vector = draw((int(np.prod(shape)),), complex_)
    return left, right, [mo] if nsite else [], shape, vector


def worker(opt):
    import renormalizer
    if opt.backend != 'numpy':
        renormalizer.set_backend(opt.backend)
    metadata = provenance(opt.source)
    import time
    import numpy as np
    from renormalizer.backend.context import internal_backend as xp, current_backend
    from renormalizer.mps.matrix import asxp, asnumpy, asworkspace
    from renormalizer.mps.hop_expr import hop_expr
    metadata.update(backend=current_backend().name, opt_einsum_module=str(current_backend().opt_einsum_module),
                    native_workspace_storage=bool(xp.native_workspace_storage))
    rows, values = [], {}
    for kind, nsite in KINDS.items():
        for dtype in opt.dtypes.split(','):
            for bond in map(int, opt.bonds.split(',')):
                seed = zlib.crc32(f'{kind}/{dtype}/{bond}'.encode())
                left, right, mos, shape, vector = operands(bond, opt.pbond, opt.mpo_bond, dtype, nsite, seed)
                # Environ.write stores asworkspace(...); Environ.read returns asxp(...).
                l_array, r_array = asxp(asworkspace(left)), asxp(asworkspace(right))
                cmo = [asxp(mo) for mo in mos]
                y = asxp(vector)
                def build():
                    return hop_expr(l_array, r_array, list(cmo), shape)
                start = time.perf_counter()
                hop = build()
                first_build = time.perf_counter() - start
                def apply():
                    result = hop(y.reshape(shape)).ravel()
                    xp.sync()
                    return result
                start = time.perf_counter()
                result = apply()
                first_apply = time.perf_counter() - start
                key = f'{kind}/{dtype}/{bond}'
                values[key] = np.asarray(asnumpy(result))
                rows.append(dict(kind=kind, dtype=dtype, bond=bond, shape=shape,
                                 operand_types=sorted({type(a).__module__ + '.' + type(a).__name__
                                                       for a in (l_array, r_array, y, *cmo)}),
                                 first_build_seconds=first_build, first_apply_seconds=first_apply,
                                 build=measure(build, max(1, opt.number // 10), opt.repeat, opt.warmup),
                                 apply=measure(apply, opt.number, opt.repeat, opt.warmup)))
                write_json(opt.out / 'results.json', dict(metadata=metadata, measurements=rows))
    np.savez(opt.out / 'values.npz', **values)


def compare(out, backends):
    import numpy as np
    loaded = {b: np.load(out / b / 'values.npz') for b in backends}
    measured = {b: {(r['kind'], r['dtype'], r['bond']): r for r in
                    json.loads((out / b / 'results.json').read_text())['measurements']}
                for b in backends}
    reference = backends[0]
    rows = []
    for key, row in measured[reference].items():
        name = '/'.join(map(str, key))
        for other in backends[1:]:
            a, b = loaded[reference][name], loaded[other][name]
            difference = np.abs(a - b)
            rows.append(dict(kind=key[0], dtype=key[1], bond=key[2], backend=other, reference=reference,
                             apply_ratio=measured[other][key]['apply']['median_seconds'] / row['apply']['median_seconds'],
                             build_ratio=measured[other][key]['build']['median_seconds'] / row['build']['median_seconds'],
                             max_abs=float(difference.max()),
                             max_rel=float((difference / np.maximum(np.abs(a), np.finfo(float).tiny)).max())))
    write_json(out / 'comparison.json', rows)


def main():
    ap = worker_parser(__doc__)
    ap.set_defaults(number=200, repeat=7, warmup=10)
    ap.add_argument('--backends', default='numpy,jax', help='first one is the reference')
    ap.add_argument('--bonds', default='8,16,32,64,128,256')
    ap.add_argument('--dtypes', default='float64,complex128')
    ap.add_argument('--pbond', type=int, default=2)
    ap.add_argument('--mpo-bond', type=int, default=5)
    ap.add_argument('--backend', help=argparse.SUPPRESS)
    opt = ap.parse_args()
    if opt.worker:
        worker(opt)
        return
    if opt.baseline:
        ap.error('hop_backends compares backends of one tree; omit --baseline')
    out = opt.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cores = core_list(opt.cores)
    if len(cores) < opt.threads:
        ap.error('not enough cores for --threads')
    prefix = pin_prefix(cores[:opt.threads], opt.unpinned)
    backends = opt.backends.split(',')
    rows = []
    for backend in backends:
        work = out / backend
        work.mkdir()
        env = environment(opt.current, opt.threads)
        env['JAX_PLATFORMS'] = 'cpu'
        if opt.threads == 1:
            env['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
        command = prefix + [interpreter(opt.python), str(Path(__file__).resolve()), '--worker',
                            '--backend', backend, '--source', str(opt.current), '--out', str(work),
                            '--bonds', opt.bonds, '--dtypes', opt.dtypes, '--pbond', str(opt.pbond),
                            '--mpo-bond', str(opt.mpo_bond), '--number', str(opt.number),
                            '--repeat', str(opt.repeat), '--warmup', str(opt.warmup)]
        row = run_process(command, work, env, work / 'output.log', opt.cap)
        row.update(backend=backend)
        rows.append(row)
        write_json(out / 'summary.json', rows)
    if any(r['returncode'] or r['timed_out'] for r in rows):
        raise SystemExit('Worker failed or timed out; see summary.json and output.log')
    if len(backends) > 1:
        compare(out, backends)
    print(out / 'summary.json')


if __name__ == '__main__':
    main()
