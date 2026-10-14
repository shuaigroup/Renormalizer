"""Measure warmed per-call backend dispatch costs in isolated CPU interpreters.

Example: python micro_dispatch.py --current . --baseline ../reno-baseline --out micro
results.json reports raw samples and medians in seconds per call. Baseline-only
legacy APIs are measured where available; absent multibackend APIs are marked
unavailable, never substituted with fabricated baseline timings.
"""
from _common import dispatch, measure, provenance, worker_parser, write_json


def main():
    opt = worker_parser(__doc__).parse_args()
    if not dispatch(opt, __file__):
        return
    metadata = provenance(opt.source)
    import numpy as np
    from renormalizer.mps.matrix import Matrix, tensordot, asxp, asnumpy
    from renormalizer.mps.backend import backend
    a = np.ones((4, 3, 4))
    m = Matrix(a)
    cases = {'numpy.tensordot attribute': lambda: np.tensordot,
             'legacy canonical_atol': lambda: backend.canonical_atol,
             'asxp(Matrix)': lambda: asxp(m), 'asnumpy(Matrix)': lambda: asnumpy(m),
             'Matrix(ndarray)': lambda: Matrix(a),
             'matrix.tensordot': lambda: tensordot(m, m, ([2], [0])),
             'numpy.tensordot': lambda: np.tensordot(a, a, ([2], [0]))}
    unavailable = []
    try:
        from renormalizer.backend.context import internal_backend as xp, current_backend
        from renormalizer.backend.execution import to_backend, to_host
    except ModuleNotFoundError as error:
        if not error.name.startswith('renormalizer.backend'):
            raise
        unavailable.append('multibackend APIs: ' + str(error))
    else:
        b = current_backend()
        cases.update({'proxy.tensordot attribute': lambda: xp.tensordot,
                      'adapter.tensordot attribute': lambda: b.tensordot,
                      'proxy.real_dtype': lambda: xp.real_dtype,
                      'adapter.real_dtype': lambda: b.real_dtype,
                      'proxy.canonical_atol': lambda: xp.canonical_atol,
                      'adapter.canonical_atol': lambda: b.canonical_atol,
                      'current_backend()': current_backend,
                      'to_backend(ndarray)': lambda: to_backend(a),
                      'to_host(ndarray)': lambda: to_host(a)})
    result = {name: measure(fn, opt.number, opt.repeat, opt.warmup) for name, fn in cases.items()}
    write_json(opt.out / 'results.json', dict(metadata=metadata, measurements=result, unavailable=unavailable))


if __name__ == '__main__':
    main()
