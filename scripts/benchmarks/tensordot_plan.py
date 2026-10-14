"""Compare the current tree's cached tensordot implementation with NumPy.

Example: python tensordot_plan.py --current . --out tensordot --number 5000
results.json gives warmed median per-call seconds, cache stats and strict byte
parity for representative small/large, real/complex and noncontiguous tensors.
This measures the actual library implementation, not a copied prototype.
"""
from _common import bitwise, dispatch, measure, provenance, worker_parser, write_json


def main():
    opt = worker_parser(__doc__).parse_args()
    if opt.baseline:
        raise SystemExit('This probe uses the current cached-plan API; omit --baseline')
    if not dispatch(opt, __file__):
        return
    metadata = provenance(opt.source)
    import numpy as np
    from renormalizer.backend import numpy_contraction as nc
    rng = np.random.default_rng(0)
    shapes = [((4, 3, 4), (4, 3, 4), ([2], [0])),
              ((16, 2, 16), (16, 2, 16), ([2], [0])),
              ((16, 2, 16), (16, 2, 2, 16), ([1, 2], [1, 0])),
              ((64, 4, 64), (64, 4, 4, 64), ([1, 2], [1, 0]))]
    rows = []
    for sa, sb, axes in shapes:
        for complex_input in (False, True):
            a, b = rng.standard_normal(sa), rng.standard_normal(sb)
            if complex_input:
                a = a + 1j * rng.standard_normal(sa)
                b = b + 1j * rng.standard_normal(sb)
            for noncontiguous in (False, True):
                x, y = (a[..., ::-1], b[..., ::-1]) if noncontiguous else (a, b)
                reference, actual = np.tensordot(x, y, axes), nc.tensordot(x, y, axes)
                rows.append(dict(shapes=[sa, sb], axes=axes, dtype=str(a.dtype), noncontiguous=noncontiguous,
                    bitwise=bitwise(reference, actual),
                    numpy=measure(lambda: np.tensordot(x, y, axes), opt.number, opt.repeat, opt.warmup),
                    cached=measure(lambda: nc.tensordot(x, y, axes), opt.number, opt.repeat, opt.warmup)))
    write_json(opt.out / 'results.json', dict(metadata=metadata, measurements=rows, cache=nc._plan.cache_info()._asdict()))
    if not all(row['bitwise'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
