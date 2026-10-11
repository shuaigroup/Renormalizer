"""Per-call cost of renormalizer's own environment update and H-v by bond dimension.

Example: python contraction_cost.py --current . --out contraction --bonds 1,2,4,8,16,32,64,128
Both operations are the package's, called as TDVP-PS calls them: the environment
update is lib.contract_one_site (what Environ.GetLR runs) on an asxp environment
and Matrix MPS/MPO sites; H-v is hop_expr applied as hop(y.reshape(shape)).ravel().
results.json reports their medians next to one current_backend() lookup, so the
share of backend resolution can be read against whole calls of known size.
"""
from _common import dispatch, measure, provenance, worker_parser, write_json


def main():
    ap = worker_parser(__doc__)
    ap.set_defaults(number=1000)
    ap.add_argument('--bonds', default='1,2,4,8,16,32,64,128')
    opt = ap.parse_args()
    if opt.baseline:
        ap.error('this probe requires current multibackend APIs; omit --baseline')
    if not dispatch(opt, __file__):
        return
    metadata = provenance(opt.source)
    import numpy as np
    from renormalizer.backend.context import current_backend
    from renormalizer.mps.matrix import Matrix, asxp, asworkspace
    from renormalizer.mps.lib import contract_one_site
    from renormalizer.mps.hop_expr import hop_expr
    rng = np.random.default_rng(0)
    lookup = measure(current_backend, opt.number, opt.repeat, opt.warmup)
    rows = []
    for bond in map(int, opt.bonds.split(',')):
        if bond < 1:
            raise ValueError('bond dimensions must be positive')
        left = rng.standard_normal((bond, 5, bond)) + 0j
        right = rng.standard_normal((bond, 5, bond)) + 0j
        state = rng.standard_normal((bond, 2, bond)) + 1j * rng.standard_normal((bond, 2, bond))
        operator = rng.standard_normal((5, 2, 2, 5))
        # As stored by Environ.write and returned by Environ.read.
        environ = asxp(asworkspace(left))
        ms, mo = Matrix(state), Matrix(operator)
        hop = hop_expr(environ, asxp(asworkspace(right)), [asxp(operator)], list(state.shape))
        y = asxp(state.ravel())
        shape = list(state.shape)
        for label, call in [('environment', lambda: contract_one_site(environ, ms, mo, 'L')),
                            ('hv', lambda: hop(y.reshape(shape)).ravel())]:
            rows.append(dict(bond=bond, operation=label,
                             call=measure(call, opt.number, opt.repeat, opt.warmup)))
    write_json(opt.out / 'results.json', dict(metadata=metadata, lookup=lookup, measurements=rows))


if __name__ == '__main__':
    main()
