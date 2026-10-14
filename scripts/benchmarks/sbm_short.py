"""Short spin-boson workload for same-batch baseline/current timing.

Example: python sbm_short.py --source . --out sbm-output --phonons 300 --evolve-time 2
Writes the SpinBosonDynamics outputs for a fixed model and adaptive propagation.
Use run_examples.py --examples sbm_short for controlled interpreter, affinity,
thread caps and baseline/current comparisons; direct execution uses your environment.
"""
import argparse
import logging
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, help='source tree containing renormalizer')
    ap.add_argument('--out', type=Path, default=Path('.'))
    ap.add_argument('--phonons', type=int, default=300)
    ap.add_argument('--evolve-time', type=float, default=2)
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--threshold', type=float, default=1e-4)
    opt = ap.parse_args()
    if opt.phonons < 1 or min(opt.evolve_time, opt.dt, opt.threshold) <= 0:
        ap.error('workload sizes and tolerances must be positive')
    if opt.source:
        tree = opt.source.resolve()
        if not (tree / 'renormalizer').is_dir():
            ap.error('--source must contain renormalizer')
        sys.path.insert(0, str(tree))
    from renormalizer.sbm import SpinBosonDynamics, param2mollist
    from renormalizer.utils import Quantity, CompressConfig, EvolveConfig, log
    log.init_log(logging.INFO)
    opt.out.mkdir(parents=True, exist_ok=True)
    model = param2mollist(0.05, Quantity(1), Quantity(20), 1, opt.phonons)
    sbm = SpinBosonDynamics(model, Quantity(0),
        compress_config=CompressConfig(threshold=opt.threshold),
        evolve_config=EvolveConfig(adaptive=True, guess_dt=opt.dt),
        dump_dir=str(opt.out), job_name='sbm')
    sbm.evolve(evolve_dt=opt.dt, evolve_time=opt.evolve_time)


if __name__ == '__main__':
    main()
