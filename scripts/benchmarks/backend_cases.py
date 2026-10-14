"""Ground-state and TDVP backend cases with optional shared initial states.

Environment:
  BENCH_BACKEND   numpy (default) | jax | torch   (only on trees with set_backend)
  BENCH_CASES     comma list from CASES below (default: all)
  BENCH_REPEAT    repetitions, best is reported (default 3)
  BENCH_INIT_DIR  directory of shared NPZ initial states (default: generate in memory)
Prints one JSON line per case, with all times and the final repetition's value.
The first timing includes first-use costs; later timings reuse process caches.
Missing shared states are generated and saved, so prepopulate this directory
when it must remain read-only. Select a source tree with PYTHONPATH.
"""
import json, logging, os, sys, time

logging.disable(logging.CRITICAL)
# renormalizer first, as in user code: it applies RENO_NUM_THREADS before NumPy loads BLAS.
import renormalizer
backend_name = os.environ.get("BENCH_BACKEND", "numpy")
if backend_name != "numpy":
    renormalizer.set_backend(backend_name)

import numpy as np

from renormalizer.model import Model, Op
from renormalizer.model.basis import BasisHalfSpin
from renormalizer.mps import Mps, Mpo
from renormalizer.mps.gs import optimize_mps
from renormalizer.utils import CompressConfig, EvolveConfig, EvolveMethod
from renormalizer.utils.configs import OptimizeConfig
from renormalizer.tests.parameter import holstein_model


def heisenberg(nsite):
    # quantum number counts up spins, so sectors are block-sparse (Sz conserved)
    basis = [BasisHalfSpin(i, [0, 1]) for i in range(nsite)]
    terms = []
    for i in range(nsite - 1):
        terms.append(Op("sigma_+ sigma_-", [i, i + 1], 0.5))
        terms.append(Op("sigma_- sigma_+", [i, i + 1], 0.5))
        terms.append(Op("sigma_z sigma_z", [i, i + 1], 0.25))
    return Model(basis, terms)


INIT_DIR = os.environ.get("BENCH_INIT_DIR")


def initial_state(tag, model, qntot, m):
    """Identical start for every backend/LAPACK: Mps.random depends on SVD
    conventions, so the state is generated once and loaded from disk."""
    if INIT_DIR is None:
        np.random.seed(7)
        return Mps.random(model, qntot, m, percent=1.0)
    fname = os.path.join(INIT_DIR, f"{tag}.npz")
    if not os.path.exists(fname):
        np.random.seed(7)
        Mps.random(model, qntot, m, percent=1.0).dump(fname)
    return Mps.load(model, fname)


def gs_job(tag, model, qntot, m, nsweep):
    mpo = Mpo(model)
    def run():
        mps = initial_state(tag, model, qntot, m)
        mps.optimize_config = OptimizeConfig(procedure=[[m, 0.2]] + [[m, 0]] * (nsweep - 1))
        energies, _ = optimize_mps(mps, mpo)
        return float(np.min(energies))
    return run


def tdvp_job(tag, model, qntot, m, nstep, dt, probe=None):
    mpo = Mpo(model)
    probe_mpo = Mpo(model, Op(*probe)) if probe else None
    def run():
        mps = initial_state(tag, model, qntot, m)
        mps.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
        mps.compress_config = CompressConfig(max_bonddim=m)
        for _ in range(nstep):
            mps = mps.evolve(mpo, dt)
        energy = complex(mps.expectation(mpo)).real
        local = complex(mps.expectation(probe_mpo)).real if probe_mpo is not None else None
        return [energy, local]
    return run


L = 32
CASES = {
    # Python-overhead regime: tiny tensors, many calls
    "small_gs":   lambda: gs_job("small_gs", holstein_model, 1, 20, 5),
    "small_tdvp": lambda: tdvp_job("small_tdvp", holstein_model, 1, 20, 10, 10.0),
    # intermediate
    "mid_gs":     lambda: gs_job("mid_gs", heisenberg(L), L // 2, 64, 3),
    "mid_tdvp":   lambda: tdvp_job("mid_tdvp", heisenberg(L), L // 2, 64, 2, 0.05, ("sigma_z", L // 2)),
    # BLAS regime
    "large_gs":   lambda: gs_job("large_gs", heisenberg(L), L // 2, 200, 2),
    "large_tdvp": lambda: tdvp_job("large_tdvp", heisenberg(L), L // 2, 160, 1, 0.05, ("sigma_z", L // 2)),
}

selected = os.environ.get("BENCH_CASES")
names = selected.split(",") if selected else list(CASES)
repeat = int(os.environ.get("BENCH_REPEAT", "3"))
for name in names:
    job = CASES[name]()
    times, value = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        value = job()
        times.append(time.perf_counter() - t0)
    print(json.dumps({"case": name, "backend": backend_name, "numpy": np.__version__,
                      "best": round(min(times), 4), "times": [round(t, 4) for t in times],
                      "value": value}), flush=True)
