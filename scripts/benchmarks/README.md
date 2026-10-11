# Renormalizer CPU and CuPy benchmarks

This collection is intended to establish, starting from baseline commit
`956ad15b6be989049c1a009a0050334f3d774d43`, that the multibackend branch's new
features are good on CPU: low backend dispatch overhead, faster cached
tensordot plans, effective contraction-path/expression caches, efficient
one-sided TDVP environments, and useful single-BLAS pixi environment and thread
scaling. The numerical target is outputs bitwise identical to the baseline.
On CuPy, the target is numerical identity in float64 and float32 for the tested
paths, with no slowdown.

These are goals to measure, not results asserted by this directory. No benchmark,
example or test was run while preparing this collection. Only syntax compilation
and static inspection were performed. Measurements apply to the selected
workloads, source revisions, dependency environments and hardware; they do not
prove equivalence for all inputs or all backend features.

## Entry points

| Script | What it shows | Main output |
| --- | --- | --- |
| `backend_cases.py` | Ground-state/TDVP backend cases; configured through `BENCH_BACKEND`, `BENCH_CASES`, `BENCH_REPEAT`, and optional `BENCH_INIT_DIR` | JSON lines with all timings and final-repeat energies/observables; see [JAX measurements](results/2026-10-pr/JAX.md) |
| `run_examples.py` | Same-batch example scheduling across source trees, interpreters and thread counts; separate cProfile jobs | `summary.csv`, `summary.json`, `medians.json`, per-job logs, GNU time, environment metadata and profiles |
| `topo_alloc.py` | Physical-core/L3/NUMA layout and allocation policy | Printed topology; also imported by the scheduler |
| `compare_outputs.py` | Same-batch baseline/current NPZ byte equality, normalized log equality and TTN matched-step throughput | `comparison.json`, normalized logs, scheduler outputs |
| `micro_dispatch.py` | Per-call proxy/adapter attributes, canonical tolerance, backend resolution, transfers, Matrix construction and small tensordot | Per-version `results.json` with raw samples and medians |
| `tensordot_plan.py` | Actual current-tree cached tensordot versus NumPy, including complex and strided inputs | `results.json` with bytes, timings and plan-cache statistics |
| `contraction_cost.py` | Environment update and H-v compute versus wrapper cost across bond dimensions | `results.json` with medians and estimated dispatch/lookup shares |
| `trace_example.py` | `hv`: expression evaluation latency; `caches`: path/expression/label/plan hit rates; `tensordot`: NumPy call sites | `trace.json`, `meta.json`, log and environment metadata |
| `cupy_parity.py` | CuPy Krylov/RK45 TDVP, complex expectations, Davidson ground state, and scalar/vector RK45 dense output | Shared fixtures, per-worker NPZ/timings/metadata, `comparison.json`, `summary.json` |
| `sbm_short.py` | Short spin-boson workload, default 300 modes and evolution time 2 | SpinBosonDynamics output files |

`_common.py` contains shared process/environment/timing utilities.
`_example_worker.py` validates source/backend provenance before launching an
example and optionally profiles it. They are helpers, not additional workloads.
Drivers have argparse help; `backend_cases.py` uses the environment variables in its docstring. Worker-only imports happen after parsing.

## Prerequisites and source trees

Drivers use the standard library, plus NumPy for output comparisons. They do not
need Renormalizer, SciPy, opt_einsum or CuPy installed in the driver interpreter.
The chosen worker interpreters need the dependencies of their respective source
trees; GPU workers additionally need a functioning CuPy/CUDA installation.
The driver and worker interpreters may be different.

The example scheduler targets Linux and requires `taskset` and GNU `time`.
Topology discovery uses `lstopo-no-graphics` or `lstopo`, falling back to Linux
`/sys`. Memory binding uses `hwloc-bind`, or `numactl` when available; the recorded
topology states whether it was enabled. `--hwloc-bin DIR` supplies an optional
hwloc installation, and `--gnu-time FILE` supplies GNU time. `--no-membind`
explicitly disables memory binding. Micro, trace and GPU drivers also use
`taskset` by default; `--unpinned` explicitly opts out on other systems. Do not
mix pinned and unpinned measurements as if they were comparable.

From the repository root, prepare a baseline tree yourself:

```sh
git worktree add ../reno-baseline 956ad15b6be989049c1a009a0050334f3d774d43
```

Each `--baseline`/`--current` path must contain a `renormalizer` package.
`--current` defaults to the repository containing these scripts. `--python`
defaults to the driver interpreter; `--baseline-python` defaults to `--python`.
Relative interpreter paths are resolved before changing worker directories.
Workers record the imported package path, interpreter, source commit, NumPy/BLAS
configuration, thread environment and affinity, and reject a different imported
tree or unexpected CPU/GPU backend. Keep trees and environments unchanged during
a batch; a commit alone does not describe uncommitted changes.

Choose an allowed CPU pool with `--cores 0-7,16-23`, for example. Omit it to use
the driver's allowed affinity. The scheduler selects one logical CPU per physical
core and runs jobs on disjoint allocations. Micro/trace/GPU runs are sequential
and take the first `--threads` CPUs from their pool. Explicitly choose physical
cores for multithreaded runs of those drivers.

All drivers create fresh output directories and refuse to reuse an existing
batch directory. The scheduler uses `OUT/RUN_ID`, with an automatically generated
run ID unless supplied. Other drivers use `OUT` directly. Benchmark results are written to
the selected output tree; bytecode generation is disabled. Existing outputs are
never automatically deleted.

## CPU comparisons and scaling

Use one common copy of the example input directory for all versions. It defaults
to `CURRENT/example`; override it with `--example-dir DIR` for a prepared workload
set. Each job gets its own copy. The bundled `sbm_short.py` is added to every copy.
An example can only support parity claims for versions that support that workload.

```sh
python scripts/benchmarks/compare_outputs.py \
  --baseline ../reno-baseline --current . \
  --python .pixi/envs/default/bin/python \
  --cores 0-7 --out benchmark-results/cpu-parity \
  --examples sbm_short,fmo,dynamics,hubbard,h2o_qc,ttns_sbm_zt \
  --cap 900 --example-cap ttns_sbm_zt=240

python scripts/benchmarks/run_examples.py \
  --baseline ../reno-baseline --current . \
  --python .pixi/envs/default/bin/python \
  --threads 1,2,4 --cores 0-15 --repeat 3 --warmup 1 \
  --profile-threads 1 --examples sbm_short,fmo,hubbard \
  --out benchmark-results/scaling
```

Example names are `fmo`, `sbm`, `sbm_short`, `h2o_qc`, `dynamics`,
`transport_kubo`, `ttns_junction_zt`, `ttns_junction_ft`, `ttns_sbm_zt`,
`ttns_sbm_ft`, `ssh`, and `hubbard`; `--examples all` selects all. The original
TTN argument sets are retained. `--dry-run` prints the schedule without launching
examples. Caps apply separately to each warmup, timed and profile process.

Additional source/interpreter configurations replace the old environment-specific
shell scripts. For example, after preparing an alternative environment:

```sh
python scripts/benchmarks/run_examples.py \
  --baseline ../reno-baseline --current . \
  --python .pixi/envs/default/bin/python \
  --version current_other . ../other-env/bin/python \
  --version baseline_other ../reno-baseline ../other-env/bin/python \
  --threads 1,2,4 --examples sbm_short,fmo --out benchmark-results/environments
```

This lets a single batch compare the pixi single-BLAS environment with another
NumPy/BLAS environment. Examine the saved environment metadata and environment
package specifications before attributing a change to BLAS. These scripts do not
install environments or prove single-BLAS linkage simply from an interpreter's
name. For numerical verification between two environments, use
`compare_outputs.py --baseline-python OTHER_PYTHON --python CURRENT_PYTHON`.
For one-sided TDVP environment changes, select the relevant before/after source
trees and compare the same `sbm_short`, `fmo`, `dynamics` or `hubbard` workloads.
Timing alone does not isolate that optimization from other branch changes.

`summary.json/csv` retain every completed job's return code, timeout flag,
placement, load averages, wall time, CPU time and peak RSS. `medians.json` includes
only successful timed jobs, with their actual completed repeat count. A missing
repeat is not replaced or treated as fast. Warmups are separate discarded
process runs; timed examples still include startup and first-use costs. cProfile
jobs are separate and excluded from medians, with full `profile.prof` and text
reports sorted by self/cumulative time plus backend/ContextVar filtering.

`compare_outputs.py` compares every timed pair. Unchanged copied input NPZ/log files are identified by
pre-run hashes and excluded from output evidence. NPZ file/key sets, shapes, dtypes
and C-order bytes must agree; signed zero and NaN payload differences matter.
Object arrays are reported as unsupported instead of comparing pointer bytes or
loading untrusted pickle payloads. Logs strip timestamps, object addresses and
recognized timing/environment noise, then require the full remaining sequences,
including lengths, to agree. Normalized files and diff previews make this filter
reviewable. Warnings are retained. Equality of printed values only covers their
printed precision, not the underlying floating-point bytes.

A failed or capped job is incomplete and causes a nonzero exit. For TTN logs,
`[INFO] (` lines are paired in order only while their result text matches. The
reported elapsed interval spans the first to last matched timestamp, excludes
startup before the first result, and records how many step intervals it covers.
A matching prefix provides throughput evidence, not full-run numerical parity.

## Focused CPU probes and instrumentation

```sh
python scripts/benchmarks/micro_dispatch.py --current . \
  --baseline ../reno-baseline --python .pixi/envs/default/bin/python \
  --number 200000 --repeat 5 --cores 0 --out benchmark-results/micro
python scripts/benchmarks/tensordot_plan.py --current . \
  --python .pixi/envs/default/bin/python --number 5000 --cores 0 \
  --out benchmark-results/tensordot
python scripts/benchmarks/contraction_cost.py --current . \
  --python .pixi/envs/default/bin/python --bonds 1,2,4,8,16,32,64,128 \
  --number 1000 --cores 0 --out benchmark-results/contractions
python scripts/benchmarks/trace_example.py caches --current . \
  --python .pixi/envs/default/bin/python --example ttns/sbm_zt.py \
  --cores 0 --cap 240 --out benchmark-results/cache-trace -- 050 001 050
```

Replace `caches` with `hv` or `tensordot` for the other trace modes. Instrumentation
uses internal APIs, so a tree without those APIs can fail explicitly. H-v mode
actually observes all `execution.contract_expression` callables; use recorded
call sites to identify which belong to H-v. NumPy tensordot tracing cannot see
calls handled wholly by the cached-plan implementation. It is not a complete
contraction census. Instrumented times contain observer overhead; compare
uninstrumented examples to evaluate speed. Trace JSON is written in a `finally`
block on ordinary exceptions and graceful SIGTERM; SIGKILL cannot guarantee it.

Micro and contraction probes exclude explicit warmup calls and report every
sample plus the median seconds per call. They do not use best-of-run minima.
`contraction_cost.py` holds the backend fixed in its prebuilt compute reference;
wrapper-minus-compute and backend-lookup shares are estimates. Noise can produce
negative differences. Tensordot plan parity covers the listed representative
cases, not exhaustive axes validation or all shapes/dtypes.

## GPU parity and performance

Use an idle GPU. Both source trees must be compatible with the selected CuPy
interpreter(s). The driver needs only NumPy; all CUDA work occurs in workers.

```sh
python scripts/benchmarks/cupy_parity.py \
  --baseline ../reno-baseline --current . \
  --python ../cupy-env/bin/python --gpu 0 --cores 0 \
  --precisions 64,32 --repeat 3 --process-repeats 2 --warmup 1 \
  --cap 900 --out benchmark-results/cupy
```

The baseline float64 setup creates one shared initial MPS, evolved complex MPS
and dense-reference fixture. Both versions and precisions load those same inputs.
`RENO_FP32=1` is set only for float32. Each worker checks `USE_GPU`, CuPy array
conversion, actual precision and CuPy dense-output arrays; silent NumPy fallback
fails the job. GPU selection uses `CUDA_VISIBLE_DEVICES`, with `RENO_GPU=0` inside
that visibility mapping. CuPy cache and temporary directories live under OUT.

Workers run sequentially, alternate version order between process repeats, and
synchronize CUDA before/after timing. Warmup is excluded; TDVP host dense-state
reconstruction occurs after the timed region. `--steps`, `--dt`, `--sweeps`,
`--number` (expectation calls per sample), `--repeat`, and `--process-repeats`
control workload sizes. Ground-state runs require and count Davidson calls.

NPZ results include trajectories, expectation values, optimization energies,
ground-state vectors, scalar/vector dense-output values and their query times,
as well as within-process replicas. Nonfinite GPU outputs fail validation. Reports compare dtype/shape/bytes, absolute
and relative differences, replica/process reproducibility, and independent dense
reference errors. When float64 is included, float32 trajectories are also compared
with the same-method baseline float64 trajectories. Equality to baseline and agreement with a dense reference are
separate checks: MPS truncation and integrator tolerances can give nonzero
reference error even when both versions are identical. Query times must match
for the dense-output comparison to be meaningful.

`summary.json` aggregates synchronized samples across the process repeats and
reports current/baseline median ratios. `--slowdown-tolerance 0.10` is the default
fractional threshold: ratios above 1.10 fail the speed gate. Use `0` for a literal
no-increase criterion, understanding that timing noise can trigger it. Raw ratios
are always reported; a tolerance is not proof of exactly equal speed. Parity,
repeatability, worker failure, timeout or speed-gate failures produce nonzero exit.

## Measurement discipline

- Compare only configurations run in the same batch on shared hosts. Expect
  roughly 10% batch-to-batch noise; use raw samples, repeated paired runs and
  workload-sized effects before drawing conclusions.
- Start with one thread per pinned job. The drivers set `OPENBLAS_NUM_THREADS`,
  `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `RENO_NUM_THREADS`,
  `VECLIB_MAXIMUM_THREADS`, and `BLIS_NUM_THREADS` to the requested count before
  importing numerical packages. They set `PYTHONHASHSEED=0` and clear inherited
  Renormalizer backend/precision settings. CPU jobs hide GPUs.
- Keep dependency versions and BLAS implementations matched when isolating a
  code change. Deliberately vary interpreters only for environment comparisons;
  bitwise equality across different NumPy/BLAS builds is not guaranteed.
- Respect affinity and NUMA boundaries. The scheduler queues paired versions
  adjacent, reverses version order on alternate repeats, and caps jobs per L3.
  Shared-host contention, clock changes and thermal effects remain possible.
- Inspect errors and incomplete jobs before interpreting medians. A short
  timeout, empty output, matching prefix or successful backend import is not
  evidence of complete workload parity.
