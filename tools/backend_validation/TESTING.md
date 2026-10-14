# Testing backend compatibility

Backend validation has three separate layers:

1. `renormalizer/backend/tests/test_*contract.py` checks the named primitive contract, including dtype, copy, device, layout and explicit unsupported cases.
2. `test_application_semantics.py` checks compatibility operations actually consumed by algorithms, and complete internal IVP paths against an analytical solution. It covers scalar/complex comparisons, RNG sizes, contractions, forward/backward integration, events and dense-output context retention.
3. `test_scientific_paths.py` checks initialization, registered MPS/TTNS evolution methods, observations, reduced density matrices, raw-environment solver boundaries and persistent storage. Dense references use independently constructed Pauli Hamiltonians and partial traces. Property/mobility regressions are in `renormalizer/property/tests/test_transition_and_mobility.py`.

A passing layer does not establish the next layer. A passing CPU test does not establish CUDA behavior. NumPy1 and NumPy2 use the same adapter but require separate dependency environments.

## Reproducible selected runs

The new application and scientific fixtures accept explicit backend/device selection:

```sh
python -m pytest renormalizer/backend/tests/test_application_semantics.py \
  renormalizer/backend/tests/test_scientific_paths.py \
  renormalizer/property/tests/test_transition_and_mobility.py \
  --reno-backend=numpy --reno-device=cpu

python -m pytest \
  renormalizer/backend/tests/test_application_semantics.py \
  renormalizer/backend/tests/test_scientific_paths.py \
  --reno-backend=jax --reno-device=cpu

CUDA_VISIBLE_DEVICES=0 python -m pytest \
  renormalizer/backend/tests/test_application_semantics.py \
  renormalizer/backend/tests/test_scientific_paths.py \
  --reno-backend=torch --reno-device=cuda:0
```

Use `cupy` or `jax` with `cuda:0` for the other device jobs. `RENO_TEST_BACKEND` and `RENO_TEST_DEVICE` are equivalent fixture defaults; CLI options take precedence. An explicitly selected missing or unusable runtime is a failing job, not a skipped success. These options select the new fixtures; they do **not** magically parameterize every historical test or change the application's global backend API. Legacy global-selection checks need a separate process and explicit `set_backend` before collection and execution.

Reno defaults to float64/complex128. Both `set_backend('jax')` and default
`make_context('jax')` enable JAX x64 support before allocating Reno arrays;
no precision environment variable is required. This is a process-wide JAX
setting and affects subsequent JAX defaults, but does not recast existing
arrays. Explicit float32 contexts and the existing presence-based `RENO_FP32`
selection remain available and never turn off an already enabled x64 setting.
If double precision cannot be enabled or verified, selection fails without
publishing the candidate backend; the previous backend and host RNG states
are preserved.

Default `python -m pytest` discovers source tests/doctests and the validation tool tests. Build outputs, environments, docs and examples are excluded. The optional JAX adapter is omitted from doctest import only when JAX is absent; an installed but broken runtime is not hidden.

## Evidence and boundaries

Scientific tests record actual contraction witnesses during candidate evolution, excluding input setup and reference computations. Where a witness exists, its adapter identity, device and result ownership are checked. Three MPS propagation/compression variants currently have no such contraction witness and are labelled accordingly; numerical agreement alone is not evidence of GPU execution. The current algorithms retain host tensor storage and explicit host solver work.

Direct private sparse block kernels are NumPy-only. Their tests explicitly enter a labelled NumPy scope; optimization fallback tests continue under the selected backend. Four TTNS evolution enum choices are not registered implementations. Dependency absence, unsupported methods, test timeouts and native process aborts must remain distinct from numerical passes.

`test_oe_wrap.py` injects deterministic allocation exceptions to check error logging and exception identity. It deliberately does not allocate impossible arrays. Real allocator/OOM behavior requires separate bounded processes and cannot be inferred from these unit tests.

Host reference comparisons explicitly convert candidate arrays at the comparison boundary. Do not replace a native candidate computation with a NumPy reference or widen tolerances merely to obtain a pass.

## Numerical behavior correction

MPS one- and two-site reduced density matrices now follow their documented ket-first definition, `rho = Tr_other |psi><psi|`, consistent with `trace(rho @ observable)`. The previous complex result was transposed; real-only and entropy comparisons could miss the error. Code that manually compensated for that transpose should remove the compensation. Shapes and return types remain unchanged. TTNS native RDM return behavior is preserved.

## Internal helpers and complete consumers

`test_consumed_surface_semantics.py` checks the consumed scalar/iterable axis forms and helper return semantics, plus dense and grouped sparse numerical Jacobians against analytic matrices. Retry tests witness a larger second perturbation and preserve the supplied factor. Dense Jacobians and RHS evaluations stay on the selected backend; sparse CSC assembly is an explicit SciPy host boundary. These internal Jacobian tests do not imply that the registered RK methods invoke this helper.

Variational-compression regressions cover MPS/MpDm/MPO, real/complex data, one/two-site procedures and both sweep directions, comparing independent dense operator action and exact input preservation. Cached-transition checks cover mixed real/complex operands and both physical and legacy bra conventions.

TTNS entropy tests compare a known Schmidt spectrum and the documented half-mutual-information convention. Public native density-matrix results remain native; entropy consumers explicitly transfer them to the existing SciPy host eigensolver. Passing an array comparison at a host boundary alone is insufficient to establish the full entropy consumer works.
