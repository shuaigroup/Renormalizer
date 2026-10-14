# Optional developer validation tools

This directory is a reproducible development aid, not a dependency of normal
Renormalizer imports and not a general AI execution platform. Run from a source
checkout. The base installation and each backend remain independently usable.

Start with `python -m tools.backend_validation.environment numpy2 --device cpu`
for an installation check, or `examples/backend_candidate_validation.py` for the
fixed two-input matrix-multiplication protocol. Algorithm and physics regressions
remain under `renormalizer/*/tests`; they are not replaced by a candidate score.

## Kept boundaries

- `environment`: installed build, actual device, synchronized numerical smoke.
- `fixtures`, `scoring`, `protocol`: fixed reference and input/output contract.
  Backend tests reuse the scorer's error gates; the worker never imports it.
- `runner`, `worker`, `report`: execution, output collection and explicit outcomes.
- `policy`, `sandbox`: optional Linux isolation. Reviewed execution trusts code;
  unavailable isolation does not silently become permission to run unknown code.
- `rawkernel`, `gpu`, `gpu_worker`, `timing`: an optional source-pinned GPU control,
  **not arbitrary CUDA submission**. Timing is not a speedup or memory-peak proof.
- `legacy_inventory`: source-history audit helper; requires historical Git objects.

Keep these execution boundaries even when consolidating small utility code.
Optional CUDA/isolation dependencies are not in the default/dev package list.
If arbitrary model-generated programs become a supported product, its isolation,
resource management and service lifecycle should be designed separately rather
than silently expanding this small scientific-library helper.
