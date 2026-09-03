# TargetedBeast

BEAST 2 package: targeted tree operators that propose from edge weights rather than uniformly.

## Style

**Be concise.** This applies to chat replies, code comments, javadoc and BEAST `Input`
descriptions alike. State the fact and stop. Explain *why* only where the reason is not obvious
from the code, and in one sentence when you do. No restating what the next line already says, no
recapping decisions already made, no summaries of work just shown in a diff.

## Build

- `ant compile`, `ant build` (produces `dist/TargetedBeast.v1.0.0.zip`). There is no `addon`
  target.
- `ant test-single -Dtest.class=<FQCN>` for one test class.
- Installed package lives at
  `/Users/nneureiter/Library/Application Support/BEAST/2.7/TargetedBeast`; unzip the dist there
  after a build or BEAST runs stale code.
- LSP diagnostics in this repo cannot resolve `beast.base` at all and are noise. ant is the
  source of truth.

## Benchmark

`scripts/Snakefile` drives a dataset registry (`scripts/datasets.yaml`: `primates`, `infb`,
`iecor` — `lassaL` is parked, its ESS never reaches usable levels at any affordable chain length,
see `docs/evaluation/where-targeting-wins.md`). `scripts/render_xml.py` dispatches to each
dataset's renderer script. Configuration belongs in the template as literal XML; the renderer only
fills declared slots — do not have Python emit whole operator elements.

Tiers (see `docs/evaluation/where-targeting-wins.md` and `docs/evaluation/baseline-tests.md`):
- `smoke` (primates, seconds) — crashes, sane acceptance, no statistical claim.
- `fast` (infB, ~90s/setting) — dev-loop directional signal only, **not decision-grade** (ESS ~13–80).
- `thorough` (infB, ~61 min for a 2-setting × 4-rep comparison, ESS-gated ≥100) — the tier the
  decision rule actually gates on. infB's confirmed baseline: `PCAWeights2` beats `ConstantWeights`
  by +7.3% chain-level ESJD/second (90% CI excludes zero) — every feature branch's weighting is
  expected to beat this.
- `cross-check` (iecor, on request only, never automatic) — confirms a `thorough` result
  generalizes to a structurally different dataset before it's trusted broadly.

`targetedbeast.benchmark.BenchmarkMCMC` (`spec="targetedbeast.benchmark.BenchmarkMCMC"` in place
of `spec="MCMC"`) runs the instrumented chain and writes `<statsOut>.opstats.tsv`
(chain-level `__CHAIN__` row = the headline ESJD/second, per-operator rows are diagnostics only)
and, when `referenceLowerBound`/`referenceUpperBound` are set, `<statsOut>.burnin.tsv`
(proposals-to-target burn-in speed, separate from steady-state mixing efficiency).
