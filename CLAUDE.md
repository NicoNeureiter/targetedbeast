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

`scripts/Snakefile` → `scripts/generate_xmls.py` → `examples/lassaL_kli22/template.xml`.
Configuration belongs in the template as literal XML; the generator only fills declared slots.
Do not have Python emit whole operator elements.
