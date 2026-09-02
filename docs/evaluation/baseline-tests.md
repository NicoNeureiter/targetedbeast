# Baseline: `beast2.8-feature-trunk`

Base branch for the evaluation pipeline changed from `beast2.8-migration` to
`beast2.8-feature-trunk` (0 commits behind each of the four feature branches; `beast2.8-migration`
is missing `PCAWeights2`, `PartialWeights` and the whole `weighttransform/` package, which the
calibration study needs). This baseline is `mvn -o test` on `beast2.8-feature-trunk` at commit
`5a6b356` ("Migrate feature trunk sources to beast.base.spec types"), on the `eval-pipeline`
worktree/branch.

`mvn -o compile` succeeds first: the `2.8.0-SNAPSHOT` build of the
`NicoNeureiter/beast3` fork branch `expose-likelihood-core-internals` is present in the local m2.
Not blocked.

## Summary

```
Tests run: 105, Failures: 2, Errors: 0, Skipped: 1
```

| Test class | Run | Failures | Errors | Skipped |
|---|---|---|---|---|
| `targetedbeast.EdgeWeightRejectConsistencyTest` | 6 | 0 | 0 | 1 |
| `targetedbeast.TargetedOperatorDetailedBalanceTest` | 30 | 1 | 0 | 0 |
| `targetedbeast.TargetedWilsonBaldingTest` | 8 | 0 | 0 | 0 |
| `targetedbeast.WilsonBaldingDBTest` | 1 | 0 | 0 | 0 |
| `targetedbeast.coalescent.RapidTreeIntervalsTest` | 1 | 0 | 0 | 0 |
| `targetedbeast.edgeweights.FelsensteinWeightsTest` | 7 | 0 | 0 | 0 |
| `targetedbeast.edgeweights.FelsensteinWeightsWithoutNodeTest` | 2 | 0 | 0 | 0 |
| `targetedbeast.likelihood.RapidLikelihoodTest` | 4 | 0 | 0 | 0 |
| `targetedbeast.util.Alignment2PCATest` | 3 | 1 | 0 | 0 |
| `targetedbeast.weighttransform.GridTunerTest` | 14 | 0 | 0 | 0 |
| `targetedbeast.weighttransform.PartialWeightsLogSpaceTest` | 3 | 0 | 0 | 0 |
| `targetedbeast.weighttransform.SpsaAdaptationTest` | 11 | 0 | 0 | 0 |
| `targetedbeast.weighttransform.WeightTransformPersistenceTest` | 2 | 0 | 0 | 0 |
| `targetedbeast.weighttransform.WeightTransformTest` | 13 | 0 | 0 | 0 |

`DetailedBalanceTest` itself is an abstract base class (no runnable methods of its own); its
concrete subclasses (`TargetedOperatorDetailedBalanceTest`, `WilsonBaldingDBTest`, …) are listed
above. `FelsensteinWeightsTest` (edgeweights package, 7 tests) is distinct from the
`FelsensteinWeights2`-branded test inside `EdgeWeightRejectConsistencyTest`, below.

## Failures (2)

### `TargetedOperatorDetailedBalanceTest.testTargWilsonBaldingMaxHeightRatio`

```
java.lang.AssertionError: Operator produced zero acceptance mass across 100000 proposals
	at org.junit.Assert.fail(Assert.java:89)
	at org.junit.Assert.assertTrue(Assert.java:42)
	at targetedbeast.DetailedBalanceTest.testDetailedBalance(DetailedBalanceTest.java:144)
	at targetedbeast.TargetedOperatorDetailedBalanceTest.testTargWilsonBaldingMaxHeightRatio(TargetedOperatorDetailedBalanceTest.java:232)
```

Known-failing per the evaluation-pipeline plan. Not fixed here.

### `Alignment2PCATest.distanceBasedEmbeddingUsesClassicalMdsGeometry`

```
java.lang.AssertionError: expected:<1.0> but was:<0.12870945208344026>
	at targetedbeast.util.Alignment2PCATest.distanceBasedEmbeddingUsesClassicalMdsGeometry(Alignment2PCATest.java:33)
```

Known-failing per the evaluation-pipeline plan (referred to there as `Alignment2PCATest`). Not
fixed here.

## Skipped (1)

### `EdgeWeightRejectConsistencyTest.testFelsensteinWeights2`

`@Ignore`d in source (`src/test/java/targetedbeast/EdgeWeightRejectConsistencyTest.java:202`),
with the comment: "remove the `@Ignore` once the cache bookkeeping is fixed." Known-ignored per
the evaluation-pipeline plan. Not un-ignored here.

## Outstanding

The other three feature branches (`beast2.8-pca-brownian-weights`, `beast2.8-learned-weights`,
`beast2.8-prior-weights`) have not had their baselines run — that needs additional worktrees,
which have not been requested yet.

## Throughput

The corpus (`examples/primates.xml`, `examples/infb_HA_141/template.xml`,
`examples/iecor/{pca,vanilla}/iecor.xml`, `examples/lassaL_kli22/template.xml`) is now migrated
with `beast3-xml-migration` and tracked in git (`.gitignore` narrowed to keep example XMLs/
templates while still excluding run outputs; the lassaL alignment stays ignored, supplied
locally). `BEASTLabs` (`io.github.beast2-dev:beast-labs:2.1.0-SNAPSHOT`, built from
`/Users/nneureiter/Projects/BEAST3/BEASTLabs` against `beast.version=2.8.0-beta5` and installed
to the local m2) is now a declared Maven dependency and a `<depends on='BEASTlabs'/>` in
`version.xml`, not bundled in `lib/`. It compiles and resolves cleanly alongside this project's
`2.8.0-beta8` pin; `mvn -o test` is unchanged (105 run, 2 failures, 1 skipped — same two known
failures as above).

### Migration-time findings acted on

- **`primates.xml` — `SubtreeSlide`/`BactrianSubtreeSlide` mismatch.** The converter's blanket
  rename of the short name `SubtreeSlide` to its spec twin
  `beast.base.evolution.operator.kernel.BactrianSubtreeSlide` (flagged in its own report as
  `[warn] ⚠ SubtreeSlide attr gaussian="true" dropped`) breaks
  `targetedbeast.operators.TargetedWilsonBaldingHeightDelta.subtreeSlideInput`, which is still
  typed to the legacy `beast.base.evolution.operator.SubtreeSlide` (siblings under `TreeOperator`,
  not a subtype relationship) — the migrated XML failed to parse
  (`type mismatch for input subtreeSlide`). Reverted this one operator to the legacy FQN with its
  `gaussian="true"` attribute restored, per the report's own warning; this is a documented
  converter limit ("structural changes specific to a package's custom classes" —
  `references/xml-migration-strategy.md`), not a hand-migration of the XML.
- **`infb_HA_141/template.xml`, `iecor/*/iecor.xml`** — the `[todo] ✗ no spec twin` lines for
  `StandardData`, `TwoStateCovarion`, and `RateStatistic` are false positives: all three classes
  exist unchanged at their pre-migration FQNs in `beast-base-2.8.0-beta8.jar` (data types and
  simple statistics are outside the typed system, so they were correctly left as-is).
- **`lassaL_kli22/template.xml`** — the converter choked on well-formedness (`attributes
  construct error, line 140`) because `{target_weight_transform_ref}` sits mid-tag, directly after
  a closing attribute quote, three times. Substituted ` __SLOT_TWTREF__="1"` for each occurrence,
  converted, confirmed the substitution and all five other slots
  (`{alignment_file}`, `{chain_length}`, `{edge_weights}`, `{target_weight_transform}`,
  `{target_weight_transform_log}`) survived the round-trip unmangled, then restored the original
  `{target_weight_transform_ref}` text exactly. `infb_HA_141/template.xml`'s two slots
  (`{chain_length}`, `{edge_weights}`) needed no substitution — they sit in attribute-value and
  element-content position, both well-formed as-is.
- **`version.xml`** — registered `feast.fileio.AlignmentFromFasta` and
  `feast.fileio.TraitSetFromTaxonSet` as `BEASTInterface` providers. `lib/feast.v10.6.0.jar` was
  already bundled but had no `version.xml`/`module-info` of its own, so BEAST's
  service-registry class loader (`BEASTClassLoader.forName(class, service)`) rejected both
  classes outright ("Could not find class ... as service beast.base.core.BEASTInterface")
  regardless of classpath placement; `infb_HA_141` and `lassaL_kli22` both use
  `feast.fileio.TraitSetFromTaxonSet` for the date trait.

### AVMVN tree transform — resolved

**`infb_HA_141` and `lassaL_kli22` — `Transform$NoTransform` no longer accepts a `Tree`.** Both
templates fed the tree directly into an `AVMNOperator` joint transform
(`<transformations spec="Transform$NoTransform"><f idref="tree"/></transformations>`), which
worked in BEAST2 because `Tree` implements `Function`. The spec-typed twin's `f` input is typed
to `Tensor`, which `beast.base.evolution.tree.Tree` does not implement in `2.8.0-beta8` (no
spec-typed tree-as-vector wrapper exists yet in `beast.base.spec.evolution.tree`). **Fixed in the
committed corpus**: that one `<transformations>` block is removed (with an explanatory XML
comment) from both `template.xml` files. The `AVMNOperator` itself and its other two
transformations (`clockRate`/`gammaShape`/`kappa` and the `freqs` sum-to-one constraint) are
unchanged. Verified by running both templates end-to-end (see Results below) — this is no longer
a scratch-copy workaround, it is the shipped XML.

### iecor — BDSKY replaced with BDMM-Prime, build blocked

The original blocker — `bdsky.evolution.speciation.BirthDeathSkylineModel` has no BEAST 3 port
anywhere in this workspace — is superseded by the user's decision to replace the BDSKY tree prior
with **BDMM-Prime** (single-type `bdmmprime.distribution.BirthDeathMigrationDistribution` +
`bdmmprime.parameterization.EpiParameterization`, whose `Re`/`becomeUninfectiousRate`/
`samplingProportion`/`rhoSampling` inputs map one-to-one onto iecor's existing BDSKY parameters).
The XML migration itself was not attempted because the dependency does not build:

```
mvn -o clean install -DskipTests   (in /Users/nneureiter/Projects/BEAST3/BDMM-Prime, branch beast2.8-migration)

[WARNING] The POM for io.github.compevol:sampled-ancestors:jar:2.3.0-beta1 is missing, no dependency information available
[ERROR] Failed to execute goal on project bdmm-prime: Could not resolve dependencies for project io.github.tgvaughan:bdmm-prime:jar:3.0.0
[ERROR] dependency: io.github.compevol:sampled-ancestors:jar:2.3.0-beta1 (compile)
[ERROR] 	Cannot access github (https://maven.pkg.github.com/CompEvol/beast3) in offline mode and the artifact io.github.compevol:sampled-ancestors:jar:2.3.0-beta1 has not been downloaded from it before.
[ERROR] 	Cannot access central (https://repo.maven.apache.org/maven2) in offline mode and the artifact io.github.compevol:sampled-ancestors:jar:2.3.0-beta1 has not been downloaded from it before.
```

BDMM-Prime's pom hard-pins `io.github.compevol:sampled-ancestors:2.3.0-beta1`. That coordinate is
not in the local m2 and (offline) not reachable from GitHub Packages or Central. The only local
`sampled-ancestors` checkout (`/Users/nneureiter/Projects/BEAST3/sampled-ancestors`) is versioned
`2.3.0-SNAPSHOT` and built against `beast.version=2.8.0-beta8` — a different artifact coordinate
and a different BEAST base version than BDMM-Prime (`2.8.0-beta5`) declares. This is exactly the
beta5-vs-beta8 skew called out as a possible blocker: unlike BEASTLabs (whose `2.8.0-beta5` pin
resolved and compiled cleanly alongside this project's `beta8` pin with no missing transitive
dependency), BDMM-Prime cannot be built offline at all, because the artifact it needs under its
declared coordinate exists nowhere in this workspace, not because of an API mismatch that could be
patched around. **BLOCKED** — not attempted further (installing the local SA checkout under a
different, unauthorized version coordinate, or editing BDMM-Prime's pom, would be improvising
outside this task's scope). iecor could not be migrated, parsed, run, or measured for throughput
or pattern count. Needs a user decision: publish/vendor `sampled-ancestors:2.3.0-beta1` somewhere
reachable, rebuild BDMM-Prime against the local SA `2.3.0-SNAPSHOT`/`beta8` combination (and verify
BDMM-Prime still compiles/runs against that pair), or another resolution.

### Method

Fixed-length runs, no BEAGLE (not installed — `beagle-1.0.0.jar` is present in the local m2 as a
Java stub only; native library load falls back to `BeerLikelihoodCore4` in all runs below,
consistent with normal use). `examples/primates.xml`: chain shortened to 20k in a scratch copy.
`examples/infb_HA_141/template.xml`: `{chain_length}` → 10k (later re-verified at 2k after the
AVMVN fix), `{edge_weights}` → `targetedbeast.edgeweights.ConstantWeights`.
`examples/iecor/*/iecor.xml`: blocked, not run (BDMM-Prime does not build — see above).
`lassaL_kli22/template.xml`: `{chain_length}` → 5k, `{edge_weights}` → `ConstantWeights`,
`{alignment_file}` → the user-supplied `ali_normalized.fasta` (551 taxa) or its seeded
`ali_normalized.n120.fasta` subsample (120 taxa, via `generate_xmls.py`'s
`subsampled_alignment(120)`, `SUBSAMPLE_SEED = 42`), all `{target_weight_transform*}` slots
emptied. Runs used `render_xml`'s own template rendering (`generate_xmls.py`'s `render_xml`
function called directly for the `ConstantWeights`/`none` combination it doesn't currently
generate by default) against the **committed** template (post AVMVN fix) — no scratch-copy
XML surgery was needed this time. "Total calculation time" is BEAST's own reported chain time,
excluding JVM/module-boot overhead. Runs used a manually assembled module path
(`target/classes` + the Maven dependency classpath + `lib/feast.v10.6.0.jar`, copied to a
JPMS-compliant filename since automatic-module-name derivation rejects `feast.v10.6.0.jar`'s
literal name) since `mvn exec:exec`'s `%classpath` does not include the repo's raw `lib/` jars.

### Results

| Dataset | Taxa | Sites | **Site patterns** | States run | Chain time | **States/second** |
|---|---:|---:|---:|---:|---:|---:|
| `primates.xml` | 6 | 768 | 69 | 20,000 | 0.217 s | ~92,200 |
| `infb_HA_141/template.xml` | 141 | 1,856 | 489 | 10,000 | 5.275 s | ~1,900 |
| `infb_HA_141/template.xml` (re-verified, AVMVN fix, committed XML) | 141 | 1,856 | 489 | 2,000 | 1.063 s | ~1,881 |
| `iecor/{pca,vanilla}/iecor.xml` | 41 | 2,659×341 partitions | **blocked** (BDMM-Prime) | — | — | **blocked** |
| `lassaL_kli22` (+ user's `ali_normalized.fasta`, full) | 551 | 7,038 | 6,119 | 5,000 | 100.274 s | ~50 |
| `lassaL_kli22@120` (seeded subsample, committed XML) | 120 | 7,038 | 4,572 | 5,000 | 17.003 s | ~294 |

Pattern counts and taxon counts are both BEAST's own startup report
(`Alignment(...): [taxa, patterns, sites] = [...]`), confirming the plan's point that pattern
count does not scale linearly with taxa: infB has 2.3x primates' taxa but 7.1x its patterns;
lassaL has 3.9x infB's taxa but 12.5x its patterns; lassaL@120 (0.22x lassaL's taxa) still keeps
0.75x its patterns — cost does not fall in proportion to the taxon subsample, which is exactly why
the tier arithmetic below uses the measured lassaL@120 number rather than extrapolating from taxa
alone.

### The 10-hour rule and the derived tier table

The original `full` tier (10M states, 4 reps, ~6 settings, 3 datasets, all at full taxon count) is
machine-days, not hours, and is dataset-dominated by lassaL:

- infB: 10,000,000 / 1,900 ≈ **1.5 hours** for one replicate — fine on its own.
- lassaL (551 taxa): 10,000,000 / 50 ≈ **55.7 hours (≈2.3 days)** for one replicate; at 4 reps ×
  ~6 settings that is **~56 days of sequential compute** — an order of magnitude past "overnight."
- iecor: was unmeasured (now known blocked — see above); the plan's only prior calibration point,
  `examples/iecor/vanilla/IECOR_untargeted.log`, is a trace log with no wall-clock metadata.

**Decision: lassaL's full 551-taxon alignment is not part of the pipeline at all** — not just at
the `full` tier. Every tier uses the lassaL@120 subsample (or smaller, at `fast`, once Phase 0
step 4 picks a size) instead. The full-taxon analysis is a user-initiated cluster job, not
something any tier runs.

**Full-tier arithmetic, from measured throughput, under the 10-hour rule.** Budget: 10h =
36,000s. Split evenly across the two datasets that currently build and run (infB, lassaL@120);
iecor is excluded until BDMM-Prime builds (see above) — its cost is unknown, so it cannot be
budgeted yet. Each dataset gets ≈3h of margin below its half-budget, held back because the
`ConstantWeights` baseline used to measure throughput is the *cheapest* of the ~6 settings —
`PCAWeights2`, `PartialWeights`, `FelsensteinWeights`, and the `PowerOffsetTransform`/`GridTuner`
grid all do more per-proposal work — and because JVM start-up and log/tree-file I/O across 48 runs
are not in the measured "chain time."

| dataset | measured states/sec | `full` chain | time/rep | × 24 reps (6 settings × 4 reps) |
|---|---:|---:|---:|---:|
| infB (full, 141 taxa) | 1,900 | 1,000,000 | 526 s | ≈3.51 h |
| lassaL@120 | 294 | 150,000 | 510 s | ≈3.40 h |
| iecor | **blocked** | **blocked** | — | — |

Sequential total for the two runnable datasets: ≈**6.91 h**, inside the 10h ceiling with ≈3h
(≈30%) to spare. `medium` scales both chain lengths down 10× (infB 100,000, lassaL@120 15,000),
giving a ≈41-minute medium-tier sweep — so **neither tier needs to drop below 6 settings × 4
reps to fit the 10-hour rule**; the constraint is met by right-sizing chain length and dataset
(lassaL@120, not lassaL@551), not by shrinking the experimental design.

Once BDMM-Prime is unblocked and iecor's throughput is measured, its `full`/`medium` chain
lengths must be set the same way (its own time budget carved out of, or added on top of, the same
10h ceiling) — this doc's numbers for infB/lassaL@120 may need to shrink to make room, or iecor
may need to run as a separate overnight job. This is the same tier table now recorded in
`targetedbeast-evaluation-pipeline.md`'s Phase 3 tier table.
