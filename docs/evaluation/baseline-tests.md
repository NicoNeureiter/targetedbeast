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

### New blockers found (outside this task's authorized scope)

- **`iecor/{pca,vanilla}/iecor.xml` — `bdsky.evolution.speciation.BirthDeathSkylineModel` is not
  available anywhere** (not in any local BEAST3 checkout, not in the local m2). This is a
  different missing package than the pre-identified `BEASTLabs`/`SimpleRandomTree` issue — no
  BEAST3 port of BDSKY exists in this workspace. **iecor could not be parsed, run, or measured for
  throughput or pattern count.** This needs a user decision (port BDSKY, find/build an existing
  BEAST3 build of it, or substitute a different tree prior in the example) before iecor is usable;
  not attempted here since only a `BEASTlabs` dependency was authorized.
- **`infb_HA_141` and `lassaL_kli22` — `Transform$NoTransform` no longer accepts a `Tree`.** Both
  templates feed the tree directly into an `AVMNOperator` joint transform
  (`<transformations spec="Transform$NoTransform"><f idref="tree"/></transformations>`), which
  worked in BEAST2 because `Tree` implements `Function`. The spec-typed twin's `f` input is typed
  to `Tensor`, which `beast.base.evolution.tree.Tree` does not implement in `2.8.0-beta8` (no
  spec-typed tree-as-vector wrapper exists yet in `beast.base.spec.evolution.tree`). This blocks
  both templates from parsing as committed. **For the throughput measurement below only**, this one
  `<transformations>` block was dropped from a disposable scratch copy of each XML (not the
  committed corpus) — the operator still has two other valid transforms (frequencies,
  clock/shape/kappa), so the run is otherwise unmodified. This is a core-level typed-system gap,
  not something fixable via XML or `version.xml` configuration.

### Method

Fixed-length runs, no BEAGLE (not installed — `beagle-1.0.0.jar` is present in the local m2 as a
Java stub only; native library load falls back to `BeerLikelihoodCore4` in all runs below,
consistent with normal use). `examples/primates.xml`: chain shortened to 20k in a scratch copy.
`examples/infb_HA_141/template.xml`: `{chain_length}` → 10k, `{edge_weights}` →
`targetedbeast.edgeweights.ConstantWeights`. `examples/iecor/*/iecor.xml`: chain shortened to 5k
(blocked before running — see above). `lassaL_kli22/template.xml`: `{chain_length}` → 5k,
`{edge_weights}` → `ConstantWeights`, `{alignment_file}` → the user-supplied
`ali_normalized.fasta` (present in the user's checkout, copied read-only into this worktree,
untracked per `.gitignore`), all `{target_weight_transform*}` slots emptied (no transform
declared — `RangeSlide`/`WeightedWideOperator`/`TargetedWilsonBalding` all still take their
untransformed defaults). "Total calculation time" is BEAST's own reported chain time, excluding
JVM/module-boot overhead.

### Results

| Dataset | Taxa | Sites | **Site patterns** | States run | Chain time | **States/second** |
|---|---:|---:|---:|---:|---:|---:|
| `primates.xml` | 6 | 768 | 69 | 20,000 | 0.217 s | ~92,200 |
| `infb_HA_141/template.xml` | 141 | 1,856 | 489 | 10,000 | 5.275 s | ~1,900 |
| `iecor/{pca,vanilla}/iecor.xml` | 41 | 2,659×341 partitions | **blocked** (BDSKY) | — | — | **blocked** |
| `lassaL_kli22` (+ user's `ali_normalized.fasta`) | 551 | 7,038 | 6,119 | 5,000 | 100.274 s | ~50 |

Pattern counts and taxon counts are both BEAST's own startup report
(`Alignment(...): [taxa, patterns, sites] = [...]`), confirming the plan's point that pattern
count does not scale linearly with taxa: infB has 2.3x primates' taxa but 7.1x its patterns;
lassaL has 3.9x infB's taxa but 12.5x its patterns.

### Tier wall-time estimates this contradicts

The plan's `full` tier (10M states, 4 reps, ~6 settings, 3 datasets) is described as "machine-days,
not hours." At the measured per-dataset throughput, a **single** 10M-state replicate would take:

- infB: 10,000,000 / 1,896 ≈ **1.5 hours** — not "machine-days" on its own.
- lassaL: 10,000,000 / 50 ≈ **55.7 hours (≈2.3 days)** — confirms and sharpens the plan's warning;
  at 4 reps × ~6 settings this dataset alone is **~56 days of sequential compute**, an order of
  magnitude past "overnight-to-multi-day" unless run with substantial parallelism
  (`--config parallel_subdirs=1`) or a smaller `full`-tier chain length for this dataset
  specifically.
- iecor: unmeasured (blocked); the plan's only calibration point,
  `examples/iecor/vanilla/IECOR_untargeted.log`, is a trace log with no wall-clock metadata, so it
  could not be used to estimate iecor throughput independently of a working run.

The `full`-tier estimate is dataset-dominated by lassaL, not uniform across datasets as the plan's
single "machine-days" framing implies — infB is cheap enough to run overnight by itself, while
lassaL is not, at these settings.
