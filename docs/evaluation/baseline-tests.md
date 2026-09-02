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

### iecor — BDSKY replaced with BDMM-Prime

**Dependency unblocked.** The original `sampled-ancestors:2.3.0-beta1` resolution failure was an
offline-mode problem, not the version skew it first looked like: that artifact lives in the
GitHub Packages repo BDMM-Prime's pom declares (`https://maven.pkg.github.com/CompEvol/beast3`),
which `mvn -o` cannot reach. Fetched with network access once, `mvn clean install -DskipTests` in
`/Users/nneureiter/Projects/BEAST3/BDMM-Prime` (branch `beast2.8-migration`) then succeeded, and
both `io.github.compevol:sampled-ancestors:2.3.0-beta1` and `io.github.tgvaughan:bdmm-prime:3.0.0`
are now in the local m2; `mvn -o` resolves them from here on. Wired in: `bdmm-prime:3.0.0` as a
Maven dependency (`pom.xml`), `<depends on='BDMM-Prime'/>` in `version.xml` (matching BDMM-Prime's
own declared package name/capitalisation), not bundled into `lib/` (the assembly's `<includes>`
allow-list already only lists `targetedbeast`/`colt`/`commons-math3`).

**Tree-prior mapping.** `examples/iecor/{vanilla,pca}/iecor.xml`'s `birthDeathPrior` is now a
single-type `bdmmprime.distribution.BirthDeathMigrationDistribution` with
`bdmmprime.parameterization.EpiParameterization`, read from BDMM-Prime's own worked XMLs
(`validation/tip_date_sampling/infer_tip_dates.xml`, the `EpiParameterization` subtemplate in
`src/main/resources/bdmmprime/fxtemplates/BDMMPrimeParameterizations.xml`) and source
(`EpiParameterization.java`, `SkylineVectorParameter.java`/`SkylineParameter.java`,
`TimedParameter.java`, `ProcessLength.java`), not guessed:

- `origin`'s domain is tightened `Real` → `NonNegativeReal`: `bdmmprime.util.ProcessLength`'s
  `origin` input requires `RealScalar<? extends NonNegativeReal>`. `origin` was already
  lower-bounded at 0.0 in the pre-migration BDSKY XML, so this restores rather than adds a
  constraint.
- `reproductiveNumber`, `becomeUninfectiousRate`, `rho` are now dimension-1 `RealVectorParam`,
  not scalars. `Re`/`becomeUninfectiousRate` need `SkylineVectorParameter`, and `rhoSampling`
  needs `TimedParameter`; both take a `RealVector`-typed values input (`SkylineParameter.java:48`,
  `TimedParameter.java:66`), which a `RealScalarParam` cannot satisfy — confirmed by reading the
  actual `Input<>` declarations, not inferred. Their existing scalar priors (`LogNormal`,
  `Exponential`, `Beta`) are preserved by wrapping each in `beast.base.spec.inference.distribution.IID`
  (`param` = the vector, `distr` = the unchanged scalar distribution with its original
  hyperparameters) — the framework's own pattern for a scalar distribution applied to a
  vector-typed parameter (`IID.java`'s own comment: "when param is vector, dist is univariate,
  then apply dist to each dim").
- `removalProb` has no equivalent on the old model (its optional `r`/removal-probability was
  commented out, i.e. `r=1`), so it is a fixed, non-estimated `SkylineVectorParameter` at 1.0.
- `rhoSampling` is a `TimedParameter` with `values="@rho"` and a fixed, non-estimated
  `times="0.0"` (a single rho-sampling event at the present).

**`reproductiveNumber` type conflict — resolved by an authorized operator-set deviation.**
`reproductiveNumber` was also used by `ScaleAll`
(`spec="targetedbeast.operators.IntervalScaleOperator"`, `<up idref="reproductiveNumber"/>`),
whose `up`/`down` inputs are hard-typed `List<RealScalarParam<PositiveReal>>`
(`IntervalScaleOperator.java:37`) — scalar-only, and structurally incompatible with the
vector-typed `Re` it must also satisfy. Reported as BLOCKED with three options; the user chose:
drop `<up idref="reproductiveNumber"/>` from `ScaleAll` (both XMLs, with an explanatory comment at
the site) and let `reproductiveNumber` become the vector `Re` needs. This is an explicit,
authorized deviation from "keep the existing operator set" — iecor is in the corpus for its data
regime (binary cognate characters, 341 partitions, relaxed clock, birth-death prior) to benchmark
tree-operator mixing, and `ScaleAll` moving one fewer scalar parameter does not affect that
measurement. The two rejected options: decoupling `Re` from a frozen copy of `reproductiveNumber`
(the operator would move a value the likelihood never reads — a silent correctness bug), or
widening `IntervalScaleOperator`'s type (editing nicfel's third-party TargetedBeast operator code,
which is out of scope for "change only the tree-prior block").

**`samplingProportion` — simplified to a constant, not ported as a skyline.** The original had 3
epochs (`samplingRateChangeTimes.t:tree` = `[4.401, 0.9, 0.0]`, values `[0.0, 0.01, 0.0]`) with a
per-array `reverseTimeArrays[2]=true` flag on the old BDSKY model. BDMM-Prime's
`SkylineVectorParameter` uses a different convention (`nIntervals = nChangeTimes + 1`, raw
`changeTimes` values must be ascending) that does not translate 1:1 from BDSKY's without knowing
BDSKY's exact interval-to-value time direction — and BDSKY has no BEAST 3 port and no source
anywhere in this workspace to check against. A plausible reconstruction exists (the trailing `0.0`
self-fixes under the shared `ScaleOperator` and carries no prior in either `xInclude` mask, so it
reads as a fixed "present" marker rather than a free change time, leaving `[0.9, 4.401]` ascending
as the two real change times) but it is unverifiable, and a silently time-reversed model is worse
than a known simplification. Per the user's decision: `samplingProportion` is now a single
dimension-1 constant (`0.01`, its one nonzero epoch), `samplingRateChangeTimes.t:tree` and its two
`ExcludablePrior`-based change-time priors and its `ScaleOperator` are removed entirely (nothing
left for them to act on). iecor is in the corpus to benchmark tree-operator mixing on its data
regime, not to reproduce the published epidemiological analysis, so this simplification does not
compromise the evaluation's purpose — but it means iecor's tree-prior no longer matches the
original publication and that must not be forgotten if this XML is ever reused for anything else.

**`samplingProportionPrior` also needed its wrapper class changed, not just its mask.**
`beastlabs.math.distributions.ExcludablePrior` turns out to `extends beast.base.inference.distribution.Prior`
— legacy-typed (`x` is `Input<Function>`, `distr` is `Input<ParametricDistribution>`), predating the
`beast.base.spec` split, and it cannot bind to a typed `RealVectorParam` or a typed `Beta` at all
(this is *not* a niche edge case — the `xInclude` input is even still typed to the legacy
`BooleanParameter`, so the pre-existing migrated XML's `spec="...BoolVectorParam"` mask was already
mismatched before this task touched it, just never caught because iecor never previously parsed).
Since `ExcludablePrior`'s per-element masking existed only to prior one of 3 skyline epochs while
leaving the other 2 fixed, and there is now only 1 epoch, the masking has nothing left to do:
`samplingProportionPrior` is now a plain `IID(Beta(1,1))` on the single element, matching the
`R0`/`becomeUninfectiousRate`/`rho` pattern above, not a hand-migration of `ExcludablePrior` itself.

**Throughput measurement is still blocked — by a different, broader problem.** With the tree-prior
swap complete, a scratch run (`chainLength` shortened, `-overwrite`) parses cleanly through the
entire prior section (confirming the BDMM-Prime wiring above is correct) and well into the
likelihood section, before hitting a *second*, unrelated wall: `iecor.xml`'s substitution/clock
model blocks have their own pre-existing, previously-undiscovered typed-migration defects, e.g.
`beast.base.spec.inference.distribution.Gamma`'s scale input renamed `beta`→`theta` (found on the
`ucldStdev` prior), bare legacy `Function$Constant` elements passed to a typed `Gamma`'s
alpha/theta (the `bcov_s` prior), `BinaryCovarion`'s `switchRate` requiring a `RealVector` while
`alpha` requires a `RealScalar` (so only one of the pair needed converting), and a bare,
spec-less `<LogNormal name='distr'>` inside the relaxed-clock model (the same "legacy map tag,
now unresolvable since the file's `<map>` declarations were dropped during Phase 0b's migration"
failure mode noted there). These predate this task — they went undiscovered because iecor could
never previously be parsed at all (blocked on `bdsky`) — and they recur throughout a 1300+ line
file with several more partitions after the one just reached. Fixing them properly means
re-running the `beast3-xml-migration` converter and working through its per-file report, per the
workspace rule ("never hand-edit BEAST XML to migrate it"), not patching each one by hand as
found; that is a distinct task from "change only the tree-prior block" and was not attempted
beyond the isolated, disposable scratch-copy patches used to reach this diagnosis (not applied to
the committed corpus). **iecor's states/sec and site-pattern count remain unmeasured.** The tier
table below reflects only the two datasets that currently run (infB, lassaL@120); iecor stays
out until this second migration pass happens.

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
- iecor: still unmeasured — BDMM-Prime now builds and the tree-prior swap is done and verified as
  far as parsing goes, but a *different*, pre-existing set of typed-migration defects elsewhere in
  the file (substitution/clock models, unrelated to the tree prior) blocks an actual run; see
  above. The plan's only prior calibration point, `examples/iecor/vanilla/IECOR_untargeted.log`,
  is a trace log with no wall-clock metadata either way.

**Decision: lassaL's full 551-taxon alignment is not part of the pipeline at all** — not just at
the `full` tier. Every tier uses the lassaL@120 subsample (or smaller, at `fast`, once Phase 0
step 4 picks a size) instead. The full-taxon analysis is a user-initiated cluster job, not
something any tier runs.

**Full-tier arithmetic, from measured throughput, under the 10-hour rule.** Budget: 10h =
36,000s. Split evenly across the two datasets that currently run end-to-end (infB, lassaL@120);
iecor is excluded because it still cannot be run at all (see above) — its cost is unknown, so it
cannot be budgeted yet. Each dataset gets ≈3h of margin below its half-budget, held back because the
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

Once iecor actually runs and its throughput is measured (its BDMM-Prime tree-prior swap no
longer blocks this; a second, unrelated migration-defect pass through its substitution/clock
models does), its `full`/`medium` chain lengths must be set the same way (its own time budget
carved out of, or added on top of, the same 10h ceiling) — this doc's numbers for infB/lassaL@120
may need to shrink to make room, or iecor may need to run as a separate overnight job. Per the
instruction not to silently cut reps or settings: **this doc does not shrink infB/lassaL@120's
reps or settings to make room for an iecor number that does not exist yet** — their current
chains (1,000,000 / 150,000, ≈6.91h total) stand unchanged until iecor is actually measured. This
is the same tier table now recorded in `targetedbeast-evaluation-pipeline.md`'s Phase 3 tier
table.
