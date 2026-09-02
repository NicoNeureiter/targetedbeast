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

### iecor — substitution/clock migration defects fixed; one new blocker found

With the tree-prior swap complete, the substitution/clock defects the previous session diagnosed
were fixed by re-running `beast3-xml-migration` per the workspace rule ("never hand-edit BEAST
XML to migrate it"), not by hand-patching each one. Two runs were needed to see the full picture:
running the converter on the **committed** (already partially migrated) `iecor.xml` reports only
16 changes — mostly `[warn]` confirmations that already-typed `ScaleOperator`/`Uniform` usages are
fine — because most of the legacy short-name classes were already renamed. Running it fresh on the
**true, untouched BEAST2 original** (`/Users/nneureiter/Projects/BEAST3/targetedbeast/examples/iecor/{pca,vanilla}/iecor.xml`,
`version="2.7"`, `<map>` declarations intact — never committed to this repo) reports the full
108-change/25-warning/14-todo picture, confirming every domain-narrowing `[todo]` already present
in the committed file (`bcov_alpha.s:five`, `bcov_s.s:five`, `ucldMean.c:clock`, `ucldStdev.c:clock`,
`reproductiveNumber`, and the R0/becomeUninfectiousRate/samplingProportion/rho prior parameters)
matches what the committed file already had, and that the `StandardData`/`TwoStateCovarion`/
`RateStatistic` `[todo]`s are the same false positives already documented above (no spec twin
needed — both classes exist unchanged in `beast-base-2.8.0-beta8.jar`).

**Neither converter run flags the four defects the previous session diagnosed** — confirmed by
inspecting both reports line-by-line. This is a real gap in the converter's rule set for this
file, not a report I overlooked: `Function$Constant` isn't in `deprecated_classes.md` (it isn't a
deprecated class, just one that fails a typed `Input`'s generic bound, so `resolve_spec()` never
even considers it); the bare `<LogNormal>` tag is a `<map name="LogNormal">` alias usage the
converter strips without rewriting the *usage site* (it only migrates `spec="ShortName"`
attributes, not tag-name-as-class references — the skill's own docs anticipate this: "verify no
usage still leans on a stripped alias ... If one remains, set its `spec=` by hand"); and neither
run's report distinguishes `RealScalar` from `RealVector` shape, only the `PositiveReal`/`Real`
*domain* — so `BinaryCovarion.switchRate` needing a vector while `.alpha` needs a scalar is
invisible to it. Each was verified against the actual compiled class (`javap`/decompiled sources
from `beast-base-2.8.0-beta6-sources.jar`, since beta8 ships no sources jar) before fixing, per
the task's "verify against the actual current file and the actual BEAST 3 class definitions"
instruction:

- **`Gamma`'s `beta` → `theta`, not `lambda`.** `beast.base.spec.inference.distribution.Gamma` has
  *two* mutually-exclusive scale/rate inputs: `theta` (shape-scale form) and an input *named*
  `"lambda"` on the Java field called `betaInput` (shape-rate form, `Input.Validate.XOR` against
  `theta`). Legacy `beast.base.inference.distribution.Gamma`'s `beta` input defaults to
  `mode=ShapeScale` (confirmed by reading its source) — i.e. legacy `beta` *is* scale, which is
  `theta` in the typed twin, not the rate-named `lambda` despite the surface-level name match.
  Fixed on `ucldStdevPrior`'s `Gamma.204` (renamed `name="beta"` → `name="theta"`, and widened its
  domain from `Real` to `PositiveReal` since `thetaInput` requires `RealScalar<PositiveReal>`, not
  just any real).
- **Bare `Function$Constant` → typed `RealScalarParam`.** `bcov_s_prior`'s `Gamma.0` fed
  `<alpha spec="Function$Constant" value="0.05"/>` / `<beta spec="Function$Constant" .../>`
  directly — `Function$Constant` implements the legacy `Function` interface, not
  `RealScalar<PositiveReal>`, so it cannot satisfy `Gamma`'s typed `alpha`/`theta` inputs. Replaced
  both with fixed, non-estimated `RealScalarParam`s (matching the file's own established pattern
  for fixed hyperparameters elsewhere), renaming `beta`→`theta` as above.
- **`BinaryCovarion.switchRate` needs `RealVector<PositiveReal>`; `.alpha` needs `RealScalar<PositiveReal>`.**
  Confirmed by reading `BinaryCovarion.java`'s `Input` declarations directly
  (`alphaInput = new Input<RealScalar<PositiveReal>>(...)`,
  `switchRateInput = new Input<RealVector<PositiveReal>>(...)`) — this is the exact vector/scalar
  split the task described, and it is real, not a false positive like the three `[todo]`s above.
  `bcov_s.s:five` is now a dimension-1 `RealVectorParam` (domain unchanged, `PositiveReal`); its
  prior (`Gamma.0`, a `ScalarDistribution`) is wrapped in `IID`, matching the same
  `IID`-wrapped-scalar-prior-on-a-vector-parameter pattern already established for
  `reproductiveNumber`/`becomeUninfectiousRate`/`rho`/`samplingProportion`. `bcov_alpha.s:five`
  needed no change (already a correctly-typed scalar).
- **Bare `<LogNormal name='distr'>` resolves to the untyped legacy class, not the typed twin.**
  `UCRelaxedClockModel`'s `distr` input requires
  `ScalarDistribution<RealScalar<? extends NonNegativeReal>, Double>` (the typed base class).
  Without its stripped `<map name="LogNormal">beast.base.inference.distribution.LogNormalDistributionModel</map>`
  alias, the bare `<LogNormal>` tag falls back to namespace-scan resolution and hits
  `beast.base.inference.distribution.LogNormal` — a `@Deprecated` one-line subclass of
  `LogNormalDistributionModel` kept only for this legacy short-name convenience, which does **not**
  extend the typed `ScalarDistribution` base and so cannot satisfy `distr`. Fixed by adding
  `spec="beast.base.spec.inference.distribution.LogNormal"` explicitly, per the skill's own
  documented handling for a stripped map alias with no inline resolution left.

None of these four needed the 341-partition/171-`TreeLikelihood` structure the plan's dataset
description mentions — **that entire structure turns out to be commented out** (a single
`<!-- ... -->` block spanning ~1,050 lines, confirmed with `lxml`: it parses only 215 elements
from the file, not the ~2,000+ the raw byte count would suggest, because everything inside the
comment is opaque text to the XML parser). The only *live* partition is `focalsubset`
(41 taxa, 2,653 sites, 1,071 patterns, one `TreeLikelihood`) — the four defects above live in that
one live partition's substitution/clock blocks and needed fixing exactly once per file, not
171 times. **This corrects the dataset description in the pipeline plan and in Phase 0's dataset
list**: iecor's live/runnable content is a single-partition analysis, not 171 `TreeLikelihood`s
over 341 partitions — the throughput and pattern-count numbers below reflect what actually runs.

**A fifth, unrelated defect was found and left BLOCKED — needs your input, not guessed at.**
`almostUniformOrigin`, live in the prior section (not commented out), is
`spec="almostbeast.math.distributions.AlmostUniform"` on the `origin` parameter. `almostbeast` has
no BEAST 3 port, no jar or source anywhere in this workspace, and — unlike `bdsky` — is not even
in `deprecated_classes.md`, so the converter has no way to flag it at all (a `[todo]` requires the
class to at least be a *recognized* deprecated one). It cannot simply be replaced with a literal
`Uniform(0, 10)`: the pre-migration `origin` stateNode had **hard bounds `[0.0, 100.0]` with an
initial value of `100.0`** — right at the *old* upper bound and an order of magnitude past
`AlmostUniform`'s own `upper="10.0"`. A hard-bounded `Uniform(0,10)` prior is inconsistent with the
model's own initial state (confirmed empirically: substituting it makes BEAST3 throw
`IllegalArgumentException: Tensor param is not valid ! origin: 100.0` at `initAndValidate`).
`AlmostUniform` must therefore be some kind of soft, non-truncating density — flat-ish near
`[0,10]` but still positive out past `100` — and its exact form cannot be reconstructed from
anything available here. Deciding what to substitute is a modelling judgement call, not a syntax
migration, so — per the task's own "never change requirements, STOP and report BLOCKED"
instruction — **the committed XML still has the original, unresolvable `AlmostUniform` element**,
now with an explanatory comment, rather than a guessed replacement. This is the one item in this
pass that is genuinely unresolved.

To get the throughput and pattern-count numbers below despite this, a **disposable, uncommitted**
scratch copy substitutes `beast.base.spec.inference.distribution.Uniform(param="@origin", upper="1.0E6")`
for `almostUniformOrigin` — a wide-enough-to-be-harmless placeholder (matching the same
`Infinity`→`1.0E6` finite-bound convention the converter itself already uses elsewhere in this
file for `bcov_alpha`'s prior), used only to unblock measurement, exactly as the previous session
used disposable, uncommitted patches to reach its diagnosis. It is not applied to the committed
corpus and is not a proposed fix.

### Method

Fixed-length runs, no BEAGLE (not installed — `beagle-1.0.0.jar` is present in the local m2 as a
Java stub only; native library load falls back to `BeerLikelihoodCore4` in all runs below,
consistent with normal use). `examples/primates.xml`: chain shortened to 20k in a scratch copy.
`examples/infb_HA_141/template.xml`: `{chain_length}` → 10k (later re-verified at 2k after the
AVMVN fix), `{edge_weights}` → `targetedbeast.edgeweights.ConstantWeights`.
`examples/iecor/{pca,vanilla}/iecor.xml`: `chainLength` shortened to 50k in a scratch copy
(`-overwrite`), plus the disposable, uncommitted `almostUniformOrigin` substitution described
above; run via `mvn -o exec:exec -Dbeast.args="-seed 1 -overwrite <scratch xml>"` directly (iecor
uses no `feast.*` classes, so it needed none of the manual module-path assembly infB/lassaL do).
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
| `iecor/vanilla/iecor.xml` (no edge weights, live `focalsubset` partition only) | 41 | 2,653 | 1,071 | 50,000 | 4.617 s | ~10,831 |
| `iecor/pca/iecor.xml` (`PCAWeights`, live `focalsubset` partition only) | 41 | 2,653 | 1,071 | 50,000 | 5.853 s | ~8,543 |
| `lassaL_kli22` (+ user's `ali_normalized.fasta`, full) | 551 | 7,038 | 6,119 | 5,000 | 100.274 s | ~50 |
| `lassaL_kli22@120` (seeded subsample, committed XML) | 120 | 7,038 | 4,572 | 5,000 | 17.003 s | ~294 |

Pattern counts and taxon counts are both BEAST's own startup report
(`Alignment(...): [taxa, patterns, sites] = [...]`), confirming the plan's point that pattern
count does not scale linearly with taxa: infB has 2.3x primates' taxa but 7.1x its patterns;
lassaL has 3.9x infB's taxa but 12.5x its patterns; lassaL@120 (0.22x lassaL's taxa) still keeps
0.75x its patterns — cost does not fall in proportion to the taxon subsample, which is exactly why
the tier arithmetic below uses the measured lassaL@120 number rather than extrapolating from taxa
alone. iecor is the outlier in the other direction: its live content (one partition, 1,071
patterns) is smaller than infB's, so it is by far the *cheapest* of the three runnable datasets,
not the most expensive as the original 341-partition dataset description implied — see the
"substitution/clock migration defects" note above for why. `vanilla` (no edge-weights operator) is
used as the tier-arithmetic baseline, matching how `ConstantWeights` is used for infB/lassaL@120;
`pca`'s ~21% slower throughput is the sensitivity check below.

### The 10-hour rule and the derived tier table

The original `full` tier (10M states, 4 reps, ~6 settings, 3 datasets, all at full taxon count) is
machine-days, not hours, and is dataset-dominated by lassaL:

- infB: 10,000,000 / 1,900 ≈ **1.5 hours** for one replicate — fine on its own.
- lassaL (551 taxa): 10,000,000 / 50 ≈ **55.7 hours (≈2.3 days)** for one replicate; at 4 reps ×
  ~6 settings that is **~56 days of sequential compute** — an order of magnitude past "overnight."
- iecor: now measured (~10,831 states/sec, `vanilla`) — cheap enough that a single 10M-state
  replicate would take ≈4.2 minutes; the constraint below is fitting 24 reps at a chain length
  worth running, not fitting within 10h at all.

**Decision: lassaL's full 551-taxon alignment is not part of the pipeline at all** — not just at
the `full` tier. Every tier uses the lassaL@120 subsample (or smaller, at `fast`, once Phase 0
step 4 picks a size) instead. The full-taxon analysis is a user-initiated cluster job, not
something any tier runs.

**Full-tier arithmetic, from measured throughput, under the 10-hour rule — now all three runnable
datasets.** Budget: 10h = 36,000s. infB and lassaL@120 keep their existing chain lengths and
budget exactly as before (their arithmetic did not change — see below); iecor's throughput is now
measured, so its cost can be carved out of the *remaining* budget rather than left open-ended:

| dataset | measured states/sec | `full` chain | time/rep | × 24 reps (6 settings × 4 reps) |
|---|---:|---:|---:|---:|
| infB (full, 141 taxa) | 1,900 | 1,000,000 | 526 s | ≈3.51 h |
| lassaL@120 | 294 | 150,000 | 510 s | ≈3.40 h |
| iecor (`vanilla`, 41 taxa, 1,071 patterns) | 10,831 | 3,000,000 | 277 s | ≈1.85 h |

- infB + lassaL@120 subtotal: 12,624 s + 12,240 s = 24,864 s ≈ **6.91 h**, unchanged from before.
- Remaining budget to the 10h ceiling: 36,000 s − 24,864 s = 11,136 s ≈ **3.09 h**.
- iecor at a 3,000,000-state `full` chain: 3,000,000 / 10,831 ≈ 277.0 s/rep × 24 reps ≈ 6,649 s ≈
  **1.85 h** — comfortably inside the remaining 3.09 h, with 4,487 s (≈1.25 h, ≈12.5% of the total
  10h budget) of margin left over.
- **Sensitivity check with the slower `pca` throughput** (8,543 states/sec, `PCAWeights` rather
  than no edge-weights operator — the same margin logic the original arithmetic used for
  `PCAWeights2`/`PartialWeights`/`FelsensteinWeights` being costlier than the `ConstantWeights`
  baseline): 3,000,000 / 8,543 ≈ 351.2 s/rep × 24 ≈ 8,428 s ≈ 2.34 h. Total across all three
  datasets at this pessimistic rate: 24,864 + 8,428 = 33,292 s ≈ **9.25 h** — still inside the 10h
  ceiling, with 2,708 s (≈0.75 h) to spare.
- **Grand total, all three datasets, 6 settings × 4 reps each: ≈8.75 h (vanilla-rate estimate) to
  ≈9.25 h (pca-rate, worst-case estimate) — inside the 10-hour ceiling either way.** No reps or
  settings need to be cut for any of the three datasets; infB and lassaL@120's chain lengths are
  unchanged (1,000,000 / 150,000, as before) and iecor's `full` chain length is set to **3,000,000
  states**.
- `medium` scales every dataset's `full` chain length down 10× (infB 100,000, lassaL@120 15,000,
  iecor 300,000): 1,262 s + 1,224 s + 665 s ≈ 3,151 s ≈ **53 minutes** for a complete medium-tier
  sweep across all three datasets, six settings, four reps each.

This is the same tier table now recorded in `targetedbeast-evaluation-pipeline.md`'s Phase 3 tier
table. The one caveat: iecor's throughput was measured with `almostUniformOrigin`'s disposable
`Uniform(0, 1.0E6)` placeholder (see above), not the original, still-unresolved `AlmostUniform`
prior — if `AlmostUniform` is eventually restored with a real implementation that is
computationally heavier than a plain `Uniform` evaluation, iecor's throughput number here would
need re-measuring, though a single scalar-parameter prior's `calculateLogP()` is very unlikely to
be the dominant cost against a 1,071-pattern tree likelihood.
