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

**Blocked.** The task asked to measure states/second for `examples/primates.xml` (20k states) and
`examples/iecor/pca/iecor.xml` (5k states). Neither file is tracked in `beast2.8-feature-trunk`
(or in any of `beast2.8-migration` / the four feature branches / `main`) — both are matched by
`.gitignore` patterns (`/examples/primates*.xml`, `/examples/iecor`) and exist only as local,
uncommitted files in the user's `beast2.8-felsenstein-weights` checkout. They were copied
read-only from that checkout into this worktree to attempt the measurement (not committed, and
the source checkout was not modified), but both fail to run against `beast2.8-feature-trunk`'s
core:

**`primates.xml`** — uses the pre-migration (untyped) likelihood spec; this branch's core expects
the `beast.base.spec.*` typed twin:

```
Error 130 parsing the xml input file
Input 102: type mismatch for input likelihood
Provide type class beast.base.evolution.likelihood.TreeLikelihood, but expected class beast.base.spec.evolution.likelihood.TreeLikelihood

Error detected about here:
  <beast>
      <run id='mcmc' spec='MCMC'>
          <distribution id='posterior' spec='CompoundDistribution'>
              <distribution id='edgeweights' spec='targetedbeast.edgeweights.PartialWeights'>
```

**`iecor/pca/iecor.xml`** — depends on `beastlabs.evolution.tree.SimpleRandomTree`, and the
`BEASTLabs` package/dependency is not on this project's module path:

```
Class beastlabs.evolution.tree.SimpleRandomTree could not be found.
This XML requires the following package that is not installed: BEASTLabs v2.0.2
See http://beast2.org/managing-packages/ for details on how to install packages.
Or perhaps there is a typo in spec and you meant beast.base.spec.evolution.tree.coalescent.RandomTree?

Error detected about here:
  <beast>
      <run id='mcmc' spec='MCMC'>
          <init id='RandomTree.t:tree' spec='beastlabs.evolution.tree.SimpleRandomTree'>
```

Per the repository's `CLAUDE.md` ("Never hand-edit BEAST XML to migrate it — run the converter
and review its per-file report"), these were not hand-patched. Both example XMLs need a proper
XML migration (`beast3-xml-migration` skill) — and, for `iecor.xml`, a `BEASTLabs` dependency
decision — before throughput can be measured. No numbers are reported; none were fabricated.
