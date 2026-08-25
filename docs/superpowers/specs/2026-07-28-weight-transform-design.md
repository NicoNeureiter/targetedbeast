# WeightTransform: adaptive reshaping of edge and target weights

Date: 2026-07-28
Status: approved, ready for implementation

## Problem

Targeted operators select a source edge and a target edge using weights supplied by an
`EdgeWeights` object. Two aspects of how those weights are shaped are currently hardcoded
and unadaptive:

1. **Pointiness.** `TargetedWilsonBalding.java:252` squares every target weight
   (`Math.pow(distance[k], 2)`) before normalizing. The exponent 2 is a fixed guess at how
   greedy the proposal should be.
2. **Floor.** Most `EdgeWeights` implementations expose a `minWeight` input that keeps
   candidates from reaching zero probability. It is a static XML constant that a user has
   to guess, and it means different things in different implementations.

Neither can respond to the chain. A `WeightTransform` makes both adaptive: it reshapes the
raw weights before normalization, and tunes its own coefficients during the run to raise
the operator's acceptance rate.

## The transform

Given a raw weight vector `w` over a candidate set, a transform produces

```
u_k  = f(w_k, b)                    pointiness
w'_k = a · mean(u) + u_k            offset
```

and the operator normalizes `w'` to sum to 1. Two concrete forms:

| class | input space | `f(w, b)` |
|---|---|---|
| `PowerOffsetTransform` | real | `w^b` |
| `ExpOffsetTransform` | log | `exp(b · (w − max w))` |

The two are the same function; `w^b` on real-space weights is exactly `exp(b · log w)`.
The max-subtraction in the log form is numerical only and cancels in the normalization.

### Why the offset is relative to the mean

An absolute offset (`w'_k = a + u_k`, as originally sketched) fails for two reasons.

**No portable magnitude.** After the pointiness step, weights sit at wildly different
scales depending on which `EdgeWeights` is plugged in. `PartialWeights` returns softmax
probabilities (`~1/K`, so `~0.02` for 50 candidates); `ConsensusWeights` returns `1/distance`
(`~O(1)`). A fixed `a = 0.1` is negligible in one case and swamps the signal in the other,
so the adaptation would have to search several orders of magnitude before `a` does anything.

**Log-weights are only defined up to an additive constant.** Every consumer normalizes, so
`w` and `w + c` denote the same distribution. But

```
a + exp(b·(w + c))  =  a + exp(bc)·exp(b·w)
```

which after normalizing is *not* the same as `a + exp(b·w)`. With an absolute offset the
meaning of `a` would silently depend on an arbitrary constant chosen inside whichever
`EdgeWeights` produced the log-weights. Scaling the offset by `mean(u)` fixes both problems:
`mean(u)` carries the same `exp(bc)` factor, so it cancels.

### Interpretation

Normalizing `w' = a·mean(u) + u` gives

```
normalize(w')  =  (a/(1+a)) · uniform  +  (1/(1+a)) · normalize(u)
```

so `a` is literally the uniform-mixture proportion: `a = 0` is purely targeted, `a → ∞` is
uniform over the candidate set. That is the adaptive replacement for `minWeight`.

For `EdgeWeights` that already return a softmax (`PartialWeights`, `FelsensteinWeights2`),
the power form is exactly temperature scaling, because

```
normalize(softmax(x)^b) == softmax(b·x)
```

so `b` is an inverse temperature with no contract change required.

### Consequence for the existing squaring

`PowerOffsetTransform` with `b = 2, a = 0` reproduces `TargetedWilsonBalding.java:252`
exactly. Therefore:

- **No transform input** → the current code path runs unchanged, bit-identical to today.
- **Transform supplied** → the hardcoded `Math.pow(w, 2)` is skipped and the transform's
  exponent takes over.

The default coefficients are `b = 2.0, a = 0.01`. `b = 2.0` starts adaptation at current
behaviour. `a` cannot default to exactly 0 because the coefficient is adapted in log space,
so the starting point with a transform supplied is *current behaviour plus 1% uniform
smoothing*, not bit-identical. This is intentional and documented, not an oversight.

## `EdgeWeights` contract change

One default method:

```java
default boolean isLogSpace() { return false; }
```

No existing implementation overrides it.

**`isLogSpace()` describes the target weights only.** Every `EdgeWeights` implementation in
the tree returns a real-space scalar from `getEdgeWeights` — an edge-mutation count or a
branch length — and none has a reason not to. That is now contract, stated in the interface
javadoc. So the operator validates:

- `targetWeightTransform.requiresLogSpace() == edgeWeights.isLogSpace()`
- `edgeWeightTransform.requiresLogSpace() == false`, unconditionally

and throws otherwise. This also covers `useEdgeLength="true"`, where the source weights are
branch lengths rather than `EdgeWeights` output.

`PartialWeights` additionally gains a `logSpace` boolean input (default `false`). When set,
it skips its final `softmax` (`PartialWeights.java:544-546`) and returns raw log-weights,
with `isLogSpace()` returning the input value. This is what makes `ExpOffsetTransform`
exercisable end to end; without it, nothing in the codebase produces log-weights. Default
off, so every existing XML is untouched.

## Class design

New package `targetedbeast.weighttransform`:

```
WeightTransform          abstract BEASTObject, implements Loggable
├── PowerOffsetTransform
└── ExpOffsetTransform
```

Both concrete transforms have exactly the same two coefficients, so the base class declares
them directly (`exponent` = `b`, `offset` = `a`), owns the mean-relative offset step, and
owns the whole adaptation mechanism. Subclasses supply only the pointiness function:

```java
protected abstract double[] pointiness(double[] raw, double b);
protected abstract boolean  requiresLogSpace();
```

The adaptation still runs over a `double[] theta` of length 2 rather than over two named
scalars, so a future three-coefficient transform needs no rework of the SPSA loop.

Public API used by operators:

```java
void     startProposal();          // draw the SPSA perturbation, freeze coefficients
double[] apply(double[] raw);      // reshape a feasible weight vector
void     optimize(double logAlpha);// one adaptation step
```

Inputs: `initialCoefficients`, `optimise` (default `true`), `stepSize` (A, default 0.05),
`perturbation` (C, default 0.2), `stabilityConstant` (n0, default 100).

`WeightTransform` implements `Loggable` so the live coefficient values can be written to
the trace log — needed to watch convergence in the benchmark.

A transform holds mutable adaptation state that is *not* part of the MCMC state, exactly
like `ScaleOperator.scaleFactor`. It is not a `CalculationNode` and takes no part in
store/restore. Referencing one transform object from both operator inputs shares its
coefficients; separate objects adapt independently.

## Adaptation: SPSA

### Why not `calcDelta`

Every adaptable operator in BEAST tunes via `OperatorSchedule.calcDelta`
(`OperatorSchedule.java:413`), which is Robbins-Monro toward a *target* acceptance of 0.234:

```
delta = (1/count) · (exp(min(logAlpha, 0)) − target)
```

That machinery drives acceptance to a fixed value; it cannot maximize. For a weight
transform, a pointier proposal generally means higher acceptance, so `calcDelta` would push
`b` downward whenever acceptance exceeds 0.234 — the opposite of the goal. A different
algorithm is required.

### The algorithm

Let `θ` be the coefficient vector in log space (so all coefficients stay positive without
constrained optimization). Each proposal:

```java
// operator.proposal(), before any weight is used:
transform.startProposal();
//   s      ~ Rademacher^d
//   θ_eff  = θ + c_n · s     frozen for this entire proposal

// operator.optimize(logAlpha):
transform.optimize(logAlpha);
//   α         = exp(min(logAlpha, 0))
//   θ        += (a_n / c_n) · (α − baseline) · s      then clip to bounds
//   baseline += λ · (α − baseline)
//   n++
```

with Spall's standard gain sequences

```
a_n = A / (n + n0)^0.602
c_n = C / (n + n0)^0.101
```

so `a_n/c_n → 0` and `c_n → 0`. This is a one-measurement simultaneous-perturbation
gradient estimate: because `s` is symmetric, `E[(α − baseline)·s] ∝ c_n · ∇_θ E[α]`. The
running `baseline` is a control variate; without it the estimator is dominated by the
magnitude of `α` rather than its dependence on `θ`.

### Validity

The proposal kernel is a valid Metropolis-Hastings kernel for each *fixed* coefficient
value, because the forward and reverse probabilities within one proposal are computed with
the same frozen `θ_eff`. The coefficients change only in `optimize()`, i.e. based purely on
the past, which is adaptive MCMC in the sense of Roberts & Rosenthal. Diminishing adaptation
holds because both the per-step change (`a_n/c_n → 0`) and the per-step jitter (`c_n → 0`)
vanish.

### Two operational details

**Aborted proposals are skipped.** `TargetedWilsonBalding` has roughly eight early
`return Double.NEGATIVE_INFINITY` paths (parent is root, no candidates, `j.isRoot()`, and
so on). Those produce `logAlpha = −inf` ⇒ `α = 0` for reasons unrelated to `θ`, and would
add pure noise. `startProposal()` clears a `used` flag, `apply()` sets it, and `optimize()`
skips the update when the flag is still clear.

**`optimize()` is not called in debug mode.** `MCMC.java:482-485` calls `operator.optimize`
only in the non-debug branch, so any benchmark measuring adaptation must not run with
`-Dbeast.debug=true`.

### Caveat to document

Maximizing acceptance alone is not automatically the same as maximizing mixing. An
extremely pointy proposal can repeatedly propose the same target, giving high acceptance
while the chain flip-flops between two topologies. `TargetedWilsonBalding` partially guards
against this by zeroing the sibling (so a proposal always changes the topology), but the
risk is real and the upper bound on `b` exists for this reason.

A prose write-up of this section goes to `docs/WeightTransformAdaptation.md`.

## Operator integration

Scope for this change: `TargetedWilsonBalding` only. Two optional inputs,
`targetWeightTransform` and `edgeWeightTransform`, both defaulting to null (identity).

### The invariant

**The transform must be applied structurally identically in the forward and reverse
blocks** — same masking order, same interval-weight multiplication, same candidate-set
membership. Detailed balance requires `q(x→y)` computed at state `x` and `q(y→x)` computed
at state `y`; each block normalizes over its own candidate set, which is correct as long as
the reverse block mirrors the forward block exactly.

### Target path

Forward at `TargetedWilsonBalding.java:249-265`, reverse at `:383-399`. Order of operations
in both:

1. Transform the full candidate weight vector.
2. Multiply by the attachment-interval weight.
3. Zero the excluded element (the sibling forward, the no-op target in reverse).

Zeroing happens *after* the transform, so the excluded element participates in `mean(u)`.
That is fine and must be true in both blocks.

### Source path

Forward at `:162-183`, reverse at `:429-435`. `selectSourceNode` currently accumulates
weights on the fly; it must be restructured into build-vector → transform → sample. The
reverse sum must build and transform the same way.

**Root masking is a correctness hazard.** `getSourceWeight` returns `0.0` for the root. A
relative offset would give the root `a·mean(u) > 0`, making it selectable and breaking
`p = i.getParent()`. The transform therefore receives only the *feasible subvector*
(non-root nodes); masked entries stay at zero and are excluded from `mean(u)`. This applies
identically in both blocks.

## Persistence

`Operator.storeToFile` persists a single coercable value `p`. `TargetedWilsonBalding`
overrides `storeToFile`/`restoreFromFile` using the nested-JSON meta-operator pattern
documented at `Operator.java:236-252`:

```json
{"id":"targetedWB", "p":"NaN", "accept":39, "reject":35,
 "transforms":[
   {"role":"target", "b":1.84, "a":0.031, "n":120000, "baseline":0.41},
   {"role":"edge",   "b":1.02, "a":0.000, "n":120000, "baseline":0.41}
 ]}
```

Restoring `n` matters as much as restoring the coefficients: without it the gain sequences
reset to their maximum and a resumed chain takes a burst of large adaptation steps.

## Testing

### Correctness

- **`WeightTransformTest`** — `normalize(softmax(x)^b) == softmax(b·x)`; shift-invariance
  of `ExpOffsetTransform` under `w → w + c`; the mixture identity
  `normalize(w') == (a/(1+a))·uniform + (1/(1+a))·normalize(u)`; and that `b=2, a=0`
  reproduces the current squaring.
- **`SpsaAdaptationTest`** — a synthetic unimodal `α(θ)` with a fixed seed; assert the
  coefficient climbs to the known optimum and that the step sizes decay.
- **`TargetedOperatorDetailedBalanceTest`** — new cases with `optimise="false"` (the kernel
  must be fixed across samples for the flow accounting to be valid) and deliberately
  non-trivial coefficients such as `b=3.7, a=0.3`. One case exercises the target path; one
  exercises the source path using the existing cache-free `StructuralSourceWeights` fixture,
  which is the only way to isolate source-selection errors from caching artifacts.
- **`EdgeWeightRejectConsistencyTest`** — unchanged and expected to stay green; transforms
  touch no cache.

Known pre-existing state to preserve, not fix: the detailed-balance suite currently has four
failures in operators this change does not touch (`testTargWilsonBaldingHeightDelta`,
`testTargWilsonBaldingFixedHeight`, `testTargWilsonBaldingMaxHeightRatio`, plus one that
rotates), and `EdgeWeightRejectConsistencyTest.testFelsensteinWeights2` is `@Ignore`d for a
pre-existing bug. The baseline is 12 pass / 4 fail.

### Performance

Extend the existing Snakemake harness in `scripts/` rather than building a second one; it
already builds the package, renders XMLs from a template, runs BEAST, and collects ESS/hour
plus per-operator acceptance.

- `generate_xmls.py` gains a `WEIGHT_TRANSFORMS` dict crossed with `EDGE_WEIGHTS`, subdirs
  named `{edgeweights}__{transform}`, and a `--chain-length` argument (currently hardcoded
  at 10M in `OTHER_BEAST_SETTINGS`).
- `examples/lassaL_kli22/template.xml` gains a `{target_weight_transform}` slot nested
  inside the `targetedWilsonBalding` operator element.
- The Snakefile gains a `--config quick=1` profile: 200k chain, 2 replicates, a single
  edge-weights entry, so a full comparison completes in minutes during development.
- `collect_stats.py` needs no change; it already records the subdir label and per-operator
  acceptance, so the transform axis appears in `mcmc_stats.tsv` automatically.

### Development loop

| purpose | command | duration |
|---|---|---|
| correctness | `ant compile compile-test` + ConsoleLauncher | seconds |
| acceptance / ESS | `snakemake -j4 --config quick=1` | minutes |
| real measurement | `snakemake -j4` | hours |

A clean rebuild (`rm -rf build build-test && ant compile compile-test`) is mandatory before
trusting any test result: `ant compile` is incremental without dependency tracking and will
happily leave stale classes behind, which has produced phantom test failures in this
codebase before.

## Out of scope

- The other nine operators taking an `edgeWeights` input. Each has its own weight
  normalization idiom and needs its own forward/reverse audit.
- Removing `minWeight` inputs from `EdgeWeights` implementations. The transform offset
  stacks on top of them for now.
- Converting the `EdgeWeights` contract wholesale to log space.
