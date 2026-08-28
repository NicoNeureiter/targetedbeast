# Adapting weight-transform coefficients

## What is being optimized

A targeted operator picks a source edge and a target edge from distributions built out of
`EdgeWeights` output. A `WeightTransform` reshapes those weights before normalization:

```
u_k  = f(w_k, b)                pointiness
w'_k = a · mean(u) + u_k        offset
q_k  = w'_k / Σ_j w'_j          selection probability
```

`b` controls how greedy the proposal is; `a` mixes in a uniform distribution, since

```
q = (a/(1+a)) · uniform + (1/(1+a)) · normalize(u)
```

The objective is the expected Metropolis-Hastings acceptance probability of the operator,

```
J(θ) = E[ min(1, exp(logAlpha)) ]
```

as a function of θ = (log b, log a). We want the argmax.

## Why BEAST's standard machinery cannot do this

Every adaptable operator in BEAST tunes through `OperatorSchedule.calcDelta`:

```
delta = (1/count) · (exp(min(logAlpha, 0)) − target)
```

with `target = 0.234`. This is Robbins-Monro applied to the equation `J(θ) = target`. It
finds a coefficient at which the acceptance rate *equals* a preset value; it has no notion
of a maximum. For a step-size parameter that is the right thing to do, because acceptance
decreases monotonically in step size and 0.234 is the known optimal trade-off between
acceptance and distance travelled.

A weight transform is not a step size. Making the proposal pointier raises acceptance
without shortening the move, so there is no trade-off for 0.234 to balance. Feeding `J` into
`calcDelta` would drive `b` *downward* whenever acceptance exceeded 0.234 — away from the
optimum. A maximizer is needed instead.

## SPSA

Simultaneous perturbation stochastic approximation estimates a gradient from a single noisy
measurement per iteration, regardless of the dimension of θ.

At iteration `n`, draw a Rademacher vector `s ∈ {−1,+1}^d`, hold the perturbed point for a
batch of `K` proposals, and average the objective over the batch:

```
θ_eff = clamp(θ_n + c_n · s)
δ     = θ_eff − θ_n                         realized perturbation
ᾱ_K   = (1/K) Σ min(1, exp(logAlpha_k))     over the K proposals at θ_eff
```

then update

```
θ_{n+1} = θ_n + (a_n / c_n²) · (ᾱ_K − ᾱ_n)/max(ᾱ_n, ε) · δ
ᾱ_{n+1} = ᾱ_n + λ · (α − ᾱ_n)               updated per proposal
```

Away from the bounds `δ = c_n · s`, and since `1/s_i = s_i` for a Rademacher sign this is the
textbook `(a_n/c_n) · Δ · s`. Stepping along `δ` rather than `s` matters only where `clamp`
truncated the perturbation. There the nominal form would take a full-size step in a direction
along which nothing was measured — at a bound with the sign pointing outward, `δ_i = 0` and the
measurement is taken at the *unperturbed* point, so the step would be pure noise, always
directed into the bound. With `δ` a half-truncated coordinate satisfies
`E[Δ · δ_i] = ½ c_n² g_i`: half magnitude, correct sign. The cross terms still vanish because
the signs remain independent, so no new bias is introduced.

Why this estimates a gradient: expand `α` around `θ_n`. Its expectation at the perturbed
point is `J(θ_n) + c_n · sᵀ∇J(θ_n) + O(c_n²)`. Multiplying by `s` and taking expectations,
the cross terms `E[s_i s_j] = 0` for `i ≠ j` and `E[s_i²] = 1`, so

```
E[(α − ᾱ) · s]  =  c_n · ∇J(θ_n)  +  O(c_n²)
```

Dividing by `c_n` leaves an estimate of `∇J` biased only at order `c_n`, which vanishes as
`c_n → 0`. The cost is one measurement per iteration in any dimension — as opposed to `2d`
for coordinate-wise finite differences.

### The baseline

`ᾱ` is an exponentially weighted running mean of the acceptance probability, used as a
control variate. It does not change the expectation, because `E[ᾱ · s] = 0` for `ᾱ`
independent of the current `s`. It does change the variance: without it, each update has
magnitude proportional to `α` itself rather than to how `α` responded to the perturbation,
and the signal is swamped.

The baseline's own variance is easy to overlook, and it can dominate. An EWMA at rate `λ` has
variance `σ²λ/(2−λ) ≈ σ²/(2/λ)`, i.e. it averages `2/λ` effective samples. It enters the
update as the subtrahend, so its variance *adds* to that of the batch mean:

```
var(ᾱ_K − ᾱ)  =  σ²/K  +  σ²λ/2      →   effective batch  K_eff = K / (1 + Kλ/2)
```

At `K = 400` and `λ = 0.01` the baseline averages only 200 samples, twice the noise of the
batch mean, and `K_eff = 133` — a third of what the 400 proposals paid for. The rate has to
satisfy `2/λ ≫ K`, hence the default `λ = 0.001`, which gives `K_eff ≈ 380`. Staleness is not
the counter-pressure it looks like: the iterate moves by `a_n/c_n ≈ 3·10⁻³` per batch, far too
slowly for a 2000-sample baseline to fall behind. **If you raise `batchSize`, lower
`baselineRate` with it.**

### Signal-to-noise at low acceptance

The naive per-proposal update fails badly on real targeted operators, and it is worth being
precise about why. Write `p` for the acceptance rate at the current coefficients. The drift
per update is `∝ a_n · p · r`, where `r` is the *fractional* sensitivity of acceptance to the
coefficient. The noise standard deviation is `∝ a_n · √(p(1−p)) ≈ a_n · √p` for small `p`,
since the per-proposal acceptance outcome is nearly Bernoulli. So

```
SNR  ∝  c_n · r · √p
```

At `p = 0.5` this is workable. At the `p ≈ 0.005` that `TargetedWilsonBalding` runs at on real
data, the signal is ~10× weaker, and the iterate random-walks instead of climbing. Observed
directly: four independent chains wandering over `b ∈ [1.83, 2.12]` with no agreement and no
convergence.

Three corrections, all of which the implementation applies:

**Batching.** Hold one perturbed point for `K` proposals and update on the batch mean. The
drift is unchanged and the noise falls by `√K`, so `SNR ∝ √K`. This is the dominant fix; `K`
in the hundreds is appropriate at sub-1% acceptance. The cost is `K`× fewer adaptation steps
per chain, which is why it pairs with the next item.

**Relative scaling.** Divide by the baseline, so the step responds to the fractional change in
acceptance rather than the absolute one. Without this, `stepSize` has to be retuned for every
acceptance regime — a value calibrated at 50% acceptance is two orders of magnitude too small
at 0.5%. The `max(ᾱ, ε)` floor caps how far a not-yet-converged baseline can amplify a step
early in the run.

**Larger perturbation.** Drift grows with `c_n` while noise does not, so `SNR ∝ c_n` directly.
The cost is gradient bias of order `c_n²`, which is a good trade while far from the optimum
and is annealed away by the decaying gain schedule.

### Tuning one coefficient instead of two

The offset `a` matters far less than the exponent `b` in practice — it converges to near zero
when tuned freely. Since SPSA perturbs every coefficient simultaneously, an uninformative
coefficient still contributes its full share of variance to the shared measurement. Pinning
the offset (`tuneOffset="false"`, the default) removes that variance for free. Set
`offset="0.0"` alongside it to drop the uniform mixture entirely.

### Gain sequences

```
a_n = A / (n + n₀)^0.602
c_n = C / (n + n₀)^0.101
```

These are Spall's recommended exponents. The requirements they satisfy:

- `Σ a_n/c_n = ∞` — the iterate can still travel an unbounded distance, so a bad starting
  point is recoverable.
- `Σ (a_n/c_n)² < ∞` — accumulated noise has bounded variance, so the iterate settles.
- `c_n → 0` — the gradient bias vanishes.

`n₀` is a stability constant that damps the first few hundred steps, when `ᾱ` has not yet
converged and the updates would otherwise be large and uninformed.

## Which operators support this

Every operator extending `TargetedWeightOperator` accepts `targetWeightTransform` and
`edgeWeightTransform`: `TargetedWilsonBalding`, `TargetedWilsonBaldingFixedHeight`,
`TargetedWilsonBaldingHeightDelta`, `WeightedWideOperator` and `RangeSlide`.

Two differences are worth knowing:

- **The no-transform default is not the same everywhere.** The four Wilson-Balding-style
  operators have always squared their target weights, so omitting `targetWeightTransform`
  keeps that squaring and `exponent="2.0"` reproduces it exactly. `RangeSlide` uses its target
  weights as they come, so there the no-transform default is the identity and `exponent="1.0"`
  is the neutral setting.
- **`RangeSlide` tunes two things at once.** Its own `size` is driven by `calcDelta` toward the
  0.234 target acceptance, while the transform climbs toward whatever coefficients maximize
  acceptance. They consume the same observation but pull on different parameters, so they do
  not fight; the transform makes the proposal pointier and `calcDelta` responds by widening
  the slide.

A transform can also be shared: reference one object from both inputs, or from several
operators. `startProposal` and `optimize` are guarded so a shared object is driven once per
proposal rather than once per input.

## Why this is a valid MCMC

Two separate claims.

**Each proposal is a correct Metropolis-Hastings move.** The coefficients are frozen at
`θ_eff` for the entire duration of one `proposal()` call. The forward probability of the
proposed move and the reverse probability of its inverse are therefore computed from the
same distribution family, and the Hastings ratio is exact. Adaptation happens only in
`optimize()`, which runs after the accept/reject decision.

**The adaptive chain has the right stationary distribution.** This is the standard
diminishing-adaptation argument (Roberts & Rosenthal 2007). Two conditions:

- *Diminishing adaptation.* The kernel used at step `n+1` differs from the one at step `n`
  by an amount tending to zero in probability. Here the coefficients move by
  `O(a_n/c_n) = O(n^−0.501)` and the perturbation jitter is `O(c_n) = O(n^−0.101)`, both
  vanishing.
- *Bounded convergence.* Each fixed-coefficient kernel converges, uniformly over the
  reachable coefficient range. The coefficients are clamped to `b ∈ [0.1, maxExponent]` and
  `a ∈ [1e-6, 100]`, so the family is compact and no member degenerates: `a > 0` keeps
  every feasible candidate reachable, and `b < ∞` keeps the proposal from collapsing onto a
  single target.

  `maxExponent` defaults to 20. What the argument requires is that the bound is *finite*, not
  that it takes any particular value, so raising it is safe for validity — but it does widen
  the family over which convergence has to hold uniformly, and it lets the proposal get
  pointier before anything stops it. Raise it when a measurement says the reward is still
  climbing at the current bound, not as a matter of course.

The clamping is not cosmetic — it is what makes the second condition hold.

## The limitation worth knowing about

Maximizing acceptance is not the same as maximizing mixing. A proposal that is extremely
pointy will keep proposing the same target and be accepted almost every time, while the
chain oscillates between two topologies and explores nothing. Acceptance rate is blind to
this.

`TargetedWilsonBalding` partially defends itself: it zeroes the sibling's selection mass, so
a proposal always changes the topology, and a pure no-op is impossible. But the degenerate
two-state oscillation is not excluded. The upper bound on `b` exists for exactly this
reason, and the benchmark reports ESS per hour rather than acceptance alone, because ESS is
the quantity that actually matters.

If the adapted `b` is sitting at its upper bound in a real run, treat that as a signal to
investigate rather than as a converged answer.

## Why SPSA struggles on this problem

At the `p ≈ 0.005` these operators run at, one batch of `K = 400` proposals carries roughly
`√(p/K) ≈ 3.5e-3` of noise on its acceptance estimate — about 70% in relative terms. The signal
a `c_n`-sized perturbation produces is `c_n · dlogα/dlogb ≈ 0.1 × 0.5`, about 7%. That is a
per-batch SNR near 0.1, so establishing even the *sign* of the gradient takes on the order of a
hundred batches, by which time the gain schedule has decayed. Observed in practice: four chains
wandering over `b ∈ [1.83, 2.12]`, less than half a factor of two from where they started, with
the trajectory driven by variance rather than by the surface.

A grid over widely separated values does not have this problem. Comparing points a factor of 2–4
apart gives an effect size 5–10× larger, needs no baseline subtraction, no correlation with a
perturbation sign, and no decaying gain — the estimate only gets better with more samples. A
grid scales badly in the number of coefficients, but there is effectively one coefficient that
matters here (`b`; the offset `a` is normally pinned at zero), which is exactly the regime where
it is the right tool.

## Tuners

The choice of pointiness function (`PowerOffsetTransform`, `ExpOffsetTransform`) and the choice
of how its coefficients are searched are independent, so the search lives in a separate
`CoefficientTuner` that the transform delegates to. `SpsaTuner` is the default and reproduces
the behaviour described above; omitting the `tuner` input gives it with its own defaults.

`GridTuner` proposes each time from a fixed set of exponents and keeps a running mean reward per
point, sampling among them in proportion to that reward:

```xml
<targetWeightTransform id="targetWeightTransform"
                       spec="targetedbeast.weighttransform.PowerOffsetTransform"
                       exponent="2.0" offset="0.0" maxExponent="64.0">
    <tuner spec="targetedbeast.weighttransform.GridTuner"
           exponents="1 2 4 8 16 32 64" explorationProb="0.1"/>
</targetWeightTransform>
```

Proportional rather than greedy so a point that scores nearly as well keeps being proposed at
nearly as often, and because normalized rewards are scale-invariant: nothing has to be calibrated
against the reward's units, which a softmax would require.

`explorationProb="1.0"` disables the adaptation and turns this into a pure measurement: equal
sample sizes per point give an unbiased reward curve, which is what says whether the surface has
an interior optimum at all. Below 1.0 it is the exploration floor, which matters here only for a
point whose mean has collapsed to zero, since proportional sampling would never revisit it. Points above the
transform's `maxExponent` are rejected at startup rather than clamped, since clamping would
silently collapse the top of the grid onto one value.

Validity is the mixture-kernel argument: each grid point is a valid MH move whose Hastings ratio
is computed from its own coefficients, frozen for the whole proposal, so detailed balance holds
per point and hence for the mixture. The requirement is that coefficients are never averaged
across points — covered by `testTargWilsonBaldingGridTransform`.

## The reward

`<reward idref="likelihood"/>` on the tuner makes the reward the squared change in likelihood
across a proposal. `MCMC` calls `operator.optimize()` *after* accepting or restoring the state
(`MCMC.java:484` versus the `state.restore()` at line 586), so reading the function at
`startProposal` and again at `optimize` gives the **realized** jump — zero for a rejection. Its
mean is therefore ESJD directly, with no acceptance factor to apply and no operator-side hook.

Omitting `reward` falls back to acceptance probability, which is the wrong objective: acceptance
rises monotonically with pointiness well past the point where the chain oscillates between a
handful of topologies. A benchmark run on the acceptance reward pinned the exponent at its
ceiling of 64.

`burnin` matters more with this reward than with acceptance. Early in a chain the likelihood
moves by thousands of log units per proposal on its way to stationarity, and a mean of squares
is dominated by a handful of those; every grid point then measures the same transient and the
curve comes out flat.

## References

- Spall, J. C. (1992). Multivariate stochastic approximation using a simultaneous
  perturbation gradient approximation. *IEEE Transactions on Automatic Control* 37(3).
- Roberts, G. O. and Rosenthal, J. S. (2007). Coupling and ergodicity of adaptive MCMC.
  *Journal of Applied Probability* 44(2).
