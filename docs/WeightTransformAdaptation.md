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

At iteration `n`, draw a Rademacher vector `s ∈ {−1,+1}^d` and evaluate the objective at the
perturbed point:

```
θ_eff = θ_n + c_n · s
α     = min(1, exp(logAlpha))       measured at θ_eff
```

then update

```
θ_{n+1} = θ_n + (a_n / c_n) · (α − ᾱ_n) · s
ᾱ_{n+1} = ᾱ_n + λ · (α − ᾱ_n)
```

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
and the signal is swamped. With acceptance around 0.4 and a perturbation effect of a few
percent, the baseline is the difference between converging and random-walking.

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
  reachable coefficient range. The coefficients are clamped to `b ∈ [0.1, 20]` and
  `a ∈ [1e-6, 100]`, so the family is compact and no member degenerates: `a > 0` keeps
  every feasible candidate reachable, and `b < ∞` keeps the proposal from collapsing onto a
  single target.

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

## References

- Spall, J. C. (1992). Multivariate stochastic approximation using a simultaneous
  perturbation gradient approximation. *IEEE Transactions on Automatic Control* 37(3).
- Roberts, G. O. and Rosenthal, J. S. (2007). Coupling and ergodicity of adaptive MCMC.
  *Journal of Applied Probability* 44(2).
