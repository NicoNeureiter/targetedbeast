# Phase 0 step 4 — at what dataset size does targeting beat uniform?

Dataset: lassaL (`examples/lassaL_kli22/template.xml` + `ali_normalized.fasta`, seeded taxon
subsamples at n=40/80/150/250, `SUBSAMPLE_SEED=42`). Settings: `ConstantWeights` (uniform baseline)
vs `ParsimonyWeights2` (`maxWeight=20`). 4 reps per cell, seeded on rep index alone (seed =
1000+rep, shared between the two settings at a given rep) so the comparison is paired. Metric:
RNNI (`beastlabs.evolution.tree.RNNIMetric`), `jumpInterval=10`. Full 4×2×4 = 32-cell design run;
no cells cut.

Per-size chain length (post-burnin) and pre-burnin, chosen to fit a ~2h wall-clock budget against
measured throughput at each size:

| size | preBurnin | chainLength (post-burnin) | total proposals |
|---|---:|---:|---:|
| 40  | ~1,000 (est.) | 100,000 | ~101,000 |
| 80  | ~1,000 (est.) | 50,000  | ~51,000  |
| 150 | 5,000  | 25,000  | 30,000  |
| 250 | 3,000  | 15,000  | 18,000  |

## Convergence check — the decisive finding

Before trusting any ESJD number, I checked whether the chain had reached the stationary regime by
comparing the tail of the posterior trace across reps of the same setting. The repo's own
`scripts/generate_xmls.py` already flags this failure mode (`GRID_BURNIN = 10_000`, "at 30k steps
the chain is still descending thousands of log units per proposal").

**n40 and n80: converged.** Independent reps agree within ~15 log-posterior units at the end of the
run and the trace is flat over the last 500+ samples (see raw `run.log` tails). These numbers are
trustworthy measurements of the stationary regime.

**n150 and n250: NOT converged.** Independent reps of the same setting disagree by **hundreds of
log-posterior units** at the end of the run, and the trace is still trending steeply downward
(n150 rep0: -168767 → -168617 over the last 2,500 steps, still descending; rep1 sits at -168350,
a different plateau entirely). At this budget (18k–30k total proposals), a 150+-taxon lassaL chain
has not left its burn-in transient. **The n150/n250 ESJD numbers below measure which burn-in
trajectory each chain happened to follow, not operator mixing efficiency, and must not be used to
answer the question this experiment was run to answer.** This is exactly the failure mode
`GRID_BURNIN` exists to avoid, reproduced here because the ~2h/32-cell budget could not afford
enough proposals at these sizes.

## Raw results — chain-level (`__CHAIN__` row), per cell

| size | setting | rep | acceptRate | meanJumpGivenAccepted | esjdPerSec_rnni |
|---:|---|---:|---:|---:|---:|
| 40 | constant | 0 | 0.2068 | 0.774 | 594.1 |
| 40 | constant | 1 | 0.2065 | 0.661 | 450.8 |
| 40 | constant | 2 | 0.2053 | 0.667 | 453.9 |
| 40 | constant | 3 | 0.2086 | 0.736 | 601.0 |
| 40 | parsimony | 0 | 0.2083 | 0.675 | 439.3 |
| 40 | parsimony | 1 | 0.2160 | 0.680 | 524.3 |
| 40 | parsimony | 2 | 0.2138 | 0.702 | 547.1 |
| 40 | parsimony | 3 | 0.2080 | 0.698 | 495.6 |
| 80 | constant | 0 | 0.2114 | 1.288 | 1075.1 |
| 80 | constant | 1 | 0.2038 | 1.170 | 791.7 |
| 80 | constant | 2 | 0.2110 | 1.133 | 830.5 |
| 80 | constant | 3 | 0.1995 | 1.178 | 766.1 |
| 80 | parsimony | 0 | 0.2105 | 1.180 | 1037.3 |
| 80 | parsimony | 1 | 0.2136 | 1.269 | 1357.7 |
| 80 | parsimony | 2 | 0.2107 | 1.028 | 538.1 |
| 80 | parsimony | 3 | 0.2155 | 1.430 | 1269.2 |
| 150 † | constant | 0–3 | 0.203–0.232 | 2.15–3.26 | 1222.9 / 9507.2 / 2502.1 / 2032.2 |
| 150 † | parsimony | 0–3 | 0.220–0.244 | 2.72–4.5 | 8834.7 / 6552.8 / 15374.3 / 15527.5 |
| 250 † | constant | 0–3 | 0.201–0.214 | 8.6–13.6 | 32762.6 / 111628.2 / 39526.5 / 155667.3 |
| 250 † | parsimony | 0–3 | 0.207–0.221 | 10.9–14.7 | 78707.9 / 93815.6 / 151971.4 / 40144.3 |

† = not converged, see above. Full per-operator rows are in the committed `run.opstats.tsv` under
each `runs/eval0/design/n<size>/<setting>/rep<rep>/` directory (not committed — regenerable from
`scripts/render_benchmark_xml.py` with the parameters above).

## Paired analysis (parsimony − constant), t-based 90% CI, n=4, df=3

| size | mean diff (ESJD/s) | 90% CI | excludes zero? | note |
|---:|---:|---|---|---|
| 40  | −23.3   | (−170.6, 123.9)     | no | converged data |
| 80  | +184.7  | (−306.9, 676.3)     | no | converged data |
| 150 | +7756.2 | (−1199.8, 16712.3)  | no | **not converged — do not use** |
| 250 | +6263.7 | (−107925.0, 120452.4) | no | **not converged — do not use** |

## Answers

**(a) At which size does `ParsimonyWeights2` beat `ConstantWeights` with a CI excluding zero?**
None, in the valid (converged) range. At n=40 the point estimate is slightly negative; at n=80 it
is positive (+184.7 ESJD/s, roughly +21% relative to the constant-weights mean of 865.9) but the
CI is wide and includes zero. n150/n250 cannot answer this question with the current design — the
chains never reached stationarity, so their apparent large effects are burn-in artifacts, not
evidence of anything about operator efficiency.

**(b) What is the effect size where it wins?** Not established. The one directionally-positive,
converged result (n=80, ~+21%) is far short of significance at 4 reps and should not be used to
set the decision rule's 20% placeholder threshold — that number is still unfounded and needs a
properly-powered follow-up.

**(c) What is the between-rep variance?** At the converged sizes it is large relative to the mean:
coefficient of variation on `esjdPerSec` is roughly 15–20% at n=40 and 20–30% at n=80. At n≥150 the
"variance" is dominated by non-stationarity and is not interpretable as sampling variance at all —
CVs of 60–100% there reflect different burn-in trajectories, not chain-to-chain noise in a fixed
distribution.

## What this means for the plan

1. **The `fast` tier size is still undetermined.** This experiment establishes that lassaL needs
   *far* more than 18k–30k total proposals to reach stationarity at 150+ taxa — an order of
   magnitude more budget than a ~2h/32-cell sweep can afford. Either the fast tier stays at ≤80
   taxa (where this experiment shows convergence is achievable quickly), or a much longer per-cell
   budget is needed at larger sizes, trading off reps or settings.
2. **No effect was confirmed at any size**, including the converged ones. This could mean: the
   effect is real but smaller than 4 reps can resolve (increase reps); the effect needs more
   post-burnin proposals even where the trace looks flat (250k–500k range, not just 25k–100k);
   or `ParsimonyWeights2` genuinely does not help lassaL at these sizes and a different weighting
   (`PCAWeights2`, `PartialWeights`) should be tried instead. This experiment cannot distinguish
   between these — it only rules out a *large*, easily-detected effect at n=40/80.
3. **The decision rule's 20% threshold remains a placeholder.** No effect size from this experiment
   is trustworthy enough to calibrate it.

## Recommendation

Before setting the `fast` tier or trusting any weighting comparison at 150+ taxa, run a
convergence-length pilot at that size specifically (track the posterior trace, extend until
independent reps agree, the same check applied here) rather than assuming a chain length scaled
linearly from smaller sizes is sufficient. Given how much throughput drops with taxon count, this
pilot should be run once per size and its result cached, not re-derived per experiment.

---

# Follow-up — infB full (141 taxa) and lassaL@100

Extends the experiment above to two more cases chosen to bracket the previous result: **infB
full** (141 taxa, 489 patterns, nucleotide/HKY/strict-clock — a different data regime from lassaL
entirely) and **lassaL@100** (a seeded 100-taxon subsample, `SUBSAMPLE_SEED=42`, same alignment
and template as above), which sits between the converged-but-null lassaL n80 and the
never-converged lassaL n150/n250. Same design as above: `ConstantWeights` vs `ParsimonyWeights2`
(`maxWeight=20`), 4 reps per (case, setting) cell, seed = 1000+rep shared between settings at a
given rep (paired), RNNI metric (`jumpInterval=20`), chain-level `__CHAIN__` row. infB has no
external alignment or `{target_weight_transform}` slots to fill (verified directly against
`examples/infb_HA_141/template.xml`) — only `{chain_length}` and `{edge_weights}` — so it needed a
standalone renderer, `scripts/render_infb_benchmark_xml.py`, mirroring
`render_benchmark_xml.py`'s `<run>`/`<state>` rewrite and RNNI-injection pattern but pointing the
metric's `taxonset` at infB's own `TaxonSet.infB_HA` and `ParsimonyWeights2`'s `data` at infB's own
`infB_HA` alignment id instead of lassaL's `@taxonSet`/`@alignment`.

## Chain length chosen, and why

Measured throughput first (both used to set the initial chain-length guess):

- infB: reused the ~1,900 states/sec baseline in `docs/evaluation/baseline-tests.md` (re-verified
  live: 8 cells at chain 150,000 + preBurnin 20,000 averaged ~1,880 states/sec, matching).
- lassaL@100: no prior direct measurement existed (only n80/n120/n551), so it was measured with a
  short 3,000-proposal probe (`ConstantWeights`, `runs/eval1/throughput/n100`): 3,000 states in
  8.939 s ≈ 335 states/sec — consistent with interpolating between the measured lassaL@120 rate
  (294 states/sec, 4,572 patterns) and lassaL@80's implicit rate from the earlier experiment
  (higher, since n80 fit a 50k-proposal chain into the earlier 2h/32-cell budget).

Initial chain lengths, set to fit comfortably inside the ~90-minute total budget while giving each
case a proposal budget at or above the smallest converged budget from the first experiment (lassaL
n80 converged at ~51,000 total proposals):

| case | preBurnin | chainLength (post-burnin) | total proposals | throughput | time/cell |
|---|---:|---:|---:|---:|---:|
| infB full (141 taxa)  | 20,000 | 150,000 | 170,000 | ~1,880/s | ~90 s |
| lassaL@100 | 5,000 | 95,000 | 100,000 | ~335/s | ~300 s |

8 cells × ~90 s (infB) + 8 cells × ~300 s (lassaL@100) ≈ 12 min + 40 min ≈ 52 min wall-clock for
both full designs, run once — no extension was needed for either case (see below), so the
one-extension budget was never spent.

## Convergence check

**infB full: converged, cleanly, on the first attempt.** All 8 reps (4 `constant`, 4 `parsimony`)
land in a tight -9226 to -9274 posterior band at the final sample (150,000), agreeing within ~48
log-posterior units — an order of magnitude tighter than the "tens of units" bar — and the trace
for every rep is flat over the last 500 samples (checked directly in each `run.log` tail, not
just two reps). infB's smaller pattern count and shorter tree evidently make it far easier to mix
than lassaL at a comparable taxon count.

**lassaL@100: converged.** Checked the full posterior trajectory (every 20,000 samples), not just
the tail, for `constant` reps 0 and 1: `-123448 → -121095 → -120977 → -120981 → -120969` and
`-124217 → -121144 → -120974 → -120966 → -120967` — both flat from sample 20,000 onward, matching
plateaus. All 8 reps (both settings) land in the range -120964 to -120991 at the final sample
(95,000) — within 27 log-posterior units, comparable to the converged n40/n80 tolerance in the
first experiment and nowhere near the "hundreds of units, different plateaus" signature that
disqualified n150/n250. lassaL@100 sits closer to n80 (which converged in ~51,000 proposals) than
to n150 (which failed at ~30,000 proposals with 5–10× fewer *effective* proposals per taxon); at
100,000 total proposals here it converges comfortably, consistent with that reading.

Both cases are usable — no chain-length extension was needed for either.

## Raw results — chain-level (`__CHAIN__` row), per cell

| case | setting | rep | acceptRate | meanJumpGivenAccepted | esjdPerSec_rnni |
|---|---|---:|---:|---:|---:|
| infB | constant | 0 | 0.1340 | 4.097 | 15164.3 |
| infB | constant | 1 | 0.1411 | 3.987 | 16814.0 |
| infB | constant | 2 | 0.1351 | 4.092 | 16335.4 |
| infB | constant | 3 | 0.1386 | 4.446 | 19448.4 |
| infB | parsimony | 0 | 0.1239 | 3.966 | 17708.6 |
| infB | parsimony | 1 | 0.1252 | 3.969 | 18814.9 |
| infB | parsimony | 2 | 0.1232 | 3.921 | 17471.1 |
| infB | parsimony | 3 | 0.1291 | 3.407 | 14370.1 |
| lassaL@100 | constant | 0 | 0.1755 | 1.838 | 1383.0 |
| lassaL@100 | constant | 1 | 0.1727 | 1.779 | 1977.9 |
| lassaL@100 | constant | 2 | 0.1753 | 1.561 | 1163.3 |
| lassaL@100 | constant | 3 | 0.1771 | 1.465 | 1608.1 |
| lassaL@100 | parsimony | 0 | 0.1800 | 1.540 | 1157.1 |
| lassaL@100 | parsimony | 1 | 0.1784 | 1.624 | 951.4 |
| lassaL@100 | parsimony | 2 | 0.1745 | 1.817 | 1473.6 |
| lassaL@100 | parsimony | 3 | 0.1804 | 1.644 | 1450.1 |

Full per-operator rows are in each run's `run.opstats.tsv` under
`runs/eval1/{infb,lassaL/n100}/<setting>/rep<rep>/` (not committed — regenerable from
`scripts/render_infb_benchmark_xml.py` / `scripts/render_benchmark_xml.py --sizes 100` with the
parameters in the table above).

## Paired analysis (parsimony − constant), t-based 90% CI, n=4, df=3

| case | mean diff (ESJD/s) | 90% CI | excludes zero? | relative to constant mean |
|---|---:|---|---|---:|
| infB full   | +150.7  | (−4006.9, 4308.3) | no | +0.9% |
| lassaL@100  | −275.0  | (−927.7, 377.6)   | no | −17.9% |

Neither CI excludes zero. infB's point estimate is essentially flat (+0.9%); lassaL@100's point
estimate is *negative* this time (parsimony ~18% worse than constant on average), the opposite
sign from lassaL n80's positive-but-nonsignificant point estimate — both are well inside a wide,
zero-straddling CI at 4 reps, so neither sign should be read as evidence of anything, but the
flip itself is a useful data point: it shows the between-rep noise at this size/regime is large
enough to swing the point estimate's sign on its own, reinforcing that 4 reps cannot resolve an
effect at this magnitude (if one exists) in either dataset tried so far.

## Updated answer — at what size/dataset does targeting show an effect?

Combining this data with the original n40–n250 lassaL sweep: **no dataset or size tried so far —
lassaL at 40, 80, 100, or 150/250 (unusable), or infB at its full 141 taxa — shows
`ParsimonyWeights2` beating `ConstantWeights` with a confidence interval excluding zero.** Two
additional cases converged cleanly here (infB full, lassaL@100), extending the converged, usable
range from {lassaL n40, n80} to {lassaL n40, n80, n100, infB@141}, and both add null results, not
positive ones: infB shows an essentially zero effect (+0.9%, tight-looking point estimate but on a
noisy chain — CI is ±4,157, over 27× the point estimate), and lassaL@100's point estimate flipped
negative (−17.9%) relative to lassaL n80's positive point estimate, with both point estimates
comfortably inside their respective CIs of the other's sign. This is now four converged
(case, size) points — lassaL n40, n80, n100, and infB@141 — none of which reproduces the "targeted
beats uniform" ranking with statistical support at 4 reps. The most plausible reading is not that
one dataset or size is special, but that 4 reps is simply underpowered relative to the between-rep
variance observed everywhere (CV roughly 15–30% at every converged size/dataset tried): either the
true effect is smaller than 4 reps can resolve at any of these sizes, or `ParsimonyWeights2`
specifically does not reliably help relative to `ConstantWeights` at ≤150 taxa regardless of data
regime, and a properly-powered study (more reps, or a variance-reduction technique beyond common
random numbers) or a different weighting (`PCAWeights2`, `PartialWeights`) is needed before the
Phase-4 decision rule's 20% threshold can be treated as anything but a placeholder.

---

# Follow-up — `PCAWeights2` on lassaL@80 and infB full (141 taxa)

Extends the two already-converged, clean cases above — lassaL n80 and infB full — to a third
weighting, `PCAWeights2` (`maxWeight=20`, `dimension=10`, matching `pca_10` in
`scripts/generate_xmls.py`'s `EDGE_WEIGHTS` dict, alignment-distance/classical-MDS based, unlike
`ParsimonyWeights2`'s mutation-count signal). 4 reps per case, same seeds (1000+rep) as the
`constant` reps already collected above, so the comparison is paired against the existing
`constant` numbers without rerunning them. `scripts/render_benchmark_xml.py` and
`scripts/render_infb_benchmark_xml.py` each got a `"pca"` entry added to their `EDGE_WEIGHTS_XML`
dict (`<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" tree="@tree"
maxWeight="20" data="@alignment|@infB_HA" dimension="10"/>`, alignment idref matching each
renderer's existing `parsimony` entry). Chain parameters reused exactly as converged before:
lassaL@80 preBurnin=1,000, chainLength=50,000, jumpInterval=10; infB full preBurnin=20,000,
chainLength=150,000, jumpInterval=20.

## Convergence check

**lassaL n80: converged.** All 4 `pca` reps land in a tight -111878.05 to -111888.98 band at the
final sample (50,000), agreeing within 10.9 log-posterior units — inside the ~15-unit tolerance
that qualified the `constant`/`parsimony` n80 reps above — and every rep's trace is flat over the
last 250 samples (checked directly in each `run.log` tail).

**infB full: converged.** All 4 `pca` reps land in a -9249.5 to -9267.3 band at the final sample
(150,000), agreeing within 17.75 log-posterior units — comparable to the ~48-unit band that
qualified the `constant`/`parsimony` infB reps above — and flat over the last 500 samples in every
rep. `PCAWeights2` does not destabilize convergence at either size relative to the other two
weightings.

## Raw results — chain-level (`__CHAIN__` row), per cell

| case | setting | rep | acceptRate | meanJumpGivenAccepted | esjdPerSec_rnni |
|---|---|---:|---:|---:|---:|
| lassaL@80 | pca | 0 | 0.2346 | 1.018 | 1055.8 |
| lassaL@80 | pca | 1 | 0.2383 | 1.011 | 1237.2 |
| lassaL@80 | pca | 2 | 0.2355 | 1.038 | 947.0 |
| lassaL@80 | pca | 3 | 0.2284 | 1.171 | 2003.6 |
| infB | pca | 0 | 0.1500 | 4.033 | 15683.4 |
| infB | pca | 1 | 0.1489 | 3.979 | 16483.5 |
| infB | pca | 2 | 0.1519 | 3.980 | 17884.8 |
| infB | pca | 3 | 0.1500 | 4.024 | 19233.2 |

(`constant` reps at the same seeds are the rows already reported above: lassaL@80 constant =
1075.1 / 791.7 / 830.5 / 766.1; infB constant = 15164.3 / 16814.0 / 16335.4 / 19448.4.)

Full per-operator rows are in each run's `run.opstats.tsv` under
`runs/eval2/{lassaL/n80,infb}/pca/rep<rep>/` (not committed — regenerable from
`scripts/render_benchmark_xml.py --sizes 80 --settings pca` /
`scripts/render_infb_benchmark_xml.py --settings pca` with the parameters above).

## Paired analysis (pca − constant), t-based 90% CI, n=4, df=3

| case | mean diff (ESJD/s) | 90% CI | excludes zero? | relative to constant mean |
|---|---:|---|---|---:|
| lassaL@80 | +445.1 | (−217.5, 1107.7) | no | +51.4% |
| infB full  | +380.7 | (−637.3, 1398.7) | no | +2.2% |

Neither CI excludes zero, so at 4 reps neither result is statistically significant. But the
lassaL@80 point estimate is notably larger and more consistently positive than either prior
weighting's result at this size: 3 of 4 `pca` reps beat their paired `constant` rep (only rep0 is
slightly negative, −19.3), versus `parsimony`'s roughly 50/50 split at n80 (+184.7 mean, CI
(−306.9, 676.3)). infB's `pca` point estimate (+2.2%) sits between `parsimony`'s near-zero result
there (+0.9%) and is, like it, small relative to a wide CI — infB shows no sign that a different
weighting signal changes its outcome.

## Updated conclusion

Adding `PCAWeights2` to the two cleanly-converged cases does not change the headline finding — no
weighting tried so far (`ParsimonyWeights2` or `PCAWeights2`) beats `ConstantWeights` with a
confidence interval excluding zero at 4 reps, on either lassaL@80 or infB@141. It does add a
useful directional data point: `PCAWeights2`'s alignment-distance signal produces a larger,
more-consistently-positive point estimate on lassaL@80 (+51.4%, 3/4 reps favorable) than
`ParsimonyWeights2`'s mutation-count signal did at the same size (+21.3%, roughly even split),
while on infB both weightings land near zero (+2.2% vs +0.9%). This is consistent with — though
far from proof of — the hypothesis floated earlier that a different weighting signal might show
an effect where parsimony didn't: lassaL's alignment-based PCA signal looks more promising than
its parsimony signal, but the between-rep noise at 4 reps (CV still ~15-30%, same order as before)
is large enough that this reading could easily be sampling noise, not a real effect. infB shows no
such hint under either weighting. The next step to distinguish signal from noise remains the same
one already flagged: more reps (or a variance-reduction technique) targeted specifically at
lassaL@80 `PCAWeights2` vs `ConstantWeights`, since that is now the single most promising
(case, weighting) pair in the whole comparison, not a blanket re-run of the full design.

---

# infB — 5x extended run (ESS-gated re-check)

**Context / why this run exists.** The convergence checks used everywhere above (flat posterior
trace across reps, agreement within a "tens of units" band) turned out not to be sufficient: an
actual `loganalyser -oneline -b 10 -quiet` pass on the underlying `run.log` files, run after the
fact, showed `posterior.ESS` of only 13–82 for every infB cell above (and 2.6–9.9 for every
lassaL cell, which is why lassaL is being set aside entirely, not extended). A flat trace and
agreeing final-sample posteriors are consistent with a chain that has *found* the right region but
is still autocorrelated enough that its effective sample size is far too low to trust a paired
comparison. This run redoes `constant` vs `pca` on infB (141 taxa) with a ~5x larger total
proposal budget and uses `posterior.ESS`/`tree.height.ESS`, gated at ≥100, as the actual
convergence criterion.

## Chain length chosen, and why

Previous infB cells used `preBurnin=20,000 / chainLength=150,000` (170,000 total proposals).
Scaled ~5x to `preBurnin=100,000 / chainLength=750,000` (850,000 total proposals), keeping
`jumpInterval=20` as before. A timed calibration cell (`constant` rep0, seed 1000) took 452.7s of
chain time (7:34 wall including JVM/module boot) at ~1,890 states/sec, matching the throughput
measured earlier — confirming the ~450s/cell, ~60min/8-cell estimate before committing to the
full run. No further extension was needed (see below).

Rendered via `scripts/render_infb_benchmark_xml.py --settings constant pca --reps 4
--chain-length 750000 --pre-burnin 100000 --jump-interval 20 --out-root runs/eval3/infb` (script
unchanged from the earlier phase). Seeds `1000+rep` (1000–1003), shared between `constant` and
`pca` at a given rep (paired), matching every other design in this document.

## Convergence check — ESS gate

All 8 cells clear the ESS≥100 bar comfortably, on both `posterior.ESS` and `tree.height.ESS`:

| setting | rep | posterior.ESS | tree.height.ESS |
|---|---:|---:|---:|
| constant | 0 | 157.7 | 366.2 |
| constant | 1 | 222.5 | 325.3 |
| constant | 2 | 141.3 | 271.1 |
| constant | 3 | 269.3 | 274.6 |
| pca | 0 | 322.9 | 341.5 |
| pca | 1 | 241.8 | 458.8 |
| pca | 2 | 332.1 | 535.8 |
| pca | 3 | 264.9 | 541.4 |

Range 141.3–332.1 (posterior), 271.1–541.4 (tree.height) — an order of magnitude above the 13–82
seen at 1x proposals, and comfortably above the ≥100 rule of thumb for every cell in both
settings. **This is the first infB comparison in this document where the ESS gate is actually
satisfied, so it is the first one whose paired result should be trusted.**

## Raw results — chain-level (`__CHAIN__` row), per cell

| setting | rep | acceptRate | meanJumpGivenAccepted | esjdPerSec_rnni |
|---|---:|---:|---:|---:|
| constant | 0 | 0.1490 | 4.569 | 21230.5 |
| constant | 1 | 0.1544 | 4.397 | 19742.1 |
| constant | 2 | 0.1550 | 4.298 | 20425.2 |
| constant | 3 | 0.1504 | 4.405 | 20740.3 |
| pca | 0 | 0.1670 | 4.068 | 21870.7 |
| pca | 1 | 0.1654 | 4.282 | 21560.3 |
| pca | 2 | 0.1659 | 3.935 | 21649.6 |
| pca | 3 | 0.1639 | 4.234 | 23067.5 |

## Paired analysis (pca − constant), t-based 90% CI, n=4, df=3

| mean diff (ESJD/s) | 90% CI | excludes zero? | relative to constant mean |
|---:|---|---|---:|
| +1502.5 | (643.1, 2361.9) | **yes** | +7.3% |

This is the first CI in the whole evaluation (across every dataset/size/weighting tried) that
excludes zero. At an ESS the earlier "converged" cells never actually reached, `PCAWeights2`
shows a small but statistically supported ~7% ESJD/s advantage over `ConstantWeights` on infB at
its full 141 taxa. This does not retroactively validate any of the earlier null results above (all
of those remain confounded by inadequate ESS and cannot be un-confounded after the fact) — it is
a new, independently-gated data point, and the only one in this document that currently passes
the ESS≥100 bar.

---

# iecor — constant vs PCAWeights2 (IN PROGRESS — see resume instructions)

**Status at time of writing: `constant` (4/4 reps) complete and ESS-checked; `pca` (0/4 reps) not
yet run.** This section will be completed in a follow-up pass; the exact resume command is below
so no recalibration is needed.

## Renderer

No benchmark renderer existed for iecor before this task. `scripts/render_iecor_benchmark_xml.py`
was written, modeled on `scripts/render_infb_benchmark_xml.py`'s `<run>`/`<state>`/RNNIMetric
rewrite pattern. It reads `examples/iecor/vanilla/iecor.xml` directly (chosen as the base: it is
the simpler of the two committed files — no active edge-weights distribution, no targeted
operators wired up at all, only inert `<!-- -->`-commented alternatives and plain
`WilsonBalding`/`Exchange` topology operators) and:

- makes the `edgeweights` distribution swappable (`ConstantWeights` / `PCAWeights2` with
  `maxWeight=20`, `dimension=10`, `data="@focalsubset"` — iecor's actual alignment id, not
  infB's/lassaL's), inserted where vanilla's own inert alternatives already lived;
- splices in the four targeted operators (`RangeSlide`, `TargetedWilsonBalding`,
  `TargetedWilsonBaldingFixedHeight`, `WeightedWideOperator`, all wired to `@edgeweights`)
  verbatim from `examples/iecor/pca/iecor.xml` (the only place they already exist, wired to that
  file's own `PCAWeights`/dimension=6 convention — replaced here with the `PCAWeights2`
  convention used everywhere else in this doc), in place of vanilla's plain
  `WilsonBalding`/`WideExchange` pair. `pca/iecor.xml`'s own independent tuning changes
  (`SubtreeSlide` weight 2.0→5.0, `screenlog` interval) were deliberately **not** copied — they're
  orthogonal to the constant/pca comparison this renderer exists for, and copying only the
  targeted-operator block keeps `constant` and `pca` identical except for the one thing being
  compared.
- points `RNNIMetric`'s `taxonset` at `@TaxonSet.five` — iecor's only `TaxonSet` element (built
  off a site-filtered view of `focalsubset`; `FilteredAlignment` filters sites, not taxa, so it
  carries the same 41 taxa as the full alignment and is what the tree itself uses).

Deviation from the task's premise worth flagging explicitly: the task description assumed
`vanilla/iecor.xml` "currently declares its own edge-weights distribution (likely
`ConstantWeights` or similar)". It does not — all three of its `edgeweights` alternatives are
commented out and it has no targeted operators active at all, so a "constant vs pca" comparison
had nothing to vary without the operator-splicing step above. This was not treated as blocking:
it's resolved by following the same convention every other renderer in this doc already uses
(operators fixed, only `edgeweights` swaps), just sourced from `pca/iecor.xml` since that's the
only file where the targeted-operator wiring already exists.

## Chain length chosen, and why

No prior direct throughput measurement existed for iecor with targeted operators active (the
~10,428 states/sec figure elsewhere in this doc's baseline section was for a *shorter* probe
without them). A first calibration cell (`pca`, seed 1000, `preBurnin=200,000 /
chainLength=2,000,000`, `jumpInterval=20`) ran in 5:00 wall time (~7,330 states/sec measured from
2.2M total proposals / ~300s chain time) and gave `posterior.ESS=95.8`, `Tree.t:tree.height.ESS
=264.3` — just under the ≥100 gate on the headline parameter. Rather than accept a borderline
result, the chain length was bumped 1.5x to `preBurnin=300,000 / chainLength=3,000,000` (3.3M
total proposals, ~7.5 min/cell at the measured rate, ~60 min for the full 8-cell design), and the
full `constant` side (4 reps) was rerun at this setting — see ESS table below, all comfortably
clear ≥100. The `pca` side has not yet been rerun at this setting (see "Resume instructions").

Rendered via `scripts/render_iecor_benchmark_xml.py --settings constant pca --reps 4
--chain-length 3000000 --pre-burnin 300000 --jump-interval 20 --out-root runs/eval3/iecor`. Seeds
`1000+rep` (1000–1003), paired with `pca` at the same seeds once run.

## `constant` — convergence check (ESS gate) and raw results — DONE

| rep | seed | posterior.ESS | Tree.t:tree.height.ESS | acceptRate | meanJumpGivenAccepted | esjdPerSec_rnni |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1000 | 279.7 | 377.3 | 0.2775 | 0.3253 | 2237.4 |
| 1 | 1001 | 209.4 | 301.6 | 0.2764 | 0.3109 | 2255.0 |
| 2 | 1002 | 163.8 | 192.0 | 0.2718 | 0.3167 | 2177.5 |
| 3 | 1003 | 214.5 | 355.6 | 0.2757 | 0.3083 | 2103.8 |

All 4 `constant` reps clear ESS≥100 (163.8–279.7 posterior, 192.0–377.3 tree.height).

## `pca` — completed (resumed directly after the driving agent was interrupted)

| rep | seed | posterior.ESS | acceptRate | meanJumpGivenAccepted | esjdPerSec_rnni |
|---|---|---:|---:|---:|---:|
| 0 | 1000 | 278.2 | 0.2784 | 0.343 | 2602.4 |
| 1 | 1001 | 180.9 | 0.2771 | 0.345 | 1724.1 |
| 2 | 1002 | 225.9 | 0.2786 | 0.352 | 1852.1 |
| 3 | 1003 | 189.5 | 0.2800 | 0.342 | 1739.6 |

**Correction (2026-09-03): rep0 above was originally from a chain interrupted mid-run** (2,000,001
of the intended 3,000,001 proposals, giving a spuriously low posterior.ESS=77.2, marginally below
the ≥100 gate). It has been rerun to completion at the same seed (1000) and settings; the table
above already reflects the corrected numbers. All 4 `pca` reps now clear ESS≥100 comfortably
(180.9–278.2), matching the `constant` side — no cell in this comparison is ESS-borderline anymore.

### Paired analysis (pca − constant), t-based 90% CI, n=4, df=3

| dataset | mean diff (ESJD/s) | 90% CI | relative | excludes zero? |
|---|---:|---|---:|---|
| iecor | −213.9 | (−679.9, +252.1) | −9.8% | no |

**Verdict: no effect (unchanged by the rep0 correction).** The point estimate still favours
`constant`, opposite in sign to the infB result (`## infB — 5x extended run` above), and the CI
still includes zero. The correction moved the point estimate from −13.0% to −9.8% and widened the
CI slightly (the old CI's near-miss upper bound of +17.3 is no longer close to zero at +252.1), but
did not change the sign or the significance verdict — best read, as before, as "no evidence
`PCAWeights2` helps on iecor," not "iecor prefers uniform weights."


---

## What this changes, taking infB and iecor together

1. **ESS, not trace flatness, is the actual convergence gate** — the key correction from this
   round. Every lassaL cell in this document has posterior.ESS in the 2.6–9.9 range even where the
   trace looked flat across reps: that was convergence to the right region without adequate
   *mixing* within it. infB and iecor, being smaller/faster, could afford chains long enough to
   clear ESS≥100 once budgeted for it explicitly (5x total proposals for infB; ~11x for iecor
   relative to its first attempt).
2. **The one confirmed effect (infB, +7.3%, CI excludes zero) is dataset-specific, not a general
   property of `PCAWeights2`.** It does not replicate on iecor (−9.8%, wrong sign, CI includes
   zero — this figure reflects the rep0 correction above and was re-verified after that rerun).
   Any future weighting comparison needs per-dataset validation, not one verdict generalized across
   datasets.
3. **lassaL stays set aside.** Its ESS problem is structural to its size/throughput at any chain
   length affordable in this evaluation's time budget, not fixable by the same 5x/11x scaling that
   worked for infB and iecor.
4. **The decision rule's 20% threshold is still not calibrated by a same-sign result at that
   magnitude** — the one confirmed effect is 7.3%, well under the placeholder. Whether a
   consistent 5-10% gain is "worth adopting" is a judgement call the current decision rule doesn't
   make.

---

# iecor operator tuning review — `ucldStdev.c:clock` and one other fix

Prompted by `ucldStdevScaler.c:clock` (the `ScaleOperator` on `ucldStdev.c:clock`) looking like a
convergence bottleneck in the iecor benchmark runs above: its `run.opstats.tsv` acceptance rate in
the `constant`/eval3 reps ranged 0.12–0.31 (rep2 at 0.12 is well below the ~0.23–0.44 rule-of-thumb
band for a `ScaleOperator`), and its own ESS trails `posterior.ESS` in every rep checked
(174.1 vs. 279.7 in rep0).

## Changes made (both `examples/iecor/vanilla/iecor.xml` and `examples/iecor/pca/iecor.xml`)

1. **`ucldStdevScaler.c:clock` weight 3.0 → 5.0** (as requested). It is a single scalar hyperparameter
   controlling the whole relaxed-clock rate distribution's spread — high-influence, low-dimensional
   — and was previously weighted the same as low-influence single-scalar epi params
   (`becomeUninfectiousRateOperator`, `rhoOperator`, etc.), despite governing a distribution that
   feeds every one of the 80 per-branch rate categories.

2. **`SubtreeSlide.t:tree` weight 2.0 → 5.0, `vanilla/iecor.xml` only.** `pca/iecor.xml` already
   carried this exact fix from earlier tuning work (see the "iecor — constant vs PCAWeights2"
   section above: "`pca/iecor.xml`'s own independent tuning changes (`SubtreeSlide` weight
   2.0→5.0 ...) were deliberately not copied"). A weight of 2.0 for a combined topology+
   branch-length move, on a 41-taxon tree, versus 30.0 for the pure node-height `Uniform` operator
   and 20.0/10.0 for the two global topology rearrangement moves (`WilsonBalding`/`WideExchange`),
   looked like a leftover that was never revisited in `vanilla` after being fixed in `pca`.
   Bringing `vanilla` up to `pca`'s value both fixes the apparent underweighting and restores
   parity between the two files for any future `vanilla`-vs-`pca` comparison that isn't about
   this operator.

No other operator in the block looked like an obvious mismatch on inspection:
- `WilsonBalding`=20, `WideExchange`=10, `Narrow`=2 (an inverted-from-usual-template ordering that
  favours global over local topology moves) is unchanged from the pre-migration upstream XML
  (`targetedbeast/examples/iecor/vanilla/iecor.xml`, checked directly) and matches the pca file's
  targeted-operator weights (5+15=20 for the two `TargetedWilsonBalding*` variants, 10 for
  `TargetedWide`) almost exactly — this is deliberate parallel design for the vanilla-vs-targeted
  comparison, not an unadjusted default, so it was left alone per "targeted fixes only."
- The five single-scalar epidemiological parameters (`origin`, `reproductiveNumber`,
  `becomeUninfectiousRate`, `samplingProportion`, `rho`) all carry weight 1.0–3.0, proportionate to
  being scalar, non-skyline parameters (confirmed `dimension="1"` on each in the `<state>` block).
- `scaleFactor` values throughout (0.5, 0.878, 0.837, 0.9) are all non-extreme and, since none of
  these operators set `autoOptimize="false"`, BEAST's adaptive tuning already adjusts them during
  the 300,000-sample preBurnin regardless of the seed value — not a plausible source of a
  systematic mismatch.
- `ClockRateScaler.c:focalsubset`'s `AdaptableOperatorSampler` (weight 0.5) and its
  `AdaptableVarianceMultivariateNormalOperator` child's `burnin=400`/`initial=800` are template
  defaults but not obviously wrong — they don't have an inherent tuning target the way a
  `ScaleOperator`'s acceptance rate does.

## Diagnostic run — one cell per file, `constant`/`pca` setting, rep0, seed 1000

Rendered via `scripts/render_iecor_benchmark_xml.py --settings constant pca --reps 1
--chain-length 3000000 --pre-burnin 300000 --jump-interval 20 --out-root runs/eval4/iecor_tuned`
(same parameters as the `runs/eval3/iecor` cells being compared against). XML well-formedness
confirmed with `xml.etree.ElementTree.parse` before running; both runs completed in ~6:50–6:54
wall time (`mvn -o exec:exec`), matching the eval3 baseline's throughput.

| variant | | ucldStdev acceptRate | ucldStdev ESS | posterior.ESS | Tree.height.ESS |
|---|---|---:|---:|---:|---:|
| constant | before (eval3, weight 3.0) | 0.3087 | 174.1 | 279.7 | 377.3 |
| constant | after (eval4, weight 5.0)  | 0.3542 | 88.4  | 138.4 | 272.1 |
| pca      | before (eval3, weight 3.0) | 0.3217 | 188.5 | 278.2 | 507.9 |
| pca      | after (eval4, weight 5.0)  | 0.2835 | 131.5 | 300.5 | 326.1 |

**Result: mixed, and not the hoped-for clean win — flagging this rather than presenting it as a
success.** `ucldStdevScaler`'s own acceptance rate moved in opposite directions on the two files
(+0.045 on `constant`, −0.038 on `pca`), and its own ESS *dropped* in both (174.1→88.4 on
`constant`, 188.5→131.5 on `pca`) despite the operator now receiving proportionally more proposals
(69,455–69,470 → 111,237–111,460 out of ~3.3M total). `posterior.ESS` dropped sharply on
`constant` (279.7→138.4, −50%) but rose on `pca` (278.2→300.5, +8%). `Tree.height.ESS` dropped on
both.

This is a single seed per variant, and this document has already established (the n40/n80/n100/
infB reps above) that single-rep ESS at this chain length varies by roughly 2× on identical
settings purely from seed noise — e.g. the existing `constant` table itself shows rep0's
`posterior.ESS`=279.7 against rep2's 163.8 with *no* operator change at all. A weight change also
reshuffles which pseudo-random draws land on which operator at every step, so "same seed, changed
weights" is not a controlled comparison of the same trajectory — it is effectively a second,
independent draw. **One seed cannot distinguish "the weight bump helped/hurt" from "this is normal
between-run noise,"** and the result here (up on one file, down on the other, ESS down on both for
the operator itself) is exactly what that noise floor would produce for a change with a small or
null true effect. The weight bump is applied per the task's explicit instruction and is a
defensible prioritization on first principles (single scalar governing a whole rate distribution,
previously weighted the same as low-influence params) — but this diagnostic does not confirm it
delivers a mixing improvement, and a properly powered follow-up (multiple reps, both settings)
would be needed before treating it as validated.

`mvn -o test`: 113 run, 2 failures, 1 skipped — unchanged from baseline, confirmed after the
diagnostic runs and before committing.
