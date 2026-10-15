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

---

# Weight-transform feature (`GridTuner`-adaptive `PowerOffsetTransform`) vs `PCAWeights2` alone

`PCAWeights2` (+7.3% over `ConstantWeights`, CI excludes zero — "infB — 5x extended run" above)
is the reference baseline every improvement is measured against. This section asks whether
layering the weight-transform feature (`targetedbeast.weighttransform.PowerOffsetTransform` +
`GridTuner`, see `docs/WeightTransformAdaptation.md`) on top of `PCAWeights2` beats
`PCAWeights2` alone, using the same paired design, seeds, and ESS gate as every result above.

## What was built

`scripts/render_infb_benchmark_xml.py` gained a `"pca_grid"` setting: the identical
`PCAWeights2` distribution as the existing `"pca"` setting (`maxWeight="20"`, `dimension="10"`),
with a shared `PowerOffsetTransform` wired into every targeted operator that accepts one
(`TargetedWide`, `targetedWilsonBalding`, `targetedWilsonBaldingFixedHeight`,
`targetedRangeSlide`) via `targetWeightTransform`. `examples/infb_HA_141/template.xml` gained the
same `{target_weight_transform}` / `{target_weight_transform_ref}` / `{target_weight_transform_log}`
slots `examples/lassaL_kli22/template.xml` already declares, filled the same way (empty strings
for every other setting, so `constant`/`parsimony`/`pca` render byte-identical to before — checked
directly by diffing a re-rendered `pca` cell against a fresh one).

**Parameters, and where they deviate from the lassaL defaults in `scripts/generate_xmls.py`'s
`_grid_transform`:**

- `exponent="2.0" offset="0.0" maxExponent="24.0"` on the transform, `GridTuner` with
  `exponents="8 12 16 24"`, `explorationProb="0.1"` — identical to lassaL, no reason to deviate.
- `<reward idref="likelihood"/>` — infB's `likelihood` `CompoundDistribution` id matches lassaL's
  exactly (checked directly against `examples/infb_HA_141/template.xml`), so no adaptation needed
  there either.
- **`burnin="5000"`, not lassaL's `10000`.** lassaL has no separate pre-burnin phase, so its
  `GRID_BURNIN` has to absorb the chain's entire transient by itself (its own comment: "at 30k
  steps the chain is still descending thousands of log units per proposal"). infB's `thorough`
  tier is different: core `MCMC.doLoop` only calls `operator.optimize()` when `sampleNr >= 0`
  (`MCMC.java:483-485`), i.e. strictly *after* `BenchmarkMCMC`'s `preBurnin` (100,000 proposals)
  has already run to completion — and the "infB — 5x extended run" section above already
  established, at this exact `preBurnin`/`chainLength`, that the posterior trace is flat within a
  few hundred post-preBurnin samples and clears `posterior.ESS`≥100 comfortably. So `GridTuner`'s
  own burn-in does not need to re-discard a steep transient the way lassaL's does; `5000` is a
  modest safety margin against residual noise right after `preBurnin` ends, not a burn-in-length
  estimate from scratch.

## Smoke check (`fast` tier, infB, `preBurnin=20000/chainLength=150000`, seed 1000)

XML parsed and rendered correctly (verified both the raw template — which, like lassaL's, is not
well-formed standalone XML until `.format()`-filled, confirmed by checking lassaL's template
fails the same way — and the rendered output, which is). The run completed without error in
84s chain time (1:26 wall including JVM/module boot).

**The tuner adapts.** `targetWeightTransform.leadingExponent` in the trace log (added via
`<log idref="targetWeightTransform"/>`, following lassaL's pattern) starts at the initial default
choice (8.0, before every grid point has cleared `GridTuner`'s own `learnin=200` floor), jumps to
24.0 around proposal ~11,000 once the burn-in margin clears, cycles down through 16.0, and
stabilizes at a leading exponent of 12.0 (highest mean reward, ~0.115) for the back half of the
150,000-proposal chain — clearly away from both the initial `exponent="2.0"` transform default
and the tuner's own starting grid choice, not stuck.

## Real comparison (`thorough` tier, `preBurnin=100000/chainLength=750000`, 4 reps, seeds 1000-1003)

Rendered via `scripts/render_infb_benchmark_xml.py --settings pca_grid --reps 4
--chain-length 750000 --pre-burnin 100000 --jump-interval 20 --out-root runs/eval5/infb`. Each
rep ran as a blocking `mvn -o exec:exec` invocation (never backgrounded), 7:03-7:27 wall time
per rep, ~29 minutes total for all 4. Paired against the **already-recorded** `pca` cells from
the "infB — 5x extended run" section above (same seeds 1000-1003, not rerun).

### ESS gate

| rep | seed | posterior.ESS | tree.height.ESS |
|---:|---:|---:|---:|
| 0 | 1000 | 158.0 | 438.5 |
| 1 | 1001 | 248.5 | 450.6 |
| 2 | 1002 | 158.5 | 398.5 |
| 3 | 1003 | 205.1 | 231.4 |

All 4 cells clear `posterior.ESS`≥100 (158.0-248.5) and `tree.height.ESS`≥100 (231.4-450.6) —
same gate, same margin as the `pca`/`constant` cells they are compared against.

### Raw results — chain-level (`__CHAIN__` row), per cell

| setting | rep | seed | acceptRate | meanJumpGivenAccepted | esjdPerSec_rnni |
|---|---:|---:|---:|---:|---:|
| pca_grid | 0 | 1000 | 0.1583 | 3.689 | 18421.4 |
| pca_grid | 1 | 1001 | 0.1638 | 3.810 | 19561.8 |
| pca_grid | 2 | 1002 | 0.1670 | 3.681 | 18608.8 |
| pca_grid | 3 | 1003 | 0.1642 | 3.938 | 19665.8 |

(`pca` reps at the same seeds are the rows already reported in "infB — 5x extended run" above:
0.1670/4.068/21870.7, 0.1654/4.282/21560.3, 0.1659/3.935/21649.6, 0.1639/4.234/23067.5.)

The transform's own logged trace (`targetWeightTransform.leadingExponent`, per-rep posterior
median over the whole post-burnin chain) settled at **8.0 in all 4 reps** — the low end of the
grid, not the 12.0 the smoke check happened to settle on at a shorter chain length/different
seed. Two reps (1 and 3) transiently favoured 12.0 for part of the run (mean 8.2-8.5, median
8.0) before reward estimates pulled the leader back to 8.0; the other two (0 and 2) sat at 8.0
essentially throughout. No rep's leader approached the grid's ceiling (24.0).

### Paired analysis (pca_grid − pca), t-based 90% CI, n=4, df=3

| mean diff (ESJD/s) | 90% CI | excludes zero? | relative to pca mean |
|---:|---|---|---:|
| −2972.6 | (−3766.2, −2179.0) | **yes** | −13.5% |

## Verdict

**No — at this sample size, layering the `GridTuner`-adaptive `PowerOffsetTransform` on top of
`PCAWeights2` makes infB's mixing significantly worse, not better.** The 90% CI excludes zero on
the negative side: `pca_grid` loses ~13.5% chain-level ESJD/second relative to `PCAWeights2`
alone, in all 4 reps individually (18421-19666 vs 21560-23068), with no overlap between the two
settings' individual-rep ranges. This is the second CI in this document (after the `pca` vs
`constant` result above) to exclude zero, and the first negative one: the weight-transform
feature, at the parameters used here, does not merely fail to help — it actively hurts. The
tuner did adapt (smoke check) and did not pin at the exponent ceiling the way an
acceptance-only reward is documented to (`docs/WeightTransformAdaptation.md`), but it also did
not climb toward higher pointiness on infB the way it did in the smoke-check run; it settled at
the grid's lowest point in every real-comparison rep. One plausible reading, consistent with
`docs/WeightTransformAdaptation.md`'s own caveat that "maximizing acceptance is not the same as
maximizing mixing": `PCAWeights2`'s target weights on infB may already be well-shaped, and
reshaping them toward the historical squaring default (`exponent=2.0`) or pointier still (the
grid never explores below 8) discards signal the untransformed `PCAWeights2` distribution
carried. This result should not be read as evidence against the weight-transform feature in
general — only against this specific combination (infB, `PCAWeights2`, this grid and burn-in) at
n=4 — but it is a clean, ESS-gated, statistically supported result at this sample size: on this
dataset, plugging the feature in as configured here is a regression, not an improvement.

---

# Weight-transform feature, re-tested with a grid spanning the operator's own default exponent

**Why this run exists.** The `pca_grid` result above used `GridTuner` exponents `[8, 12, 16, 24]`
— confirmed after the fact to be an old leftover from unrelated exploration, not a deliberate
choice — which only ever let the tuner explore exponents pointier than the targeted operators'
own hardcoded default (`exponent="2.0"`, plain squaring). It could never test flatter-than-default
or the default itself, so the −13.5% result above cannot distinguish "the transform hurts" from
"this specific grid never let the tuner find the default/flatter region that would have helped."
This section reruns the same comparison with `exponents="0.5 1 2 4"` — flatter-than-default (0.5),
raw/neutral (1), the historical default (2), and moderately pointier (4) — against the same
already-recorded `pca` cells, so `pca_grid` and this new `pca_grid_low` setting are both judged
against one shared baseline.

## What was built

`scripts/render_infb_benchmark_xml.py` gained a `"pca_grid_low"` setting: identical
`PCAWeights2` distribution to `"pca"`/`"pca_grid"`, wired to a second `PowerOffsetTransform`/
`GridTuner` variant (`"grid_low"`) that is otherwise identical to `"grid"` (`exponent="2.0"
offset="0.0"`, `explorationProb="0.1"`, `burnin="5000"`, `<reward idref="likelihood"/>` — the
`burnin=5000` reasoning from the `pca_grid` section above applies unchanged, since it depends only
on infB's own convergence behaviour post-`preBurnin`, not on which grid is being explored) except:

- `exponents="0.5 1 2 4"` in place of `"8 12 16 24"`.
- `maxExponent="4.0"` in place of `"24.0"`. `maxExponent` must be ≥ the grid's largest value (4);
  it is set to exactly 4, matching the grid's own top value, the same pattern the original `grid`
  setting used (`maxExponent == max(GRID_EXPONENTS)`, i.e. also no headroom above its own grid's
  ceiling) — kept for consistency rather than introducing an unexamined asymmetry between the two
  transform variants.

## Smoke check (`fast` tier, infB, `preBurnin=20000/chainLength=150000`, seed 1000)

XML parsed (`xml.etree.ElementTree`) and rendered correctly. Run completed without error in 82.9s
chain time (1:23 wall including JVM/module boot).

**The tuner explores the new grid.** The trace log's `targetWeightTransform.reward.*` columns
confirm the four reward trackers are exactly `0.5`, `1.0`, `2.0`, `4.0` (not the old `8/12/16/24`).
`targetWeightTransform.leadingExponent` starts at `0.5` (the grid's lowest value, the same
"smallest grid point first" pattern the `pca_grid` smoke check showed with `8.0`), then visits
`1.0`, back to `0.5`, then `4.0`, back to `0.5`, then `4.0` again — three of the four grid values
visited within a 150,000-proposal chain, clearly moving among grid points rather than stuck at the
initial default or at `2.0`.

## Real comparison (`thorough` tier, `preBurnin=100000/chainLength=750000`, 4 reps, seeds 1000-1003)

Rendered via `scripts/render_infb_benchmark_xml.py --settings pca_grid_low --reps 4
--chain-length 750000 --pre-burnin 100000 --jump-interval 20 --out-root runs/eval5_low/infb`.
Each rep ran as a blocking `mvn -o exec:exec` invocation, sequential (not parallelized — infB's
likelihood already uses `useThreads="true"`, so concurrent chains would contend for the same core
pool), 407.9-435.7s chain time per rep (~6:48-7:17 wall including JVM/module boot), ~28.4 minutes
chain time total for all 4. Paired against the **already-recorded** `pca` cells from the "infB —
5x extended run" section above (same seeds 1000-1003, not rerun).

### ESS gate

| rep | seed | posterior.ESS | tree.height.ESS |
|---:|---:|---:|---:|
| 0 | 1000 | 211.0 | 384.7 |
| 1 | 1001 | 311.2 | 349.1 |
| 2 | 1002 | 363.7 | 425.3 |
| 3 | 1003 | 223.1 | 407.4 |

All 4 cells clear `posterior.ESS`≥100 (211.0-363.7) and `tree.height.ESS`≥100 (349.1-425.3) — same
gate, same margin as every other `thorough`-tier result in this document.

### Raw results — chain-level (`__CHAIN__` row), per cell

| setting | rep | seed | acceptRate | meanJumpGivenAccepted | esjdPerSec_rnni |
|---|---:|---:|---:|---:|---:|
| pca_grid_low | 0 | 1000 | 0.1626 | 4.186 | 22746.6 |
| pca_grid_low | 1 | 1001 | 0.1615 | 4.199 | 23186.4 |
| pca_grid_low | 2 | 1002 | 0.1640 | 4.159 | 21409.6 |
| pca_grid_low | 3 | 1003 | 0.1611 | 4.151 | 22078.8 |

(`pca` reps at the same seeds are the rows already reported in "infB — 5x extended run" above:
0.1670/4.068/21870.7, 0.1654/4.282/21560.3, 0.1659/3.935/21649.6, 0.1639/4.234/23067.5.)

The transform's own logged trace (`targetWeightTransform.leadingExponent`, per-rep posterior
median over the whole post-burnin chain) settled at **4.0 — the grid's ceiling — in all 4 reps**,
the mirror image of `pca_grid`'s result (which settled at 8.0, its grid's floor, in all 4 reps).
Per-rep posterior means were 3.46, 3.14, 4.00, and 3.31 — all below the median of 4.0, meaning
each chain spent real time exploring lower grid points (0.5/1/2) before its reward estimates
pulled the leader up to 4.0, not that 4.0 was fixed from the start. No rep's leader ever settled at
the flatter-than-default (0.5) or default (2.0) points that motivated this rerun, nor at the
grid's raw/neutral midpoint (1.0).

### Paired analysis (pca_grid_low − pca), t-based 90% CI, n=4, df=3

| mean diff (ESJD/s) | 90% CI | excludes zero? | relative to pca mean |
|---:|---|---|---:|
| +318.3 | (−1047.4, 1684.0) | no | +1.4% |

## Verdict

**No effect, in either direction.** The 90% CI includes zero — `pca_grid_low` is statistically
indistinguishable from `PCAWeights2` alone at 4 reps, with a small positive point estimate
(+1.4%). This is a materially different result from the high-grid `pca_grid` run: that comparison
had a CI excluding zero on the *negative* side (−13.5%), with `pca_grid`'s per-rep ESJD/s
(18421-19666) not overlapping `pca`'s (21560-23068) at all. Here, `pca_grid_low`'s per-rep range
(21410-23186) sits squarely inside/overlapping `pca`'s own range, and the point estimate is barely
above zero rather than clearly below it.

Read together with `pca_grid`, this replicates the earlier section's own tentative reading: the
regression previously observed was not an inherent property of "adding an adaptive power
transform to `PCAWeights2` on infB" — it was specific to a grid that only ever pushed the
transform pointier than the operators' built-in default. A grid that includes the default (2.0)
and flatter options (0.5) removes the harm (no more significant negative effect) but does not
produce a significant improvement either: the tuner, given the option to settle near the default
or flatten, instead settled at this grid's own ceiling (4.0) — moderately pointier than default,
but far short of the pointiness (`pca_grid`'s floor of 8.0) that hurt before. This is consistent
with the discussion in `docs/WeightTransformAdaptation.md` that pointiness beyond some
dataset-specific optimum stops helping and starts hurting mixing; on infB with `PCAWeights2`, that
optimum appears to sit somewhere in or near `[2, 4]`, not near `[8, 24]`, and (per this null
result) does not clearly beat leaving `PCAWeights2` untransformed at all. As with every other null
result in this document, this does not rule out a smaller true effect that 4 reps cannot resolve;
it rules out the specific large effect (positive or negative) that would be needed to justify
recommending the transform, at this grid, on this dataset, right now.

`mvn -o test`: 123 run, 2 failures, 1 skipped, 0 errors — unchanged from baseline, confirmed after
these runs and before committing.

---

# Weight-transform feature — reference-point checks and a grid spanning the gap

## Reference-point sanity check: `exponent=2, offset=0, optimise="false"`

Before trusting `pca_grid`/`pca_grid_low` any further, a cheap end-to-end check: a
`PowerOffsetTransform` pinned at the operators' own hardcoded default, with no tuner at all
(`optimise="false"`), should behave like plain `pca` — this is already proven at the unit level
(`WeightTransformTest`, "b=2, a=0 reproduces the current squaring"), but a pipeline bug in
rendering or wiring would not show up there. `scripts/render_infb_benchmark_xml.py` gained a
`"pca_fixed2"` setting for this (`PowerOffsetTransform exponent="2.0" offset="0.0"
optimise="false"`, no `tuner` element, wired only to `targetWeightTransform` as in every other
setting here).

One `fast`-tier cell (`preBurnin=20000/chainLength=150000`, seed 1000) run for `pca_fixed2` and
compared directly against the existing `pca` fast-tier cell at the same seed. Per-operator accept
counts for `targetedWilsonBalding` (the operator the transform is nested inside) are close but not
identical (1142 both runs — exact match on this operator specifically), while other operators
using the shared transform object drift apart (`TargetedWide`: 1833 vs 1794; `FixedHeight`: 1941 vs
1772) and the final log-likelihoods diverge (-9262.10 vs -9249.52 at sample 150,000). This is the
expected signature of a chaotic system given a correct implementation, not a bug: `Math.pow(x,
2.0)` and `x*x` are not bit-identical in Java's floating-point semantics, so a tiny last-bit
difference in one transformed weight eventually flips an accept/reject decision, and MCMC
trajectories diverge exponentially from there — the same reason two `constant` reps at different
seeds land on different posterior values despite sampling the same distribution. What matters is
that the two runs are statistically unremarkable relative to each other (similar acceptance rates,
similar posterior range, no crash, no NaN, no degenerate weight vector) — nothing here suggests a
wiring or math bug upstream of the chaotic-divergence point. This check does not replace a paired
`thorough`-tier comparison (not run, since the unit test already covers the underlying algebra and
this was only meant to rule out a *rendering*-level mistake cheaply), but it found none.

## A grid spanning the gap between `pca_grid` and `pca_grid_low`

**Why this run exists.** `pca_grid` (exponents 8/12/16/24) settled at its floor, 8.0, in all 4
reps. `pca_grid_low` (exponents 0.5/1/2/4) settled at its ceiling, 4.0, in all 4 reps. Two boundary
solutions on either side of the same gap point at an interior optimum somewhere in `(4, 8)` or
close to it — this grid, `exponents="2 3 4 6"` (`maxExponent="6.0"`), spans that gap directly and
retains the historical default (2) as an internal point, so a leader that lands away from *both*
edges (2 and 6) would confirm the earlier boundary solutions were genuine signal, not grid-geometry
artifacts. Setting name: `"pca_grid_mid"` in `scripts/render_infb_benchmark_xml.py`, otherwise
identical wiring to `pca_grid`/`pca_grid_low` (`burnin="5000"`, `explorationProb="0.1"`,
`<reward idref="likelihood"/>`, same `PCAWeights2` edge weights as every `pca*` setting).

### Smoke check (`fast` tier, seed 1000)

Ran cleanly in 84.7s chain time. Reward trace at the final sample: `reward.2.0=0.113`,
`reward.3.0=0.118`, `reward.4.0=0.123` (highest), `reward.6.0=0.102` (lowest) — an interior optimum
at 4.0 already visible at this budget, not pinned to either edge of `[2, 6]`.

### Real comparison (`thorough` tier, `preBurnin=100000/chainLength=750000`, 4 reps, seeds 1000-1003)

Each rep run as a blocking, sequential `mvn -o exec:exec` invocation (406-434s chain time each,
~27.5 minutes total), paired against the **already-recorded** `pca` cells from the "infB — 5x
extended run" section above (same seeds, not rerun).

#### ESS gate

| rep | seed | posterior.ESS | tree.height.ESS |
|---:|---:|---:|---:|
| 0 | 1000 | 154.0 | 425.5 |
| 1 | 1001 | 181.5 | 436.4 |
| 2 | 1002 | 294.1 | 469.7 |
| 3 | 1003 | 310.6 | 487.7 |

All 4 cells clear `posterior.ESS`≥100 (154.0-310.6) and `tree.height.ESS`≥100 (425.5-487.7).

#### Raw results — chain-level (`__CHAIN__` row), per cell

| setting | rep | seed | acceptRate | meanJumpGivenAccepted | esjdPerSec_rnni |
|---|---:|---:|---:|---:|---:|
| pca_grid_mid | 0 | 1000 | 0.1675 | 4.076 | 22040.6 |
| pca_grid_mid | 1 | 1001 | 0.1651 | 3.979 | 22712.0 |
| pca_grid_mid | 2 | 1002 | 0.1658 | 3.963 | 22595.7 |
| pca_grid_mid | 3 | 1003 | 0.1701 | 4.247 | 25139.5 |

(`pca` reps at the same seeds are the rows already reported in "infB — 5x extended run" above:
21870.7, 21560.3, 21649.6, 23067.5.)

Per-rep posterior-median leader (`targetWeightTransform.leadingExponent`, last 90% of the
post-burnin chain): rep0 = 4.0, rep1 = 4.0, rep2 = 3.0, rep3 = 4.0 — an interior point (3.0 or 4.0)
in every rep, never the grid's edges (2.0 or 6.0), matching the smoke check.

#### Paired analysis (pca_grid_mid − pca), t-based 90% CI, n=4, df=3

| mean diff (ESJD/s) | 90% CI | excludes zero? | relative to pca mean |
|---:|---|---|---:|
| +1084.9 | (164.7, 2005.1) | **yes** | +4.9% |

### Verdict

**Yes — this is the first positive, ESS-gated result for the weight-transform feature on infB.**
`pca_grid_mid` beats `PCAWeights2` alone by ~4.9% chain-level ESJD/second, 90% CI excluding zero on
the positive side, at n=4. Read together with the two grids that bracket it: `pca_grid` (8-24, all
4 reps pinned at the floor, −13.5%, CI excludes zero on the negative side) and `pca_grid_low`
(0.5-4, all 4 reps pinned at the ceiling, +1.4%, CI includes zero), the three results trace out a
coherent, unimodal-looking picture on this dataset: harm above ~8, no effect when the tuner is
capped at 4 without room to explore past it, and a real gain once the grid actually brackets the
interior region around 3-4 that both boundary results were pointing at. This reconciles the earlier
`pca_grid` regression with the design docs' framing (`docs/WeightTransformAdaptation.md`:
"pointiness beyond some dataset-specific optimum stops helping and starts hurting") rather than
contradicting it: the earlier grid never sampled the region where the actual optimum lives, and the
low grid clipped it at exactly the ceiling. This result should be read as dataset/grid-specific,
like every other result in this document — the next section raises an independent reason not to
treat it as fully settled yet.

## Cross-check: `AdaptableOperatorSampler` over fixed exponents

**Why this exists.** `GridTuner`'s reward is squared change in *log-likelihood* per proposal
(`<reward idref="likelihood"/>`) — a proxy for mixing, not the RNNI-distance ESJD that this whole
document's headline metric (`esjdPerSec_rnni`) actually measures. `AdaptableOperatorSampler`
(`beast.base.evolution.operator.AdaptableOperatorSampler`, vendored copy at
`src/main/java/targetedbeast/operators/AdaptableOperatorSampler.java`) is a structurally different,
more established mechanism: it selects among *distinct candidate operators* by
`acceptanceProb × (meanSquaredJump / runtime / parameterVariance)` (`getZ`,
`getOperatorCumulativeProbs`), and critically, with a `TreeMetric` supplied it can be pointed
directly at RNNI distance instead of likelihood. Wiring it up gives an independent read on the same
question `pca_grid_mid` just answered, using the metric that actually matters instead of
`GridTuner`'s likelihood proxy.

### What was built

A one-off diagnostic XML (not integrated into the renderer — this is a single cross-check, not a
repeatable benchmark setting): took a rendered `pca` `fast`-tier cell and replaced the
`targetedWilsonBalding` operator with an `AdaptableOperatorSampler` (`metric="@rnni"`, the same
`RNNIMetric` instance the benchmark's own RNNI logging already uses, `burnin="10000"`) wrapping
four fixed `TargetedWilsonBalding` candidates, each `weight="1"`, each with its own
`PowerOffsetTransform exponent="{2,3,4,6}.0" offset="0.0" optimise="false"` — the same four grid
points as `pca_grid_mid`, but each one pinned rather than tuned online, so the *sampler* is doing
the searching instead of `GridTuner`. Every other operator (including the other three
transform-bearing ones — `TargetedWide`, `targetedWilsonBaldingFixedHeight`, `targetedRangeSlide`)
was left as plain `pca` (no transform) so this isolates the question to
`targetedWilsonBalding` specifically, matching the scope the design doc puts this operator's target
path in.

### Run and result (`fast` tier, `preBurnin=20000/chainLength=150000`, seeds 1000 and 1001)

Both seeds ran cleanly (~80s chain time each). `storeToFile`'s JSON dump gives the sampler's
learned selection weights directly:

| seed | exponent 2 | exponent 3 | exponent 4 | exponent 6 |
|---:|---:|---:|---:|---:|
| 1000 | **0.331** | 0.265 | 0.202 | 0.202 |
| 1001 | **0.318** | 0.211 | 0.274 | 0.196 |

`mean_SS` (mean squared RNNI jump per proposal, the numerator of the sampler's own reward) is
monotonically *decreasing* in exponent in both seeds (seed 1000: 25.4 / 22.0 / 13.0 / 13.9; seed
1001: 33.1 / 19.9 / 22.7 / 20.0) — exponent 2, the least pointy option and the operators'
historical default, produces the largest per-proposal RNNI jump in both runs, and is the clear
leader by the sampler's own combined acceptance×jump score in both seeds. Exponent 6 is
last or tied-last in both.

### This disagrees with `GridTuner`

`GridTuner`'s likelihood-reward search (`pca_grid_mid`, above) converged on an *interior* optimum
at 3-4, away from the default. `AdaptableOperatorSampler`'s RNNI-reward search, given the identical
four candidate exponents, favors exponent 2 — the flattest option offered, the opposite end of the
range from `pca_grid_mid`'s leader. This is a real, reproducible-in-2-seeds directional
disagreement between the two mechanisms, not noise in one reading: `mean_SS` drops by nearly half
from exponent 2 to exponent 4 in both seeds independently.

**This does not on its own overturn the `pca_grid_mid` result above** — that result was measured
directly in the metric that matters (`esjdPerSec_rnni`, ESS-gated, `thorough` tier, CI excludes
zero) and stands on its own evidence. But it raises a specific, testable concern the design docs
already anticipated in a different form (acceptance vs. mixing): here it is *squared
log-likelihood jump* vs. *squared RNNI-distance jump* that disagree about which exponent is best,
one level up from the acceptance-vs-likelihood distinction `docs/WeightTransformAdaptation.md`
already documents. A pointier target selection can produce a bigger likelihood swing while landing
on a topologically *closer* rearrangement than a flatter selection would, which is exactly the kind
of proxy mismatch that would make `GridTuner` converge on a leader that is good for the reward it is
given but not for RNNI-space mixing specifically. Two important caveats on this cross-check itself,
both against reading too much into it:

- Two seeds, `fast` tier (no ESS gate, ~1,900-3,150 proposals per candidate) — this is a cheap
  directional probe, not a decision-grade measurement, and 2/2 agreement at this sample size is
  suggestive, not conclusive.
- The comparison is confounded by more than just the tuning *algorithm*: `GridTuner` explores
  online with a moving leader and 10% forced exploration, while every `AdaptableOperatorSampler`
  candidate here is a *pinned*, non-adaptive kernel — the two setups differ in reward signal and in
  adaptivity at once, so this cross-check cannot cleanly attribute the disagreement to "RNNI vs
  likelihood as a reward" alone.

The clean way to resolve this: give `GridTuner` itself an RNNI-based reward (a `TreeMetric`-driven
`Function` analogous to `<reward idref="likelihood"/>` but for squared RNNI jump) and rerun
`pca_grid_mid`'s `thorough`-tier comparison — if the leader moves down toward 2 and the ESJD/s
advantage shrinks or disappears, that would show the `pca_grid_mid` win is an artifact of the
likelihood-reward proxy specifically; if the leader stays at 3-4 and the advantage holds, the two
online mechanisms would then agree once put on the same reward, and the disagreement above would be
attributable to `GridTuner`'s adaptivity/exploration rather than its reward choice. This is flagged
as the most direct next step, not yet run — no `Function` computing squared RNNI-metric jump exists
in the codebase today, and building one is a modest but nontrivial addition (`RNNIMetric` measures
between two trees, not a running "jump since last proposal" the way `likelihood`'s `Function`
interface exposes for free).

`mvn -o test`: 123 run, 2 failures, 1 skipped, 0 errors — unchanged from baseline, confirmed after
these runs and before committing.

### Resolving the disagreement: `GridTuner` on an RNNI reward (`pca_grid_mid_rnni`)

Built `targetedbeast.weighttransform.TreeJumpFunction`, the `Function` the previous section called
for: wraps a `Tree` and a `TreeMetric`, snapshotting the tree (`.copy()`) on the first call of each
before/after pair and returning the metric distance from that snapshot on the second. Wired as
`GridTuner`'s `<reward>` in place of `<reward idref="likelihood"/>`, this makes `GridTuner`'s
existing before/after squared-difference reward mechanism compute squared RNNI-distance jump
instead of squared likelihood jump, with zero changes to `GridTuner` itself. `pca_grid_mid_rnni`
reruns `pca_grid_mid`'s exact grid (`exponents="2 3 4 6"`, `explorationProb="0.1"`, same burnin) with
only the reward swapped, `@rnni` pointed at the same `RNNIMetric` instance the benchmark's own
logging already uses.

`thorough` tier, 4 reps, seeds 1000-1003 (paired with `pca_grid_mid`'s reps). ESS gate:

| rep | seed | posterior.ESS | tree.height.ESS |
|---:|---:|---:|---:|
| 0 | 1000 | 305.6 | 367.8 |
| 1 | 1001 | 215.5 | 522.6 |
| 2 | 1002 | 204.8 | 464.4 |
| 3 | 1003 | 218.6 | 313.2 |

All 4 reps clear both gates.

| rep | seed | leadingExponent (RNNI reward) | leadingExponent (likelihood reward, `pca_grid_mid`) | esjdPerSec_rnni (RNNI reward) | esjdPerSec_rnni (likelihood reward) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1000 | 2.0 | 4.0 | 19753.1 | 22040.6 |
| 1 | 1001 | 4.0 | 4.0 | 21506.6 | 22712.0 |
| 2 | 1002 | 4.0 | 3.0 | 19351.6 | 22595.7 |
| 3 | 1003 | 4.0 | 4.0 | 20047.1 | 25139.5 |

Paired difference (RNNI-reward minus likelihood-reward), t-based 90% CI (n=4, df=3, t≈2.353):
mean −2957.3, CI (−4897.5, −1017.1) — **excludes zero, in the negative direction**. The
likelihood-reward `GridTuner` wins on the actual `esjdPerSec_rnni` benchmark metric in all 4 of 4
reps.

### This answers the open question, and not the way either candidate mechanism predicted

Two things were live going in: would an RNNI-based reward pull `GridTuner`'s leader down toward
exponent 2 (matching `AdaptableOperatorSampler`'s sampler-weight leader), and would it change
whether `GridTuner`'s chosen exponent actually performs better on `esjdPerSec_rnni`. Neither
happened the way the reward-proxy hypothesis predicted:

- **The leader barely moved.** 3 of 4 reps still land on exponent 4 under the RNNI reward — the
  same leader `pca_grid_mid`'s likelihood reward picked in 3 of 4 reps. Making the reward match the
  metric we actually care about did not push `GridTuner` toward `AdaptableOperatorSampler`'s
  exponent-2 answer.
- **Performance got worse, not better or unchanged.** Reward-proxy mismatch would predict the
  likelihood-reward leader looks good on its own reward but mediocre on RNNI-space mixing — instead
  the likelihood-reward run *beats* the RNNI-reward run on `esjdPerSec_rnni` itself, in every rep,
  with a CI that excludes zero.

So the earlier cross-check's disagreement is not explained by "wrong reward signal." The likelier
explanation is the second confound that section flagged: `GridTuner` explores online with a moving
leader and forced exploration, while the `AdaptableOperatorSampler` cross-check used four *pinned*
non-adaptive candidates — the disagreement traces to the tuning mechanism/exploration dynamics, not
to which jump measure the reward is built from. A plausible mechanical reason performance dropped
under the RNNI reward specifically: RNNI distance is a coarser, more discrete signal per proposal
than a continuous likelihood delta (many proposals move the tree by the same small number of NNI
steps), so the reward `GridTuner` optimizes against is noisier per-observation even though it is
computed from the metric that ultimately matters — online tuning against a noisier reward can
converge less efficiently, independent of where the "true" optimum sits.

**Practical takeaway**: keep `GridTuner` on the likelihood reward. It is both cheaper (no
per-proposal tree copy or `TreeMetric` call) and, on this evidence, at least as good — here strictly
better — at finding an exponent that performs well on the RNNI-space benchmark metric that actually
matters.

`mvn -o test`: 126 run, 2 failures, 1 skipped, 0 errors (123 baseline + 3 new `TreeJumpFunctionTest`
cases, all passing) — unchanged known-failure set, confirmed before committing.

---

### Offset floor and an independently-tuned edgeWeightTransform (pca_grid_mid_dual)

**What was built.** `pca_grid_mid_dual` extends `pca_grid_mid` (the `exponents="2 3 4 6"`,
likelihood-reward `GridTuner` result above, +4.9%, CI excludes zero) two ways at once: (a)
`targetWeightTransform`'s `PowerOffsetTransform` gets `offset="0.1111111111111111"` (≈1/9) instead
of `0.0` — a fixed, non-tuned floor that mixes in 10% uniform proposal mass via
`WeightTransform.apply()`'s `a/(1+a)` mixture (`a=1/9` → 0.1); and (b) a second, independently
tuned `PowerOffsetTransform`+`GridTuner` (same grid, same `burnin="5000"`,
`explorationProb="0.1"`, `<reward idref="likelihood"/>`, own id `edgeWeightTransform`, same
`0.1111111111111111` offset) is wired onto `edgeWeightTransform`, previously always left unset
(source-edge weights untransformed). `pca_grid_mid` itself is unchanged and is the baseline this
section compares against, rerun at the same seeds rather than reusing old numbers, per the task's
paired design. This section is about the result, not the design — the XMLs already existed
(commit 16f432f) — so no smoke check is repeated here.

Rendered via `scripts/render_infb_benchmark_xml.py --settings pca_grid_mid_dual pca_grid_mid
--reps 4 --chain-length 750000 --pre-burnin 100000 --jump-interval 20 --out-root
runs/eval7_dual/infb`, `thorough` tier, 4 reps, seeds 1000-1003 shared between the two settings at
a given rep (paired). Each of the 8 cells ran as a blocking, sequential `mvn -o exec:exec`
invocation, 400-441s chain time for 7 of 8 cells and one outlier at 641s (`pca_grid_mid_dual`
rep3, evidently slowed by transient system contention rather than anything about the run itself —
its own posterior/likelihood trajectory and final ESS look unremarkable next to the other three
reps), ~59 minutes chain time total for all 8.

#### ESS gate

| setting | rep | posterior.ESS | tree.height.ESS |
|---|---:|---:|---:|
| pca_grid_mid_dual | 0 | 225.4 | 751.4 |
| pca_grid_mid_dual | 1 | 245.4 | 626.8 |
| pca_grid_mid_dual | 2 | 281.3 | 749.9 |
| pca_grid_mid_dual | 3 | 236.4 | 618.0 |
| pca_grid_mid | 0 | 123.3 | 367.9 |
| pca_grid_mid | 1 | 283.1 | 462.6 |
| pca_grid_mid | 2 | 290.2 | 546.2 |
| pca_grid_mid | 3 | 309.4 | 506.2 |

All 8 cells clear `posterior.ESS`≥100 (123.3-309.4) and `tree.height.ESS`≥100 (367.9-751.4).

#### Raw results — chain-level (`__CHAIN__` row), per cell

| setting | rep | seed | esjdPerSec_rnni | targetWeightTransform.leadingExponent (median) | edgeWeightTransform.leadingExponent (median) |
|---|---:|---:|---:|---:|---:|
| pca_grid_mid_dual | 0 | 1000 | 21867.9 | 4.0 | 3.0 |
| pca_grid_mid_dual | 1 | 1001 | 20016.1 | 4.0 | 3.0 |
| pca_grid_mid_dual | 2 | 1002 | 20041.1 | 3.0 | 2.0 |
| pca_grid_mid_dual | 3 | 1003 | 21817.8 | 4.0 | 3.0 |
| pca_grid_mid | 0 | 1000 | 21409.8 | — | — |
| pca_grid_mid | 1 | 1001 | 23480.0 | — | — |
| pca_grid_mid | 2 | 1002 | 23915.4 | — | — |
| pca_grid_mid | 3 | 1003 | 20587.6 | — | — |

`targetWeightTransform.leadingExponent`'s per-rep leaders (4.0, 4.0, 3.0, 4.0) exactly match
`pca_grid_mid`'s own already-recorded leaders from the RNNI-reward section above (rep0=4.0,
rep1=4.0, rep2=3.0, rep3=4.0) — the offset floor and the added `edgeWeightTransform` did not move
what the target transform converges to. `edgeWeightTransform.leadingExponent` settles one grid
step flatter in every rep (3.0, 3.0, 2.0, 3.0), i.e. its own tuner independently prefers a less
pointy exponent for source-edge weighting than `targetWeightTransform` prefers for target-edge
weighting on the same dataset.

#### Paired analysis (pca_grid_mid_dual − pca_grid_mid), t-based 90% CI, n=4, df=3

| mean diff (ESJD/s) | 90% CI | excludes zero? | relative to pca_grid_mid mean |
|---:|---|---|---:|
| −1412.5 | (−4506.7, 1681.7) | no | −6.3% |

#### Verdict

**No detectable difference.** The 90% CI includes zero, and the per-rep picture is genuinely
mixed rather than a weak-but-consistent trend: `pca_grid_mid_dual` beats its paired
`pca_grid_mid` rep in 2 of 4 reps (rep0: +458.1, rep3: +1230.2) and loses in the other 2 (rep1:
−3463.9, rep2: −3874.3), with the losses noticeably larger in magnitude than the gains — hence the
negative point estimate (−6.3%) despite an even win/loss split. This adds neither support nor a
clear caveat to `pca_grid_mid`'s own standalone result (+4.9% over plain `PCAWeights2`, CI
excludes zero): nothing here contradicts that result, since `pca_grid_mid` was rerun fresh at the
same seeds and landed close to its previously-recorded numbers (21409.8-23915.4 here vs.
22040.6-25139.5 previously — same ballpark, ordinary seed-to-seed variation).

The obvious caveat applies and is worth stating plainly: `pca_grid_mid_dual` bundles two changes
at once (the fixed 0.1-mixture offset floor, and the independently-tuned `edgeWeightTransform`),
and this result cannot on its own attribute the (null) effect to either one individually — it only
says that *together*, at n=4, they do not measurably help or hurt relative to `pca_grid_mid`
alone. A follow-up isolating each change (offset floor alone, `edgeWeightTransform` alone) would
be needed to say anything about either piece specifically, and given how close this result already
sits to a coin flip, such a follow-up would need to budget for an effect this small being genuinely
undetectable at 4 reps rather than assume more reps will necessarily resolve it.

`mvn -o test`: 126 run, 2 failures, 0 errors, 1 skipped — unchanged from baseline, confirmed after
these runs and before committing.

### Regularized edge/target coupling via a ratio grid, and a smaller offset floor (pca_grid_mid_ratio)

**What was built.** `pca_grid_mid_dual` above gave `edgeWeightTransform` its own fully independent
`GridTuner` and found no detectable benefit over `pca_grid_mid` alone. `pca_grid_mid_ratio` is a
regularized alternative: `edgeWeightTransform`'s exponent still tunes independently via its own
`GridTuner`, but `targetWeightTransform`'s exponent is no longer tuned independently — instead a
new `CoefficientTuner` (`RatioGridTuner`) ties it to `edgeWeightTransform`'s live, per-proposal
exponent through a small ratio grid (`0.7 1.0 1.4`), so `targetWeightTransform`'s exponent always
equals `edgeWeightTransform`'s current exponent times whichever ratio is currently selected. This
also drops the offset floor from `pca_grid_mid_dual`'s 0.1-uniform-mixture (`offset=1/9`) to a
smaller 0.05-uniform-mixture (`OFFSET_MIXTURE_0_05 = 1/19`) on both transforms. The `RatioGridTuner`
class, the `TargetedWeightOperator` reorder it needed, its unit tests, and detailed-balance
verification are already committed (this section is about the result, not the design).

Rendered via `scripts/render_infb_benchmark_xml.py --settings pca_grid_mid_ratio --reps 4
--chain-length 750000 --pre-burnin 100000 --jump-interval 20 --out-root runs/eval8_ratio/infb`,
`thorough` tier, 4 reps, seeds 1000-1003, paired against `pca_grid_mid`'s already-recorded
`runs/eval7_dual/infb/pca_grid_mid/rep{0..3}/summary.tsv` (same seeds, reused rather than rerun,
per the task). Each of the 4 `pca_grid_mid_ratio` cells ran as its own blocking, sequential
`mvn -o exec:exec` invocation, 432-511s chain time, ~33 minutes total.

#### ESS gate

| setting | rep | posterior.ESS | tree.height.ESS |
|---|---:|---:|---:|
| pca_grid_mid_ratio | 0 | 226.6 | 788.6 |
| pca_grid_mid_ratio | 1 | 221.4 | 707.9 |
| pca_grid_mid_ratio | 2 | 245.1 | 734.1 |
| pca_grid_mid_ratio | 3 | 184.7 | 703.2 |
| pca_grid_mid | 0 | 123.3 | 367.9 |
| pca_grid_mid | 1 | 283.1 | 462.6 |
| pca_grid_mid | 2 | 290.2 | 546.2 |
| pca_grid_mid | 3 | 309.4 | 506.2 |

All 8 cells clear `posterior.ESS`≥100 (123.3-309.4) and `tree.height.ESS`≥100 (367.9-788.6).

#### Raw results — chain-level (`__CHAIN__` row), per cell

| setting | rep | seed | esjdPerSec_rnni | targetWeightTransform.leadingRatio (median) | edgeWeightTransform.leadingExponent (median) |
|---|---:|---:|---:|---:|---:|
| pca_grid_mid_ratio | 0 | 1000 | 19112.5 | 1.0 | 3.0 |
| pca_grid_mid_ratio | 1 | 1001 | 18605.3 | 0.7 | 2.0 |
| pca_grid_mid_ratio | 2 | 1002 | 20455.5 | 1.4 | 3.0 |
| pca_grid_mid_ratio | 3 | 1003 | 15800.1 | 1.4 | 3.0 |
| pca_grid_mid | 0 | 1000 | 21409.8 | — | — |
| pca_grid_mid | 1 | 1001 | 23480.0 | — | — |
| pca_grid_mid | 2 | 1002 | 23915.4 | — | — |
| pca_grid_mid | 3 | 1003 | 20587.6 | — | — |

`targetWeightTransform.leadingRatio` lands on a different grid point in 3 of 4 reps (1.0, 0.7,
1.4, 1.4) rather than converging consistently — with `edgeWeightTransform.leadingExponent` itself
settling at 2.0 or 3.0 depending on the rep, the effective target exponent (ratio × edge exponent)
varies from rep to rep (2.0, 1.4, 4.2, 4.2) rather than tracking a single stable value.

#### Paired analysis (pca_grid_mid_ratio − pca_grid_mid), t-based 90% CI, n=4, df=3

| mean diff (ESJD/s) | 90% CI | excludes zero? | relative to pca_grid_mid mean |
|---:|---|---|---:|
| −3854.9 | (−5294.5, −2415.3) | yes | −17.2% |

#### Verdict

**Worse, and detectably so.** Unlike `pca_grid_mid_dual`'s null result, this CI excludes zero on
the losing side: `pca_grid_mid_ratio` loses to its paired `pca_grid_mid` rep in all 4 reps (−2297.3,
−4874.7, −3459.9, −4787.6), a consistent rather than mixed pattern. The regularization this section
set out to test — tying `targetWeightTransform` to `edgeWeightTransform` via a ratio grid instead of
giving it a fully independent grid — does not recover `pca_grid_mid_dual`'s null result, let alone
improve on it; it costs measurable ESJD/second instead. Combined with the per-rep ratio not settling
on a consistent grid point (0.7, 1.0, 1.4, 1.4), the coupling does not appear to be finding a stable
regime, unlike `pca_grid_mid_dual`'s independent tuner, whose `edgeWeightTransform.leadingExponent`
converged predictably one grid step below `targetWeightTransform`'s in every rep. As with
`pca_grid_mid_dual`, this result bundles two simultaneous changes (the ratio-grid coupling and the
smaller 0.05-mixture offset floor) and cannot on its own attribute the loss to either one
individually — but given the clean, one-sided, CI-excludes-zero result, there is no ambiguity about
whether this specific combination is worth keeping: it is not.

`mvn -o test`: 134 run, 2 failures, 0 errors, 1 skipped — unchanged from baseline, confirmed after
these runs and before committing.

---

# `AdaptableOperatorSampler` baseline, and ESJD per step as a second metric

## Setup

`pca_aos_mid` replaces each of the four transform-bearing operators (`TargetedWide`,
`targetedWilsonBalding`, `targetedWilsonBaldingFixedHeight`, `targetedRangeSlide`) with an
`AdaptableOperatorSampler` over four copies of itself, target-weight exponent pinned at 2/3/4/6
(`pca_grid_mid`'s grid), scored by squared RNNI jump, `burnin=5000`, `uniformp=0.1` (matching
`GridTuner`'s burnin and `explorationProb`). Operators are literal XML in
`examples/infb_HA_141/template_aos.xml`; `render_infb_benchmark_xml.load_template` refuses it if it
differs from `template.xml` outside the `targeted-operators` markers. Unlike `pca_grid_mid`, each
operator learns its own exponent rather than sharing one.

`snakemake aos_baseline` runs `pca`, `pca_grid_mid` and `pca_aos_mid` at the `thorough` tier
(4 reps, seeds 1000–1003, shared checkpoint) in one invocation, so all three share timing
conditions; `scripts/paired_compare.py` does the paired analysis. Run with `-j4`.

## Why ESJD per step

Earlier verdicts paired cells from different sessions on `esjdPerSec_rnni`. The same
`pca_grid_mid` seeds differ by up to 18% between `eval6` and `eval7` (rep 3: 25,140 vs 20,588), so
timing drift is comparable to the effects measured. `esjd_rnni` (squared RNNI jump per proposal,
chain level) is unaffected by it, and a pinned or grid-tuned exponent costs the same per proposal,
so for weight transforms it is the cleaner headline. It does not see the sampler's own per-call
overhead (tree copy and RNNI distance), so `esjdPerSec_rnni` stays alongside it.

Re-analysing the existing `thorough` cells per step (vs `eval7_dual`'s `pca_grid_mid`, n=4):

| setting | per second | per step |
|---|---|---|
| `pca_grid_mid_rnni` | −9.8%, CI excludes 0 | −5.6%, CI (−1.62, +0.38) |
| `pca_grid_mid_dual` | −6.3%, CI includes 0 | −9.6%, CI (−2.93, +0.80) |
| `pca_grid_mid_ratio` | −17.2%, CI excludes 0 | −9.9%, CI (−2.62, +0.42) |

None is detectably worse per step; "no better than `pca_grid_mid`" stands, "detectably worse"
does not. (`pca_grid_mid_rnni` was originally paired with `eval6`'s reps, not `eval7`'s.)

## Result

ESS gate: 11 of 12 cells pass; `pca_aos_mid` rep 2 has `posterior.ESS` 87.8. Excluding it does not
change the conclusions. Runtimes per cell: `pca` 312–325 s, `pca_grid_mid` 321–325 s,
`pca_aos_mid` 330–339 s.

Paired against `pca`, t-based 90% CI, n=4:

| setting | `esjd_rnni` | `esjdPerSec_rnni` |
|---|---|---|
| `pca_grid_mid` | +4.8%, (+0.19, +0.84), excludes 0 | +3.4%, (−361, +2194) |
| `pca_aos_mid` | +0.0%, (−0.73, +0.73) | −5.6%, (−2798, −214), excludes 0 |

- **`pca_grid_mid` beats `pca` on chain-level ESJD, but not through the transform.** See "Where
  chain-level ESJD comes from" below: the gain sits entirely in operators the transform does not
  touch. On the targeted operators it is −4.1%, CI includes 0.
- **`AdaptableOperatorSampler` adds nothing.** Per step it equals `pca` at chain level and is
  −9.7% (CI includes 0) on the targeted operators; per second it also pays its per-call overhead.
  Its learned weights (exponents 2/3/4/6, min–max over reps) never commit:

  | operator | 2 | 3 | 4 | 6 |
  |---|---|---|---|---|
  | `TargetedWide` | 0.24–0.26 | 0.24–0.27 | 0.24–0.26 | 0.23–0.26 |
  | `targetedWilsonBalding` | 0.28–0.29 | 0.25–0.27 | 0.22–0.26 | 0.21–0.23 |
  | `targetedWilsonBaldingFixedHeight` | 0.24–0.26 | 0.25–0.28 | 0.25–0.26 | 0.22–0.24 |
  | `targetedRangeSlide` | 0.27–0.31 | 0.24–0.31 | 0.17–0.24 | 0.19–0.25 |

  A near-uniform mixture of exponents is no better than exponent 2 alone; the slight lean to 2
  matches the targeted-operator result below.
- **`GridTuner` is effectively a uniform mixture.** It draws each proposal's exponent in
  proportion to the mean reward per grid point, and those means are nearly equal (final values
  0.106–0.121 across all points and reps). Each exponent is therefore drawn about a quarter of the
  time. The logged `leadingExponent` is the argmax of four near-ties and says nothing about an
  optimum; `eval6`'s "leader at 3–4" and this run's "leader at 6" are both noise.

## Fixed exponents

`pca_fixed4`, `pca_fixed6`, `pca_fixed8`: one shared `PowerOffsetTransform` pinned at that
exponent, no tuner (`pca_fixed2`'s wiring). `snakemake fixed_exponents` ran them at the `thorough`
tier, reusing `aos_baseline`'s `pca`/`pca_grid_mid` cells (so judged per step). All 20 cells pass
the ESS gate.

```bash
snakemake -s scripts/Snakefile -j4 fixed_exponents
python3 scripts/paired_compare.py runs/infb/thorough/fixed_exponents.summary.tsv \
    --baseline pca --settings pca_fixed4 pca_fixed6 pca_fixed8 pca_grid_mid
```

`esjd_rnni`, paired, t-based 90% CI, n=4:

| setting | mean | vs `pca` (exponent 2) | vs `pca_grid_mid` |
|---|---:|---|---|
| `pca` | 10.74 | — | −4.6%, (−0.84, −0.19) |
| `pca_fixed4` | 10.53 | −1.9%, (−0.71, +0.31) | −6.4%, (−1.08, −0.35) |
| `pca_fixed6` | 10.57 | −1.6%, (−0.90, +0.56) | −6.1%, (−1.22, −0.15) |
| `pca_fixed8` | 10.99 | +2.4%, (−0.11, +0.62) | −2.3%, (−0.81, +0.29) |
| `pca_grid_mid` | 11.25 | +4.8%, (+0.19, +0.84) | — |

Chain level shows no exponent effect, yet puts `pca_grid_mid` above every fixed exponent except 8.
A state-independent mixture of exponents 2/3/4/6 (each operator redraws the exponent at the start
of every proposal) cannot do that: its expected squared jump is the weighted mean of its
components'. The next section resolves this.

Posterior summaries agree across all six settings (likelihood, prior, tree height and length,
clock rate, popSize, kappa, gammaShape: means within 0.3%). This is too coarse to rule out a small
bias: reps share the checkpoint, so their spread understates the error.

## Where chain-level ESJD comes from

Chain-level `esjd_rnni` is Σᵢ nᵢ·ESJDᵢ / N over all operators. Split into the four operators a
weight transform acts on (`targeted_operators` in `datasets.yaml`) and the rest, mean over reps:

| setting | targeted | rest | chain |
|---|---:|---:|---:|
| `pca` (exponent 2) | 1.49 | 9.25 | 10.74 |
| `pca_fixed4` | 1.45 | 9.09 | 10.54 |
| `pca_fixed6` | 1.24 | 9.33 | 10.57 |
| `pca_fixed8` | 1.34 | 9.65 | 10.99 |
| `pca_grid_mid` | 1.43 | 9.83 | 11.25 |
| `pca_aos_mid` | 1.35 | 9.39 | 10.74 |

The targeted operators supply ~13% of chain-level ESJD. Most of the rest comes from height
operators (`strictClockUpDownOperatorX` ≈3.4, `ScaleIntervalsRandomly` ≈2.7, `ScaleAll` ≈1.9):
RNNI distance is defined on ranked trees, so reordering node heights counts as a jump. Chain level
therefore dilutes a transform's effect about eightfold and adds the noise of operators the
transform does not touch. `pca_grid_mid`'s chain-level gain is entirely in the "rest" column.

`collect_stats.py` now writes `esjd_<metric>_targeted` (Σ nᵢ·ESJDᵢ / Σ nᵢ over the targeted
operators, i.e. per targeted proposal) and `esjdPerSec_<metric>_targeted`;
`paired_compare.py` reports them first. Paired against `pca`, n=4, 90% CI:

| setting | mean | per step | per second |
|---|---:|---|---|
| `pca` | 3.24 | — | — |
| `pca_fixed4` | 3.16 | −2.7%, (−0.46, +0.29) | −5.6%, CI includes 0 |
| `pca_fixed6` | 2.69 | −17.1%, (−1.10, −0.003) | −19.2%, CI excludes 0 |
| `pca_fixed8` | 2.93 | −9.8%, (−0.72, +0.09) | −12.7%, CI excludes 0 |
| `pca_grid_mid` | 3.11 | −4.1%, (−0.54, +0.28) | −4.8%, CI includes 0 |
| `pca_aos_mid` | 2.93 | −9.7%, (−0.94, +0.31) | −19.7%, CI excludes 0 |

- **Exponent 2, the operators' default, is best measured so far**; 6 is detectably worse.
- **Both mixtures are consistent with linearity** on this metric: `pca_grid_mid` (3.11) and
  `pca_aos_mid` (2.93) lie within their components' range.
- **`GridTuner` has nothing to learn.** Its selection probabilities (mean rewards normalised) are
  scattered at 2% of the chain, within 0.21–0.30 by 10% and 0.23–0.27 from 50% on, in every rep:
  adaptation finishes within ~75k steps, at "no preference", because squared likelihood change per
  proposal is nearly flat across 2–6.

RNNI still counts height reorderings within topology moves; a height-free topology metric would
isolate topology mixing further. ESS remains the final arbiter, since the targeted metric ignores
indirect effects on the rest of the chain.

## Exponent sweep, 8 reps

`pca_fixed1`, `pca_fixed1p5`, `pca_fixed3` added; `snakemake exponent_sweep` ran exponents 1, 1.5, 2,
3, 4, 6, 8 and `pca_grid_mid` at the `thorough` tier, 8 reps (seeds 1000–1007; reps 0–3 reused from
the runs above). All 64 cells pass the ESS gate.

```bash
snakemake -s scripts/Snakefile -j4 exponent_sweep
python3 scripts/paired_compare.py runs/infb/thorough/exponent_sweep.summary.tsv --baseline pca \
    --settings pca_fixed1 pca_fixed1p5 pca_fixed3 pca_fixed4 pca_fixed6 pca_fixed8 pca_grid_mid
```

Paired against `pca` (exponent 2), t-based 90% CI, n=8:

| exponent | `esjd_rnni_targeted` (mean, sd) | per step | per second | chain-level per step |
|---|---|---|---|---|
| 1 | 3.05, 0.23 | −7.6%, (−0.58, +0.08) | −6.5% | −2.3% |
| 1.5 | 3.23, 0.34 | −2.2%, (−0.40, +0.26) | −4.0% | −1.4% |
| 2 (`pca`) | 3.30, 0.39 | — | — | — |
| 3 | 3.27, 0.15 | −1.1%, (−0.31, +0.24) | −3.5% | −1.1% |
| 4 | 3.24, 0.22 | −2.0%, (−0.30, +0.17) | −0.2% | +0.5% |
| 6 | 2.88, 0.33 | −12.8%, (−0.78, −0.06), excludes 0 | −10.1% | −0.2% |
| 8 | 2.84, 0.17 | −14.0%, (−0.71, −0.22), excludes 0 | −14.7%, excludes 0 | −0.9% |
| `pca_grid_mid` | 3.10, 0.15 | −6.0%, (−0.44, +0.05) | −5.1% | +0.7% |

- **A plateau from 1.5 to 4, centred on the default.** Nothing in it differs detectably from
  exponent 2; 1 tends lower, and 6 and above are clearly worse. There is no better fixed exponent
  for a tuner to find on infB.
- **`pca_grid_mid` is what its mixture predicts.** Its grid is 2/3/4/6 drawn about uniformly; the
  mean of those components' effects is about −4%, and it measures −6%. Including 6 costs it.
- **The chain-level +4.8% for `pca_grid_mid` was a 4-rep fluke**: at 8 reps it is +0.7%, CI
  (−0.50, +0.65).
- **Noise:** rep-to-rep sd of the targeted metric is 0.15–0.39, i.e. 5–12% of the mean. At n=8 the
  CI half-width is about ±8%, so effects under ~5% remain unresolved.

For the target exponent on infB, the transform feature has nothing to offer over the operators'
built-in squaring: a tuner can at best avoid exponents ≥6, which the default already does.

## Burn-in speed

The `burnin_speed` target was unusable as built. `BenchmarkMCMC`'s burn-in tracker only counts
proposals after `preBurnin`, and the cells were rendered with the `thorough` tier's
`preBurnin=100000`, so the chain had converged before counting began. Burn-in cells now take their
chain shape from the dataset's `burnin_speed` entry in `datasets.yaml` (`pre_burnin: 0`,
`chain_length: 300000`, 8 reps), which also lists their settings: `constant`, `pca`, and exponents
1 and 8 as the flat and sharp extremes (2 is `pca`).

Metric: proposals from the UPGMA start until the log-posterior enters, and stays in for 5
proposals, the reference band (5th–95th percentile of the reference chain's post-burn-in
log-posterior). All 32 cells reached it.

```bash
snakemake -s scripts/Snakefile -j4 burnin_speed
python3 scripts/paired_compare.py runs/infb/burnin_speed_summary.tsv --baseline pca \
    --settings constant pca_fixed1 pca_fixed8 --metrics proposalsToTarget
```

Paired against `pca`, t-based 90% CI, n=8:

| setting | proposals to target (range) | vs `pca` |
|---|---|---|
| `pca` (exponent 2) | 4,686–10,150, mean 6,824 | — |
| `constant` | 6,823–16,390 | +68.9%, (+1,738, +7,671), excludes 0 |
| `pca_fixed1` | 5,516–10,560 | +12.6%, (−900, +2,624) |
| `pca_fixed8` | 6,080–9,444 | +14.3%, (−614, +2,568) |

- **Targeting speeds up burn-in substantially**: `constant` needs ~1.7× the proposals `pca` does.
- **The exponent matters little here too**; 1 and 8 are both slower than 2 but not detectably.
- In absolute terms infB burns in within ~1% of a `thorough` chain (a few seconds), so on this
  dataset the gain is real but cheap either way.

## Distance norm

`pca_l2`: `PCAWeights2` with `useOneNorm="false"`, i.e. Euclidean distance in PCA space instead of
the 1-norm. `snakemake norm_compare` ran it at the `thorough` tier, 8 reps, against `pca`'s 8 reps.
All 16 cells pass the ESS gate.

```bash
snakemake -s scripts/Snakefile -j4 norm_compare
python3 scripts/paired_compare.py runs/infb/thorough/norm_compare.summary.tsv --baseline pca \
    --settings pca_l2
python3 scripts/paired_compare.py runs/infb/thorough/norm_compare.summary.tsv --baseline pca \
    --settings pca_l2 --metrics tree.height.ESS posterior.ESS tree.treeLength.ESS likelihood.ESS
```

Paired `pca_l2` − `pca`, t-based 90% CI, n=8:

| metric | difference |
|---|---|
| `esjd_rnni_targeted` | −23.2%, (−1.10, −0.44), excludes 0 |
| `esjdPerSec_rnni_targeted` | −10.2%, excludes 0 |
| `esjd_rnni` (chain) | −6.7%, excludes 0 |
| `tree.height.ESS` | **+25.7%**, (+61, +179), excludes 0 |
| `tree.treeLength.ESS` | −17.1%, (−128, −0.05), excludes 0 |
| `posterior.ESS` | −5.8%, CI includes 0 |
| `likelihood.ESS` | −6.7%, CI includes 0 |

Per targeted operator (mean over reps), `pca` → `pca_l2`:

| operator | acceptance | jump given accepted |
|---|---|---|
| `TargetedWide` | 0.105 → 0.066 | 4.23 → 4.81 |
| `targetedWilsonBalding` | 0.065 → 0.041 | 4.87 → 5.56 |
| `targetedWilsonBaldingFixedHeight` | 0.105 → 0.066 | 4.35 → 4.82 |
| `targetedRangeSlide` | 0.047 → 0.069 | 3.11 → 2.07 |

- **On the target metric the 1-norm is clearly better.** With the 2-norm the three topology
  operators lose about 37% of their acceptances, and slightly larger accepted jumps do not make
  up for it.
- **ESS disagrees in part.** `tree.height` ESS rises 26% while `tree.treeLength` ESS falls 17%
  (posterior and likelihood unchanged). The height gain plausibly comes from `targetedRangeSlide`,
  the one operator that accepts more often under the 2-norm. ESS at 150–700 is noisy and four ESS
  columns were tested, so treat this as a flag, not a result: ESJD over RNNI distance is not a
  faithful proxy for every ESS dimension.
- Per second the loss is smaller (−10%) because `pca_l2`'s targeted proposals are 12–22% cheaper,
  consistent with fewer accepted moves.

### 2-norm at other exponents

`norm_compare` also runs `pca_l2_fixed1`, `pca_l2_fixed1p5` and `pca_l2_fixed3` (8 reps each, all
pass the ESS gate). Paired against `pca` (1-norm, exponent 2), n=8, 90% CI:

| 2-norm exponent | `esjd_rnni_targeted` | `tree.height.ESS` | `tree.treeLength.ESS` | `posterior.ESS` |
|---|---|---|---|---|
| 1 | −30.5%, excludes 0 | +42.8%, excludes 0 | −3.2% | −8.4% |
| 1.5 | −23.7%, excludes 0 | +34.7%, excludes 0 | −13.4% | −25.7%, excludes 0 |
| 2 | −23.2%, excludes 0 | +25.7%, excludes 0 | −17.1%, excludes 0 | −5.8% |
| 3 | −18.5%, excludes 0 | +33.5%, excludes 0 | −2.7% | +10.0% |

- **The 2-norm wants a sharper exponent, not a flatter one**: per-step ESJD improves steadily from
  1 to 3. That contradicts the guess that its acceptance drop came from over-sharp weights. A
  plausible reading: a 1-norm sum over 10 PCA axes spreads distances out more than the Euclidean
  norm, which is dominated by the largest axis, so 2-norm weights are flatter at equal exponent.
  Even at 3 it is 18.5% below the 1-norm; the curve may keep rising above 3, but closing that gap
  would take a 20%+ gain.
- **`tree.height` ESS is 26–43% higher at every 2-norm exponent**, each CI excluding zero, while
  the 1-norm exponent sweep shows no such shift. Four independent settings agreeing makes this the
  most consistent ESS signal so far, and it runs opposite to ESJD over RNNI distance. Other ESS
  columns show no consistent direction.
- **Per-second figures for these 24 cells are unusable.** Runtimes for identical work ranged from
  280 s to 1,000 s, i.e. the machine was heavily loaded for part of the batch. Per-step figures
  are unaffected.

## Tree-space ESS

Ported from the learned-weights branch: `scripts/tree_pseudo_ess.py` computes Lanfear et al.'s
(2016) pseudo-ESS from `run.trees` — distance from every sampled tree (10% burn-in) to 100 random
reference trees, univariate ESS of each resulting trace — for two distances: Robinson–Foulds (RF,
topology only) and Kuhner–Felsenstein (KF, branch-score, so branch lengths over matched clades).
`collect_stats.py --trees` adds `treeESS_{min,p10,median,max}_{rf,kf}`; the Snakefile passes it for
ESS-gated tiers, and `paired_compare.py` reports the p10 columns by default. It runs under
TreeTracer's venv (`/Users/nneureiter/Projects/treetracer`, for its compiled `rapidtrees`), about
5 s per cell.

Building it exposed a bug in the summary aggregation: `concat_tsv` concatenated cells by position
under the first file's header, but transform settings log extra `targetWeightTransform.*` ESS
columns, so every column after them was shifted. It now aligns by column name. Every column used
in the conclusions above (the ESJD metrics and `posterior`/`tree.height`/`tree.treeLength` ESS)
precedes those extra columns, so those results are unaffected.

`pca` levels (8 reps, mean ± sd): RF p10 157 ± 15, median 218 ± 15; KF p10 424 ± 40, median
521 ± 39. Topology mixes 2–3× more slowly than branch lengths.

Paired against `pca`, n=8, 90% CI (n=4 for `pca_aos_mid`):

| setting | `treeESS_p10_rf` | `treeESS_p10_kf` |
|---|---|---|
| exponent 1 | −5.8% | −4.2% |
| exponent 1.5 | −3.9% | −8.2% |
| exponent 3 | +1.4% | +3.2% |
| exponent 4 | −0.3% | −0.9% |
| exponent 6 | **−18.7%**, excludes 0 | +2.0% |
| exponent 8 | **−12.5%**, excludes 0 | −3.5% |
| `pca_grid_mid` | +0.1% | −0.9% |
| `pca_aos_mid` (n=4) | −1.9% | −9.1% |
| 2-norm, exponent 1 | **−38.7%**, excludes 0 | **+21.4%**, excludes 0 |
| 2-norm, exponent 1.5 | **−30.1%**, excludes 0 | **+14.6%**, excludes 0 |
| 2-norm, exponent 2 | **−38.6%**, excludes 0 | **+14.6%**, excludes 0 |
| 2-norm, exponent 3 | **−17.7%**, excludes 0 | **+20.8%**, excludes 0 |

(Unmarked entries: CI includes 0. Medians give the same picture.)

- **RF tree ESS agrees with targeted ESJD on the exponent**: a plateau from 1 to 4, with 6 and 8
  worse. KF shows no exponent effect. `GridTuner` and AOS are indistinguishable from `pca`.
- **The 2-norm trades topology mixing for branch-length mixing.** RF tree ESS falls 18–39% and KF
  rises 15–21% at every exponent. This resolves the apparent conflict above: targeted ESJD over
  RNNI distance tracks topology, `tree.height` ESS tracks branch lengths, and the two norms are
  better at different things.
- **Since topology is the bottleneck** (its ESS is 2–3× lower than KF's) and is what the targeted
  operators exist for, the 1-norm at exponent 2 stays the default. The 2-norm's KF gain would
  matter only where branch-length mixing limits a run.

## `RangeSlide` detailed-balance bug: all `PCAWeights2` results above are affected

`targetedRangeSlide` (`useWeightedStep="true"`) did not satisfy detailed balance with
`PCAWeights2`, with or without a transform. Fixed in `76fc1e7`; the plain `Range` operator
(`useWeightedStep` false: uniform targets) was not affected.

**Cause.** `RangeSlide` rebuilt the "without i" cache only for ancestors of the moved node i that
were also slide targets. A `PCAWeights2` node's point is a branch-length-weighted average of its
children's, so the targets above a skipped ancestor were rebuilt from cached values that still
described the tree before the move. The reverse target weights therefore differed from the
forward weights a proposal from the moved tree computes: in ~10% of proposals, by up to 10× for
single candidates. Forward weights were always correct. It now rebuilds all of i's ancestors.

**How it surfaced.** `testRangeSlideTargetTransform` failed intermittently (4 of 6 runs on the
unfixed code, at 4.2–4.7σ against a 4σ threshold): a sharper exponent magnifies the weight error
enough to cross the threshold, which is why only transform runs showed it. A new deterministic
test, `testRangeSlideTargetWeightsMatchFreshCache`, checks every target-weight vector against a
freshly built `PCAWeights2`; it fails on the unfixed code within one proposal batch. After the fix,
the statistical `RangeSlide` tests (including temporary exponent-3 and -6 variants that failed 2/4
and 3/4 before) passed 5 of 5 runs each.

The same commit series brings `PartialWeights`' detailed-balance fix (`3a79c17`, from
`beast2.8-felsenstein-weights`) and `RangeSlide`'s stale-cache fix for live-tree weightings
(`d3bdfe9`).

**Consequence.** Every `PCAWeights2` cell in this document, including the `pca` baselines, ran a
slightly wrong `targetedRangeSlide` kernel (~10% of all proposals). Posterior summaries agreed
across settings within 0.3%, so the bias is small, but the effect sizes for `pca`-family settings
(exponent sweep, norms, tuners, AOS) should be treated as provisional until rerun on the fixed
kernel. The fix also changes `PCAWeights2`'s proposal distribution in `RangeSlide` (all ancestors
now exclude i), so old and new cells must not be paired.

## `PartialWeights` exponent sweep (fixed kernel)

> **INVALIDATED — rerun before citing.** These cells ran with the `RangeSlide` restore bug and
> the `PCAWeights2` cache drift described under "Edge-weight cache bugs" below.

First `PartialWeights` results on a valid kernel (`3a79c17`, `d3bdfe9`, `76fc1e7`); the
`beast2.8-felsenstein-weights` branch's `partial`/`partial_grid` numbers ran on the broken one.
`PartialWeights` returns a softmax over candidate log-likelihoods, so a pinned exponent b is an
inverse temperature, softmax(b · loglik); the sweep flattens (b ≤ 1, 0.1 being the transform's
lower bound). `partial` is the operators' default: squared for `TargetedWide` and both Wilson–
Balding operators, as-is for `targetedRangeSlide`. `pca_nt` is `pca` on the same likelihood.

`PartialWeights` needs `TreeLikelihood.getLikelihoodCore()`, which is null whenever BEAGLE is in
use (BEAGLE 4.0.0 is now installed and active), so these settings run with BEAST's `-java` flag
(`java_only_settings` in `datasets.yaml`) on a non-threaded `TreeLikelihood`.

```bash
snakemake -s scripts/Snakefile -j4 --rerun-incomplete partial_sweep
python3 scripts/paired_compare.py runs/infb/thorough/partial_sweep.summary.tsv --baseline pca_nt \
    --settings partial partial_fixed1 partial_fixed0p5 partial_fixed0p2 partial_fixed0p1
```

`thorough` tier, 4 reps; all 24 cells pass the ESS gate. Paired, t-based 90% CI, n=4:

| setting | targeted ESJD/step vs `pca_nt` | vs `partial` | RF tree ESS p10 vs `partial` | KF tree ESS p10 vs `pca_nt` |
|---|---|---|---|---|
| `partial` (default) | −16.5%, excl. 0 | — | — | +21.2% |
| b = 1 | −5.5%, excl. 0 | +13.2%, excl. 0 | +13.2% | +33.4%, excl. 0 |
| b = 0.5 | **+14.8%**, excl. 0 | **+37.6%**, excl. 0 | **+42.1%**, excl. 0 | +36.9%, excl. 0 |
| b = 0.2 | **+20.9%**, excl. 0 | **+44.8%**, excl. 0 | **+38.4%**, excl. 0 | +14.9% |
| b = 0.1 | +10.9%, excl. 0 | +32.8%, excl. 0 | +37.8%, excl. 0 | +31.8%, excl. 0 |

(Unmarked: CI includes 0. RF tree ESS of every `partial` setting vs `pca_nt` has a CI including 0,
from −21% for the default to +12% for b = 0.5.)

- **Unlike `PCAWeights2`, the exponent matters a lot here.** The default is far too sharp:
  flattening to b ≈ 0.2–0.5 raises targeted ESJD per step by 38–45% and RF (topology) tree ESS by
  ~40%. The optimum is interior (0.1 is worse than 0.2), consistent with the branch's ~0.2.
- **Per step, flattened `PartialWeights` beats `PCAWeights2`** on targeted ESJD (+15–21%) and on
  KF tree ESS; RF tree ESS is not distinguishable from `pca_nt` at n=4.
- **Per second it loses by 81–86%.** A targeted proposal costs far more: `targetedWilsonBalding`
  14 ms vs 0.35 ms, `targetedRangeSlide` 4.1 ms vs 0.30 ms, `TargetedWide` 1.6 ms vs 0.31 ms. No
  per-step gain here comes close to paying for that. (Per-second values are wall-clock and some
  cells ran under contention, e.g. `pca_nt` rep 1 at 2,322 against 3,900–5,100 for the others;
  the gap is far larger than that noise.)

On infB, `PartialWeights` is a better proposal per step once flattened, but not a better sampler:
`PCAWeights2` at its default delivers several times more topology movement per CPU-second. If
`PartialWeights` is pursued, b ≈ 0.2–0.5 should be its default, and the work is in its cost.

## `PCAWeights2` vs `ConstantWeights` on the fixed kernel

> **INVALIDATED — rerun before citing.** These cells ran with the `RangeSlide` restore bug and
> the `PCAWeights2` cache drift described under "Edge-weight cache bugs" below.

The project's headline comparison, rerun after the `RangeSlide` fix (`76fc1e7`) at the operators'
defaults: `snakemake pca_vs_constant`, `thorough` tier, 8 reps (seeds 1000–1007), from a shared
checkpoint rebuilt on the fixed code. All 16 cells pass the ESS gate. This replaces the earlier
+7.3% (`esjdPerSec_rnni`, 4 reps, broken kernel, cells from different sessions).

```bash
snakemake -s scripts/Snakefile -j4 pca_vs_constant
python3 scripts/paired_compare.py runs/infb/thorough/pca_vs_constant.summary.tsv \
    --baseline constant --settings pca
```

`pca` − `constant`, paired, t-based 90% CI, n=8:

| metric | difference |
|---|---|
| `treeESS_p10_rf` (topology) | **+46.8%**, (+35, +72), excludes 0 |
| `treeESS_median_rf` | +44.5%, excludes 0 |
| `treeESS_p10_kf` (branch lengths) | **+37.7%**, (+80, +140), excludes 0 |
| `tree.height.ESS` | +28.1%, excludes 0 |
| `posterior.ESS` | +22.4%, (−19, +121), CI includes 0 |
| `esjd_rnni_targeted` | +7.9%, excludes 0 |
| `esjdPerSec_rnni_targeted` | +11.4%, excludes 0 |
| `esjd_rnni` (chain) | +4.8%, excludes 0 |
| `esjdPerSec_rnni` (chain) | +5.3%, excludes 0 |

Targeting with `PCAWeights2` improves mixing clearly, and tree-space ESS shows it far more
strongly than ESJD: topology ESS rises by nearly half. RNNI ESJD understates the effect for the
reason found above — it is dominated by height reorderings the targeted operators barely touch.
The ESS-type metrics agree in direction everywhere, which makes this the most solid result in the
document. Per CPU-second the gain holds (`PCAWeights2` costs little per proposal).

## Edge-weight cache bugs: every `PCAWeights2` result above is affected

Three defects let the edge-weight cache disagree with the tree. All `PCAWeights2` cells in this
document, and the `PartialWeights` sweep, ran with at least one of them.

1. **`PCAWeights2` ignored height changes** (`efea525`). A node's point is a branch-length-weighted
   average of its children's, but the resync after a move followed only "filthy" nodes (dirty > 1)
   and stopped climbing at the move's MRCA (dirty 3). A plain height change (dirty 1) therefore left
   that node and all its ancestors stale, and so did every node above an MRCA. It now resyncs any
   dirty node and propagates to the root.
2. **`RangeSlide` left forward-pass values active** (`99f11c8`). The stale-cache fix ported from the
   felsenstein branch (`d3bdfe9`) added a second `prestore()` while the forward "without i" pass was
   never rolled back, so after a rejected move the cache was not restored. It now resets after the
   forward weights.
3. **Late direct rejects skipped the restore** (`832feda`). The four targeted operators can return
   −∞ after updating the cache (degenerate reverse source weights). BEAST does not restore calculation
   nodes on a direct reject when `requiresStateInitialisation()` is true (the default), so the cache
   kept describing the rejected tree. They now reset before returning; `TargetedWilsonBalding` had
   one more such return before its first reset.

**How they were found.** Old and new code gave different chains from the same seed although the
change only skipped unread values. A probe comparing the cache with a freshly built `PCAWeights2`
after every step found it stale at 95% of proposal starts on infB, growing over the run, even with
only a standard `SubtreeSlide` moving the tree. BEAST's `-Dbeast.debug=true` (which validates
`getChecksum()` after every reject and rechecks the posterior every step) caught defect 2 within
250 steps and, after the first fix, defect 3 at step ~5,750. The likelihood itself was never wrong:
BEAST rechecks it by full recomputation, but the edge weights add 0 to the posterior and escape
that check.

**Why the detailed-balance tests missed them.** They rebuild the cache for every sample, so they
check a single proposal from a fresh cache, never the state a real chain carries from step to step.
`EdgeWeightCacheFreshnessTest` now runs a real MCMC per targeted operator, plus one with infB's full
tree-operator mix for 20,000 steps, with reject validation on, and compares the cache with a fresh
instance after every step. It fails within 10 steps on the old code. A 20,000-step infB run in
debug mode is clean on the fixed code. `PartialWeights` is not covered by this test yet.

**Trimming the "without i" rebuild** (`46eb08f`, `8bbdc3e`). `RangeSlide` and `TargetedWilsonBalding`
rebuilt the cache for all of i's ancestors up to the root; only those up to the highest
(feasible) target are read. Run with constant identity hash codes (`-XX:hashCode=2`), old and new code
give identical 5,000-step chains: the trimming changes only speed. Under normal hashing chains differ,
because `RangeSlide` picks targets by iterating a `HashMap<Node, …>`, whose order depends on when
nodes were first hashed. Fast tier, `partial_fixed0p2`: `targetedWilsonBalding` −21% per proposal,
`targetedRangeSlide` −46%, whole chain −13%; for `PCAWeights2` the rebuild is too cheap to matter.
`TargetedWilsonBaldingFixedHeight` and `TargetedWide` already rebuilt only what they read.

**Reruns.** The `PartialWeights` sweep (`partial_sweep`, ~8–10 CPU-hours) needs rerunning if its
results are still wanted.

## `PCAWeights2` vs `ConstantWeights` on the final code

`snakemake pca_vs_constant` rerun after all three cache fixes and the trimming (through
`8bbdc3e`), `thorough` tier, 8 reps, from a shared checkpoint rebuilt on that code. All 16 cells
pass the ESS gate. `pca` − `constant`, paired, t-based 90% CI, n=8:

| metric | difference |
|---|---|
| `treeESS_p10_rf` (topology) | **+31.4%**, (+22, +50), excludes 0 |
| `treeESS_median_rf` | +27.4%, excludes 0 |
| `treeESS_p10_kf` (branch lengths) | **+53.7%**, (+97, +189), excludes 0 |
| `tree.height.ESS` | +58.9%, excludes 0 |
| `posterior.ESS` | +28.6%, (+1, +115), excludes 0 |
| `esjd_rnni_targeted` | +7.9%, excludes 0 |
| `esjdPerSec_rnni_targeted` | +11.7%, excludes 0 |
| `esjd_rnni` (chain) | +2.0%, CI includes 0 |
| `esjdPerSec_rnni` (chain) | +3.7%, CI includes 0 |

Targeting with `PCAWeights2` improves mixing on every ESS-type metric, posterior ESS included, and
per CPU-second. The magnitudes moved against the invalidated run (topology ESS +47% → +31%,
branch-length ESS +38% → +54%), so the bugs shifted the size of the effect, not its direction.
Chain-level RNNI ESJD barely registers it, for the reason found above.

Editing `datasets.yaml` or the renderer makes Snakemake treat existing cells, including the shared
checkpoint, as out of date. If their rendered XML is unchanged, mark them current with
`snakemake <target> --touch` rather than rerunning; a rerun checkpoint changes every cell's
starting state.

## Threaded `PartialWeights`

`ThreadedPartialWeights` scores each `ThreadedTreeLikelihood` slice in parallel and sums the
slices' scores, so its weights equal `PartialWeights` on the whole alignment. `snakemake
partial_threaded` runs it at the sweep's two best exponents (b = 0.2, 0.5) against `pca_thr`
(`pca` on the same likelihood), all with `-java -threads 2`; `pca` (BEAGLE, one thread) is the
existing rep 0–3 cells. `thorough` tier, 4 reps, from the shared checkpoint; all 16 cells pass the
ESS gate. Two threads, because on infB's 489 patterns four gain only 5% more.

```bash
snakemake -s scripts/Snakefile --cores 4 partial_threaded
python3 scripts/paired_compare.py runs/infb/thorough/partial_threaded.summary.tsv \
    --baseline pca_thr --settings partial_thr_fixed0p2 partial_thr_fixed0p5 pca
```

Per-hour columns are the ESS divided by `runtime_hours`. Paired, t-based 90% CI, n=4, vs `pca_thr`:

| metric | b = 0.2 | b = 0.5 | `pca` (BEAGLE) |
|---|---|---|---|
| runtime | +49.5%, excl. 0 | +43.3%, excl. 0 | −17.7%, excl. 0 |
| `treeESS_p10_rf` (topology) | +4.2% | +4.2% | −4.2%, excl. 0 |
| `treeESS_p10_kf` (branch lengths) | **+46.7%**, excl. 0 | **+40.1%**, excl. 0 | +5.7% |
| `tree.height.ESS` | **+94.6%**, excl. 0 | **+119.7%**, excl. 0 | +5.8% |
| `posterior.ESS` | −13.3% | −18.6% | +28.3%, excl. 0 |
| `esjd_rnni_targeted` | +11.9% | +7.8% | +1.3% |
| `treeESS_p10_rf` per hour | −30.6%, excl. 0 | −25.1%, excl. 0 | +16.2%, excl. 0 |
| `treeESS_p10_kf` per hour | −1.9% | −0.2% | +28.4%, excl. 0 |
| `tree.height.ESS` per hour | +29.6%, excl. 0 | +56.3%, excl. 0 | +28.3% |
| `posterior.ESS` per hour | −41.8% | −42.6%, excl. 0 | +56.3%, excl. 0 |

(Unmarked: CI includes 0.)

- **Threading and the earlier optimisations cut the cost from 5–7× to 1.4–1.5×** `pca`'s on the
  same likelihood (cells of 7–10 min against 30–60 min in the unthreaded sweep).
- **Per step, the gain is in branch lengths and tree height, not topology.** KF tree ESS rises by
  40–47% and tree-height ESS doubles; RF tree ESS moves by +4%, CI including 0.
- **Per hour it still loses on topology** (−25–31%) and posterior ESS, breaks even on branch
  lengths, and wins only on tree height. Against BEAGLE `pca`, the realistic baseline, the gap is
  larger: BEAGLE alone is 18% faster than the two-thread Java likelihood here. (`pca`'s cells ran
  in an earlier session, so its runtimes are less comparable than the other three settings'.)

On infB, a BEAGLE PR to expose partials would remove the Java-likelihood penalty but not the
scoring cost (about a third of `PartialWeights`' time before threading), and the per-step topology
gain it would have to amplify is not distinguishable from zero.

## `PartialWeights` reverse-pass bug: every `PartialWeights` result above is affected

**Cause.** In the reverse pass, `TargetedWilsonBalding` and the weighted `RangeSlide` rebuilt the
"without i" partials on the moved tree T′ from the likelihood core, which still holds the partials
of T: the likelihood runs only after the proposal. Every node on i's old path to the root still
contained i, and so did every value built from them. In a probe of 300 random Wilson–Balding moves
(12 taxa), 80% of moves had wrong reverse scores; the largest error in a reverse target probability
averaged 0.37 per move. Moves that change the root were wrong in a second way: `Tree.setRoot` swaps
node numbers, so the core's partials and the tree no longer used the same numbering.

**Fix.** Pruning i and its parent gives the same tree from T and from T′, so the forward pass's
"without i" partials are exactly what the reverse pass needs. `PartialWeights` now computes them for
all of i's ancestors (ignoring the operator's node list), keys them on that pruned tree, and reuses
them when the reverse pass asks for the same one. It reads the core under the numbers the core was
computed with, and forgets the cache in `store()`/`restore()`, which MCMC calls before the
likelihood updates the core.

**Why nothing caught it.** The detailed-balance harness never recomputed the likelihood, so the core
was equally out of date in the forward and reverse passes and the kernel looked consistent. It now
recomputes the likelihood for every sample; on the old code the three `PartialWeights`
detailed-balance tests then fail at up to 8.3σ. `EdgeWeightCacheFreshnessTest` covered only
`PCAWeights2`.

**General check.** `EdgeWeightOracleTest` runs a real MCMC per weight class (`PCAWeights2`,
`ConstantWeights`, `PartialWeights`, `ThreadedPartialWeights`) and targeted operator (all five in
the benchmark), with subtree slides, kappa, gamma-shape, clock-rate and up-down moves alongside.
`CheckedEdgeWeights` compares every weight an operator reads with a fresh instance built on a
snapshot of the current tree and parameters, rebuilding "without i" values for all ancestors. On
the old code it flags exactly the two reverse passes; on the fixed code all 20 chains (≈1.8M reads)
are clean. New weight classes and operators should be added to it.

**Consequence.** All `PartialWeights` cells (the exponent sweep and the threaded comparison) ran
with a wrong Hastings ratio and must be rerun. Posterior means (likelihood, tree height, posterior)
agreed with `pca` within two standard errors, a weak check. The fix also changes the cost: the
forward pass now rebuilds the whole ancestor chain, and the reverse pass rebuilds nothing.

## Threaded `PartialWeights` on the fixed code

`partial_threaded` rerun after the reverse-pass fix: `pca_thr`, `partial_thr_fixed0p2` and
`partial_thr_fixed0p5` (4 reps each, same seeds, two cells at a time); the BEAGLE `pca` cells and
the reference checkpoint were kept (`--touch`), since the fix does not touch their code. All 12 new
cells pass the ESS gate. Paired, t-based 90% CI, n=4, vs `pca_thr`; * = CI excludes 0:

| metric | b = 0.2 | b = 0.5 | `pca` (BEAGLE) |
|---|---|---|---|
| runtime | +92.9% * | +89.1% * | +9.1% * |
| `treeESS_p10_rf` (topology) | +14.4% | +15.4% | −0.9% |
| `treeESS_p10_kf` (branch lengths) | **+47.6%** * | **+43.5%** * | +10.9% |
| `tree.height.ESS` | **+103.8%** * | **+114.1%** * | +11.1% |
| `posterior.ESS` | −33.2% | −4.0% | +17.9% |
| `esjd_rnni_targeted` | +21.4% | +14.4% * | +5.6% |
| `treeESS_p10_rf` per hour | −40.8% * | −38.6% * | −9.1% |
| `treeESS_p10_kf` per hour | −23.5% * | −24.1% * | +1.8% |
| `tree.height.ESS` per hour | +5.6% | +13.1% | +1.9% |
| `posterior.ESS` per hour | −65.4% * | −48.9% * | +8.4% |

- **Per step, the picture survives the fix and the topology gain grows**: RF tree ESS +14–15%
  (was +4%), still with CI including 0; KF tree ESS +44–48% and tree-height ESS ≈2×, both
  significant as before. Posterior means agree with `pca` within two standard errors.
- **The fix is cost-neutral**: partial cells take 533–548 s, as before (≈540 s).
- **The runtime ratio moved because `pca_thr` did**: its cells took 277–283 s now against
  348–389 s in the first run, with the same scheduling (each paired with a partial cell). The cause
  is not known; most likely other load on the machine during the first run. On this run's
  same-session timing, `PartialWeights` costs ≈1.9× `pca_thr`, not 1.45×, and loses per hour on
  every metric but tree height. `pca` (BEAGLE) ran in an earlier session, so its runtime row is not
  comparable either.
