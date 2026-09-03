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
