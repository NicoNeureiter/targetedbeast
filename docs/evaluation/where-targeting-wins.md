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
