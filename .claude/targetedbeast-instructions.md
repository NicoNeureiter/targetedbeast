# TargetedBEAST: Targeted Operators — Theory, Tests, and Debugging Strategy

This document synthesizes the theory of targeted operators, the current test suite, and the debugging strategy for likelihood-informed proposals, incorporating recent lessons from development and troubleshooting.

## 1) Theory Overview (Targeted Operators)

### Goal
Targeted operators bias tree proposals toward regions that are more informative for the likelihood, approximating Gibbs-like behavior while keeping computation affordable.

### Core idea
Instead of proposing uniformly, sample proposal targets according to **edge weights** computed from likelihood-related summaries. This increases proposal quality and acceptance.

Given a moving node $i$ and candidate destination edges/nodes $c$, proposals are weighted by approximations of

$$P(\text{data} \mid \text{tree with } i \text{ reattached at } c).$$

In current Felsenstein-based targeted operators, this is assembled from:
- inside partials for the moving subtree,
- inside partials on the destination side in the tree with $i$ ignored,
- outside partials for data above the local attachment context,
- transition matrices at candidate-specific heights.

### Likelihood-informed weights
Weights are derived from Felsenstein-style partial likelihoods:
- **Inside/partials**: $L_i(s) = P(D_{below} | s_i=s)$ (bottom-up)
- **Outside partials**: $O_i(s) = P(D_{above} | s_i=s)$ (top-down)
- **Full conditional**: $L_i(s) \cdot O_i(s)$

Equivalent decomposition statement used throughout debugging/tests:

$$P(D \mid s_n=s) = L_n(s)\,O_n(s).$$

Integrating across states and rate categories should recover the tree log-likelihood when messages are computed consistently.

### “Without node” recursion
To compute candidate weights for reattachment, we need partials **excluding** the moving subtree:
- removing a node only affects **ancestors** (upward pass)
- outside partials are computed top-down and do **not** use the `nodes` list

This enables fast reuse and avoids full recomputation for every candidate.

### Similarity / agreement measures
Options used or explored:
- **Dot-product similarity** over normalized partials (stable, fast)
- **Partial agreement** via a hypothetical parent at candidate height
- **Full conditional agreement** (inside × outside) — attempted but unstable in practice
- **Full proposed tree likelihood** (current direction)

### Effective-parent logic
When ignoring a subtree, its parent can become unary and should be treated as “transparent.” Any height or parent reference in agreement/likelihood computations must skip unary parents to maintain symmetry across forward/backward moves.

---

## 2) Test Suite (Current)

### Unit/behavior tests (FelsensteinWeights)
- **Invariant target weights under move** (`targetWeightsInvariantAfterMove`):
  - Verifies symmetry when ignoring a node and moving it via Wilson–Balding
   - Uses current ancestor paths for the ignored node after topology changes
   - Candidate set must remain unchanged across forward/backward comparisons

- **Stability assertions** added during debugging:
  - Node indices and branch rates should remain invariant across move

- **Tree-likelihood consistency checks**:
   - Full conditional reconstruction equals tree log-likelihood when not ignoring
   - Pruned-tree likelihood consistency across aggregation points when ignoring

- **Status**:
   - `FelsensteinWeightsTest` passes
   - `FelsensteinWeightsWithoutNodeTest` passes

### Detailed balance tests
- **DetailedBalanceTest** for targeted operators
   - Confirms proposal symmetry across many samples
   - Passes for Felsenstein-based targeted operators

---

## 3) Debugging Strategy (Updated)

### A. Identify asymmetry sources
1. **Candidate set changes** across forward/backward moves
   - If candidate selection depends on the current parent height, original targets may disappear in reverse.
2. **Unary parent transparency**
   - If a parent becomes unary when ignoring a node, height references must skip it.
3. **Back-probability computed on post-move state**
   - Candidate sets and probabilities must be consistent with the *post-move* topology.

### B. Compare direct likelihood vs. approximation
For a single candidate:
- Compute `computeProposedTreeLogLikelihood`
- Compare against explicit TreeLikelihood on a reconstructed temporary tree
- Large mismatch indicates bug in outside partial usage or transition-matrix direction

### C. Sanity checks for partials
- Ensure `partialsWithoutNode` exist for all nodes **needed** by the method used
- For *full proposed tree likelihood*, verify `outsidePartials[grandparent]` is valid

### D. Numerical stability
- Large weight gaps can collapse the softmax and yield near-deterministic proposals
- If necessary, clamp, temper, or rescale weights for stability

---

## 4) Key Insights from Recent Debugging

1. **Effective parent heights are critical**
   - When ignoring a subtree, unary nodes appear. Any to-height computation must skip them.

2. **Outside partials do not depend on `nodes`**
   - Passing only ancestors to `updateByOperatorWithoutNode` is correct, because:
     - Only ancestors need “without node” partials
     - Outside partials are computed top-down regardless of the list

3. **Candidate-set differences can invalidate symmetry tests**
   - If the candidate set changes after a move, weight comparisons are not meaningful
   - Tests should compare only candidates present in both sets or restore topology

4. **Full conditional likelihood mixing is fragile**
   - The inside/outside combination is sensitive to topological changes; partial agreement proved more stable.

5. **Leaf-partial mismatch can mimic deep algorithmic bugs**
   - A small inconsistency in leaf partial encoding can break invariance and detailed-balance diagnostics.
   - Always verify base-case semantics before further recursion-level debugging.

6. **Current theory is empirically supported on simple examples**
   - Including the current node in candidate sets yields near-1 acceptance for targeted Wilson–Balding, consistent with the intended "almost Gibbs" behavior.

7. **LearnedWeights Brownian surrogates use diffusion time, not raw time**
   - For `LearnedWeights` and `PCAWeightsBrownianFullLikelihood`, branch-rate-aware Brownian motion remains Gaussian.
   - The runtime and training surrogates should therefore operate on integrated diffusion length, i.e. branch contributions of `branchRate * branchLength`.
   - In the trainer, covariance entries should be built from cumulative diffusion length to each leaf and to each MRCA, not from raw node heights alone.
   - This applies to strict and relaxed clocks as long as rates are branchwise constant over each edge.

8. **Posterior LearnedWeights training should restore sampled clock state when possible**
   - If posterior `.trees` files contain per-branch `rate` metadata, use those sampled branch rates directly for both covariance construction and recomputed target likelihoods.
   - When rates are not encoded on the tree, pair `trainingTreeFile` with a matching `trainingLogFile` and restore logged state-node values whenever the corresponding parameter IDs are present in the log. This covers strict-clock mean rates and also relaxed-clock latent state such as logged category or quantile parameters.
   - If the paired training log contains the target likelihood column, prefer that logged value over recomputing the likelihood from the trainer-local evaluator so posterior training also reflects variation from non-tree, non-clock parameters.
   - Tree-embedded branch rates take precedence over logged strict-clock means because they preserve branch-specific variation.

9. **IECoR diagnostics must resolve root-relative training files from the workspace root**
   - `LearnedWeightsFitCli` loads `trainingTreeFile` / `trainingLogFile` paths relative to the current process working directory.
   - The `targetedbeast` Ant `fit-report` target forces `dir="${basedir}/.."`, so running it from any shell location still resolves `IECOR_pca.trees` and related files against `/Users/nneureiter/Projects/BEAST`.
   - If a future diagnostic is added outside Ant, keep this cwd requirement in mind or use absolute file paths in the XML.

10. **Quadratic-only LearnedWeights currently does not change the learned IECoR solution**
   - The fresh IECoR `scaleQuadraticPartOnly=true` fit-report converged to the same mixed-sign per-dimension slope vector seen in the corresponding MCMC run: `[0.00754, -0.62096, -0.78814, -0.57373, 0.68933, -0.63222, -0.63093, 0.67850]`.
   - Tree-level fit metrics were unchanged relative to a fresh sign-flip baseline on the same 19945 posterior-tree samples (`learned` Pearson about `0.4269`, Spearman about `0.4073`, RMSE about `118.08`).

11. **Removing the scaled runtime base term hurts local candidate ranking on IECoR**
   - Comparing fresh candidate diagnostics from the same CLI path showed `scaleQuadraticPartOnly=true` made `learned-log` ordering metrics worse than the sign-flip baseline even though the slope vector stayed the same.
   - IECoR candidate summary delta: mean KL improved only trivially (`4.2178 -> 4.2077`), but mean Spearman dropped (`0.1615 -> 0.1492`), top-1 accuracy dropped (`0.1282 -> 0.1026`), and mean exact-top probability dropped (`0.1071 -> 0.1047`).
   - Current working interpretation: moving `logDetPart` to the regression target is neutral for training on IECoR, but unscaling the runtime base/log-det-like term removes a component that still helps the local proposal ranking, despite the mixed-sign learned slopes.

12. **Use labeled source nodes for cross-run candidate-breakdown comparisons**
   - The new `LearnedWeightsFitCli --candidate-breakdown-source <nodeNr-or-id>` diagnostic works, but source node numbers for unlabeled internal nodes are not a stable cross-run selector.
   - Fresh parses can assign different internal-node numbers, so `--candidate-breakdown-source 63` is not reliable for comparing two separate runs unless the node is labeled externally.
   - For reproducible comparisons across runs, prefer leaf/source IDs like `Punjabi`, or compare internal-node diagnostics only within the same parsed run.

13. **On Punjabi, the learned misranking is dominated by the scaled quadratic term, not the base term**
   - Fresh sign-flip and quadratic-only breakdown runs for `--candidate-breakdown-source Punjabi` both place almost all learned probability on `Danish` while the exact distribution is essentially all on `Kashmiri`.
   - In those breakdown TSVs, the raw/scaled base contribution varies only modestly across candidates (roughly `18` to `37`), while the scaled quadratic contribution spans hundreds and determines the ordering.
   - So at least for this representative IECoR case, the local proposal failure is not caused by the base/log-det-like term overwhelming the ranking; the dominant mismatch is in the scaled quadratic term itself.

14. **IECoR embedding geometry is worse than PCA, but Punjabi is not literally embedded near Danish**
   - The new `embedding-diagnostics` report compares Jukes-Cantor corrected sequence distances, PCA distances, and current learned Euclidean distances across all leaf pairs.
   - On IECoR, learned geometry is materially worse than PCA: sequence-vs-learned Pearson `0.6246` and Spearman `0.2881`, versus sequence-vs-PCA Pearson `0.8804` and Spearman `0.5730`.
   - Nearest-neighbor recovery also drops for learned embeddings: exact nearest-neighbor match rate `0.3659` and mean rank of the true sequence-nearest neighbor `4.12`, versus PCA `0.4634` and `2.49`.
   - For Punjabi specifically, learned nearest leaves are still `Bengali`, `Urdu`, and `Kashmiri` (distances about `1.43`, `1.80`, `2.88`), while `Punjabi`-`Danish` is far apart in learned space (`10.79`). This argues against a simple “Punjabi was embedded next to Danish” explanation.
   - Current interpretation: the learned leaf geometry is degraded relative to PCA, but the dramatic Punjabi candidate misattachment to `Danish` is primarily a tree-context / surrogate-scoring problem on top of that imperfect geometry, not a direct leaf-nearest-neighbor collapse.

15. **Runtime sign reversal helps IECoR candidate ranking, but it is not the main fix and it does not rescue Punjabi**
   - A fresh `learnedweights-iecor-signreversal-check` run added direct `reversed-learned-log` diagnostics by negating the learned runtime proposal-log scales while keeping the same fitted embeddings.
   - Reversal improves the learned candidate metrics substantially over the original learned-log baseline on IECoR: mean KL `4.1764 -> 3.8067`, mean Spearman `0.1521 -> 0.3154`, top-1 accuracy `0.1282 -> 0.2564`, and mean exact-top probability `0.1190 -> 0.1588`.
   - The improvement is broad but incomplete: reversed-learned-log beats learned-log on `45/78` sources in Spearman and raises exact-top probability on `56/78`, but it still remains clearly worse than `pca-log` (`meanSpearman 0.8555`, `top1 0.3077`) and also worse than `learned-inverse` on top-1 (`0.3718`).
   - Punjabi is an important counterexample: the exact best destination is still `Kashmiri`, learned-log picks `FrancoProvencal`, and reversed-learned-log flips to `Danish`, with Punjabi's exact-top probability dropping from about `0.0106` to `0.00775`.
   - Current interpretation: there is a real sign component in the learned runtime scoring, but the IECoR failure is not mainly “just flip the sign.” Sign reversal helps globally yet still misses the best available baselines and can actively worsen the signature Punjabi/Danish pathology.

16. **Keep LearnedWeights diagnostics on the native Java path**
   - The Python reference trainer and the Java import/export bridge for externally trained models were removed to keep the code path smaller and avoid maintaining two optimization implementations.
   - `LearnedWeightsFitCli` is the supported fit diagnostic entry point, and it reports only the core tree-level fit summary and per-sample output from the Java model itself.
   - `targetedbeast.tools.LearnedWeightsSampleCli` remains the supported way to export posterior training samples for reuse via `trainingSamplesFile`.

---

## 5) Current baseline for future work

Treat the following as non-regression requirements:

1. Felsenstein unit tests stay green:
   - `FelsensteinWeightsTest`
   - `FelsensteinWeightsWithoutNodeTest`
2. Detailed balance remains green for Felsenstein-based targeted operators.
3. Candidate-set symmetry assumptions in targeted operators remain valid for compared forward/backward scenarios.
4. Leaf-partial semantics remain aligned with tree-likelihood core behavior.

---

## 6) File Map

- **targetedbeast/src/targetedbeast/edgeweights/FelsensteinWeights.java**
  - Partial/likelihood-based edge weights, without-node recursion, outside partials

- **targetedbeast/src/targetedbeast/operators/TargetedWilsonBalding.java**
  - Targeted WB operator, Hastings ratio logic

- **targetedbeast/test/targetedbeast/edgeweights/FelsensteinWeightsWithoutNodeTest.java**
  - Invariance/symmetry tests for partials-based weights

- **targetedbeast/test/targetedbeast/edgeweights/FelsensteinWeightsTest.java**
   - Core property and reconstruction tests for FelsensteinWeights

- **targetedbeast/test/targetedbeast/DetailedBalanceTest.java**
  - Detailed balance validation for targeted operators
