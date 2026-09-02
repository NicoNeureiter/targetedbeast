package targetedbeast.benchmark;

/**
 * Per-operator accumulator for {@link BenchmarkMCMC}.
 *
 * <p>Follows the running-mean bookkeeping style of
 * {@code targetedbeast.operators.AdaptableOperatorSampler} ({@code recordRuntime}, {@code mean_SS}):
 * every quantity is kept as an incrementally-updated mean, so no per-proposal history is retained.
 *
 * <p>Two means are kept over two different counts, matching the subsampling design in
 * {@code targetedbeast-evaluation-pipeline.md} (Phase 1, "instrument must not dominate"):
 * <ul>
 * <li>{@code proposals} - every post-burnin proposal of this operator; backs acceptance rate,
 * mean operator+likelihood time, and the (free) squared-Delta-log-posterior diagnostic.</li>
 * <li>{@code sampledProposals} - only every {@code jumpInterval}-th proposal; backs the ESJD
 * columns and the instrumentation-overhead column, since those require an expensive tree copy.</li>
 * </ul>
 *
 * <p>{@code instrumentationOverheadMeanMs} covers <em>all</em> per-sample instrumentation cost -
 * the tree copy (dominant: 2n-1 {@code Node} allocations) plus every metric's
 * {@code distance(...)} call - not just the metric computation. The tree copy is what makes this
 * cost worth showing at all: it is what {@code jumpInterval} subsamples away from most proposals,
 * and it grows with taxon count exactly where the tier decision (Phase 3) gets made.
 */
public class OperatorStats {

    final String name;
    private final int numMetrics;

    long proposals = 0;
    long accepts = 0;
    private double meanTimeMs = 0;
    private double meanSqDeltaLogPosterior = 0;

    long sampledProposals = 0;
    long sampledAccepts = 0;
    private final double[] meanSqJump;
    private double instrumentationOverheadMeanMs = 0;
    private double primaryJumpSumGivenAccepted = 0;

    OperatorStats(String name, int numMetrics) {
        this.name = name;
        this.numMetrics = numMetrics;
        this.meanSqJump = new double[numMetrics];
    }

    /** Record timing/acceptance for one proposal. Call once per propagateState call. */
    void recordProposal(double operatorPlusLikelihoodTimeMs, boolean accepted) {
        proposals++;
        if (accepted) {
            accepts++;
        }
        meanTimeMs += (operatorPlusLikelihoodTimeMs - meanTimeMs) / proposals;
    }

    /**
     * Record the (free) squared Delta-log-posterior diagnostic. Must be called after
     * {@link #recordProposal} in the same step, since it divides by the updated {@code proposals}.
     */
    void recordSqDeltaLogPosterior(double sqDelta) {
        meanSqDeltaLogPosterior += (sqDelta - meanSqDeltaLogPosterior) / proposals;
    }

    /**
     * Record a jump-metric sample (only called on sampled proposals).
     *
     * @param accepted whether this proposal was accepted
     * @param jump     per-metric jump distance (not squared), same order as the metric list;
     *                 {@code jump[0]} is the primary metric, used for meanJumpGivenAccepted
     * @param instrumentationTimeMs wall time spent computing these distances (tree copy + all metrics)
     */
    void recordSample(boolean accepted, double[] jump, double instrumentationTimeMs) {
        sampledProposals++;
        instrumentationOverheadMeanMs += (instrumentationTimeMs - instrumentationOverheadMeanMs) / sampledProposals;
        for (int i = 0; i < numMetrics; i++) {
            double sq = jump[i] * jump[i];
            meanSqJump[i] += (sq - meanSqJump[i]) / sampledProposals;
        }
        if (accepted) {
            sampledAccepts++;
            if (numMetrics > 0) {
                primaryJumpSumGivenAccepted += jump[0];
            }
        }
    }

    double acceptRate() {
        return proposals == 0 ? Double.NaN : (double) accepts / proposals;
    }

    double meanTimeMs() {
        return meanTimeMs;
    }

    double meanSqDeltaLogPosterior() {
        return meanSqDeltaLogPosterior;
    }

    /**
     * Mean jump distance under the primary (list position 0) metric, conditional on acceptance.
     * {@code NaN} when undefined: no metric configured (so nothing is ever sampled), or no sampled
     * proposal for this operator was accepted yet.
     */
    double meanJumpGivenAccepted() {
        return sampledAccepts == 0 ? Double.NaN : primaryJumpSumGivenAccepted / sampledAccepts;
    }

    /** Mean squared jump for metric {@code i} - ESJD, but per-proposal, not per-second. */
    double esjd(int i) {
        return meanSqJump[i];
    }

    /**
     * ESJD per second of operator+likelihood time - never per second of total instrumented time
     * (Phase 1 constraint 2: "separate clocks").
     */
    double esjdPerSec(int i) {
        return meanTimeMs == 0 ? Double.NaN : esjd(i) / (meanTimeMs / 1000.0);
    }

    double instrumentationOverheadMeanMs() {
        return instrumentationOverheadMeanMs;
    }

    double primaryJumpSumGivenAccepted() {
        return primaryJumpSumGivenAccepted;
    }
}
