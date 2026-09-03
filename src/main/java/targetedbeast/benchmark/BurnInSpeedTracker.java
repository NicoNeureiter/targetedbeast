package targetedbeast.benchmark;

/**
 * Tracks proposals-to-target for the burn-in-speed metric (Phase 3c of
 * {@code targetedbeast-evaluation-pipeline.md}).
 *
 * <p>Target: the first proposal after which the log-posterior stays within a reference band -
 * the reference chain's own 5th-95th percentile of post-burnin log-posterior, computed externally
 * and supplied as {@code lowerBound}/{@code upperBound} - for 5 consecutive proposals. Kept as its
 * own small class, separate from {@link BenchmarkMCMC}, so the "5 consecutive, not 5 total" logic
 * is unit-testable against a synthetic sequence of log-posterior values, without needing to run an
 * actual MCMC chain.
 */
class BurnInSpeedTracker {

    static final int REQUIRED_CONSECUTIVE = 5;

    private final double lowerBound;
    private final double upperBound;

    private boolean reached = false;
    private long target = -1;
    private int consecutiveInBand = 0;
    private long consecutiveStartIndex = -1;
    private long proposalsSeen = 0;

    BurnInSpeedTracker(double lowerBound, double upperBound) {
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
    }

    /**
     * Records one more proposal's log-posterior. Call once per post-burnin proposal, in order; a
     * no-op once the target has already been reached.
     */
    void update(double logPosterior) {
        proposalsSeen++;
        if (reached) {
            return;
        }
        final boolean inBand = logPosterior >= lowerBound && logPosterior <= upperBound;
        if (inBand) {
            if (consecutiveInBand == 0) {
                consecutiveStartIndex = proposalsSeen;
            }
            consecutiveInBand++;
            if (consecutiveInBand >= REQUIRED_CONSECUTIVE) {
                target = consecutiveStartIndex;
                reached = true;
            }
        } else {
            consecutiveInBand = 0;
        }
    }

    boolean isReached() {
        return reached;
    }

    /** 1-based post-burnin proposal index of the first proposal of the qualifying run of 5; -1 if not (yet) reached. */
    long getTarget() {
        return reached ? target : -1;
    }

    long getProposalsSeen() {
        return proposalsSeen;
    }
}
