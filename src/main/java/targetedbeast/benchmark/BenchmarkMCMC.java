package targetedbeast.benchmark;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeMetric;
import beast.base.inference.MCMC;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;
import targetedbeast.operators.AdaptableOperatorSampler;

/**
 * An {@link MCMC} that measures itself while it runs, without perturbing the chain it measures.
 *
 * <p>Overrides only {@link #propagateState(long)} (already the entire propose/accept/reject step
 * in core, see {@code MCMC.java:512}) and {@link #close()}; {@code doLoop()} is untouched, so this
 * class cannot drift from core's debug/checkpoint/error-handling logic. Everything this class
 * records - acceptance, timing, tree jump - is read from state that core already keeps
 * ({@code logAlpha}, {@code oldLogLikelihood}, {@code newLogLikelihood}, and each {@link Operator}'s
 * own accept/reject counters) or copied only on a sparse subsample of proposals.
 *
 * <p>Design is pinned to {@code targetedbeast-evaluation-pipeline.md} Phase 1:
 * <ul>
 * <li>tree copies (the expensive part) happen only on every {@code jumpInterval}-th proposal;</li>
 * <li>ESJD/second is computed against operator+likelihood time only, never against time spent
 * computing metrics (that cost is its own column, {@code metricOverheadMs});</li>
 * <li>the headline is the chain-level row (operator {@value #CHAIN_ROW_NAME}): sum-of(n_i * ESJD_i)
 * over total operator+likelihood time; per-operator rows are diagnostics;</li>
 * <li>the {@link Randomizer} seed is recorded, and no metric may call {@link Randomizer}.</li>
 * </ul>
 */
@Description("MCMC that also writes a '<statsOut>.opstats.tsv' benchmark report: per-operator "
        + "acceptance/timing/tree-jump statistics plus a chain-level headline row. Instrumentation "
        + "reads only what beast.base.inference.MCMC already tracks (logAlpha, "
        + "oldLogLikelihood/newLogLikelihood, per-operator accept/reject counts) and adds timing and "
        + "sparsely-subsampled tree-distance metrics; the chain it runs is otherwise unchanged.")
public class BenchmarkMCMC extends MCMC {

    public static final String CHAIN_ROW_NAME = "__CHAIN__";

    final public Input<String> statsOutInput = new Input<>("statsOut",
            "output prefix; benchmark statistics are written to '<statsOut>.opstats.tsv'. "
            + "If not specified, statistics are accumulated in memory but not written to file.");

    final public Input<List<TreeMetric>> metricInput = new Input<>("metric",
            "tree metrics used to measure jump distance between consecutive states. The first "
            + "metric in this list is primary: it backs the meanJumpGivenAccepted column. "
            + "Invariant: no metric implementation may call Randomizer, or this benchmark would "
            + "perturb the chain it measures.",
            new ArrayList<>());

    final public Input<Integer> jumpIntervalInput = new Input<>("jumpInterval",
            "compute tree-jump metrics on only every jumpInterval-th post-burnin proposal; "
            + "acceptance and timing are still recorded for every proposal. ESJD is a mean, so "
            + "this costs sqrt(jumpInterval) precision in exchange for not letting the instrument "
            + "(a full tree copy every proposal) dominate what it measures.", 10);

    final public Input<String> jumpTraceInput = new Input<>("jumpTrace",
            "optional per-proposal trace file: sampleNr, operator, accepted, then one jump column "
            + "per metric (only written on sampled proposals).");

    // ---- resolved config ----
    private List<TreeMetric> metrics;
    private int jumpInterval;

    // ---- operator bookkeeping (built lazily on first propagateState call, once
    //      operatorSchedule.init()/TargetedOperatorSchedule's operator substitution has run) ----
    private List<Operator> trackedOperators;
    private Map<Operator, Integer> operatorIndex;
    private long[] acceptedCountBefore;
    private long[] rejectedCountBefore;
    private List<Tree> stateTrees;
    private final Map<Operator, OperatorStats> statsByOperator = new LinkedHashMap<>();

    private long postBurninProposalIndex = 0;
    private long recordedSeed = Long.MIN_VALUE;
    private PrintWriter jumpTraceWriter;

    @Override
    public void initAndValidate() {
        super.initAndValidate();
        metrics = metricInput.get();
        jumpInterval = Math.max(1, jumpIntervalInput.get());
    }

    @Override
    protected Operator propagateState(final long sampleNr) {
        ensureTrackingInitialised();

        final boolean postBurnin = sampleNr >= 0;
        final boolean sampled = postBurnin && !metrics.isEmpty()
                && (postBurninProposalIndex % jumpInterval == 0);

        final List<Tree> treesBefore = sampled ? copyStateTrees() : null;
        if (postBurnin) {
            snapshotOperatorCounts();
        }
        final double oldLL = oldLogLikelihood;

        final long t0 = System.nanoTime();
        final Operator op = super.propagateState(sampleNr);
        final long t1 = System.nanoTime();

        if (!postBurnin) {
            // Burn-in: Operator.accept()/reject() are not called by core during burn-in
            // (MCMC.java:550,556), so there is nothing consistent to attribute here either.
            return op;
        }

        if (recordedSeed == Long.MIN_VALUE) {
            recordedSeed = Randomizer.getSeed();
        }

        final Operator leaf = resolveLeaf(op);
        final boolean accepted = leafAccepted(leaf);

        final OperatorStats stats = statsByOperator.computeIfAbsent(leaf,
                o -> new OperatorStats(o.getName(), metrics.size()));
        stats.recordProposal((t1 - t0) / 1e6, accepted);

        // Diagnostic only (GridTuner already optimises this internally - see class javadoc of
        // targetedbeast.weighttransform.GridTuner - so it must not be used to rank weightings).
        // logAlpha == NEGATIVE_INFINITY marks MCMC's "direct reject" branch (MCMC.java:567-582),
        // where newLogLikelihood is never (re)computed and so is stale; no likelihood evaluation
        // was attempted, so the delta is defined as zero.
        final double sqDeltaLogPosterior = (logAlpha == Double.NEGATIVE_INFINITY) ? 0.0
                : (newLogLikelihood - oldLL) * (newLogLikelihood - oldLL);
        stats.recordSqDeltaLogPosterior(sqDeltaLogPosterior);

        if (sampled) {
            final long m0 = System.nanoTime();
            final double[] jump = new double[metrics.size()];
            for (int i = 0; i < metrics.size(); i++) {
                final TreeMetric metric = metrics.get(i);
                double sumSq = 0;
                for (int j = 0; j < stateTrees.size(); j++) {
                    final double d = metric.distance(treesBefore.get(j), stateTrees.get(j));
                    sumSq += d * d;
                }
                jump[i] = Math.sqrt(sumSq);
            }
            final long m1 = System.nanoTime();
            stats.recordSample(accepted, jump, (m1 - m0) / 1e6);
            writeJumpTraceRow(sampleNr, leaf, accepted, jump);
        }

        postBurninProposalIndex++;
        return op;
    }

    @Override
    public void close() {
        super.close();
        writeStatsFile();
        if (jumpTraceWriter != null) {
            jumpTraceWriter.close();
        }
    }

    // ---- operator-tree bookkeeping ----

    private void ensureTrackingInitialised() {
        if (trackedOperators != null) {
            return;
        }
        trackedOperators = new ArrayList<>();
        final Set<Operator> seen = new LinkedHashSet<>();
        collectOperators(operatorSchedule.operators, seen);
        trackedOperators.addAll(seen);

        operatorIndex = new HashMap<>();
        for (int i = 0; i < trackedOperators.size(); i++) {
            operatorIndex.put(trackedOperators.get(i), i);
        }
        acceptedCountBefore = new long[trackedOperators.size()];
        rejectedCountBefore = new long[trackedOperators.size()];

        stateTrees = new ArrayList<>();
        for (StateNode sn : state.stateNodeInput.get()) {
            if (sn instanceof Tree) {
                stateTrees.add((Tree) sn);
            }
        }

        final String jumpTracePath = jumpTraceInput.get();
        if (jumpTracePath != null) {
            try {
                jumpTraceWriter = new PrintWriter(new FileWriter(jumpTracePath));
            } catch (IOException e) {
                throw new RuntimeException("Could not open jumpTrace file: " + jumpTracePath, e);
            }
            final StringBuilder header = new StringBuilder("sampleNr\toperator\taccepted");
            for (TreeMetric metric : metrics) {
                header.append('\t').append("jump_").append(metricName(metric));
            }
            jumpTraceWriter.println(header);
        }
    }

    /**
     * Registers every operator reachable from the schedule, expanding
     * {@link AdaptableOperatorSampler} recursively so its candidates are tracked individually -
     * per Phase 1: "unwrap AdaptableOperatorSampler ... so nested operators are attributed
     * individually rather than to the meta-operator." {@code TargetedOperatorSchedule} needs no
     * separate unwrapping: by the time this runs (lazily, on the first propagateState call) it has
     * already replaced/flattened operators into {@code operatorSchedule.operators} directly.
     */
    private void collectOperators(List<Operator> ops, Set<Operator> seen) {
        for (Operator op : ops) {
            if (seen.add(op) && op instanceof AdaptableOperatorSampler) {
                collectOperators(((AdaptableOperatorSampler) op).operatorsInput.get(), seen);
            }
        }
    }

    private void snapshotOperatorCounts() {
        for (int i = 0; i < trackedOperators.size(); i++) {
            final Operator op = trackedOperators.get(i);
            acceptedCountBefore[i] = op.get_m_nNrAccepted();
            rejectedCountBefore[i] = op.get_m_nNrRejected();
        }
    }

    /**
     * Resolves the operator returned by {@code propagateState} down to the leaf that actually ran,
     * by finding which candidate's own accept/reject counters changed - since
     * {@code AdaptableOperatorSampler.accept()/reject()} always delegates to the chosen candidate
     * (see {@code AdaptableOperatorSampler.java:530,543}) in addition to updating its own. Falls
     * back to the sampler itself if no candidate changed (its internal proposal-count guard can
     * reject before delegating - {@code AdaptableOperatorSampler.java:263-264}).
     */
    private Operator resolveLeaf(Operator op) {
        Operator current = op;
        while (current instanceof AdaptableOperatorSampler) {
            final AdaptableOperatorSampler sampler = (AdaptableOperatorSampler) current;
            Operator next = null;
            for (Operator sub : sampler.operatorsInput.get()) {
                final Integer idx = operatorIndex.get(sub);
                if (idx == null) {
                    continue;
                }
                if (sub.get_m_nNrAccepted() != acceptedCountBefore[idx]
                        || sub.get_m_nNrRejected() != rejectedCountBefore[idx]) {
                    next = sub;
                    break;
                }
            }
            if (next == null) {
                break;
            }
            current = next;
        }
        return current;
    }

    private boolean leafAccepted(Operator leaf) {
        final Integer idx = operatorIndex.get(leaf);
        // leaf is always in trackedOperators: it is either a top-level schedule operator, or a
        // candidate reachable by recursing into an AdaptableOperatorSampler - both are covered by
        // collectOperators.
        return leaf.get_m_nNrAccepted() != acceptedCountBefore[idx];
    }

    private List<Tree> copyStateTrees() {
        final List<Tree> copies = new ArrayList<>(stateTrees.size());
        for (Tree t : stateTrees) {
            copies.add(t.copy());
        }
        return copies;
    }

    private void writeJumpTraceRow(long sampleNr, Operator leaf, boolean accepted, double[] jump) {
        if (jumpTraceWriter == null) {
            return;
        }
        final StringBuilder row = new StringBuilder();
        row.append(sampleNr).append('\t').append(leaf.getID()).append('\t').append(accepted);
        for (double j : jump) {
            row.append('\t').append(j);
        }
        jumpTraceWriter.println(row);
    }

    private static String metricName(TreeMetric metric) {
        if (metric instanceof beast.base.core.BEASTInterface bi && bi.getID() != null) {
            return bi.getID();
        }
        return metric.getClass().getSimpleName();
    }

    /** Test-only access to the per-operator accumulators, keyed by the leaf operator. */
    Map<Operator, OperatorStats> getOperatorStatsForTesting() {
        return statsByOperator;
    }

    List<TreeMetric> getMetricsForTesting() {
        return metrics;
    }

    // ---- output ----

    private void writeStatsFile() {
        final String prefix = statsOutInput.get();
        if (prefix == null) {
            return;
        }
        final String path = prefix + ".opstats.tsv";
        final long seed = recordedSeed == Long.MIN_VALUE ? Randomizer.getSeed() : recordedSeed;

        try (PrintWriter out = new PrintWriter(new FileWriter(path))) {
            out.println(header());

            long totalProposals = 0;
            long totalAccepts = 0;
            double totalTimeMs = 0;
            double totalSqDeltaLogPosterior = 0;
            long totalSampledProposals = 0;
            long totalSampledAccepts = 0;
            double totalMetricOverheadMs = 0;
            double totalPrimaryJumpSumGivenAccepted = 0;
            final double[] totalWeightedEsjd = new double[metrics.size()];

            for (OperatorStats s : statsByOperator.values()) {
                out.println(formatRow(seed, s.name, s.proposals, s.accepts, s.acceptRate(),
                        s.meanTimeMs(), s.meanJumpGivenAccepted(), esjdArray(s), esjdPerSecArray(s),
                        s.metricOverheadMeanMs(), s.meanSqDeltaLogPosterior()));

                totalProposals += s.proposals;
                totalAccepts += s.accepts;
                totalTimeMs += s.meanTimeMs() * s.proposals;
                totalSqDeltaLogPosterior += s.meanSqDeltaLogPosterior() * s.proposals;
                totalSampledProposals += s.sampledProposals;
                totalSampledAccepts += s.sampledAccepts;
                totalMetricOverheadMs += s.metricOverheadMeanMs() * s.sampledProposals;
                totalPrimaryJumpSumGivenAccepted += s.primaryJumpSumGivenAccepted();
                for (int i = 0; i < metrics.size(); i++) {
                    totalWeightedEsjd[i] += s.esjd(i) * s.proposals;
                }
            }

            // Headline: chain-level row. Sum_i(n_i . ESJD_i) / T_total, T_total = operator+
            // likelihood time only (never total instrumented time - Phase 1 constraint 2).
            final double chainAcceptRate = totalProposals == 0 ? Double.NaN
                    : (double) totalAccepts / totalProposals;
            final double chainMeanTimeMs = totalProposals == 0 ? Double.NaN
                    : totalTimeMs / totalProposals;
            final double chainMeanJumpGivenAccepted = totalSampledAccepts == 0 ? Double.NaN
                    : totalPrimaryJumpSumGivenAccepted / totalSampledAccepts;
            final double chainMetricOverheadMs = totalSampledProposals == 0 ? Double.NaN
                    : totalMetricOverheadMs / totalSampledProposals;
            final double chainSqDeltaLogPosterior = totalProposals == 0 ? Double.NaN
                    : totalSqDeltaLogPosterior / totalProposals;
            final double[] chainEsjd = new double[metrics.size()];
            final double[] chainEsjdPerSec = new double[metrics.size()];
            final double totalTimeSec = totalTimeMs / 1000.0;
            for (int i = 0; i < metrics.size(); i++) {
                chainEsjd[i] = totalProposals == 0 ? Double.NaN : totalWeightedEsjd[i] / totalProposals;
                chainEsjdPerSec[i] = totalTimeSec == 0 ? Double.NaN : totalWeightedEsjd[i] / totalTimeSec;
            }

            out.println(formatRow(seed, CHAIN_ROW_NAME, totalProposals, totalAccepts, chainAcceptRate,
                    chainMeanTimeMs, chainMeanJumpGivenAccepted, chainEsjd, chainEsjdPerSec,
                    chainMetricOverheadMs, chainSqDeltaLogPosterior));
        } catch (IOException e) {
            throw new RuntimeException("Could not write benchmark statistics to " + path, e);
        }
    }

    private double[] esjdArray(OperatorStats s) {
        final double[] a = new double[metrics.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = s.esjd(i);
        }
        return a;
    }

    private double[] esjdPerSecArray(OperatorStats s) {
        final double[] a = new double[metrics.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = s.esjdPerSec(i);
        }
        return a;
    }

    private String header() {
        final StringBuilder h = new StringBuilder(
                "seed\toperator\tproposals\taccepts\tacceptRate\tmeanTimeMs\tmeanJumpGivenAccepted");
        for (TreeMetric metric : metrics) {
            h.append("\tesjd_").append(metricName(metric));
        }
        for (TreeMetric metric : metrics) {
            h.append("\tesjdPerSec_").append(metricName(metric));
        }
        h.append("\tmetricOverheadMs\tsqDeltaLogPosterior_DIAGNOSTIC_NOT_A_HEADLINE");
        return h.toString();
    }

    private String formatRow(long seed, String name, long proposals, long accepts, double acceptRate,
            double meanTimeMs, double meanJumpGivenAccepted, double[] esjd, double[] esjdPerSec,
            double metricOverheadMs, double sqDeltaLogPosterior) {
        final StringBuilder row = new StringBuilder();
        row.append(seed).append('\t').append(name).append('\t').append(proposals).append('\t')
                .append(accepts).append('\t').append(acceptRate).append('\t').append(meanTimeMs)
                .append('\t').append(meanJumpGivenAccepted);
        for (double v : esjd) {
            row.append('\t').append(v);
        }
        for (double v : esjdPerSec) {
            row.append('\t').append(v);
        }
        row.append('\t').append(metricOverheadMs).append('\t').append(sqDeltaLogPosterior);
        return row.toString();
    }
}
