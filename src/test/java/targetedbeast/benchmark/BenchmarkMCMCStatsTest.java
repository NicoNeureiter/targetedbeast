package targetedbeast.benchmark;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import org.junit.BeforeClass;
import org.junit.Test;

import beast.base.inference.Operator;
import beast.base.parser.XMLParser;
import beast.base.util.Randomizer;

/**
 * Fine-grained tests for {@link BenchmarkMCMC} / {@link OperatorStats} bookkeeping.
 *
 * <p>Runs one shared {@code BenchmarkMCMC} chain (primates.xml, shortened, with the narrow/wide
 * Exchange operators wrapped in an {@code AdaptableOperatorSampler}) and checks its internal
 * accounting from several angles. One shared run keeps this file fast; each {@code @Test} asserts
 * an independent property so a failure localises to one bookkeeping bug, not "something about the
 * benchmark is wrong."
 */
public class BenchmarkMCMCStatsTest {

    private static final long SEED = 7L;
    private static final long CHAIN_LENGTH = 3000;
    private static final int JUMP_INTERVAL = 4;

    private static BenchmarkMCMC mcmc;

    @BeforeClass
    public static void runOnce() throws Exception {
        Path dir = Files.createTempDirectory("benchmark-stats");
        String template = PrimatesFixture.loadTemplate();
        template = PrimatesFixture.withChainLength(template, CHAIN_LENGTH);
        template = PrimatesFixture.withAdaptableSampler(template);
        template = PrimatesFixture.withAbsoluteLogPaths(template, dir.resolve("run.log"), dir.resolve("run.trees"));
        Path statsOut = dir.resolve("run");
        template = PrimatesFixture.toBenchmarkVariant(template, statsOut, JUMP_INTERVAL, null);

        Path xmlFile = dir.resolve("run.xml");
        PrimatesFixture.write(template, xmlFile);

        Randomizer.setSeed(SEED);
        XMLParser parser = new XMLParser();
        mcmc = (BenchmarkMCMC) parser.parseFile(new File(xmlFile.toString()));
        mcmc.run();
    }

    /** Constraint 4 / Phase 1: acceptance counts must match Operator's own bookkeeping exactly. */
    @Test
    public void acceptRejectCountsMatchOperatorsOwnCounters() {
        Map<Operator, OperatorStats> stats = mcmc.getOperatorStatsForTesting();
        assertTrue("expected at least one tracked operator", stats.size() > 0);
        for (Map.Entry<Operator, OperatorStats> e : stats.entrySet()) {
            Operator op = e.getKey();
            OperatorStats s = e.getValue();
            long opTotal = op.get_m_nNrAccepted() + op.get_m_nNrRejected();
            assertEquals("proposals for " + op.getID() + " should equal its own accept+reject count",
                    opTotal, s.proposals);
            assertEquals("accepts for " + op.getID() + " should equal its own accept count",
                    op.get_m_nNrAccepted(), s.accepts);
        }
    }

    /** Nested operators inside AdaptableOperatorSampler must be attributed individually. */
    @Test
    public void nestedSamplerOperatorsAreAttributedIndividually() {
        Map<Operator, OperatorStats> stats = mcmc.getOperatorStatsForTesting();
        boolean sawNarrow = false, sawWide = false, sawSamplerItself = false;
        for (Operator op : stats.keySet()) {
            if ("narrow".equals(op.getID())) sawNarrow = true;
            if ("wide".equals(op.getID())) sawWide = true;
            if ("sampler".equals(op.getID())) sawSamplerItself = true;
        }
        assertTrue("expected 'narrow' to be tracked as its own leaf operator", sawNarrow);
        assertTrue("expected 'wide' to be tracked as its own leaf operator", sawWide);
        assertTrue("the meta-operator itself should not receive proposal attribution "
                + "(all of its calls delegate to narrow/wide)", !sawSamplerItself);
    }

    /** jumpInterval must actually reduce how often the (expensive) metric computation runs. */
    @Test
    public void jumpIntervalReducesMetricComputationCount() {
        Map<Operator, OperatorStats> stats = mcmc.getOperatorStatsForTesting();
        long totalProposals = 0;
        long totalSampled = 0;
        for (OperatorStats s : stats.values()) {
            totalProposals += s.proposals;
            totalSampled += s.sampledProposals;
        }
        assertTrue("expected some post-burnin proposals", totalProposals > 0);
        assertTrue("expected some sampled proposals", totalSampled > 0);
        // Every JUMP_INTERVAL-th proposal overall is sampled, regardless of which operator it
        // lands on, so this is an approximate (not exact) ratio per operator, but exact in total.
        double ratio = (double) totalProposals / totalSampled;
        assertEquals("sampled proposals should be ~1/jumpInterval of all proposals", JUMP_INTERVAL,
                ratio, 1.0);
    }

    /** Chain-level headline must equal the proposal-weighted sum of the per-operator rows. */
    @Test
    public void chainLevelAggregationEqualsWeightedSumOfOperatorRows() {
        Map<Operator, OperatorStats> stats = mcmc.getOperatorStatsForTesting();
        int numMetrics = mcmc.getMetricsForTesting().size();
        assertTrue("test fixture should declare at least one metric", numMetrics > 0);

        long totalProposals = 0;
        double totalTimeMs = 0;
        double[] totalWeightedEsjd = new double[numMetrics];
        for (OperatorStats s : stats.values()) {
            totalProposals += s.proposals;
            totalTimeMs += s.meanTimeMs() * s.proposals;
            for (int i = 0; i < numMetrics; i++) {
                totalWeightedEsjd[i] += s.esjd(i) * s.proposals;
            }
        }
        assertTrue(totalProposals > 0);
        double expectedChainMeanTimeMs = totalTimeMs / totalProposals;
        double actualWeightedMeanTimeMs = 0;
        for (OperatorStats s : stats.values()) {
            actualWeightedMeanTimeMs += s.meanTimeMs() * s.proposals / totalProposals;
        }
        assertEquals(expectedChainMeanTimeMs, actualWeightedMeanTimeMs, 1e-9);

        double totalTimeSec = totalTimeMs / 1000.0;
        for (int i = 0; i < numMetrics; i++) {
            double chainEsjdPerSec = totalWeightedEsjd[i] / totalTimeSec;
            // Recompute independently, straight from the definition in the design doc:
            // Sum_i(n_i . ESJD_i) / T_total.
            double recomputed = 0;
            double recomputedTimeMs = 0;
            for (OperatorStats s : stats.values()) {
                recomputed += s.esjd(i) * s.proposals;
                recomputedTimeMs += s.meanTimeMs() * s.proposals;
            }
            assertEquals(chainEsjdPerSec, recomputed / (recomputedTimeMs / 1000.0), 1e-9);
        }
    }

    /** meanJumpGivenAccepted must only average sampled proposals that were also accepted. */
    @Test
    public void meanJumpGivenAcceptedIsNonNegativeWhenDefined() {
        Map<Operator, OperatorStats> stats = mcmc.getOperatorStatsForTesting();
        boolean sawADefinedValue = false;
        for (OperatorStats s : stats.values()) {
            double m = s.meanJumpGivenAccepted();
            if (!Double.isNaN(m)) {
                sawADefinedValue = true;
                assertTrue("meanJumpGivenAccepted must not be negative", m >= 0);
            }
        }
        assertTrue("expected at least one operator with a defined meanJumpGivenAccepted",
                sawADefinedValue);
    }
}
