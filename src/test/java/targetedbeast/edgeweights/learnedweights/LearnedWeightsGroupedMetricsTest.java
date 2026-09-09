package targetedbeast.edgeweights.learnedweights;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.likelihood.TreeLikelihood;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.inference.parameter.RealScalarParam;
import targetedbeast.edgeweights.LearnedWeights;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsGroupedMetrics.GroupedMetrics;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsGroupedMetrics.GroupedReport;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsTreeTrainer.TrainingTreeSample;

/**
 * Hand-checkable ground truth for {@link LearnedWeightsGroupedMetrics#computeGroupedMetrics}.
 *
 * <p>All groups below use 3 items with true target values t = [10, 30, 20] for items [0, 1, 2],
 * so the true (descending) ranking is: item1 (rank1, gain3) > item2 (rank2, gain2) > item0 (rank3, gain1).
 */
public class LearnedWeightsGroupedMetricsTest {

    private static final double[] TRUE_VALUES_3 = {10.0, 30.0, 20.0};

    @Test
    public void computeGroupedMetrics_perfectPrediction_hitsAllMetricsExactly() {
        long[] groupIds = {1, 1, 1};
        double[] predicted = TRUE_VALUES_3.clone();

        GroupedMetrics m = LearnedWeightsGroupedMetrics.computeGroupedMetrics(groupIds, predicted, TRUE_VALUES_3);

        assertEquals(1, m.groupsSeen);
        assertEquals(1, m.groupsUsable);
        assertEquals(1, m.spearmanGroupsUsable);
        assertEquals(1.0, m.top1Accuracy, 1e-12);
        assertEquals(1.0, m.meanSpearman, 1e-12);
        assertEquals(1.0, m.medianSpearman, 1e-12);
        assertEquals(1.0, m.meanNdcg, 1e-12);
        assertEquals(1.0, m.medianNdcg, 1e-12);
    }

    @Test
    public void computeGroupedMetrics_fullyReversedPrediction_hitsAnalyticWorstCase() {
        long[] groupIds = {1, 1, 1};
        // Predicted order is the exact reverse of the true order: item0 highest predicted (true
        // rank 3), item2 middle (true rank 2, unchanged), item1 lowest predicted (true rank 1).
        double[] predicted = {100.0, 1.0, 50.0};

        GroupedMetrics m = LearnedWeightsGroupedMetrics.computeGroupedMetrics(groupIds, predicted, TRUE_VALUES_3);

        assertEquals(1, m.groupsUsable);
        // argmax(predicted) = item0, argmax(true) = item1: miss.
        assertEquals(0.0, m.top1Accuracy, 1e-12);
        // Rank correlation of a fully reversed 3-item ranking is exactly -1 (by hand: ranks of
        // predicted = [2,0,1], ranks of true = [0,2,1], Pearson correlation of those = -1).
        assertEquals(-1.0, m.meanSpearman, 1e-12);

        // NDCG lower bound, derived independently of the implementation: with gain_i = n - trueRank_i + 1
        // (gains 1,3,2 for items 0,1,2) and denominators log2(rank+1), the fully-reversed arrangement
        // gives DCG = 1/log2(2) + 3/log2(4) + 2/log2(3), IDCG = 1/log2(4) + 3/log2(2) + 2/log2(3).
        double log2_3 = Math.log(3.0) / Math.log(2.0);
        double dcg = 1.0 / 1.0 + 3.0 / 2.0 + 2.0 / log2_3;
        double idcg = 1.0 / 2.0 + 3.0 / 1.0 + 2.0 / log2_3;
        double expectedNdcg = dcg / idcg;
        assertEquals(expectedNdcg, m.meanNdcg, 1e-12);
        assertTrue("reversed ranking must score strictly below perfect NDCG", m.meanNdcg < 1.0);
    }

    @Test
    public void computeGroupedMetrics_meanAndMedianAcrossMultipleGroups() {
        // Group 1: perfect prediction (NDCG=1, top1 hit). Group 2: fully reversed (NDCG < 1, top1 miss).
        long[] groupIds = {1, 1, 1, 2, 2, 2};
        double[] trueValues = {10.0, 30.0, 20.0, 10.0, 30.0, 20.0};
        double[] predicted = {10.0, 30.0, 20.0, 100.0, 1.0, 50.0};

        GroupedMetrics m = LearnedWeightsGroupedMetrics.computeGroupedMetrics(groupIds, predicted, trueValues);

        assertEquals(2, m.groupsSeen);
        assertEquals(2, m.groupsUsable);
        assertEquals(0.5, m.top1Accuracy, 1e-12);
        assertEquals(0.0, m.meanSpearman, 1e-12); // (1 + -1) / 2
        assertEquals(0.0, m.medianSpearman, 1e-12);
        assertTrue(m.meanNdcg > 0.5 && m.meanNdcg < 1.0);
    }

    @Test
    public void computeGroupedMetrics_excludesNoGroupAndSingletonGroups() {
        long[] groupIds = {
                TrainingTreeSample.NO_GROUP, TrainingTreeSample.NO_GROUP,
                5, // singleton group: excluded from groupsUsable
                7, 7, 7, // usable group of 3
        };
        double[] trueValues = {1.0, 2.0, 3.0, 10.0, 30.0, 20.0};
        double[] predicted = {1.0, 2.0, 3.0, 10.0, 30.0, 20.0};

        GroupedMetrics m = LearnedWeightsGroupedMetrics.computeGroupedMetrics(groupIds, predicted, trueValues);

        // groupsSeen counts distinct groupIds excluding NO_GROUP: {5, 7} = 2.
        assertEquals(2, m.groupsSeen);
        // Only group 7 (size 3) is usable; group 5 (size 1) is excluded.
        assertEquals(1, m.groupsUsable);
        assertEquals(1, m.spearmanGroupsUsable);
        assertEquals(1.0, m.top1Accuracy, 1e-12);
        assertEquals(1.0, m.meanSpearman, 1e-12);
        assertEquals(1.0, m.meanNdcg, 1e-12);
    }

    @Test
    public void computeGroupedMetrics_sizeTwoGroup_countsForTop1AndNdcgButNotSpearman() {
        long[] groupIds = {1, 1};
        double[] trueValues = {10.0, 30.0};
        double[] predicted = {10.0, 30.0};

        GroupedMetrics m = LearnedWeightsGroupedMetrics.computeGroupedMetrics(groupIds, predicted, trueValues);

        assertEquals(1, m.groupsSeen);
        assertEquals(1, m.groupsUsable);
        assertEquals(0, m.spearmanGroupsUsable);
        assertEquals(1.0, m.top1Accuracy, 1e-12);
        assertTrue(Double.isNaN(m.meanSpearman));
        assertTrue(Double.isNaN(m.medianSpearman));
        assertEquals(1.0, m.meanNdcg, 1e-12);
    }

    @Test
    public void computeGroupedMetrics_noUsableGroups_returnsNanAndZeroCounts() {
        long[] groupIds = {TrainingTreeSample.NO_GROUP, 3};
        double[] trueValues = {1.0, 2.0};
        double[] predicted = {1.0, 2.0};

        GroupedMetrics m = LearnedWeightsGroupedMetrics.computeGroupedMetrics(groupIds, predicted, trueValues);

        assertEquals(1, m.groupsSeen);
        assertEquals(0, m.groupsUsable);
        assertEquals(0, m.spearmanGroupsUsable);
        assertTrue(Double.isNaN(m.top1Accuracy));
        assertTrue(Double.isNaN(m.meanSpearman));
        assertTrue(Double.isNaN(m.meanNdcg));
    }

    @Test
    public void evaluate_wiresLearnedAndPcaPredictionsFromRealSamples() {
        Fixture fixture = createFixture();

        LearnedWeights learnedWeights = new LearnedWeights();
        learnedWeights.initByName(
                "data", fixture.alignment,
                "tree", fixture.tree,
                "likelihood", fixture.likelihood,
                "dimension", 2,
                "nTrainingTrees", 6,
                "maxIterations", 10,
                "learningRate", 0.05,
                "offset", 0.01);

        List<TrainingTreeSample> baseSamples = learnedWeights.getTrainingSamples();
        assertTrue(baseSamples.size() >= 4);

        // Re-tag the (ungrouped) posterior samples into synthetic groups of 3 so evaluate()
        // has grouped data to work with; this only exercises the wiring, not exact ranking values.
        List<TrainingTreeSample> grouped = new ArrayList<>();
        long groupId = 0;
        for (int i = 0; i < baseSamples.size(); i++) {
            if (i % 3 == 0) {
                groupId++;
            }
            TrainingTreeSample s = baseSamples.get(i);
            grouped.add(new TrainingTreeSample(
                    s.targetLogLikelihood,
                    s.precisionMatrix,
                    s.logDeterminant,
                    s.logNormalizerDimension,
                    s.rootFreeCovarianceCholeskyLower,
                    s.rootFreeLogDeterminant,
                    s.rootFreeLogNormalizerDimension,
                    s.rootHeight,
                    s.totalBranchLength,
                    s.totalDiffusionLength,
                    groupId));
        }

        GroupedReport report = LearnedWeightsGroupedMetrics.evaluate(learnedWeights, grouped);

        assertTrue(report.learned.groupsSeen > 0);
        assertTrue(report.pca.groupsSeen > 0);
        assertEquals(report.learned.groupsSeen, report.pca.groupsSeen);
        assertTrue(report.learned.groupsUsable <= report.learned.groupsSeen);
        assertTrue(report.pca.groupsUsable <= report.pca.groupsSeen);
    }

    private static Fixture createFixture() {
        Sequence seqA = new Sequence("A", "ACGTACGTAA");
        Sequence seqB = new Sequence("B", "ACGTACGTAT");
        Sequence seqC = new Sequence("C", "ACGTACGTGA");
        Sequence seqD = new Sequence("D", "ACGTACGTGT");

        Alignment alignment = new Alignment();
        alignment.initByName("sequence", seqA,
                "sequence", seqB,
                "sequence", seqC,
                "sequence", seqD);

        Tree tree = new TreeParser();
        ((TreeParser) tree).initByName(
                "newick", "((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);",
                "IsLabelledNewick", true,
                "taxa", alignment);

        JukesCantor jc = new JukesCantor();
        jc.initByName();

        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", jc);

        StrictClockModel strictClock = new StrictClockModel();
        strictClock.initByName("clock.rate", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

        TreeLikelihood likelihood = new TreeLikelihood();
        likelihood.initByName(
                "data", alignment,
                "tree", tree,
                "siteModel", siteModel,
                "branchRateModel", strictClock);

        return new Fixture(alignment, tree, likelihood);
    }

    private record Fixture(Alignment alignment, Tree tree, TreeLikelihood likelihood) {
    }
}
