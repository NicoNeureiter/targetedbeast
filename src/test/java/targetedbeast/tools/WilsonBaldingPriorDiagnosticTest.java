package targetedbeast.tools;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.junit.Before;
import org.junit.Test;

import beast.base.spec.evolution.speciation.YuleModel;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.CompoundDistribution;
import targetedbeast.edgeweights.CombinedEdgeWeights;
import targetedbeast.edgeweights.ConstantWeights;
import targetedbeast.edgeweights.PriorHeuristicWeights;

public class WilsonBaldingPriorDiagnosticTest {

    private static final String TEST_NEWICK = "(((A:0.1,B:0.1):0.2,C:0.3):0.2,(D:0.15,E:0.15):0.35);";

    private Tree tree;
    private YuleModel prior;
    private ConstantWeights likelihoodWeights;
    private PriorHeuristicWeights heuristicWeights;
    private CombinedEdgeWeights combinedWeights;

    @Before
    public void setUp() {
        tree = parseTree(TEST_NEWICK);

        prior = new YuleModel();
        prior.initByName("tree", tree, "birthDiffRate", "2.0");

        likelihoodWeights = new ConstantWeights();
        likelihoodWeights.initByName("tree", tree);

        heuristicWeights = new PriorHeuristicWeights();
        heuristicWeights.initByName(
                "tree", tree,
                "temperature", 1.0,
                "sourceBranchScale", 1.0,
                "attachmentAgeScale", 0.25,
                "heightIncreaseScale", 1.0,
                "maxNewBranchScale", 1.0,
                "minWeight", 1e-6,
                "maxWeight", 1e6);

        combinedWeights = new CombinedEdgeWeights();
        combinedWeights.initByName("edgeWeights", likelihoodWeights, "edgeWeights", heuristicWeights);
    }

    @Test
    public void diagnosticRestoresPriorAndProducesRows() {
        double baseline = prior.calculateLogP();

        WilsonBaldingPriorDiagnostic.Report report = WilsonBaldingPriorDiagnostic.analyze(
                tree,
                prior,
                likelihoodWeights,
                heuristicWeights,
                combinedWeights,
                new WilsonBaldingPriorDiagnostic.Config(2.0, true, 7L, 0));

        assertFalse(report.rows().isEmpty());
        assertTrue(report.aggregateSummary().sourceCount() > 0);
        assertTrue(report.aggregateSummary().candidateRowCount() == report.rows().size());
        String consoleSummary = WilsonBaldingPriorDiagnostic.formatConsoleSummary(report);
        assertTrue(consoleSummary.contains("average rank of exact best"));
        assertTrue(consoleSummary.contains("average Spearman vs exact order"));
        assertTrue(consoleSummary.contains("average Pearson vs exact prior delta"));
        assertTrue(Math.abs(prior.calculateLogP() - baseline) < 1e-10);
    }

    @Test
    public void combinedRankingMatchesHeuristicWhenLikelihoodIsConstant() {
        WilsonBaldingPriorDiagnostic.Report report = WilsonBaldingPriorDiagnostic.analyze(
                tree,
                prior,
                likelihoodWeights,
                heuristicWeights,
                combinedWeights,
                new WilsonBaldingPriorDiagnostic.Config(2.0, true, 7L, 0));

        Map<Integer, List<WilsonBaldingPriorDiagnostic.Row>> grouped = report.rows().stream()
                .collect(Collectors.groupingBy(WilsonBaldingPriorDiagnostic.Row::sourceNr));

        boolean sawMultiCandidateSource = false;
        for (List<WilsonBaldingPriorDiagnostic.Row> rows : grouped.values()) {
            if (rows.size() < 2) {
                continue;
            }
            sawMultiCandidateSource = true;
            WilsonBaldingPriorDiagnostic.Row heuristicBest = rows.stream()
                    .max(Comparator.comparingDouble(WilsonBaldingPriorDiagnostic.Row::heuristicSelectionMass))
                    .orElseThrow();
            WilsonBaldingPriorDiagnostic.Row combinedBest = rows.stream()
                    .max(Comparator.comparingDouble(WilsonBaldingPriorDiagnostic.Row::combinedSelectionMass))
                    .orElseThrow();
            assertTrue(heuristicBest.candidateNr() == combinedBest.candidateNr());
        }
        assertTrue(sawMultiCandidateSource);
            assertEquals(
                report.aggregateSummary().averageHeuristicRankOfExactBest(),
                report.aggregateSummary().averageCombinedRankOfExactBest(),
                1e-12);
            assertEquals(
                report.aggregateSummary().averageHeuristicSpearman(),
                report.aggregateSummary().averageCombinedSpearman(),
                1e-12);
            assertEquals(
                report.aggregateSummary().averageHeuristicPearson(),
                report.aggregateSummary().averageCombinedPearson(),
                1e-12);
    }

    @Test
    public void cliRejectsCompoundDistributionPrior() {
        CompoundDistribution compoundPrior = new CompoundDistribution();
        compoundPrior.setID("prior");
        compoundPrior.initByName("distribution", prior);

        try {
            WilsonBaldingPriorDiagnosticCli.validateDiagnosticPrior(compoundPrior, "prior");
            fail("Expected compound diagnostic prior to be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("exact tree prior"));
            assertTrue(expected.getMessage().contains("--prior-id"));
        }
    }

    private static Tree parseTree(String newick) {
        Tree parsedTree = new TreeParser();
        ((TreeParser) parsedTree).initByName(
                "newick", newick,
                "IsLabelledNewick", true,
                "taxa", null);
        return parsedTree;
    }
}