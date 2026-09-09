package targetedbeast.edgeweights.learnedweights;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.likelihood.TreeLikelihood;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.inference.parameter.RealScalarParam;
import targetedbeast.edgeweights.LearnedWeights;

public class LearnedWeightsFitReporterTest {

    @Test
    public void evaluate_reportsConsistentLogDetAndQuadraticDecomposition() {
        Fixture fixture = createFixture();

        LearnedWeights learnedWeights = new LearnedWeights();
        learnedWeights.initByName(
                "data", fixture.alignment,
                "tree", fixture.tree,
                "likelihood", fixture.likelihood,
                "dimension", 2,
                "nTrainingTrees", 2,
                "maxIterations", 20,
                "learningRate", 0.05,
                "offset", 0.01);

        LearnedWeightsFitReporter.Report report = LearnedWeightsFitReporter.evaluate(learnedWeights);

        assertTrue(report.rows.size() > 0);
        assertTrue(Double.isFinite(report.learned.rawPearson));
        assertTrue(Double.isFinite(report.learned.logDetPearson));
        assertTrue(Double.isFinite(report.learned.quadraticPearson));
        assertTrue(Double.isFinite(report.learned.targetVsRootHeightPearson));
        assertTrue(Double.isFinite(report.learned.targetVsTotalBranchLengthPearson));
        assertTrue(Double.isFinite(report.learned.targetVsTotalDiffusionLengthPearson));
        assertTrue(Double.isFinite(report.learned.logDetVsRootHeightPearson));
        assertTrue(Double.isFinite(report.learned.logDetVsTotalBranchLengthPearson));
        assertTrue(Double.isFinite(report.learned.logDetVsTotalDiffusionLengthPearson));
        for (LearnedWeightsFitReporter.Row row : report.rows) {
            assertEquals(row.predictedLearned, row.logDetPart + row.learnedQuadraticPart, 1e-10);
            assertEquals(row.predictedPca, row.logDetPart + row.pcaQuadraticPart, 1e-10);
            assertTrue(Double.isFinite(row.rootHeight));
            assertTrue(Double.isFinite(row.totalBranchLength));
            assertTrue(Double.isFinite(row.totalDiffusionLength));
        }
    }

    @Test
    public void writeOutputs_includeDecompositionColumns() throws Exception {
        Fixture fixture = createFixture();

        LearnedWeights learnedWeights = new LearnedWeights();
        learnedWeights.initByName(
                "data", fixture.alignment,
                "tree", fixture.tree,
                "likelihood", fixture.likelihood,
                "dimension", 2,
                "nTrainingTrees", 2,
                "maxIterations", 10,
                "learningRate", 0.05,
                "offset", 0.01);

        LearnedWeightsFitReporter.Report report = LearnedWeightsFitReporter.evaluate(learnedWeights);
        Path rowsFile = Files.createTempFile("learnedweights-fit-reporter", ".csv");
        Path summaryFile = Files.createTempFile("learnedweights-fit-summary", ".tsv");
        try {
            LearnedWeightsFitReporter.writeRowsCsv(rowsFile, report);
            LearnedWeightsFitReporter.writeSummaryTsv(summaryFile, report);

            String rowsHeader = Files.readAllLines(rowsFile).get(0);
            String summaryHeader = Files.readAllLines(summaryFile).get(0);
            String consoleSummary = LearnedWeightsFitReporter.formatConsoleSummary(report);

            assertTrue(rowsHeader.contains("logDetPart"));
            assertTrue(rowsHeader.contains("learnedQuadraticPart"));
            assertTrue(rowsHeader.contains("pcaQuadraticPart"));
            assertTrue(rowsHeader.contains("rootHeight"));
            assertTrue(rowsHeader.contains("totalBranchLength"));
            assertTrue(rowsHeader.contains("totalDiffusionLength"));
            assertTrue(summaryHeader.contains("rawPearson"));
            assertTrue(summaryHeader.contains("logDetPearson"));
            assertTrue(summaryHeader.contains("quadraticPearson"));
            assertTrue(summaryHeader.contains("targetVsRootHeightPearson"));
            assertTrue(summaryHeader.contains("targetVsTotalBranchLengthPearson"));
            assertTrue(summaryHeader.contains("targetVsTotalDiffusionLengthPearson"));
            assertTrue(summaryHeader.contains("logDetVsRootHeightPearson"));
            assertTrue(summaryHeader.contains("logDetVsTotalBranchLengthPearson"));
            assertTrue(summaryHeader.contains("logDetVsTotalDiffusionLengthPearson"));
            assertTrue(consoleSummary.contains("rawPearson="));
            assertTrue(consoleSummary.contains("logDetPearson="));
            assertTrue(consoleSummary.contains("quadraticPearson="));
            assertTrue(consoleSummary.contains("targetVsRootHeightPearson="));
            assertTrue(consoleSummary.contains("targetVsTotalDiffusionLengthPearson="));
            assertTrue(consoleSummary.contains("logDetVsTotalDiffusionLengthPearson="));
        } finally {
            Files.deleteIfExists(rowsFile);
            Files.deleteIfExists(summaryFile);
        }
    }

    @Test
    public void withGrouped_appendsGroupedBlockWithoutChangingExistingOutput() throws Exception {
        Fixture fixture = createFixture();

        LearnedWeights learnedWeights = new LearnedWeights();
        learnedWeights.initByName(
                "data", fixture.alignment,
                "tree", fixture.tree,
                "likelihood", fixture.likelihood,
                "dimension", 2,
                "nTrainingTrees", 2,
                "maxIterations", 10,
                "learningRate", 0.05,
                "offset", 0.01);

        LearnedWeightsFitReporter.Report ungroupedReport = LearnedWeightsFitReporter.evaluate(learnedWeights);
        assertEquals(null, ungroupedReport.grouped);

        LearnedWeightsGroupedMetrics.GroupedMetrics fakeMetrics = new LearnedWeightsGroupedMetrics.GroupedMetrics(
                3, 2, 1, 0.5, 0.25, 0.25, 0.75, 0.75);
        LearnedWeightsGroupedMetrics.GroupedReport fakeGrouped =
                new LearnedWeightsGroupedMetrics.GroupedReport(fakeMetrics, fakeMetrics);
        LearnedWeightsFitReporter.Report groupedReport = ungroupedReport.withGrouped(fakeGrouped);

        // The ungrouped-path outputs must be identical whether or not grouped data is later attached.
        Path summaryUngrouped = Files.createTempFile("learnedweights-fit-summary-ungrouped", ".tsv");
        Path summaryGrouped = Files.createTempFile("learnedweights-fit-summary-grouped", ".tsv");
        try {
            LearnedWeightsFitReporter.writeSummaryTsv(summaryUngrouped, ungroupedReport);
            LearnedWeightsFitReporter.writeSummaryTsv(summaryGrouped, groupedReport);

            java.util.List<String> ungroupedLines = Files.readAllLines(summaryUngrouped);
            java.util.List<String> groupedLines = Files.readAllLines(summaryGrouped);
            assertEquals(ungroupedLines, groupedLines.subList(0, ungroupedLines.size()));
            assertTrue(groupedLines.size() > ungroupedLines.size());
            assertTrue(groupedLines.get(ungroupedLines.size() + 1).contains("groupsSeen"));

            String consoleUngrouped = LearnedWeightsFitReporter.formatConsoleSummary(ungroupedReport);
            String consoleGrouped = LearnedWeightsFitReporter.formatConsoleSummary(groupedReport);
            assertTrue(consoleGrouped.startsWith(consoleUngrouped));
            assertTrue(consoleGrouped.contains("meanNdcg="));
        } finally {
            Files.deleteIfExists(summaryUngrouped);
            Files.deleteIfExists(summaryGrouped);
        }
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