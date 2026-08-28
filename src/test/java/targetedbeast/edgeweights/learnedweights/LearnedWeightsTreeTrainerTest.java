package targetedbeast.edgeweights.learnedweights;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.spec.evolution.branchratemodel.Base;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.branchratemodel.UCRelaxedClockModel;
import beast.base.spec.evolution.likelihood.GenericTreeLikelihood;
import beast.base.spec.evolution.likelihood.ThreadedTreeLikelihood;
import beast.base.spec.evolution.likelihood.TreeLikelihood;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.inference.distribution.LogNormal;
import beast.base.spec.domain.NonNegativeInt;
import beast.base.spec.inference.parameter.IntVectorParam;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.domain.Real;
import beast.base.spec.inference.parameter.RealScalarParam;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.CholeskyDecomposition;

public class LearnedWeightsTreeTrainerTest {

    @Test
        public void computeCurrentTreeLogLikelihood_reusesExistingLikelihoodAndRestoresRuntimeTree() {
        Fixture fixture = createFixture();
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                fixture.templateLikelihood,
                fixture.alignment,
                1);

                double originalLogLikelihood = fixture.templateLikelihood.calculateLogP();
                String originalTree = fixture.tree.toString();

                Tree alternateTree = new TreeParser();
                ((TreeParser) alternateTree).initByName(
                                "newick", "((A:0.05,C:0.05):0.25,(B:0.10,D:0.10):0.20);",
                                "IsLabelledNewick", true,
                                "taxa", fixture.alignment);

                Double alternateLogLikelihood = trainer.computeCurrentTreeLogLikelihood(alternateTree);

                assertTrue(alternateLogLikelihood != null && Double.isFinite(alternateLogLikelihood));
                assertEquals(originalTree, fixture.tree.toString());
                assertEquals(originalLogLikelihood, fixture.templateLikelihood.calculateLogP(), 1e-12);
    }

    @Test
    public void buildCovarianceMatrix_matchesSharedRootPathLengths() {
        Fixture fixture = createFixture();
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                fixture.templateLikelihood,
                fixture.alignment,
                1);

        double[][] covariance = trainer.buildCovarianceMatrix(fixture.tree);

        double[][] expected = new double[][] {
                {0.30, 0.20, 0.00, 0.00},
                {0.20, 0.30, 0.00, 0.00},
                {0.00, 0.00, 0.30, 0.15},
                {0.00, 0.00, 0.15, 0.30}
        };

        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals("row " + i, expected[i], covariance[i], 1e-12);
        }
    }

        @Test
        public void buildCovarianceMatrix_scalesSharedPathLengthsByStrictClockRate() {
                Fixture fixture = createFixture("2.0");
                LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                                fixture.templateLikelihood,
                                fixture.alignment,
                                1);

                double[][] covariance = trainer.buildCovarianceMatrix(fixture.tree);

                double[][] expected = new double[][] {
                                {0.60, 0.40, 0.00, 0.00},
                                {0.40, 0.60, 0.00, 0.00},
                                {0.00, 0.00, 0.60, 0.30},
                                {0.00, 0.00, 0.30, 0.60}
                };

                for (int i = 0; i < expected.length; i++) {
                        assertArrayEquals("row " + i, expected[i], covariance[i], 1e-12);
                }
        }

        @Test
        public void buildCovarianceMatrix_usesPerBranchDiffusionLengthsForNonUniformClock() {
                Fixture fixture = createFixture((Base) null);
                LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                                fixture.templateLikelihood,
                                fixture.alignment,
                                1);

                double[][] covariance = trainer.buildCovarianceMatrix(fixture.tree);

                double[][] expected = new double[][] {
                                {1.00, 0.80, 0.00, 0.00},
                                {0.80, 1.10, 0.00, 0.00},
                                {0.00, 0.00, 1.80, 1.05},
                                {0.00, 0.00, 1.05, 1.95}
                };

                for (int i = 0; i < expected.length; i++) {
                        assertArrayEquals("row " + i, expected[i], covariance[i], 1e-12);
                }
        }

        @Test
        public void buildCovarianceMatrix_usesRootToTipPathLengthOnDiagonalForSerialSamples() {
                Fixture fixture = createFixture();
                fixture.tree.getNode(0).setHeight(0.05);
                fixture.tree.getNode(1).setHeight(0.0);
                fixture.tree.getNode(2).setHeight(0.10);
                fixture.tree.getNode(3).setHeight(0.20);
                fixture.tree.getNode(4).setHeight(0.30);
                fixture.tree.getNode(5).setHeight(0.35);
                fixture.tree.getRoot().setHeight(0.50);

                LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                                fixture.templateLikelihood,
                                fixture.alignment,
                                1);

                double[][] covariance = trainer.buildCovarianceMatrix(fixture.tree);

                assertEquals(0.45, covariance[0][0], 1e-12);
                assertEquals(0.50, covariance[1][1], 1e-12);
                assertEquals(0.40, covariance[2][2], 1e-12);
                assertEquals(0.30, covariance[3][3], 1e-12);
                assertEquals(0.20, covariance[0][1], 1e-12);
                assertEquals(0.15, covariance[2][3], 1e-12);
                assertEquals(0.0, covariance[0][2], 1e-12);
        }

    @Test
    public void buildTrainingSample_producesSymmetricPrecisionAndCorrectLogDeterminant() {
        Fixture fixture = createFixture();
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                fixture.templateLikelihood,
                fixture.alignment,
                1);

        LearnedWeightsTreeTrainer.TrainingTreeSample sample = trainer.buildTrainingSample(fixture.tree, -123.0);

        assertEquals(-123.0, sample.targetLogLikelihood, 0.0);
        for (int i = 0; i < sample.precisionMatrix.length; i++) {
            for (int j = 0; j < sample.precisionMatrix.length; j++) {
                assertEquals(sample.precisionMatrix[i][j], sample.precisionMatrix[j][i], 1e-12);
            }
        }

                double expectedDeterminant = 0.003375;
        assertEquals(Math.log(expectedDeterminant), sample.logDeterminant, 1e-12);
    }

        @Test
        public void buildTrainingSample_storesPrecisionNotCovariance() {
                Fixture fixture = createFixture();
                LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                                fixture.templateLikelihood,
                                fixture.alignment,
                                1);

                double[][] covariance = trainer.buildCovarianceMatrix(fixture.tree);
                LearnedWeightsTreeTrainer.TrainingTreeSample sample = trainer.buildTrainingSample(fixture.tree, -123.0);
                double[][] recoveredCovariance = new CholeskyDecomposition(
                                new Array2DRowRealMatrix(sample.precisionMatrix, false)).getSolver().getInverse().getData();

                assertTrue(sample.precisionMatrix[0][1] < 0.0);
                assertTrue(covariance[0][1] >= 0.0);
                for (int row = 0; row < covariance.length; row++) {
                        assertArrayEquals(covariance[row], recoveredCovariance[row], 1e-12);
                }
        }

        @Test
        public void buildCovarianceMatrix_rejectsNegativeBranchRateMetadata() throws Exception {
                Fixture fixture = createFixture("1.0");
                LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                                fixture.templateLikelihood,
                                fixture.alignment,
                                1);

                Path treeFile = Files.createTempFile("learnedweights-negative-rate", ".trees");
                Path logFile = Files.createTempFile("learnedweights-negative-rate", ".log");
                try {
                        Files.writeString(treeFile,
                                        "#NEXUS\n"
                                                        + "Begin taxa;\n"
                                                        + "Dimensions ntax=4;\n"
                                                        + "Taxlabels A B C D;\n"
                                                        + "End;\n"
                                                        + "Begin trees;\n"
                                                        + "Tree STATE_7 = ((A[&rate=-2.0]:0.10,B[&rate=2.0]:0.10)[&rate=2.0]:0.20,(C[&rate=2.0]:0.15,D[&rate=2.0]:0.15)[&rate=2.0]:0.15);\n"
                                                        + "End;\n");
                        Files.writeString(logFile,
                                        "# posterior parameter log\n"
                                                        + "Sample\ttreeLikelihood\n"
                                                        + "7\t-10.0\n");

                        try {
                                trainer.generateTrainingSamplesFromTreeFiles(
                                                List.of(treeFile.toFile()),
                                                List.of(logFile.toFile()),
                                                0,
                                                1,
                                                1);
                                fail("Expected negative branch rate metadata to be rejected");
                        } catch (IllegalStateException expected) {
                                assertTrue(expected.getMessage().contains("Negative branch rate"));
                        }
                } finally {
                        Files.deleteIfExists(treeFile);
                        Files.deleteIfExists(logFile);
                }
        }

        @Test
        public void buildTrainingSample_withFiniteRootPriorUsesRankOneAdjustedCovariance() {
                Fixture fixture = createFixture();
                LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                                fixture.templateLikelihood,
                                fixture.alignment,
                                1,
                                0.5);

                double[][] baseCovariance = trainer.buildCovarianceMatrix(fixture.tree);
                double[][] adjustedCovariance = new double[baseCovariance.length][baseCovariance.length];
                for (int row = 0; row < baseCovariance.length; row++) {
                        for (int column = 0; column < baseCovariance[row].length; column++) {
                                adjustedCovariance[row][column] = baseCovariance[row][column] + 0.5;
                        }
                }

                CholeskyDecomposition decomposition = new CholeskyDecomposition(new Array2DRowRealMatrix(adjustedCovariance, false));
                double[][] expectedPrecision = decomposition.getSolver().getInverse().getData();
                double expectedLogDeterminant = Math.log(decomposition.getDeterminant());

                LearnedWeightsTreeTrainer.TrainingTreeSample sample = trainer.buildTrainingSample(fixture.tree, -123.0);

                assertEquals(baseCovariance.length, sample.logNormalizerDimension);
                assertEquals(expectedLogDeterminant, sample.logDeterminant, 1e-12);
                for (int row = 0; row < expectedPrecision.length; row++) {
                        assertArrayEquals(expectedPrecision[row], sample.precisionMatrix[row], 1e-12);
                }
        }

        @Test
        public void buildTrainingSample_withInfiniteRootPriorUsesProjectedPrecision() {
                Fixture fixture = createFixture();
                LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                                fixture.templateLikelihood,
                                fixture.alignment,
                                1,
                                Double.POSITIVE_INFINITY);

                LearnedWeightsTreeTrainer.TrainingTreeSample sample = trainer.buildTrainingSample(fixture.tree, -123.0);

                assertEquals(fixture.alignment.getTaxonCount() - 1, sample.logNormalizerDimension);
                for (int row = 0; row < sample.precisionMatrix.length; row++) {
                        double rowSum = 0.0;
                        for (int column = 0; column < sample.precisionMatrix[row].length; column++) {
                                rowSum += sample.precisionMatrix[row][column];
                        }
                        assertEquals(0.0, rowSum, 1e-10);
                }
        }

        @Test
        public void buildTrainingSample_withManyTinyBranchesKeepsFiniteLogDeterminant() {
                Alignment alignment = createStarAlignment(80);
                Tree tree = createStarTree(alignment, 1e-5);
                LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                                null,
                                alignment,
                                0,
                                100.0);

                LearnedWeightsTreeTrainer.TrainingTreeSample sample = trainer.buildTrainingSample(tree, -123.0);

                assertTrue(Double.isFinite(sample.logDeterminant));
        }

                @Test
                public void computeCurrentTreeLogLikelihood_supportsThreadedTemplateWithScaling() {
                        Fixture fixture = createThreadedFixture();
                        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                                        fixture.templateLikelihood,
                                        fixture.alignment,
                                        1);

                        double expected = fixture.templateLikelihood.calculateLogP();
                        Double actual = trainer.computeCurrentTreeLogLikelihood(fixture.tree);

                        assertTrue(actual != null && Double.isFinite(actual));
                        assertEquals(expected, actual, 1e-12);
                }

        @Test
        public void generateTrainingSamplesFromTreeFiles_combinesMultiplePosteriorLogs() throws Exception {
                Fixture fixture = createFixture();
                LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                                fixture.templateLikelihood,
                                fixture.alignment,
                                10);

                Path firstTreeFile = Files.createTempFile("learnedweights-training-1", ".trees");
                Path secondTreeFile = Files.createTempFile("learnedweights-training-2", ".trees");
                Path firstLogFile = Files.createTempFile("learnedweights-training-1", ".log");
                Path secondLogFile = Files.createTempFile("learnedweights-training-2", ".log");
                try {
                        Files.writeString(firstTreeFile,
                                        "#NEXUS\n"
                                                        + "Begin taxa;\n"
                                                        + "Dimensions ntax=4;\n"
                                                        + "Taxlabels A B C D;\n"
                                                        + "End;\n"
                                                        + "Begin trees;\n"
                                                        + "Tree STATE_0 = ((A:0.10,B:0.10):0.20,(C:0.15,D:0.15):0.15);\n"
                                                        + "Tree STATE_1 = ((A:0.12,B:0.12):0.18,(C:0.14,D:0.14):0.16);\n"
                                                        + "End;\n");
                        Files.writeString(secondTreeFile,
                                        "#NEXUS\n"
                                                        + "Begin taxa;\n"
                                                        + "Dimensions ntax=4;\n"
                                                        + "Taxlabels A B C D;\n"
                                                        + "End;\n"
                                                        + "Begin trees;\n"
                                                        + "Tree STATE_0 = ((A:0.10,C:0.10):0.20,(B:0.15,D:0.15):0.15);\n"
                                                        + "Tree STATE_1 = (((A:0.08,B:0.08):0.1,C:0.18):0.12,D:0.3);\n"
                                                        + "End;\n");
                        Files.writeString(firstLogFile,
                                        "# posterior parameter log\n"
                                                        + "Sample\ttreeLikelihood\n"
                                                        + "0\t-10.0\n"
                                                        + "1\t-11.0\n");
                        Files.writeString(secondLogFile,
                                        "# posterior parameter log\n"
                                                        + "Sample\ttreeLikelihood\n"
                                                        + "0\t-12.0\n"
                                                        + "1\t-13.0\n");

                        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = trainer.generateTrainingSamplesFromTreeFiles(
                                        List.of(firstTreeFile.toFile(), secondTreeFile.toFile()),
                                        List.of(firstLogFile.toFile(), secondLogFile.toFile()),
                                        0,
                                        1,
                                        10);

                        assertEquals(4, samples.size());
                        for (LearnedWeightsTreeTrainer.TrainingTreeSample sample : samples) {
                                assertTrue(Double.isFinite(sample.targetLogLikelihood));
                        }
                } finally {
                        Files.deleteIfExists(firstTreeFile);
                        Files.deleteIfExists(secondTreeFile);
                        Files.deleteIfExists(firstLogFile);
                        Files.deleteIfExists(secondLogFile);
                }
        }

        @Test
        public void generateTrainingSamplesFromTreeFiles_parsesTranslatedNexusTrees() throws Exception {
                Fixture fixture = createFixture();
                LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                                fixture.templateLikelihood,
                                fixture.alignment,
                                1);

                Path treeFile = Files.createTempFile("learnedweights-translated", ".trees");
                Path logFile = Files.createTempFile("learnedweights-translated", ".log");
                try {
                        Files.writeString(treeFile,
                                        "#NEXUS\n"
                                                        + "Begin taxa;\n"
                                                        + "Dimensions ntax=4;\n"
                                                        + "Taxlabels A B C D;\n"
                                                        + "End;\n"
                                                        + "Begin trees;\n"
                                                        + "Translate\n"
                                                        + "  1 A,\n"
                                                        + "  2 B,\n"
                                                        + "  3 C,\n"
                                                        + "  4 D\n"
                                                        + ";\n"
                                                        + "Tree STATE_7 = ((1:0.10,2:0.10):0.20,(3:0.15,4:0.15):0.15);\n"
                                                        + "End;\n");
                        Files.writeString(logFile,
                                        "# posterior parameter log\n"
                                                        + "Sample\ttreeLikelihood\n"
                                                        + "7\t-10.0\n");

                        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = trainer.generateTrainingSamplesFromTreeFiles(
                                        List.of(treeFile.toFile()),
                                        List.of(logFile.toFile()),
                                        0,
                                        1,
                                        1);

                        assertEquals(1, samples.size());
                        assertTrue(Double.isFinite(samples.get(0).targetLogLikelihood));
                        assertTrue(Double.isFinite(samples.get(0).logDeterminant));
                } finally {
                        Files.deleteIfExists(treeFile);
                        Files.deleteIfExists(logFile);
                }
        }

        @Test
        public void generateTrainingSamplesFromTreeFiles_capsRetainedSamplesAfterBurninAndThinning() throws Exception {
                Fixture fixture = createFixture("1.0");
                fixture.templateLikelihood.setID("treeLikelihood");
                LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                                fixture.templateLikelihood,
                                fixture.alignment,
                                2);

                Path treeFile = Files.createTempFile("learnedweights-capped-selection", ".trees");
                Path logFile = Files.createTempFile("learnedweights-capped-selection", ".log");
                try {
                        Files.writeString(treeFile,
                                        "#NEXUS\n"
                                                        + "Begin taxa;\n"
                                                        + "Dimensions ntax=4;\n"
                                                        + "Taxlabels A B C D;\n"
                                                        + "End;\n"
                                                        + "Begin trees;\n"
                                                        + "Tree STATE_0 = ((A:0.10,B:0.10):0.20,(C:0.15,D:0.15):0.15);\n"
                                                        + "Tree STATE_1 = ((A:0.11,B:0.11):0.19,(C:0.14,D:0.14):0.16);\n"
                                                        + "Tree STATE_2 = ((A:0.12,B:0.12):0.18,(C:0.13,D:0.13):0.17);\n"
                                                        + "Tree STATE_3 = ((A:0.13,B:0.13):0.17,(C:0.12,D:0.12):0.18);\n"
                                                        + "Tree STATE_4 = ((A:0.14,B:0.14):0.16,(C:0.11,D:0.11):0.19);\n"
                                                        + "Tree STATE_5 = ((A:0.15,B:0.15):0.15,(C:0.10,D:0.10):0.20);\n"
                                                        + "End;\n");
                        Files.writeString(logFile,
                                        "# posterior parameter log\n"
                                                        + "Sample\ttreeLikelihood\n"
                                                        + "0\t-100.0\n"
                                                        + "1\t-101.0\n"
                                                        + "2\t-102.0\n"
                                                        + "3\t-103.0\n"
                                                        + "4\t-104.0\n"
                                                        + "5\t-105.0\n");

                        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = trainer.generateTrainingSamplesFromTreeFiles(
                                        List.of(treeFile.toFile()),
                                        List.of(logFile.toFile()),
                                        20,
                                        2,
                                        2);

                        assertEquals(2, samples.size());
                        assertEquals(-101.0, samples.get(0).targetLogLikelihood, 1e-12);
                        assertEquals(-103.0, samples.get(1).targetLogLikelihood, 1e-12);
                } finally {
                        Files.deleteIfExists(treeFile);
                        Files.deleteIfExists(logFile);
                }
        }

    @Test
    public void generateTrainingSamplesFromTreeFiles_usesBranchRatesEncodedInTreeMetadata() throws Exception {
        Fixture fixture = createFixture("1.0");
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                fixture.templateLikelihood,
                fixture.alignment,
                2);

        Path baselineTreeFile = Files.createTempFile("learnedweights-baseline", ".trees");
        Path metadataTreeFile = Files.createTempFile("learnedweights-metadata", ".trees");
        Path logFile = Files.createTempFile("learnedweights-metadata", ".log");
        try {
            Files.writeString(baselineTreeFile,
                    "#NEXUS\n"
                            + "Begin taxa;\n"
                            + "Dimensions ntax=4;\n"
                            + "Taxlabels A B C D;\n"
                            + "End;\n"
                            + "Begin trees;\n"
                            + "Tree STATE_7 = ((A:0.10,B:0.10):0.20,(C:0.15,D:0.15):0.15);\n"
                            + "End;\n");
            Files.writeString(metadataTreeFile,
                    "#NEXUS\n"
                            + "Begin taxa;\n"
                            + "Dimensions ntax=4;\n"
                            + "Taxlabels A B C D;\n"
                            + "End;\n"
                            + "Begin trees;\n"
                            + "Tree STATE_7 = ((A[&rate=2.0]:0.10,B[&rate=2.0]:0.10)[&rate=2.0]:0.20,(C[&rate=2.0]:0.15,D[&rate=2.0]:0.15)[&rate=2.0]:0.15);\n"
                            + "End;\n");
            Files.writeString(logFile,
                    "# posterior parameter log\n"
                            + "Sample\ttreeLikelihood\n"
                            + "7\t-10.0\n");

            List<LearnedWeightsTreeTrainer.TrainingTreeSample> baselineSamples = trainer.generateTrainingSamplesFromTreeFiles(
                    List.of(baselineTreeFile.toFile()),
                    List.of(logFile.toFile()),
                    0,
                    1,
                    1);
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> metadataSamples = trainer.generateTrainingSamplesFromTreeFiles(
                    List.of(metadataTreeFile.toFile()),
                    List.of(logFile.toFile()),
                    0,
                    1,
                    1);

            assertEquals(1, baselineSamples.size());
            assertEquals(1, metadataSamples.size());
            assertEquals(baselineSamples.get(0).targetLogLikelihood, metadataSamples.get(0).targetLogLikelihood, 1e-12);
            assertTrue(Math.abs(baselineSamples.get(0).logDeterminant - metadataSamples.get(0).logDeterminant) > 1e-10);
        } finally {
            Files.deleteIfExists(baselineTreeFile);
            Files.deleteIfExists(metadataTreeFile);
            Files.deleteIfExists(logFile);
        }
    }

                @Test
        public void generateTrainingSamplesFromTreeFiles_acceptsPosteriorTreeRatesEncodedInMetadata() throws Exception {
                Fixture fixture = createFixture("1.0");
                LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                                fixture.templateLikelihood,
                                fixture.alignment,
                                1);

        Path treeFile = Files.createTempFile("learnedweights-runtime-rates", ".trees");
        Path logFile = Files.createTempFile("learnedweights-runtime-rates", ".log");
        try {
            String posteriorNewick = "((C[&rate=11.0]:0.15,D[&rate=13.0]:0.15)[&rate=17.0]:0.15,(A[&rate=2.0]:0.10,B[&rate=3.0]:0.10)[&rate=5.0]:0.20);";
            Files.writeString(treeFile,
                    "#NEXUS\n"
                            + "Begin taxa;\n"
                            + "Dimensions ntax=4;\n"
                            + "Taxlabels A B C D;\n"
                            + "End;\n"
                            + "Begin trees;\n"
                            + "Tree STATE_7 = " + posteriorNewick + "\n"
                            + "End;\n");
            Files.writeString(logFile,
                    "# posterior parameter log\n"
                            + "Sample\ttreeLikelihood\n"
                            + "7\t-123.0\n");

            List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = trainer.generateTrainingSamplesFromTreeFiles(
                    List.of(treeFile.toFile()),
                                        List.of(logFile.toFile()),
                    0,
                    1,
                    1);

            assertEquals(1, samples.size());
            assertEquals(-123.0, samples.get(0).targetLogLikelihood, 1e-12);
            assertTrue(Double.isFinite(samples.get(0).logDeterminant));
        } finally {
            Files.deleteIfExists(treeFile);
                        Files.deleteIfExists(logFile);
        }
        }

    @Test
    public void generateTrainingSamplesFromTreeFiles_usesStrictClockMeanRateFromTrainingLog() throws Exception {
        Fixture fixture = createFixture("1.0");
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                fixture.templateLikelihood,
                fixture.alignment,
                2);

        Path treeFile = Files.createTempFile("learnedweights-strict-clock", ".trees");
        Path logFile = Files.createTempFile("learnedweights-strict-clock", ".log");
        try {
            Files.writeString(treeFile,
                    "#NEXUS\n"
                            + "Begin taxa;\n"
                            + "Dimensions ntax=4;\n"
                            + "Taxlabels A B C D;\n"
                            + "End;\n"
                            + "Begin trees;\n"
                            + "Tree STATE_7 = ((A:0.10,B:0.10):0.20,(C:0.15,D:0.15):0.15);\n"
                            + "End;\n");
            Files.writeString(logFile,
                    "# posterior parameter log\n"
                            + "Sample\tclock.rate\ttreeLikelihood\n"
                            + "7\t2.0\t-10.0\n");

            List<LearnedWeightsTreeTrainer.TrainingTreeSample> loggedSamples = trainer.generateTrainingSamplesFromTreeFiles(
                    List.of(treeFile.toFile()),
                    List.of(logFile.toFile()),
                    0,
                    1,
                    1);

            Tree posteriorTree = new TreeParser();
            ((TreeParser) posteriorTree).initByName(
                    "newick",
                    "((A:0.10,B:0.10):0.20,(C:0.15,D:0.15):0.15);",
                    "IsLabelledNewick", true,
                    "taxa", fixture.alignment);
            Fixture expectedFixture = createFixture("2.0");
            LearnedWeightsTreeTrainer expectedTrainer = new LearnedWeightsTreeTrainer(
                    expectedFixture.templateLikelihood,
                    expectedFixture.alignment,
                    1);
            LearnedWeightsTreeTrainer.TrainingTreeSample expectedSample = expectedTrainer.buildTrainingSample(posteriorTree, -10.0);

            assertEquals(1, loggedSamples.size());
            assertEquals(expectedSample.targetLogLikelihood, loggedSamples.get(0).targetLogLikelihood, 1e-12);
            assertEquals(expectedSample.logDeterminant, loggedSamples.get(0).logDeterminant, 1e-12);
        } finally {
            Files.deleteIfExists(treeFile);
            Files.deleteIfExists(logFile);
        }
    }

    @Test
    public void generateTrainingSamplesFromTreeFiles_usesRelaxedClockStateFromTrainingLog() throws Exception {
        Fixture fixture = createRelaxedClockFixture();
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                fixture.templateLikelihood,
                fixture.alignment,
                2);

        Path treeFile = Files.createTempFile("learnedweights-relaxed-clock", ".trees");
        Path logFile = Files.createTempFile("learnedweights-relaxed-clock", ".log");
        try {
            Files.writeString(treeFile,
                    "#NEXUS\n"
                            + "Begin taxa;\n"
                            + "Dimensions ntax=4;\n"
                            + "Taxlabels A B C D;\n"
                            + "End;\n"
                            + "Begin trees;\n"
                            + "Tree STATE_7 = ((A:0.10,B:0.10):0.20,(C:0.15,D:0.15):0.15);\n"
                            + "End;\n");
            Files.writeString(logFile,
                    "# posterior parameter log\n"
                            + "Sample\trateCategories.c:clock.1\trateCategories.c:clock.2\trateCategories.c:clock.3\trateCategories.c:clock.4\trateCategories.c:clock.5\trateCategories.c:clock.6\ttreeLikelihood\n"
                            + "7\t0\t1\t2\t3\t4\t5\t-10.0\n");

            List<LearnedWeightsTreeTrainer.TrainingTreeSample> loggedSamples = trainer.generateTrainingSamplesFromTreeFiles(
                    List.of(treeFile.toFile()),
                    List.of(logFile.toFile()),
                    0,
                    1,
                    1);

            Path alternateLogFile = Files.createTempFile("learnedweights-relaxed-clock-alt", ".log");
            try {
                Files.writeString(alternateLogFile,
                        "# posterior parameter log\n"
                                + "Sample\trateCategories.c:clock.1\trateCategories.c:clock.2\trateCategories.c:clock.3\trateCategories.c:clock.4\trateCategories.c:clock.5\trateCategories.c:clock.6\ttreeLikelihood\n"
                                + "7\t5\t4\t3\t2\t1\t0\t-10.0\n");

                List<LearnedWeightsTreeTrainer.TrainingTreeSample> alternateSamples = trainer.generateTrainingSamplesFromTreeFiles(
                        List.of(treeFile.toFile()),
                        List.of(alternateLogFile.toFile()),
                        0,
                        1,
                        1);

                assertEquals(1, loggedSamples.size());
                assertEquals(1, alternateSamples.size());
                assertEquals(loggedSamples.get(0).targetLogLikelihood, alternateSamples.get(0).targetLogLikelihood, 1e-12);
                assertTrue(Math.abs(loggedSamples.get(0).logDeterminant - alternateSamples.get(0).logDeterminant) > 1e-10);
            } finally {
                Files.deleteIfExists(alternateLogFile);
            }
        } finally {
            Files.deleteIfExists(treeFile);
            Files.deleteIfExists(logFile);
        }
    }

    @Test
    public void generateTrainingSamplesFromTreeFiles_usesLoggedTargetLikelihoodWhenAvailable() throws Exception {
        Fixture fixture = createFixture("1.0");
        fixture.templateLikelihood.setID("treeLikelihood");
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                fixture.templateLikelihood,
                fixture.alignment,
                2);

        Path treeFile = Files.createTempFile("learnedweights-logged-target", ".trees");
        Path logFile = Files.createTempFile("learnedweights-logged-target", ".log");
        try {
            Files.writeString(treeFile,
                    "#NEXUS\n"
                            + "Begin taxa;\n"
                            + "Dimensions ntax=4;\n"
                            + "Taxlabels A B C D;\n"
                            + "End;\n"
                            + "Begin trees;\n"
                            + "Tree STATE_7 = ((A:0.10,B:0.10):0.20,(C:0.15,D:0.15):0.15);\n"
                            + "End;\n");
            Files.writeString(logFile,
                    "# posterior parameter log\n"
                            + "Sample\tclock.rate\ttreeLikelihood\n"
                            + "7\t2.0\t-123.456\n");

            List<LearnedWeightsTreeTrainer.TrainingTreeSample> loggedSamples = trainer.generateTrainingSamplesFromTreeFiles(
                    List.of(treeFile.toFile()),
                    List.of(logFile.toFile()),
                    0,
                    1,
                    1);

            assertEquals(1, loggedSamples.size());
            assertEquals(-123.456, loggedSamples.get(0).targetLogLikelihood, 1e-12);
        } finally {
            Files.deleteIfExists(treeFile);
            Files.deleteIfExists(logFile);
        }
    }

        @Test
        public void generateTrainingSamplesFromTreeFiles_requiresConfiguredTargetLikelihoodColumn() throws Exception {
                Fixture fixture = createFixture("1.0");
                fixture.templateLikelihood.setID("treeLikelihood");
                LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                                fixture.templateLikelihood,
                                fixture.alignment,
                                2);

                Path treeFile = Files.createTempFile("learnedweights-generic-target-ignored", ".trees");
                Path logFile = Files.createTempFile("learnedweights-generic-target-ignored", ".log");
                try {
                        Files.writeString(treeFile,
                                        "#NEXUS\n"
                                                        + "Begin taxa;\n"
                                                        + "Dimensions ntax=4;\n"
                                                        + "Taxlabels A B C D;\n"
                                                        + "End;\n"
                                                        + "Begin trees;\n"
                                                        + "Tree STATE_7 = ((A:0.10,B:0.10):0.20,(C:0.15,D:0.15):0.15);\n"
                                                        + "End;\n");
                        Files.writeString(logFile,
                                        "# posterior parameter log\n"
                                                        + "Sample\tclock.rate\tlikelihood\n"
                                                        + "7\t2.0\t-123.456\n");

                        try {
                                trainer.generateTrainingSamplesFromTreeFiles(
                                                List.of(treeFile.toFile()),
                                                List.of(logFile.toFile()),
                                                0,
                                                1,
                                                1);
                                fail("Expected missing required target likelihood column to be rejected");
                        } catch (IllegalArgumentException expected) {
                                assertTrue(expected.getMessage().contains("treeLikelihood"));
                        }
                } finally {
                        Files.deleteIfExists(treeFile);
                        Files.deleteIfExists(logFile);
                }
        }

        @Test
        public void generateTrainingSamplesFromTreeFiles_requiresExactSampleMatching() throws Exception {
                Fixture fixture = createFixture("1.0");
                fixture.templateLikelihood.setID("treeLikelihood");
                LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                                fixture.templateLikelihood,
                                fixture.alignment,
                                2);

                Path treeFile = Files.createTempFile("learnedweights-exact-match", ".trees");
                Path logFile = Files.createTempFile("learnedweights-exact-match", ".log");
                try {
                        Files.writeString(treeFile,
                                        "#NEXUS\n"
                                                        + "Begin taxa;\n"
                                                        + "Dimensions ntax=4;\n"
                                                        + "Taxlabels A B C D;\n"
                                                        + "End;\n"
                                                        + "Begin trees;\n"
                                                        + "Tree STATE_7 = ((A:0.10,B:0.10):0.20,(C:0.15,D:0.15):0.15);\n"
                                                        + "End;\n");
                        Files.writeString(logFile,
                                        "# posterior parameter log\n"
                                                        + "Sample\ttreeLikelihood\n"
                                                        + "8\t-123.456\n");

                        try {
                                trainer.generateTrainingSamplesFromTreeFiles(
                                                List.of(treeFile.toFile()),
                                                List.of(logFile.toFile()),
                                                0,
                                                1,
                                                1);
                                fail("Expected sample-number mismatch to be rejected");
                        } catch (IllegalArgumentException expected) {
                                assertTrue(expected.getMessage().contains("missing rows")
                                                || expected.getMessage().contains("sample 7"));
                        }
                } finally {
                        Files.deleteIfExists(treeFile);
                        Files.deleteIfExists(logFile);
                }
        }

        private static Fixture createFixture() {
                return createFixture("1.0");
        }

        private static Fixture createFixture(String clockRate) {
                        RealScalarParam<PositiveReal> meanRate =
                                new RealScalarParam<>(Double.parseDouble(clockRate), PositiveReal.INSTANCE);
                        meanRate.setID("clock.rate");
                StrictClockModel clock = new StrictClockModel();
                        clock.initByName("clock.rate", meanRate);
                return createFixture(clock);
        }

        private static Fixture createThreadedFixture() {
                Fixture baseFixture = createFixture();

                ThreadedTreeLikelihood threadedLikelihood = new ThreadedTreeLikelihood();
                threadedLikelihood.initByName(
                        "data", baseFixture.alignment,
                        "tree", baseFixture.tree,
                        "siteModel", baseFixture.siteModel,
                        "branchRateModel", baseFixture.clock,
                        "threads", 1,
                        "scaling", "always");

                return new Fixture(
                        baseFixture.alignment,
                        baseFixture.tree,
                        baseFixture.siteModel,
                        baseFixture.clock,
                        threadedLikelihood);
        }

        private static Alignment createStarAlignment(int taxonCount) {
                Alignment alignment = new Alignment();
                List<Object> initArgs = new java.util.ArrayList<>();
                for (int taxonIndex = 0; taxonIndex < taxonCount; taxonIndex++) {
                        String taxon = "T" + taxonIndex;
                        Sequence sequence = new Sequence(taxon, "ACGTACGTAA");
                        initArgs.add("sequence");
                        initArgs.add(sequence);
                }
                alignment.initByName(initArgs.toArray());
                return alignment;
        }

        private static Tree createStarTree(Alignment alignment, double branchLength) {
                StringBuilder newick = new StringBuilder("(");
                for (int taxonIndex = 0; taxonIndex < alignment.getTaxonCount(); taxonIndex++) {
                        if (taxonIndex > 0) {
                                newick.append(',');
                        }
                        newick.append(alignment.getTaxaNames().get(taxonIndex))
                                .append(':')
                                .append(branchLength);
                }
                newick.append(");");

                Tree tree = new TreeParser();
                ((TreeParser) tree).initByName(
                        "newick", newick.toString(),
                        "IsLabelledNewick", true,
                        "taxa", alignment);
                return tree;
        }

        private static Fixture createRelaxedClockFixture() {
                RealScalarParam<Real> meanRate = new RealScalarParam<>(1.0, Real.INSTANCE);
                meanRate.setID("ucldMean.c:clock");
                RealScalarParam<PositiveReal> stdev = new RealScalarParam<>(0.5, PositiveReal.INSTANCE);
                stdev.setID("ucldStdev.c:clock");
                LogNormal distribution = new LogNormal();
                distribution.initByName("M", meanRate, "S", stdev, "meanInRealSpace", true);

                IntVectorParam<NonNegativeInt> categories =
                                new IntVectorParam<>(new int[]{0}, NonNegativeInt.INSTANCE);
                categories.setID("rateCategories.c:clock");

                Sequence seqA = new Sequence("A", "ACGTACGTAA");
                Sequence seqB = new Sequence("B", "ACGTACGTAT");
                Sequence seqC = new Sequence("C", "ACGTACGTGA");
                Sequence seqD = new Sequence("D", "ACGTACGTGT");

                Alignment alignment = new Alignment();
                alignment.initByName(
                        "sequence", seqA,
                        "sequence", seqB,
                        "sequence", seqC,
                        "sequence", seqD);

                Tree tree = new TreeParser();
                ((TreeParser) tree).initByName(
                        "newick", "((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);",
                        "IsLabelledNewick", true,
                        "taxa", alignment);

                UCRelaxedClockModel clock = new UCRelaxedClockModel();
                clock.initByName(
                        "clock.rate", meanRate,
                        "distr", distribution,
                        "rateCategories", categories,
                        "tree", tree,
                        "numberOfDiscreteRates", 8);

                JukesCantor jc = new JukesCantor();
                jc.initAndValidate();

                SiteModel siteModel = new SiteModel();
                siteModel.initByName(
                        "substModel", jc,
                        "gammaCategoryCount", 4,
                        "shape", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

                TreeLikelihood templateLikelihood = new TreeLikelihood();
                templateLikelihood.initByName(
                        "data", alignment,
                        "tree", tree,
                        "siteModel", siteModel,
                        "branchRateModel", clock,
                        "implementation", "BeerLikelihoodCore4");

                return new Fixture(alignment, tree, siteModel, clock, templateLikelihood);
        }

        private static Fixture createFixture(Base branchRateModel) {
        Sequence seqA = new Sequence("A", "ACGTACGTAA");
        Sequence seqB = new Sequence("B", "ACGTACGTAT");
        Sequence seqC = new Sequence("C", "ACGTACGTGA");
        Sequence seqD = new Sequence("D", "ACGTACGTGT");

        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", seqA,
                "sequence", seqB,
                "sequence", seqC,
                "sequence", seqD);

        Tree tree = new TreeParser();
        ((TreeParser) tree).initByName(
                "newick", "((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);",
                "IsLabelledNewick", true,
                "taxa", alignment);

                if (branchRateModel == null) {
                        branchRateModel = createDeterministicBranchRateModel(tree);
                }

        JukesCantor jc = new JukesCantor();
        jc.initAndValidate();

        SiteModel siteModel = new SiteModel();
        siteModel.initByName(
                "substModel", jc,
                "gammaCategoryCount", 4,
                "shape", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

        TreeLikelihood templateLikelihood = new TreeLikelihood();
        templateLikelihood.initByName(
                "data", alignment,
                "tree", tree,
                "siteModel", siteModel,
                                "branchRateModel", branchRateModel,
                "implementation", "BeerLikelihoodCore4");

                                return new Fixture(alignment, tree, siteModel, branchRateModel, templateLikelihood);
    }

        private static Base createDeterministicBranchRateModel(Tree tree) {
                double[] rates = new double[tree.getNodeCount()];
                for (int i = 0; i < rates.length; i++) {
                        rates[i] = 1.0;
                }
                rates[findNodeNrById(tree, "A")] = 2.0;
                rates[findNodeNrById(tree, "B")] = 3.0;
                rates[findNodeNrById(tree, "C")] = 5.0;
                rates[findNodeNrById(tree, "D")] = 6.0;
                rates[findInternalNodeWithChildren(tree, "A", "B").getNr()] = 4.0;
                rates[findInternalNodeWithChildren(tree, "C", "D").getNr()] = 7.0;
                return new DeterministicBranchRateModel(rates);
        }

        private static int findNodeNrById(Tree tree, String id) {
                for (Node node : tree.getNodesAsArray()) {
                        if (id.equals(node.getID())) {
                                return node.getNr();
                        }
                }
                throw new IllegalArgumentException("Could not find node with id " + id);
        }

        private static Node findInternalNodeWithChildren(Tree tree, String leftId, String rightId) {
                for (Node node : tree.getNodesAsArray()) {
                        if (node.isLeaf()) {
                                continue;
                        }
                        if (node.getLeft().isLeaf() && node.getRight().isLeaf()) {
                                String left = node.getLeft().getID();
                                String right = node.getRight().getID();
                                if ((leftId.equals(left) && rightId.equals(right)) || (leftId.equals(right) && rightId.equals(left))) {
                                        return node;
                                }
                        }
                }
                throw new IllegalArgumentException("Could not find internal node for children " + leftId + " and " + rightId);
        }

        private static final class DeterministicBranchRateModel extends Base {
                private final double[] rates;

                private DeterministicBranchRateModel(double[] rates) {
                        this.rates = rates.clone();
                }

                @Override
                public void initAndValidate() {
                        // No-op: branch rates are fixed by construction for the test.
                }

                @Override
                public double getRateForBranch(Node node) {
                        return rates[node.getNr()];
                }
        }

    private record Fixture(
            Alignment alignment,
                        Tree tree,
            SiteModel siteModel,
                        Base clock,
						GenericTreeLikelihood templateLikelihood) {
    }
}