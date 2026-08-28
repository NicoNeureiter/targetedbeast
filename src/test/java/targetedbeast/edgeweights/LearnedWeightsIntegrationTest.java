package targetedbeast.edgeweights;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.likelihood.TreeLikelihood;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.inference.parameter.RealScalarParam;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsModelIO;

/**
 * Integration-style component tests for LearnedWeights init path (C5).
 */
public class LearnedWeightsIntegrationTest {

    @Test
    public void initAndValidate_withValueInput_skipsLearningAndUsesProvidedLeafEmbeddings() {
        Fixture f = createFixture();

        LearnedWeights lw = new LearnedWeights();
        lw.initByName(
                "data", f.alignment,
                "tree", f.tree,
                "likelihood", f.likelihood,
                "dimension", 2,
                "value", "A=0.0 0.1,B=1.0 0.2,C=0.3 1.0,D=1.2 1.4",
            "offset", 0.01);

        // Value-input mode should skip trainer/optimizer path.
        assertTrue(lw.getTrainingSamples().isEmpty());

        int aNr = findNodeNrById(f.tree, "A");
        int dNr = findNodeNrById(f.tree, "D");

        assertEquals(0.0, lw.getPoints(aNr)[0], 1e-12);
        assertEquals(0.1, lw.getPoints(aNr)[1], 1e-12);
        assertEquals(1.2, lw.getPoints(dNr)[0], 1e-12);
        assertEquals(1.4, lw.getPoints(dNr)[1], 1e-12);

        for (Node n : f.tree.getNodesAsArray()) {
            if (n.isRoot()) {
                continue;
            }
            double w = lw.getEdgeWeights(n.getNr());
            assertTrue(Double.isFinite(w));
            assertTrue(w >= lw.minEdgeWeight());
        }
    }

    @Test
    public void initAndValidate_withoutValue_runsLearningAndProducesUsableState() {
        Fixture f = createFixture();

        LearnedWeights lw = new LearnedWeights();
        lw.initByName(
                "data", f.alignment,
                "tree", f.tree,
                "likelihood", f.likelihood,
                "dimension", 2,
                "nTrainingTrees", 1,
                "maxIterations", 50,
                "learningRate", 0.05,
                "offset", 0.01);

        assertNotNull(lw.getTrainingSamples());
        assertFalse("Expected training samples from learning path", lw.getTrainingSamples().isEmpty());

        int fromA = findNodeNrById(f.tree, "A");
        int b = findNodeNrById(f.tree, "B");
        int c = findNodeNrById(f.tree, "C");

        double[] targetWeights = lw.getTargetWeightsInteger(fromA, List.of(b, c));
        assertEquals(2, targetWeights.length);
        assertTrue(Double.isFinite(targetWeights[0]));
        assertTrue(Double.isFinite(targetWeights[1]));
        assertTrue(targetWeights[0] > 0.0);
        assertTrue(targetWeights[1] > 0.0);
        assertFalse(lw.usesInverseMeanDistanceProposal());
    }

    @Test
    public void initAndValidate_appliesFiniteLearnedSlopeAtRuntimeWhenUsingLogScores() {
        Fixture f = createFixture();

        LearnedWeights lw = new LearnedWeights();
        lw.initByName(
                "data", f.alignment,
                "tree", f.tree,
                "likelihood", f.likelihood,
                "dimension", 2,
                "nTrainingTrees", 1,
                "maxIterations", 20,
            "learningRate", 0.05,
                "useInverseMeanDistanceProposal", false,
                "offset", 0.01);

        assertTrue(Double.isFinite(lw.getLearnedSlope()));
        for (double slope : lw.getLearnedSlopes()) {
            assertTrue(Double.isFinite(slope));
        }
        assertTrue(Double.isFinite(lw.getRuntimeProposalScale()));
        for (double scale : lw.getRuntimeProposalScales()) {
            assertTrue(Double.isFinite(scale));
        }
        assertTrue(Double.isFinite(lw.getLearnedIntercept()));
        assertTrue(Double.isFinite(lw.getTargetWeightsInteger(findNodeNrById(f.tree, "A"),
                List.of(findNodeNrById(f.tree, "B"), findNodeNrById(f.tree, "C")))[0]));
    }

    @Test
    public void initAndValidate_exposesInverseDistanceProposalOption() {
        Fixture f = createFixture();

        LearnedWeights lw = new LearnedWeights();
        lw.initByName(
                "data", f.alignment,
                "tree", f.tree,
                "likelihood", f.likelihood,
                "dimension", 2,
                "nTrainingTrees", 1,
                "maxIterations", 10,
            "learningRate", 0.05,
                "useInverseMeanDistanceProposal", true,
                "offset", 0.01);

        assertTrue(lw.usesInverseMeanDistanceProposal());
    }

    @Test
    public void initAndValidate_acceptsTraitSpecificBrownianRates() {
        Fixture f = createFixture();

        LearnedWeights lw = new LearnedWeights();
        lw.initByName(
                "data", f.alignment,
                "tree", f.tree,
                "likelihood", f.likelihood,
                "dimension", 2,
                "brownianTraitRates", "0.5 1.5",
            "value", "A=0.0 0.1,B=1.0 0.2,C=0.3 1.0,D=1.2 1.4",
                "offset", 0.01);

        assertArrayEquals(new double[] {0.5, 1.5}, lw.getBrownianTraitRates(), 1e-12);
        assertTrue(Double.isFinite(lw.getTargetWeightsInteger(
                findNodeNrById(f.tree, "A"),
                List.of(findNodeNrById(f.tree, "B"), findNodeNrById(f.tree, "C")))[0]));
    }

    @Test
    public void initAndValidate_withTrainedWeightsFileLoadsCachedLearnedState() throws Exception {
        Fixture f = createFixture();

        Path trainedWeightsFile = Files.createTempFile("learnedweights-trained", ".tsv");
        try {
            LearnedWeightsModelIO.writeModel(
                    trainedWeightsFile,
                    new String[] {"A", "B", "C", "D"},
                    new double[] {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                    new double[] {0.5, 1.5},
                    new double[] {0.25, 0.75},
                    new double[] {0.6, 1.4},
                    new double[] {1.2, 0.8},
                    -0.25,
                    true,
                    false,
                    false,
                    0.0,
                    0.0);

            LearnedWeights lw = new LearnedWeights();
            lw.initByName(
                    "data", f.alignment,
                    "tree", f.tree,
                    "likelihood", f.likelihood,
                    "dimension", 2,
                        "separateBaseAndQuadraticCalibration", true,
                    "trainedWeightsFile", trainedWeightsFile.toString(),
                    "offset", 0.01);

            assertArrayEquals(new double[] {0.5, 1.5}, lw.getBrownianTraitRates(), 1e-12);
            assertArrayEquals(new double[] {0.6, 1.4}, lw.getLearnedBaseScales(), 1e-12);
            assertArrayEquals(new double[] {1.2, 0.8}, lw.getLearnedSlopes(), 1e-12);
            assertEquals(-0.25, lw.getLearnedIntercept(), 1e-12);
            assertArrayEquals(new double[] {0.1, 0.2}, lw.getCurrentLeafEmbeddingsForTaxa(new String[] {"A"}), 1e-12);
            assertArrayEquals(new double[] {0.7, 0.8}, lw.getCurrentLeafEmbeddingsForTaxa(new String[] {"D"}), 1e-12);
        } finally {
            Files.deleteIfExists(trainedWeightsFile);
        }
    }

    @Test
    public void initAndValidate_withContinueTrainingRejectsDimensionMismatch() throws Exception {
        Fixture f = createFixture();

        Path trainedWeightsFile = Files.createTempFile("learnedweights-trained", ".tsv");
        try {
            LearnedWeightsModelIO.writeModel(
                    trainedWeightsFile,
                    new String[] {"A", "B", "C", "D"},
                    new double[] {
                        0.1, 0.2, 0.3,
                        0.4, 0.5, 0.6,
                        0.7, 0.8, 0.9,
                        1.0, 1.1, 1.2
                    },
                    new double[] {0.5, 1.5, 2.5},
                    new double[] {0.25, 0.75, 1.25},
                    new double[] {0.6, 1.4, 0.9},
                    new double[] {1.2, 0.8, 1.1},
                    -0.25,
                    true,
                    false,
                    false,
                    0.0,
                    0.0);

                RuntimeException exception = assertThrows(
                    RuntimeException.class,
                    () -> {
                        LearnedWeights lw = new LearnedWeights();
                        lw.initByName(
                                "data", f.alignment,
                                "tree", f.tree,
                                "likelihood", f.likelihood,
                                "dimension", 2,
                                "separateBaseAndQuadraticCalibration", true,
                                "trainedWeightsFile", trainedWeightsFile.toString(),
                                "continueTrainingFromTrainedWeights", true,
                                "offset", 0.01);
                    });

            assertTrue(exception.getMessage().contains("dimension mismatch"));
        } finally {
            Files.deleteIfExists(trainedWeightsFile);
        }
    }

    @Test
    public void initAndValidate_withTrainingTreeFile_usesPosteriorTreesAsTrainingSet() throws Exception {
        Fixture f = createFixture();

        Path treeFile = Files.createTempFile("learnedweights-training", ".trees");
        Path logFile = Files.createTempFile("learnedweights-training", ".log");
        try {
            Files.writeString(treeFile,
                    "#NEXUS\n"
                            + "Begin taxa;\n"
                            + "Dimensions ntax=4;\n"
                            + "Taxlabels A B C D;\n"
                            + "End;\n"
                            + "Begin trees;\n"
                            + "Tree STATE_0 = ((A:0.10,B:0.10):0.20,(C:0.15,D:0.15):0.15);\n"
                            + "Tree STATE_1 = ((A:0.12,B:0.12):0.18,(C:0.14,D:0.14):0.16);\n"
                            + "End;\n");
            Files.writeString(logFile,
                "Sample\ttreeLikelihood\n"
                    + "0\t-10.0\n"
                    + "1\t-11.0\n");

            LearnedWeights lw = new LearnedWeights();
            lw.initByName(
                    "data", f.alignment,
                    "tree", f.tree,
                    "likelihood", f.likelihood,
                    "dimension", 2,
                    "nTrainingTrees", 10,
                    "trainingTreeFile", treeFile.toString(),
                "trainingLogFile", logFile.toString(),
                    "trainingTreeBurninPercentage", 0,
                    "trainingTreeThinInterval", 1,
                    "maxIterations", 10,
                    "learningRate", 0.05,
                    "offset", 0.01);

            assertNotNull(lw.getTrainingSamples());
            assertEquals(2, lw.getTrainingSamples().size());
        } finally {
            Files.deleteIfExists(treeFile);
            Files.deleteIfExists(logFile);
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
        jc.initAndValidate();

        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", jc,
                "gammaCategoryCount", 4,
                "shape", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

        StrictClockModel clock = new StrictClockModel();
        clock.initByName("clock.rate", "1.0");

        TreeLikelihood likelihood = new TreeLikelihood();
        likelihood.initByName(
                "data", alignment,
                "tree", tree,
                "siteModel", siteModel,
                "branchRateModel", clock,
            "implementation", "BeerLikelihoodCore4");

        return new Fixture(alignment, tree, siteModel, clock, likelihood);
    }

    private static int findNodeNrById(Tree tree, String id) {
        for (Node n : tree.getNodesAsArray()) {
            if (id.equals(n.getID())) {
                return n.getNr();
            }
        }
        throw new IllegalArgumentException("Node not found: " + id);
    }

    private record Fixture(
            Alignment alignment,
            Tree tree,
            SiteModel siteModel,
            StrictClockModel clock,
            TreeLikelihood likelihood) {
    }
}
