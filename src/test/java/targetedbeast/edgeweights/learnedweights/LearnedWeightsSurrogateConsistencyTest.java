package targetedbeast.edgeweights.learnedweights;

import static org.junit.Assert.assertEquals;

import java.util.List;

import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import targetedbeast.edgeweights.PCAWeightsBrownianFullLikelihood;

public class LearnedWeightsSurrogateConsistencyTest {

    private static final double LOG_TWO_PI = Math.log(2.0 * Math.PI);
    private static final double SCORE_TOLERANCE = 1e-9;

    @Test
    public void covarianceSurrogateMatchesPruningSurrogateUpToGaussianNormalizationConstant_fixedRoot() {
        assertTrainingAndRuntimeAgreeUpToConstant(0.0);
    }

    @Test
    public void covarianceSurrogateMatchesPruningSurrogateUpToGaussianNormalizationConstant_flatRoot() {
        assertTrainingAndRuntimeAgreeUpToConstant(Double.POSITIVE_INFINITY);
    }

    @Test
    public void covarianceAndPruningSurrogatesHaveMatchingCrossTreeDifferences_fixedRoot() {
        assertTrainingAndRuntimeHaveMatchingCrossTreeDifferences(0.0);
    }

    @Test
    public void covarianceAndPruningSurrogatesHaveMatchingCrossTreeDifferences_flatRoot() {
        assertTrainingAndRuntimeHaveMatchingCrossTreeDifferences(Double.POSITIVE_INFINITY);
    }

    @Test
    public void covarianceAndPruningSurrogatesHaveMatchingEmbeddingDifferences_fixedRoot() {
        assertTrainingAndRuntimeHaveMatchingEmbeddingDifferences(0.0);
    }

    @Test
    public void covarianceAndPruningSurrogatesHaveMatchingEmbeddingDifferences_flatRoot() {
        assertTrainingAndRuntimeHaveMatchingEmbeddingDifferences(Double.POSITIVE_INFINITY);
    }

        @Test
        public void covarianceSurrogateMatchesPruningSurrogateWithTraitSpecificRates() {
        Alignment alignment = createAlignment();
        Tree tree = createTree(alignment);
        double[] embeddings = new double[] {
            0.0, 0.4,
            0.7, -0.2,
            -0.3, 1.1,
            1.1, 0.2
        };
        double[] brownianTraitRates = new double[] {0.5, 2.0};

        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0, 0.0);
        LearnedWeightsTreeTrainer.TrainingTreeSample sample = trainer.buildTrainingSample(tree, 0.0);
        double covarianceScore = LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihood(
            sample, embeddings, 2, brownianTraitRates);

        PCAWeightsBrownianFullLikelihood runtime = new PCAWeightsBrownianFullLikelihood();
        runtime.initByName(
            "tree", tree,
            "data", alignment,
            "dimension", 2,
            "brownianTraitRates", "0.5 2.0",
            "value", toValueString(embeddings, 2),
            "rootPriorVariance", 0.0,
            "offset", 0.01);
        double pruningScore = runtime.getApproximateTreeLogLikelihood();

        double expectedDifference = -0.5 * sample.logNormalizerDimension * 2 * LOG_TWO_PI;
        assertEquals(expectedDifference, covarianceScore - pruningScore, SCORE_TOLERANCE);
        }

    @Test
    public void covarianceSurrogateMatchesPruningSurrogateWithTraitSpecificRootPriorVariances() {
        Alignment alignment = createAlignment();
        Tree tree = createTree(alignment);
        double[] embeddings = new double[] {
            0.0, 0.4,
            0.7, -0.2,
            -0.3, 1.1,
            1.1, 0.2
        };
        double[] rootPriorVariances = new double[] {0.5, 2.0};

        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0, 0.0);
        LearnedWeightsTreeTrainer.TrainingTreeSample sample = trainer.buildTrainingSample(tree, 0.0);
        double covarianceScore = LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihood(
            sample, embeddings, 2, new double[] {1.0, 1.0}, rootPriorVariances);

        PCAWeightsBrownianFullLikelihood runtime = new PCAWeightsBrownianFullLikelihood();
        runtime.initByName(
            "tree", tree,
            "data", alignment,
            "dimension", 2,
            "value", toValueString(embeddings, 2),
            "rootPriorVariances", "0.5 2.0",
            "offset", 0.01);
        double pruningScore = runtime.getApproximateTreeLogLikelihood();

        double expectedDifference = -0.5 * sample.rootFreeLogNormalizerDimension * 2 * LOG_TWO_PI;
        assertEquals(expectedDifference, covarianceScore - pruningScore, SCORE_TOLERANCE);
    }

    private void assertTrainingAndRuntimeAgreeUpToConstant(double rootPriorVariance) {
        Alignment alignment = createAlignment();
        Tree tree = createTree(alignment);
        double[] embeddings = new double[] {
                0.0,
                0.7,
                -0.3,
                1.1
        };

        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0, rootPriorVariance);
        LearnedWeightsTreeTrainer.TrainingTreeSample sample = trainer.buildTrainingSample(tree, 0.0);
        double covarianceScore = LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihood(sample, embeddings, 1, 1.0);

        PCAWeightsBrownianFullLikelihood runtime = new PCAWeightsBrownianFullLikelihood();
        runtime.initByName(
                "tree", tree,
                "data", alignment,
                "dimension", 1,
                "value", "A=0.0,B=0.7,C=-0.3,D=1.1",
                "rootPriorVariance", rootPriorVariance,
                "offset", 0.01);
        double pruningScore = runtime.getApproximateTreeLogLikelihood();

        double expectedDifference = -0.5 * sample.logNormalizerDimension * LOG_TWO_PI;
        assertEquals(expectedDifference, covarianceScore - pruningScore, SCORE_TOLERANCE);
    }

    private void assertTrainingAndRuntimeHaveMatchingCrossTreeDifferences(double rootPriorVariance) {
        Alignment alignment = createAlignment();
        List<Tree> trees = createTrees(alignment);
        double[] embeddings = new double[] {
                0.0, 0.4,
                0.7, -0.2,
                -0.3, 1.1,
                1.1, 0.2
        };

        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0, rootPriorVariance);
        double baselineTraining = computeTrainingScore(trainer, trees.get(0), embeddings, 2);
        double baselineRuntime = computeRuntimeScore(alignment, trees.get(0), embeddings, 2, rootPriorVariance);

        for (int i = 1; i < trees.size(); i++) {
            double trainingDifference = computeTrainingScore(trainer, trees.get(i), embeddings, 2) - baselineTraining;
            double runtimeDifference = computeRuntimeScore(alignment, trees.get(i), embeddings, 2, rootPriorVariance)
                    - baselineRuntime;
            assertEquals(trainingDifference, runtimeDifference, SCORE_TOLERANCE);
        }
    }

    private void assertTrainingAndRuntimeHaveMatchingEmbeddingDifferences(double rootPriorVariance) {
        Alignment alignment = createAlignment();
        List<Tree> trees = createTrees(alignment);
        double[] baselineEmbeddings = new double[] {
                0.0, 0.4,
                0.7, -0.2,
                -0.3, 1.1,
                1.1, 0.2
        };
        double[] alternateEmbeddings = new double[] {
                0.2, -0.1,
                0.5, 0.9,
                -0.8, 0.7,
                1.4, -0.3
        };

        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0, rootPriorVariance);
        for (Tree tree : trees) {
            double trainingDifference = computeTrainingScore(trainer, tree, alternateEmbeddings, 2)
                    - computeTrainingScore(trainer, tree, baselineEmbeddings, 2);
            double runtimeDifference = computeRuntimeScore(alignment, tree, alternateEmbeddings, 2, rootPriorVariance)
                    - computeRuntimeScore(alignment, tree, baselineEmbeddings, 2, rootPriorVariance);
            assertEquals(trainingDifference, runtimeDifference, SCORE_TOLERANCE);
        }
    }

    private double computeTrainingScore(
            LearnedWeightsTreeTrainer trainer,
            Tree tree,
            double[] embeddings,
            int dim) {
        LearnedWeightsTreeTrainer.TrainingTreeSample sample = trainer.buildTrainingSample(tree, 0.0);
        return LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihood(sample, embeddings, dim, 1.0);
    }

    private double computeRuntimeScore(
            Alignment alignment,
            Tree tree,
            double[] embeddings,
            int dim,
            double rootPriorVariance) {
        PCAWeightsBrownianFullLikelihood runtime = new PCAWeightsBrownianFullLikelihood();
        runtime.initByName(
                "tree", tree,
                "data", alignment,
                "dimension", dim,
                "value", toValueString(embeddings, dim),
                "rootPriorVariance", rootPriorVariance,
                "offset", 0.01);
        return runtime.getApproximateTreeLogLikelihood();
    }

    private String toValueString(double[] embeddings, int dim) {
        String[] taxa = new String[] {"A", "B", "C", "D"};
        StringBuilder builder = new StringBuilder();
        for (int taxon = 0; taxon < taxa.length; taxon++) {
            if (taxon > 0) {
                builder.append(',');
            }
            builder.append(taxa[taxon]).append('=');
            int offset = taxon * dim;
            for (int axis = 0; axis < dim; axis++) {
                if (axis > 0) {
                    builder.append(' ');
                }
                builder.append(embeddings[offset + axis]);
            }
        }
        return builder.toString();
    }

    private static Alignment createAlignment() {
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
        return alignment;
    }

    private static Tree createTree(Alignment alignment) {
        Tree tree = new TreeParser();
        ((TreeParser) tree).initByName(
                "newick", "((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);",
                "IsLabelledNewick", true,
                "taxa", alignment,
                "adjustTipHeights", false);
        return tree;
    }

    private static List<Tree> createTrees(Alignment alignment) {
        return List.of(
                createTree(alignment),
                parseTree("((A:0.12,C:0.12):0.18,(B:0.10,D:0.10):0.20);", alignment),
                parseTree("(((A:0.08,B:0.08):0.10,C:0.18):0.12,D:0.30);", alignment));
    }

    private static Tree parseTree(String newick, Alignment alignment) {
        Tree tree = new TreeParser();
        ((TreeParser) tree).initByName(
                "newick", newick,
                "IsLabelledNewick", true,
                "taxa", alignment,
                "adjustTipHeights", false);
        return tree;
    }
}