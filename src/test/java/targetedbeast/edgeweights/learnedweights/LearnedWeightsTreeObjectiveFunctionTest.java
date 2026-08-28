package targetedbeast.edgeweights.learnedweights;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import targetedbeast.edgeweights.FiniteDifferenceGradientUtil;

public class LearnedWeightsTreeObjectiveFunctionTest {

    @Test
    public void fitAffineCalibration_recoversExactAffineRelationship() {
        double[] surrogate = new double[] {-3.0, -1.0, 2.0, 4.0};
        double[] target = new double[surrogate.length];
        for (int i = 0; i < surrogate.length; i++) {
            target[i] = 1.7 * surrogate[i] - 0.4;
        }

        double[] affine = LearnedWeightsTreeObjectiveFunction.fitAffineCalibration(surrogate, target);

        assertEquals(1.7, affine[0], 1e-12);
        assertEquals(-0.4, affine[1], 1e-12);
    }

    @Test
    public void createInitialParameters_usesZeroSlopeWhenAffineFitIsDegenerate() {
        Alignment alignment = createAlignment();
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0);
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = new ArrayList<>();
        for (Tree tree : createTrees(alignment)) {
            LearnedWeightsTreeTrainer.TrainingTreeSample baseSample = trainer.buildTrainingSample(tree, 5.0);
            samples.add(new LearnedWeightsTreeTrainer.TrainingTreeSample(
                    10.0,
                    baseSample.precisionMatrix,
                    baseSample.logDeterminant));
        }

        LearnedWeightsTreeObjectiveFunction objective = new LearnedWeightsTreeObjectiveFunction(
                samples,
                alignment.getTaxonCount(),
                2,
                1.0);

        double[] initialEmbeddings = new double[] {
                0.0, 0.2,
                0.8, -0.1,
                -0.3, 1.0,
                1.1, 1.3
        };
        double[] parameters = objective.createInitialParameters(initialEmbeddings);

        assertArrayEquals(initialEmbeddings, objective.extractEmbeddings(parameters), 0.0);
        assertArrayEquals(new double[] {1.0, 1.0}, objective.extractBrownianRates(parameters), 1e-12);
        assertArrayEquals(new double[] {0.0, 0.0}, objective.extractSlopes(parameters), 1e-12);
        assertEquals(10.0, objective.extractIntercept(parameters), 1e-12);
    }

    @Test
    public void createInitialParameters_preservesConfiguredBrownianRates() {
        Alignment alignment = createAlignment();
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0);
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = new ArrayList<>();
        for (Tree tree : createTrees(alignment)) {
            samples.add(trainer.buildTrainingSample(tree, 0.0));
        }

        LearnedWeightsTreeObjectiveFunction objective = new LearnedWeightsTreeObjectiveFunction(
                samples,
                alignment.getTaxonCount(),
                2,
                new double[] {0.5, 1.5});

        double[] initialEmbeddings = new double[] {
                0.0, 0.2,
                0.8, -0.1,
                -0.3, 1.0,
                1.1, 1.3
        };
        double[] parameters = objective.createInitialParameters(initialEmbeddings);

        assertArrayEquals(new double[] {0.5, 1.5}, objective.extractBrownianRates(parameters), 1e-12);
    }

    @Test
    public void objectiveCanReachZeroLossForMatchingEmbeddingsAndCalibration() {
        Alignment alignment = createAlignment();
        double[] theta = new double[] {
                0.0, 0.2,
                0.8, -0.1,
                -0.3, 1.0,
                1.1, 1.3
        };
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0);
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = new ArrayList<>();
        for (Tree tree : createTrees(alignment)) {
            LearnedWeightsTreeTrainer.TrainingTreeSample baseSample = trainer.buildTrainingSample(tree, 0.0);
            double surrogate = LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihood(baseSample, theta, 2, 1.0);
            samples.add(new LearnedWeightsTreeTrainer.TrainingTreeSample(
                    1.7 * surrogate - 0.4,
                    baseSample.precisionMatrix,
                    baseSample.logDeterminant));
        }

        LearnedWeightsTreeObjectiveFunction objective = new LearnedWeightsTreeObjectiveFunction(
                samples,
                alignment.getTaxonCount(),
                2,
                1.0);

        double[] parameters = objective.createInitialParameters(theta);

        assertEquals(0.0, objective.value(parameters), 1e-10);
    }

    @Test
    public void perTraitBrownianRatesAdjustAxisContributionsIndependently() {
    Alignment alignment = createAlignment();
    double[] theta = new double[] {
        0.0, 0.2,
        0.8, -0.1,
        -0.3, 1.0,
        1.1, 1.3
    };
    LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0);
    LearnedWeightsTreeTrainer.TrainingTreeSample sample = trainer.buildTrainingSample(createTrees(alignment).get(0), 0.0);

    double[] unitContributions = LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihoodContributions(
        sample, theta, 2, 1.0);
    double[] unitQuadratic = LearnedWeightsTreeObjectiveFunction.computeQuadraticLogLikelihoodContributions(
        sample, theta, 2, 1.0);
    double[] traitRates = new double[] {0.5, 2.0};
    double[] traitContributions = LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihoodContributions(
        sample, theta, 2, traitRates);

    for (int axis = 0; axis < traitRates.length; axis++) {
        double expected = unitContributions[axis]
            - 0.5 * sample.logNormalizerDimension * Math.log(traitRates[axis])
            + unitQuadratic[axis] * ((1.0 / traitRates[axis]) - 1.0);
        assertEquals(expected, traitContributions[axis], 1e-12);
    }
    }

    @Test
    public void pcaRegularizationPenalty_isZeroAtReferenceAndPositiveAwayFromIt() {
        Alignment alignment = createAlignment();
        double[] theta = new double[] {
                0.0, 0.2,
                0.8, -0.1,
                -0.3, 1.0,
                1.1, 1.3
        };
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0);
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = new ArrayList<>();
        for (Tree tree : createTrees(alignment)) {
            samples.add(trainer.buildTrainingSample(tree, 0.0));
        }

        LearnedWeightsTreeObjectiveFunction objective = new LearnedWeightsTreeObjectiveFunction(
                samples,
                alignment.getTaxonCount(),
                2,
                1.0,
                theta,
                5.0,
                0.0);

        double[] atReference = objective.createInitialParameters(theta);
        objective.value(atReference);
        assertEquals(0.0, objective.getLastPcaPenalty(), 1e-12);

        double[] away = atReference.clone();
        away[0] += 0.5;
        away[3] -= 0.25;
        objective.value(away);
        assertTrue(objective.getLastPcaPenalty() > 0.0);
    }

    @Test
    public void slopeRegularizationPenalty_isZeroAtSlopeOneAndPositiveAwayFromIt() {
        Alignment alignment = createAlignment();
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0);
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = new ArrayList<>();
        for (Tree tree : createTrees(alignment)) {
            samples.add(trainer.buildTrainingSample(tree, 0.0));
        }

        double[] reference = new double[] {
                0.0, 0.2,
                0.8, -0.1,
                -0.3, 1.0,
                1.1, 1.3
        };
        LearnedWeightsTreeObjectiveFunction objective = new LearnedWeightsTreeObjectiveFunction(
                samples,
                alignment.getTaxonCount(),
                2,
                1.0,
                reference,
                0.0,
                3.0);

        double[] parameters = objective.createInitialParameters(reference);
        parameters[parameters.length - 1] = 1.0;
        parameters[parameters.length - 2] = 1.0;
        objective.value(parameters);
        assertEquals(0.0, objective.getLastSlopePenalty(), 1e-12);

        parameters[parameters.length - 1] = -0.2;
        parameters[parameters.length - 2] = 1.3;
        objective.value(parameters);
        assertTrue(objective.getLastSlopePenalty() > 0.0);
    }

    @Test
    public void createInitialParameters_preservesNegativeAffineSlope() {
        Alignment alignment = createAlignment();
        double[] theta = new double[] {
                0.0, 0.2,
                0.8, -0.1,
                -0.3, 1.0,
                1.1, 1.3
        };
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0);
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = new ArrayList<>();
        for (Tree tree : createTrees(alignment)) {
            LearnedWeightsTreeTrainer.TrainingTreeSample baseSample = trainer.buildTrainingSample(tree, 0.0);
            double surrogate = LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihood(baseSample, theta, 2, 1.0);
            samples.add(new LearnedWeightsTreeTrainer.TrainingTreeSample(
                    -1.7 * surrogate - 0.4,
                    baseSample.precisionMatrix,
                    baseSample.logDeterminant));
        }

        LearnedWeightsTreeObjectiveFunction objective = new LearnedWeightsTreeObjectiveFunction(
                samples,
                alignment.getTaxonCount(),
                2,
                1.0);

        double[] parameters = objective.createInitialParameters(theta);

        assertArrayEquals(new double[] {-1.7, -1.7}, objective.extractSlopes(parameters), 1e-10);
        assertEquals(-0.4, objective.extractIntercept(parameters), 1e-10);
    }

        @Test
        public void createInitialParameters_canFitQuadraticSlopeWithoutScalingLogDetPart() {
        Alignment alignment = createAlignment();
        double[] theta = new double[] {
            0.0, 0.2,
            0.8, -0.1,
            -0.3, 1.0,
            1.1, 1.3
        };
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0);
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = new ArrayList<>();
        for (Tree tree : createTrees(alignment)) {
            LearnedWeightsTreeTrainer.TrainingTreeSample baseSample = trainer.buildTrainingSample(tree, 0.0);
            LearnedWeightsTreeObjectiveFunction.SurrogateDecomposition decomposition =
                LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihoodDecomposition(baseSample, theta, 2, 1.0);
            samples.add(new LearnedWeightsTreeTrainer.TrainingTreeSample(
                decomposition.logDetPart + 1.7 * decomposition.quadraticPart - 0.4,
                baseSample.precisionMatrix,
                baseSample.logDeterminant));
        }

        LearnedWeightsTreeObjectiveFunction objective = new LearnedWeightsTreeObjectiveFunction(
            samples,
            alignment.getTaxonCount(),
            2,
            1.0,
            null,
            0.0,
            0.0,
            0,
            1L,
            false,
            true);

        double[] parameters = objective.createInitialParameters(theta);

        assertArrayEquals(new double[] {1.7, 1.7}, objective.extractSlopes(parameters), 1e-10);
        assertEquals(-0.4, objective.extractIntercept(parameters), 1e-10);
        assertEquals(0.0, objective.value(parameters), 1e-10);
        }

    @Test
    public void createInitialParameters_canFitFullTargetUsingQuadraticOnlyTrainingAblation() {
        Alignment alignment = createAlignment();
        double[] theta = new double[] {
            0.0, 0.2,
            0.8, -0.1,
            -0.3, 1.0,
            1.1, 1.3
        };
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0);
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = new ArrayList<>();
        for (Tree tree : createTrees(alignment)) {
            LearnedWeightsTreeTrainer.TrainingTreeSample baseSample = trainer.buildTrainingSample(tree, 0.0);
            LearnedWeightsTreeObjectiveFunction.SurrogateDecomposition decomposition =
                LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihoodDecomposition(baseSample, theta, 2, 1.0);
            samples.add(new LearnedWeightsTreeTrainer.TrainingTreeSample(
                1.7 * decomposition.quadraticPart - 0.4,
                baseSample.precisionMatrix,
                baseSample.logDeterminant));
        }

        LearnedWeightsTreeObjectiveFunction objective = new LearnedWeightsTreeObjectiveFunction(
            samples,
            alignment.getTaxonCount(),
            2,
            1.0,
            null,
            0.0,
            0.0,
            0,
            1L,
            false,
            false,
            true);

        double[] parameters = objective.createInitialParameters(theta);

        assertArrayEquals(new double[] {1.7, 1.7}, objective.extractSlopes(parameters), 1e-10);
        assertEquals(-0.4, objective.extractIntercept(parameters), 1e-10);
        assertEquals(0.0, objective.value(parameters), 1e-10);
    }

        @Test
        public void createInitialParameters_canFitSeparateBaseAndQuadraticCalibration() {
        Alignment alignment = createAlignment();
        double[] theta = new double[] {
                0.0,
                0.8,
                -0.3,
                1.1
        };
            double[] baseScales = new double[] {0.6};
            double[] quadraticScales = new double[] {1.7};
        double intercept = -0.4;
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0);
        List<Tree> trees = createTrees(alignment);
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> baseSamples = new ArrayList<>();
        for (Tree tree : trees) {
            baseSamples.add(trainer.buildTrainingSample(tree, 0.0));
        }
        LearnedWeightsTreeObjectiveFunction featureObjective = new LearnedWeightsTreeObjectiveFunction(
            baseSamples,
            alignment.getTaxonCount(),
            1,
            1.0);
        double[][] baseContributions = featureObjective.evaluateBaseContributions(theta);
        double[][] quadraticContributions = featureObjective.evaluateQuadraticContributions(theta);

        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = new ArrayList<>();
        for (int i = 0; i < trees.size(); i++) {
            LearnedWeightsTreeTrainer.TrainingTreeSample baseSample = baseSamples.get(i);
            double target = intercept
                + dot(baseScales, baseContributions[i])
                + dot(quadraticScales, quadraticContributions[i]);
            samples.add(new LearnedWeightsTreeTrainer.TrainingTreeSample(
                target,
                baseSample.precisionMatrix,
                baseSample.logDeterminant));
        }

        LearnedWeightsTreeObjectiveFunction objective = new LearnedWeightsTreeObjectiveFunction(
            samples,
            alignment.getTaxonCount(),
            1,
            new double[] {1.0},
            null,
            0.0,
            0.0,
            0,
            1L,
            false,
            true,
            false,
            false);

        double[] parameters = objective.createInitialParameters(theta);

        assertArrayEquals(baseScales, objective.extractBaseScales(parameters), 1e-10);
        assertArrayEquals(quadraticScales, objective.extractSlopes(parameters), 1e-10);
        assertEquals(intercept, objective.extractIntercept(parameters), 1e-10);
        assertEquals(0.0, objective.value(parameters), 1e-10);
        }

    @Test
    public void analyticGradientMatchesFiniteDifference() {
    Alignment alignment = createAlignment();
    LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0);
    List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = new ArrayList<>();
    for (Tree tree : createTrees(alignment)) {
        samples.add(trainer.buildTrainingSample(tree, -10.0 + tree.getRoot().getHeight()));
    }

    LearnedWeightsTreeObjectiveFunction objective = new LearnedWeightsTreeObjectiveFunction(
        samples,
        alignment.getTaxonCount(),
        2,
        1.0);

    double[] parameters = new double[] {
        0.1, 0.2,
        -0.4, 0.7,
        0.5, -0.3,
        0.9, 1.1,
        -0.2,
        Math.log(0.6), Math.log(1.7),
        1.3,
        0.7
    };
    double[] analytic = new double[parameters.length];
    objective.valueAndGradient(parameters, analytic);
    double[] numerical = FiniteDifferenceGradientUtil.centralDifference(objective, parameters, 1e-6);

    for (int i = 0; i < analytic.length; i++) {
        assertTrue("gradient mismatch at index " + i + ": analytic=" + analytic[i] + ", numerical=" + numerical[i],
            Math.abs(analytic[i] - numerical[i]) < 1e-4);
    }
    }

    @Test
    public void miniBatchObjective_cyclesThroughDistinctSamplesWithinAnEpoch() {
        Alignment alignment = createAlignment();
        double[] theta = new double[] {
                0.0, 0.2,
                0.8, -0.1,
                -0.3, 1.0,
                1.1, 1.3
        };
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0);
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = new ArrayList<>();
        for (Tree tree : createTrees(alignment)) {
            LearnedWeightsTreeTrainer.TrainingTreeSample baseSample = trainer.buildTrainingSample(tree, 0.0);
            double surrogate = LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihood(baseSample, theta, 2, 1.0);
            samples.add(new LearnedWeightsTreeTrainer.TrainingTreeSample(
                    1.5 * surrogate - 0.25,
                    baseSample.precisionMatrix,
                    baseSample.logDeterminant));
        }

        LearnedWeightsTreeObjectiveFunction miniBatchObjective = new LearnedWeightsTreeObjectiveFunction(
                samples,
                alignment.getTaxonCount(),
                2,
                1.0,
                null,
                0.0,
                0.0,
                1,
                7L);
        miniBatchObjective.beginOptimizationRun("test");

        double[] params = miniBatchObjective.createInitialParameters(theta);
        double[] observedLosses = new double[samples.size()];
        for (int iteration = 0; iteration < samples.size(); iteration++) {
            miniBatchObjective.beginIteration(iteration);
            observedLosses[iteration] = miniBatchObjective.value(params);
        }

        double[] expectedLosses = new double[samples.size()];
        for (int sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++) {
            LearnedWeightsTreeObjectiveFunction singletonObjective = new LearnedWeightsTreeObjectiveFunction(
                    List.of(samples.get(sampleIndex)),
                    alignment.getTaxonCount(),
                    2,
                    1.0);
            expectedLosses[sampleIndex] = singletonObjective.value(params);
        }

        Arrays.sort(observedLosses);
        Arrays.sort(expectedLosses);
        assertArrayEquals(expectedLosses, observedLosses, 1e-12);
    }

    private static Alignment createAlignment() {
        Sequence seqA = new Sequence("A", "ACGTACGTAA");
        Sequence seqB = new Sequence("B", "ACGTACGTAT");
        Sequence seqC = new Sequence("C", "ACGTACGTGA");
        Sequence seqD = new Sequence("D", "ACGTACGTGT");

        Alignment alignment = new Alignment();
        alignment.initByName("sequence", seqA,
                "sequence", seqB,
                "sequence", seqC,
                "sequence", seqD);
        return alignment;
    }

    private static List<Tree> createTrees(Alignment alignment) {
        List<Tree> trees = new ArrayList<>();
        trees.add(parse("((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);", alignment));
        trees.add(parse("((A:0.12,C:0.12):0.18,(B:0.1,D:0.1):0.2);", alignment));
        trees.add(parse("(((A:0.08,B:0.08):0.1,C:0.18):0.12,D:0.3);", alignment));
        return trees;
    }

    private static Tree parse(String newick, Alignment alignment) {
        Tree tree = new TreeParser();
        ((TreeParser) tree).initByName(
                "newick", newick,
                "IsLabelledNewick", true,
                "taxa", alignment);
        return tree;
    }

    private static double dot(double[] left, double[] right) {
        double total = 0.0;
        for (int i = 0; i < left.length; i++) {
            total += left[i] * right[i];
        }
        return total;
    }
}
