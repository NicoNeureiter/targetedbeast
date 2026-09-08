package targetedbeast.edgeweights.learnedweights;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
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

    @Test
    public void validationSplit_isDeterministicForFixedSeed() {
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = createManySamples(10);

        LearnedWeightsTreeObjectiveFunction first = newObjectiveWithSamples(samples, 42L);
        first.configureValidation(0.3, 0, 1);

        LearnedWeightsTreeObjectiveFunction second = newObjectiveWithSamples(samples, 42L);
        second.configureValidation(0.3, 0, 1);

        assertArrayEquals(first.getTrainIndices(), second.getTrainIndices());
        assertArrayEquals(first.getValidationIndices(), second.getValidationIndices());
    }

    @Test
    public void validationSplit_isDisjointAndExhaustive() {
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = createManySamples(10);
        LearnedWeightsTreeObjectiveFunction objective = newObjectiveWithSamples(samples, 3L);
        objective.configureValidation(0.3, 0, 1);

        int[] trainIndices = objective.getTrainIndices();
        int[] validationIndices = objective.getValidationIndices();

        assertEquals(10, trainIndices.length + validationIndices.length);
        assertEquals(3, validationIndices.length);

        boolean[] seen = new boolean[10];
        for (int index : trainIndices) {
            assertTrue("train index out of range: " + index, index >= 0 && index < 10);
            assertTrue("index " + index + " appears more than once", !seen[index]);
            seen[index] = true;
        }
        for (int index : validationIndices) {
            assertTrue("validation index out of range: " + index, index >= 0 && index < 10);
            assertTrue("index " + index + " appears more than once", !seen[index]);
            seen[index] = true;
        }
        for (int index = 0; index < seen.length; index++) {
            assertTrue("index " + index + " missing from split", seen[index]);
        }
    }

    @Test
    public void validationFractionZero_disablesValidationAndKeepsFullTrainingPool() {
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = createManySamples(10);
        LearnedWeightsTreeObjectiveFunction objective = newObjectiveWithSamples(samples, 42L);
        objective.configureValidation(0.0, 0, 1);

        assertTrue(!objective.isValidationEnabled());
        assertEquals(10, objective.getTrainIndices().length);
        assertEquals(0, objective.getValidationIndices().length);
    }

    @Test
    public void miniBatch_neverDrawsValidationSamples() {
        Alignment alignment = createAlignment();
        double[] theta = new double[] {
                0.0, 0.2,
                0.8, -0.1,
                -0.3, 1.0,
                1.1, 1.3
        };
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0);
        List<Tree> trees = createTrees(alignment);
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            LearnedWeightsTreeTrainer.TrainingTreeSample baseSample = trainer.buildTrainingSample(trees.get(i % trees.size()), 0.0);
            double surrogate = LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihood(baseSample, theta, 2, 1.0);
            samples.add(new LearnedWeightsTreeTrainer.TrainingTreeSample(
                    (1.0 + 0.1 * i) * surrogate - 0.25 * i,
                    baseSample.precisionMatrix,
                    baseSample.logDeterminant));
        }

        LearnedWeightsTreeObjectiveFunction miniBatchObjective = new LearnedWeightsTreeObjectiveFunction(
                samples, alignment.getTaxonCount(), 2, 1.0, null, 0.0, 0.0, 1, 7L);
        miniBatchObjective.configureValidation(0.25, 0, 1);
        miniBatchObjective.beginOptimizationRun("test");

        int[] validationIndices = miniBatchObjective.getValidationIndices();
        assertTrue("test fixture must actually hold out samples", validationIndices.length > 0);
        double[] params = miniBatchObjective.createInitialParameters(theta);

        double[] lossPerIndex = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            LearnedWeightsTreeObjectiveFunction singleton = new LearnedWeightsTreeObjectiveFunction(
                    List.of(samples.get(i)), alignment.getTaxonCount(), 2, 1.0);
            lossPerIndex[i] = singleton.value(params);
        }

        for (int iteration = 0; iteration < 3 * samples.size(); iteration++) {
            miniBatchObjective.beginIteration(iteration);
            double observed = miniBatchObjective.value(params);
            for (int validationIndex : validationIndices) {
                assertNotEquals(
                        "mini-batch must never draw a validation sample",
                        lossPerIndex[validationIndex], observed, 1e-9);
            }
        }
    }

    @Test
    public void bestParameters_followValidationLossRatherThanTrainingLossWhenEnabled() {
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> baseSamples = createManySamples(10);

        // Learn the deterministic split first, then build final samples whose targets are
        // rigged so that intercept=5 fits the train group perfectly (and is bad on validation)
        // while intercept=-5 fits the validation group perfectly (and is bad on training).
        LearnedWeightsTreeObjectiveFunction splitProbe = newObjectiveWithSamples(baseSamples, 42L);
        splitProbe.configureValidation(0.3, 0, 1);
        boolean[] isValidation = new boolean[baseSamples.size()];
        for (int index : splitProbe.getValidationIndices()) {
            isValidation[index] = true;
        }

        double[] theta = new double[] {
                0.0, 0.2,
                0.8, -0.1,
                -0.3, 1.0,
                1.1, 1.3
        };
        int dim = 2;
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> riggedSamples = new ArrayList<>();
        for (int i = 0; i < baseSamples.size(); i++) {
            LearnedWeightsTreeTrainer.TrainingTreeSample base = baseSamples.get(i);
            double surrogate = LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihood(base, theta, dim, 1.0);
            double groupOffset = isValidation[i] ? -5.0 : 5.0;
            riggedSamples.add(new LearnedWeightsTreeTrainer.TrainingTreeSample(
                    surrogate + groupOffset, base.precisionMatrix, base.logDeterminant));
        }

        LearnedWeightsTreeObjectiveFunction objective = newObjectiveWithSamples(riggedSamples, 42L);
        objective.configureValidation(0.3, 0, 1);
        objective.beginOptimizationRun("test");

        double[] rates = new double[] {1.0, 1.0};
        double[] slopes = new double[] {1.0, 1.0};
        double[] trainBestParameters = objective.createParameters(theta, rates, null, null, slopes, 5.0);
        double[] validationBestParameters = objective.createParameters(theta, rates, null, null, slopes, -5.0);

        double[] gradient = new double[objective.getParameterCount()];
        objective.valueAndGradient(trainBestParameters, gradient);
        objective.valueAndGradient(validationBestParameters, gradient);

        assertArrayEquals(validationBestParameters, objective.getBestParameters(), 1e-10);
        assertEquals(0.0, objective.getBestValidationLoss(), 1e-8);
    }

    @Test
    public void shouldStopEarly_firesAfterConfiguredPatienceOfNonImprovement() {
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = createManySamples(10);
        LearnedWeightsTreeObjectiveFunction objective = newObjectiveWithSamples(samples, 42L);
        objective.configureValidation(0.3, 3, 1);
        objective.beginOptimizationRun("test");

        double[] theta = new double[] {
                0.0, 0.2,
                0.8, -0.1,
                -0.3, 1.0,
                1.1, 1.3
        };
        double[] parameters = objective.createInitialParameters(theta);
        double[] gradient = new double[objective.getParameterCount()];

        objective.valueAndGradient(parameters, gradient); // establishes the first validation checkpoint
        assertTrue(!objective.shouldStopEarly());

        objective.valueAndGradient(parameters, gradient); // 1st non-improvement (identical loss)
        assertEquals(1, objective.getConsecutiveValidationNonImprovements());
        assertTrue(!objective.shouldStopEarly());

        objective.valueAndGradient(parameters, gradient); // 2nd non-improvement
        assertEquals(2, objective.getConsecutiveValidationNonImprovements());
        assertTrue(!objective.shouldStopEarly());

        objective.valueAndGradient(parameters, gradient); // 3rd non-improvement -> patience reached
        assertEquals(3, objective.getConsecutiveValidationNonImprovements());
        assertTrue(objective.shouldStopEarly());
    }

    private static List<LearnedWeightsTreeTrainer.TrainingTreeSample> createManySamples(int n) {
        Alignment alignment = createAlignment();
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0);
        List<Tree> trees = createTrees(alignment);
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            samples.add(trainer.buildTrainingSample(trees.get(i % trees.size()), 0.1 * i));
        }
        return samples;
    }

    private static LearnedWeightsTreeObjectiveFunction newObjectiveWithSamples(
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples, long seed) {
        return new LearnedWeightsTreeObjectiveFunction(
                samples,
                createAlignment().getTaxonCount(),
                2,
                1.0,
                null,
                0.0,
                0.0,
                0,
                seed);
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
