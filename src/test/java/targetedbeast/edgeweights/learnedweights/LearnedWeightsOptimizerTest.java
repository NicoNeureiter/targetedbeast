package targetedbeast.edgeweights.learnedweights;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Random;

import org.junit.Test;

/**
 * Component tests for {@link LearnedWeightsOptimizer}.
 */
public class LearnedWeightsOptimizerTest {

    @Test
    public void optimize_decreasesObjectiveAndReturnsSaneMetadata() {
        double[] center = new double[]{0.7, -1.2, 0.3, 1.1};
        QuadraticObjective objective = new QuadraticObjective(center);

        double[] start = new double[]{2.0, 2.0, -2.0, -2.0};

        LearnedWeightsOptimizer optimizer = new LearnedWeightsOptimizer(
                new LearnedWeightsOptimizer.Config()
                        .withMaxIter(500)
                        .withLearningRate(0.1)
                        .withGradientTolerance(1e-8));

        LearnedWeightsOptimizer.Result result = optimizer.optimize(start, objective);

        assertTrue(Double.isFinite(result.initialLoss));
        assertTrue(Double.isFinite(result.finalLoss));
        assertTrue("Final loss should not be greater than initial loss", result.finalLoss <= result.initialLoss + 1e-10);

        assertTrue("Iterations should be non-negative", result.iterations >= 0);
        assertTrue("Evaluations should be positive", result.evaluations > 0);
        assertTrue(result.status.equals("success") || result.status.equals("partial"));

        for (int i = 0; i < center.length; i++) {
            assertEquals(center[i], result.optimizedTheta[i], 2e-2);
        }
    }

    @Test
    public void optimize_stopsImmediatelyWhenGradientIsZero() {
        double[] center = new double[]{1.0, -2.0, 0.5};
        QuadraticObjective objective = new QuadraticObjective(center);

        LearnedWeightsOptimizer optimizer = new LearnedWeightsOptimizer(
                new LearnedWeightsOptimizer.Config()
                        .withMaxIter(50)
                        .withLearningRate(0.1)
                        .withGradientTolerance(1e-10));

        LearnedWeightsOptimizer.Result result = optimizer.optimize(center, objective);

        assertEquals("success", result.status);
        assertTrue(result.stopReason.contains("gradient norm"));
        assertEquals(0.0, result.finalLoss, 1e-12);
        assertEquals(0.0, result.initialGradientNorm, 1e-12);
        assertEquals(0.0, result.finalGradientNorm, 1e-12);
    }

    @Test
    public void stochasticOptimize_convergesWithSingleSampleBatches() {
        double[][] sampleCenters = new double[][] {
                {1.0, -2.0, 0.5},
                {0.8, -1.8, 0.7},
                {1.2, -2.1, 0.6},
                {0.9, -1.9, 0.4}
        };
        double[] expectedCenter = mean(sampleCenters);
        MiniBatchQuadraticObjective stochasticObjective = new MiniBatchQuadraticObjective(sampleCenters, 1, 7L);
        FullBatchQuadraticObjective fullObjective = new FullBatchQuadraticObjective(sampleCenters);
        double[] start = new double[] {4.0, 3.0, -2.0};

        LearnedWeightsStochasticOptimizer optimizer = new LearnedWeightsStochasticOptimizer(
                new LearnedWeightsStochasticOptimizer.Config()
                        .withMaxIter(2_000)
                        .withLearningRate(0.05)
                        .withGradientTolerance(1e-6)
                        .withFullEvaluationFrequency(20));

        LearnedWeightsOptimizer.Result result = optimizer.optimize(start, stochasticObjective, fullObjective);

        assertTrue(Double.isFinite(result.initialLoss));
        assertTrue(Double.isFinite(result.finalLoss));
        assertTrue(result.finalLoss <= result.initialLoss + 1e-8);
        assertTrue(result.evaluations > 0);
        for (int i = 0; i < expectedCenter.length; i++) {
            assertEquals(expectedCenter[i], result.optimizedTheta[i], 5e-2);
        }
    }

    @Test
    public void optimize_stopsEarlyAndReturnsValidationSelectedParametersWhenObjectiveSignalsEarlyStopping() {
        double[] center = new double[]{0.7, -1.2, 0.3, 1.1};
        double[] start = new double[]{2.0, 2.0, -2.0, -2.0};
        double[] sentinelParameters = new double[]{0.1, 0.2, 0.3, 0.4};
        double sentinelLoss = -123.0;
        String sentinelReason = "early stopping: validation loss did not improve for 3 evaluations (patience=3)";
        EarlyStoppingObjective objective = new EarlyStoppingObjective(
                new QuadraticObjective(center), 1, sentinelParameters, sentinelLoss, sentinelReason);

        LearnedWeightsOptimizer optimizer = new LearnedWeightsOptimizer(
                new LearnedWeightsOptimizer.Config()
                        .withMaxIter(500)
                        .withLearningRate(0.1)
                        .withGradientTolerance(1e-8));

        LearnedWeightsOptimizer.Result result = optimizer.optimize(start, objective);

        assertEquals("success", result.status);
        assertEquals(sentinelReason, result.stopReason);
        assertEquals(sentinelLoss, result.finalLoss, 0.0);
        assertArrayEquals(sentinelParameters, result.optimizedTheta, 0.0);
    }

    @Test
    public void stochasticOptimize_stopsEarlyAndReturnsValidationSelectedParametersWhenObjectiveSignalsEarlyStopping() {
        double[][] sampleCenters = new double[][] {
                {1.0, -2.0, 0.5},
                {0.8, -1.8, 0.7},
                {1.2, -2.1, 0.6},
                {0.9, -1.9, 0.4}
        };
        double[] start = new double[] {4.0, 3.0, -2.0};
        double[] sentinelParameters = new double[]{0.1, 0.2, 0.3};
        double sentinelLoss = -456.0;
        String sentinelReason = "early stopping: validation loss did not improve for 2 evaluations (patience=2)";
        MiniBatchQuadraticObjective stochasticObjective = new MiniBatchQuadraticObjective(sampleCenters, 1, 7L);
        EarlyStoppingObjective fullObjective = new EarlyStoppingObjective(
                new FullBatchQuadraticObjective(sampleCenters), 2, sentinelParameters, sentinelLoss, sentinelReason);

        LearnedWeightsStochasticOptimizer optimizer = new LearnedWeightsStochasticOptimizer(
                new LearnedWeightsStochasticOptimizer.Config()
                        .withMaxIter(2_000)
                        .withLearningRate(0.05)
                        .withGradientTolerance(1e-6)
                        .withFullEvaluationFrequency(20));

        LearnedWeightsOptimizer.Result result = optimizer.optimize(start, stochasticObjective, fullObjective);

        assertEquals("success", result.status);
        assertEquals(sentinelReason, result.stopReason);
        assertEquals(sentinelLoss, result.finalLoss, 0.0);
        assertArrayEquals(sentinelParameters, result.optimizedTheta, 0.0);
    }

    /** Wraps a delegate objective, signalling early stopping (with fixed sentinel results) after a fixed number of valueAndGradient calls. */
    private static final class EarlyStoppingObjective implements LearnedWeightsOptimizer.DifferentiableFunction {
        private final LearnedWeightsOptimizer.DifferentiableFunction delegate;
        private final int callsBeforeStop;
        private final double[] sentinelParameters;
        private final double sentinelLoss;
        private final String sentinelReason;
        private int calls = 0;

        EarlyStoppingObjective(
                LearnedWeightsOptimizer.DifferentiableFunction delegate,
                int callsBeforeStop,
                double[] sentinelParameters,
                double sentinelLoss,
                String sentinelReason) {
            this.delegate = delegate;
            this.callsBeforeStop = callsBeforeStop;
            this.sentinelParameters = sentinelParameters;
            this.sentinelLoss = sentinelLoss;
            this.sentinelReason = sentinelReason;
        }

        @Override
        public void beginIteration(int iteration) {
            delegate.beginIteration(iteration);
        }

        @Override
        public double value(double[] theta) {
            return delegate.value(theta);
        }

        @Override
        public double valueAndGradient(double[] theta, double[] gradient) {
            calls++;
            return delegate.valueAndGradient(theta, gradient);
        }

        @Override
        public boolean shouldStopEarly() {
            return calls >= callsBeforeStop;
        }

        @Override
        public double[] getEarlyStoppingBestParameters() {
            return shouldStopEarly() ? sentinelParameters : null;
        }

        @Override
        public double getEarlyStoppingBestLoss() {
            return shouldStopEarly() ? sentinelLoss : Double.NaN;
        }

        @Override
        public String getEarlyStoppingStopReason() {
            return sentinelReason;
        }
    }

    private static final class QuadraticObjective implements LearnedWeightsOptimizer.DifferentiableFunction {
        private final double[] center;

        QuadraticObjective(double[] center) {
            this.center = center.clone();
        }

        @Override
        public double value(double[] theta) {
            double sum = 0.0;
            for (int i = 0; i < theta.length; i++) {
                double d = theta[i] - center[i];
                sum += d * d;
            }
            return sum;
        }

        @Override
        public double valueAndGradient(double[] theta, double[] gradient) {
            double sum = 0.0;
            for (int i = 0; i < theta.length; i++) {
                double d = theta[i] - center[i];
                gradient[i] = 2.0 * d;
                sum += d * d;
            }
            return sum;
        }
    }

    private static final class FullBatchQuadraticObjective implements LearnedWeightsOptimizer.DifferentiableFunction {
        private final double[][] sampleCenters;

        FullBatchQuadraticObjective(double[][] sampleCenters) {
            this.sampleCenters = copy(sampleCenters);
        }

        @Override
        public double value(double[] theta) {
            double[] gradient = new double[theta.length];
            return valueAndGradient(theta, gradient);
        }

        @Override
        public double valueAndGradient(double[] theta, double[] gradient) {
            Arrays.fill(gradient, 0.0);
            double loss = 0.0;
            double weight = 1.0 / sampleCenters.length;
            for (double[] center : sampleCenters) {
                for (int i = 0; i < theta.length; i++) {
                    double delta = theta[i] - center[i];
                    loss += weight * delta * delta;
                    gradient[i] += 2.0 * weight * delta;
                }
            }
            return loss;
        }
    }

    private static final class MiniBatchQuadraticObjective implements LearnedWeightsOptimizer.DifferentiableFunction {
        private final double[][] sampleCenters;
        private final int batchSize;
        private final Random random;
        private final int[] order;
        private int cursor;
        private int activeStart;
        private int activeCount;

        MiniBatchQuadraticObjective(double[][] sampleCenters, int batchSize, long seed) {
            this.sampleCenters = copy(sampleCenters);
            this.batchSize = batchSize;
            this.random = new Random(seed);
            this.order = new int[sampleCenters.length];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            reshuffle();
            this.activeStart = 0;
            this.activeCount = Math.min(batchSize, sampleCenters.length);
        }

        @Override
        public void beginIteration(int iteration) {
            if (cursor >= order.length) {
                reshuffle();
                cursor = 0;
            }
            activeStart = cursor;
            activeCount = Math.min(batchSize, order.length - cursor);
            cursor += activeCount;
        }

        @Override
        public double value(double[] theta) {
            double[] gradient = new double[theta.length];
            return valueAndGradient(theta, gradient);
        }

        @Override
        public double valueAndGradient(double[] theta, double[] gradient) {
            Arrays.fill(gradient, 0.0);
            double loss = 0.0;
            double weight = 1.0 / activeCount;
            for (int batchIndex = 0; batchIndex < activeCount; batchIndex++) {
                double[] center = sampleCenters[order[activeStart + batchIndex]];
                for (int i = 0; i < theta.length; i++) {
                    double delta = theta[i] - center[i];
                    loss += weight * delta * delta;
                    gradient[i] += 2.0 * weight * delta;
                }
            }
            return loss;
        }

        private void reshuffle() {
            for (int i = order.length - 1; i > 0; i--) {
                int swapIndex = random.nextInt(i + 1);
                int tmp = order[i];
                order[i] = order[swapIndex];
                order[swapIndex] = tmp;
            }
        }
    }

    private static double[] mean(double[][] sampleCenters) {
        double[] mean = new double[sampleCenters[0].length];
        for (double[] center : sampleCenters) {
            for (int i = 0; i < center.length; i++) {
                mean[i] += center[i];
            }
        }
        for (int i = 0; i < mean.length; i++) {
            mean[i] /= sampleCenters.length;
        }
        return mean;
    }

    private static double[][] copy(double[][] values) {
        double[][] copy = new double[values.length][];
        for (int i = 0; i < values.length; i++) {
            copy[i] = Arrays.copyOf(values[i], values[i].length);
        }
        return copy;
    }
}
