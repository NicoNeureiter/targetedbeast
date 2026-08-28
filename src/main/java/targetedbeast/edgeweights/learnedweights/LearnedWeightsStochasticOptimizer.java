package targetedbeast.edgeweights.learnedweights;

import java.util.Arrays;

/**
 * Adam-style optimizer for LearnedWeights training.
 *
 * Designed for noisy mini-batch gradients. Updates are driven by the
 * stochastic objective while periodic full-objective evaluations track the
 * best parameters and stopping criterion.
 */
public final class LearnedWeightsStochasticOptimizer {

    private static final int DEFAULT_MAX_ITER = 20_000;
    private static final int DEFAULT_FULL_EVALUATION_FREQUENCY = 50;

    public static final class Config {
        public int maxIter = DEFAULT_MAX_ITER;
        public double learningRate = 0.01;
        public double gradientTolerance = 1e-8;
        public double maxGradientNorm = 100.0;
        public double beta1 = 0.9;
        public double beta2 = 0.999;
        public double epsilon = 1e-8;
        public int fullEvaluationFrequency = DEFAULT_FULL_EVALUATION_FREQUENCY;

        public Config withMaxIter(int value) {
            this.maxIter = value;
            return this;
        }

        public Config withLearningRate(double value) {
            this.learningRate = value;
            return this;
        }

        public Config withGradientTolerance(double value) {
            this.gradientTolerance = value;
            return this;
        }

        public Config withMaxGradientNorm(double value) {
            this.maxGradientNorm = value;
            return this;
        }

        public Config withBeta1(double value) {
            this.beta1 = value;
            return this;
        }

        public Config withBeta2(double value) {
            this.beta2 = value;
            return this;
        }

        public Config withEpsilon(double value) {
            this.epsilon = value;
            return this;
        }

        public Config withFullEvaluationFrequency(int value) {
            this.fullEvaluationFrequency = value;
            return this;
        }
    }

    private final Config config;

    public LearnedWeightsStochasticOptimizer() {
        this(new Config());
    }

    public LearnedWeightsStochasticOptimizer(Config config) {
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }
        if (config.maxIter <= 0) {
            throw new IllegalArgumentException("maxIter must be > 0");
        }
        if (!(config.learningRate > 0.0)) {
            throw new IllegalArgumentException("learningRate must be > 0");
        }
        if (!(config.gradientTolerance > 0.0)) {
            throw new IllegalArgumentException("gradientTolerance must be > 0");
        }
        if (!(config.maxGradientNorm > 0.0)) {
            throw new IllegalArgumentException("maxGradientNorm must be > 0");
        }
        if (!(config.beta1 >= 0.0 && config.beta1 < 1.0)) {
            throw new IllegalArgumentException("beta1 must be in [0, 1)");
        }
        if (!(config.beta2 >= 0.0 && config.beta2 < 1.0)) {
            throw new IllegalArgumentException("beta2 must be in [0, 1)");
        }
        if (!(config.epsilon > 0.0)) {
            throw new IllegalArgumentException("epsilon must be > 0");
        }
        if (config.fullEvaluationFrequency <= 0) {
            throw new IllegalArgumentException("fullEvaluationFrequency must be > 0");
        }
        this.config = config;
    }

    public LearnedWeightsOptimizer.Result optimize(
            double[] initialTheta,
            LearnedWeightsOptimizer.DifferentiableFunction stochasticObjective,
            LearnedWeightsOptimizer.DifferentiableFunction fullObjective) {
        if (initialTheta == null || initialTheta.length == 0) {
            throw new IllegalArgumentException("initialTheta must not be null or empty");
        }
        if (stochasticObjective == null) {
            throw new IllegalArgumentException("stochasticObjective must not be null");
        }

        LearnedWeightsOptimizer.DifferentiableFunction evaluationObjective =
                fullObjective == null ? stochasticObjective : fullObjective;
        int parameterCount = initialTheta.length;
        double[] point = Arrays.copyOf(initialTheta, parameterCount);
        double[] firstMoment = new double[parameterCount];
        double[] secondMoment = new double[parameterCount];

        EvaluationSnapshot initialSnapshot = evaluate(evaluationObjective, point, 0);
        int evaluations = 1;
        double initialLoss = initialSnapshot.loss;
        double initialGradientNorm = initialSnapshot.gradientNorm;

        if (!Double.isFinite(initialLoss)) {
            throw new IllegalStateException("Initial LearnedWeights loss is not finite: " + initialLoss);
        }

        if (initialGradientNorm <= config.gradientTolerance) {
            return new LearnedWeightsOptimizer.Result(
                    point,
                    initialLoss,
                    initialLoss,
                    0,
                    evaluations,
                    "success",
                    String.format(
                            "full gradient norm %.3g <= tolerance %.3g",
                            initialGradientNorm,
                            config.gradientTolerance),
                    initialGradientNorm,
                    initialGradientNorm);
        }

        double[] bestPoint = Arrays.copyOf(point, point.length);
        double bestLoss = initialLoss;
        double finalGradientNorm = initialGradientNorm;
        String status = "partial";
        String stopReason = String.format("reached maxIter=%d", config.maxIter);
        int completedIterations = 0;

        try {
            for (int iteration = 1; iteration <= config.maxIter; iteration++) {
                stochasticObjective.beginIteration(iteration);
                double[] batchGradient = new double[parameterCount];
                double batchLoss = stochasticObjective.valueAndGradient(point, batchGradient);
                evaluations++;

                if (!Double.isFinite(batchLoss) || !allFinite(batchGradient)) {
                    stopReason = String.format(
                            "stochastic objective produced non-finite values at iteration %d",
                            iteration);
                    break;
                }

                clipGradient(batchGradient, config.maxGradientNorm);
                applyAdamUpdate(point, firstMoment, secondMoment, batchGradient, iteration);
                completedIterations = iteration;

                boolean shouldFullEvaluate = iteration == config.maxIter
                        || iteration % config.fullEvaluationFrequency == 0;
                if (!shouldFullEvaluate) {
                    continue;
                }

                EvaluationSnapshot fullSnapshot = evaluate(evaluationObjective, point, iteration);
                evaluations++;
                finalGradientNorm = fullSnapshot.gradientNorm;

                if (!Double.isFinite(fullSnapshot.loss) || !allFinite(fullSnapshot.gradient)) {
                    stopReason = String.format(
                            "full objective produced non-finite values at iteration %d",
                            iteration);
                    break;
                }

                if (fullSnapshot.loss < bestLoss) {
                    bestLoss = fullSnapshot.loss;
                    bestPoint = Arrays.copyOf(point, point.length);
                }

                if (fullSnapshot.gradientNorm <= config.gradientTolerance) {
                    return new LearnedWeightsOptimizer.Result(
                            bestPoint,
                            initialLoss,
                            bestLoss,
                            completedIterations,
                            evaluations,
                            "success",
                            String.format(
                                    "full gradient norm %.3g <= tolerance %.3g",
                                    fullSnapshot.gradientNorm,
                                    config.gradientTolerance),
                            initialGradientNorm,
                            fullSnapshot.gradientNorm);
                }
            }
        } catch (RuntimeException e) {
            stopReason = "optimizer failed: " + e.getClass().getSimpleName() + " - " + e.getMessage();
        }

        if (completedIterations > 0 && (completedIterations % config.fullEvaluationFrequency) != 0) {
            EvaluationSnapshot finalSnapshot = evaluate(evaluationObjective, point, completedIterations);
            evaluations++;
            if (Double.isFinite(finalSnapshot.loss) && allFinite(finalSnapshot.gradient)) {
                finalGradientNorm = finalSnapshot.gradientNorm;
                if (finalSnapshot.loss < bestLoss) {
                    bestLoss = finalSnapshot.loss;
                    bestPoint = Arrays.copyOf(point, point.length);
                }
            }
        }

        return new LearnedWeightsOptimizer.Result(
                bestPoint,
                initialLoss,
                bestLoss,
                completedIterations,
                evaluations,
                status,
                stopReason,
                initialGradientNorm,
                finalGradientNorm);
    }

    private EvaluationSnapshot evaluate(
            LearnedWeightsOptimizer.DifferentiableFunction objective,
            double[] point,
            int iteration) {
        objective.beginIteration(iteration);
        double[] gradient = new double[point.length];
        double loss = objective.valueAndGradient(point, gradient);
        return new EvaluationSnapshot(loss, gradient, l2Norm(gradient));
    }

    private void applyAdamUpdate(
            double[] point,
            double[] firstMoment,
            double[] secondMoment,
            double[] gradient,
            int iteration) {
        double beta1Correction = 1.0 - Math.pow(config.beta1, iteration);
        double beta2Correction = 1.0 - Math.pow(config.beta2, iteration);
        for (int i = 0; i < point.length; i++) {
            double g = gradient[i];
            firstMoment[i] = config.beta1 * firstMoment[i] + (1.0 - config.beta1) * g;
            secondMoment[i] = config.beta2 * secondMoment[i] + (1.0 - config.beta2) * g * g;
            double mHat = firstMoment[i] / beta1Correction;
            double vHat = secondMoment[i] / beta2Correction;
            point[i] -= config.learningRate * mHat / (Math.sqrt(vHat) + config.epsilon);
        }
    }

    private static void clipGradient(double[] gradient, double maxGradientNorm) {
        double norm = l2Norm(gradient);
        if (!(norm > maxGradientNorm)) {
            return;
        }
        double scale = maxGradientNorm / norm;
        for (int i = 0; i < gradient.length; i++) {
            gradient[i] *= scale;
        }
    }

    private static boolean allFinite(double[] values) {
        for (double value : values) {
            if (!Double.isFinite(value)) {
                return false;
            }
        }
        return true;
    }

    private static double l2Norm(double[] values) {
        double sumSquares = 0.0;
        for (double value : values) {
            sumSquares += value * value;
        }
        return Math.sqrt(sumSquares);
    }

    private record EvaluationSnapshot(double loss, double[] gradient, double gradientNorm) {
    }
}