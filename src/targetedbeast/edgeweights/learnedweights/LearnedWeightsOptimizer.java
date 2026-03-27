package targetedbeast.edgeweights.learnedweights;

import java.util.Arrays;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.exception.TooManyEvaluationsException;
import org.apache.commons.math3.exception.TooManyIterationsException;
import org.apache.commons.math3.optim.ConvergenceChecker;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.MaxIter;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient;
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.Preconditioner;

/**
 * Gradient-based optimizer for LearnedWeights.
 *
 * Delegates the optimization loop and line search to Apache Commons Math's
 * nonlinear conjugate gradient implementation while preserving the existing
 * result metadata contract used by this package.
 */
public final class LearnedWeightsOptimizer {

    private static final double DEFAULT_LINE_SEARCH_RELATIVE_TOLERANCE = 1e-8;
    private static final double DEFAULT_LINE_SEARCH_ABSOLUTE_TOLERANCE = 1e-8;
    private static final int DEFAULT_MAX_EVALUATIONS_PER_ITERATION = 50;

    public interface DifferentiableFunction extends MultivariateFunction {
        default void beginIteration(int iteration) {
        }

        double valueAndGradient(double[] point, double[] gradient);
    }

    private static final int DEFAULT_MAX_ITER = 2_000;

    public static final class Config {
        public int maxIter = DEFAULT_MAX_ITER;
        public double learningRate = 0.02;
        public double gradientTolerance = 1e-12;
        public double maxGradientNorm = 100.0;

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
    }

    public static final class Result {
        public final double[] optimizedTheta;
        public final double initialLoss;
        public final double finalLoss;
        public final int iterations;
        public final int evaluations;
        public final String status;
        public final String stopReason;
        public final double initialGradientNorm;
        public final double finalGradientNorm;

        public Result(
                double[] optimizedTheta,
                double initialLoss,
                double finalLoss,
                int iterations,
                int evaluations,
                String status,
                String stopReason,
                double initialGradientNorm,
                double finalGradientNorm) {
            this.optimizedTheta = Arrays.copyOf(optimizedTheta, optimizedTheta.length);
            this.initialLoss = initialLoss;
            this.finalLoss = finalLoss;
            this.iterations = iterations;
            this.evaluations = evaluations;
            this.status = status;
            this.stopReason = stopReason;
            this.initialGradientNorm = initialGradientNorm;
            this.finalGradientNorm = finalGradientNorm;
        }
    }

    private final Config config;

    public LearnedWeightsOptimizer() {
        this(new Config());
    }

    public LearnedWeightsOptimizer(Config config) {
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }
        if (config.maxIter <= 0) {
            throw new IllegalArgumentException("maxIter must be > 0");
        }
        this.config = config;
    }

    public Result optimize(double[] initialTheta, DifferentiableFunction objective) {
        if (initialTheta == null || initialTheta.length == 0) {
            throw new IllegalArgumentException("initialTheta must not be null or empty");
        }
        if (objective == null) {
            throw new IllegalArgumentException("objective must not be null");
        }

        int n = initialTheta.length;
        double[] point = Arrays.copyOf(initialTheta, n);
        double[] gradient = new double[n];

        objective.beginIteration(0);
        double currentLoss = objective.valueAndGradient(point, gradient);
        double initialLoss = currentLoss;
        double initialGradientNorm = l2Norm(gradient);
        int evaluations = 1;

        if (!Double.isFinite(currentLoss)) {
            throw new IllegalStateException("Initial LearnedWeights loss is not finite: " + currentLoss);
        }

        if (initialGradientNorm <= config.gradientTolerance) {
            return new Result(
                    point,
                    initialLoss,
                    initialLoss,
                    0,
                    evaluations,
                    "success",
                    String.format("gradient norm %.3g <= tolerance %.3g", initialGradientNorm, config.gradientTolerance),
                    initialGradientNorm,
                    initialGradientNorm);
        }

        EvaluationAdapter adapter = new EvaluationAdapter(objective, point, currentLoss, gradient, 1);
        GradientNormChecker checker = new GradientNormChecker(adapter, config.gradientTolerance);
        NonLinearConjugateGradientOptimizer optimizer = new NonLinearConjugateGradientOptimizer(
                NonLinearConjugateGradientOptimizer.Formula.POLAK_RIBIERE,
                checker,
                DEFAULT_LINE_SEARCH_RELATIVE_TOLERANCE,
                DEFAULT_LINE_SEARCH_ABSOLUTE_TOLERANCE,
                config.learningRate,
                new ClippedGradientPreconditioner(config.maxGradientNorm));

        try {
            PointValuePair optimum = optimizer.optimize(
                    new MaxIter(config.maxIter),
                    new MaxEval(computeMaxEvaluations(config.maxIter)),
                    new ObjectiveFunction(adapter::value),
                    new ObjectiveFunctionGradient(adapter::gradient),
                    GoalType.MINIMIZE,
                    new InitialGuess(point));

            double[] optimizedPoint = optimum.getPoint();
            double finalGradientNorm = adapter.getGradientNorm(optimizedPoint);
            return new Result(
                    optimizedPoint,
                    initialLoss,
                    optimum.getValue(),
                    optimizer.getIterations(),
                    adapter.getEvaluationCount(),
                    "success",
                    String.format("gradient norm %.3g <= tolerance %.3g", finalGradientNorm, config.gradientTolerance),
                    initialGradientNorm,
                    finalGradientNorm);
        } catch (TooManyIterationsException e) {
            double[] bestPoint = adapter.getBestPoint();
            double finalGradientNorm = adapter.getGradientNorm(bestPoint);
            return new Result(
                    bestPoint,
                    initialLoss,
                    adapter.getBestValue(),
                    config.maxIter,
                    adapter.getEvaluationCount(),
                    "partial",
                    String.format("reached maxIter=%d", config.maxIter),
                    initialGradientNorm,
                    finalGradientNorm);
        } catch (TooManyEvaluationsException e) {
            double[] bestPoint = adapter.getBestPoint();
            double finalGradientNorm = adapter.getGradientNorm(bestPoint);
            return new Result(
                    bestPoint,
                    initialLoss,
                    adapter.getBestValue(),
                    optimizer.getIterations(),
                    adapter.getEvaluationCount(),
                    "partial",
                    String.format("reached maxEval=%d", computeMaxEvaluations(config.maxIter)),
                    initialGradientNorm,
                    finalGradientNorm);
        } catch (RuntimeException e) {
            double[] bestPoint = adapter.getBestPoint();
            double finalGradientNorm = adapter.getGradientNorm(bestPoint);
            return new Result(
                    bestPoint,
                    initialLoss,
                    adapter.getBestValue(),
                    optimizer.getIterations(),
                    adapter.getEvaluationCount(),
                    "partial",
                    "optimizer failed: " + e.getClass().getSimpleName() + " - " + e.getMessage(),
                    initialGradientNorm,
                    finalGradientNorm);
        }
    }

    private static int computeMaxEvaluations(int maxIter) {
        long maxEvaluations = (long) maxIter * DEFAULT_MAX_EVALUATIONS_PER_ITERATION;
        return (int) Math.min(Integer.MAX_VALUE, Math.max(maxIter, maxEvaluations));
    }

    private static double l2Norm(double[] values) {
        double sumSquares = 0.0;
        for (double value : values) {
            sumSquares += value * value;
        }
        return Math.sqrt(sumSquares);
    }

    private static final class GradientNormChecker implements ConvergenceChecker<PointValuePair> {
        private final EvaluationAdapter adapter;
        private final double gradientTolerance;

        GradientNormChecker(EvaluationAdapter adapter, double gradientTolerance) {
            this.adapter = adapter;
            this.gradientTolerance = gradientTolerance;
        }

        @Override
        public boolean converged(int iteration, PointValuePair previous, PointValuePair current) {
            double gradientNorm = adapter.getGradientNorm(current.getPoint());
            return Double.isFinite(gradientNorm) && gradientNorm <= gradientTolerance;
        }
    }

    private static final class ClippedGradientPreconditioner implements Preconditioner {
        private final double maxGradientNorm;

        ClippedGradientPreconditioner(double maxGradientNorm) {
            this.maxGradientNorm = maxGradientNorm;
        }

        @Override
        public double[] precondition(double[] variables, double[] r) {
            double norm = l2Norm(r);
            double scale = norm > maxGradientNorm ? maxGradientNorm / norm : 1.0;
            double[] clipped = new double[r.length];
            for (int i = 0; i < r.length; i++) {
                clipped[i] = scale * r[i];
            }
            return clipped;
        }
    }

    private static final class EvaluationAdapter {
        private final DifferentiableFunction objective;

        private double[] cachedPoint;
        private double cachedValue;
        private double[] cachedGradient;
        private double[] bestPoint;
        private double bestValue;
        private int evaluationCount;
        private int nextIterationIndex;

        EvaluationAdapter(
                DifferentiableFunction objective,
                double[] initialPoint,
                double initialValue,
                double[] initialGradient,
                int nextIterationIndex) {
            this.objective = objective;
            this.cachedPoint = Arrays.copyOf(initialPoint, initialPoint.length);
            this.cachedValue = initialValue;
            this.cachedGradient = Arrays.copyOf(initialGradient, initialGradient.length);
            this.bestPoint = Arrays.copyOf(initialPoint, initialPoint.length);
            this.bestValue = initialValue;
            this.evaluationCount = 1;
            this.nextIterationIndex = nextIterationIndex;
        }

        double value(double[] point) {
            if (matchesCachedPoint(point)) {
                return cachedValue;
            }
            double value = objective.value(point);
            cacheValue(point, value);
            return value;
        }

        double[] gradient(double[] point) {
            if (!matchesCachedGradient(point)) {
                objective.beginIteration(nextIterationIndex++);
                double[] gradient = new double[point.length];
                double value = objective.valueAndGradient(point, gradient);
                cacheValueAndGradient(point, value, gradient);
            }
            return Arrays.copyOf(cachedGradient, cachedGradient.length);
        }

        double getGradientNorm(double[] point) {
            if (!matchesCachedGradient(point)) {
                double[] gradient = new double[point.length];
                double value = objective.valueAndGradient(point, gradient);
                cacheValueAndGradient(point, value, gradient);
            }
            return l2Norm(cachedGradient);
        }

        int getEvaluationCount() {
            return evaluationCount;
        }

        double[] getBestPoint() {
            return Arrays.copyOf(bestPoint, bestPoint.length);
        }

        double getBestValue() {
            return bestValue;
        }

        private boolean matchesCachedPoint(double[] point) {
            return cachedPoint != null && Arrays.equals(cachedPoint, point);
        }

        private boolean matchesCachedGradient(double[] point) {
            return matchesCachedPoint(point) && cachedGradient != null;
        }

        private void cacheValue(double[] point, double value) {
            cachedPoint = Arrays.copyOf(point, point.length);
            cachedValue = value;
            cachedGradient = null;
            evaluationCount++;
            updateBest(point, value);
        }

        private void cacheValueAndGradient(double[] point, double value, double[] gradient) {
            cachedPoint = Arrays.copyOf(point, point.length);
            cachedValue = value;
            cachedGradient = Arrays.copyOf(gradient, gradient.length);
            evaluationCount++;
            updateBest(point, value);
        }

        private void updateBest(double[] point, double value) {
            if (Double.isFinite(value) && value < bestValue) {
                bestValue = value;
                bestPoint = Arrays.copyOf(point, point.length);
            }
        }
    }
}
