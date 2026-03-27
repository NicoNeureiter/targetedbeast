package targetedbeast.edgeweights.learnedweights;

import java.util.Arrays;
import java.util.Random;
import java.util.List;

import beast.base.core.Log;

public class LearnedWeightsTreeObjectiveFunction implements LearnedWeightsOptimizer.DifferentiableFunction {

    private static final double VARIANCE_EPS = 1e-12;
    private static final double LOG_TWO_PI = Math.log(2.0 * Math.PI);

    private final List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples;
    private final int taxonCount;
    private final int dim;
    private final double brownianRate;
    private final double[] referenceEmbeddings;
    private final double pcaRegularizationWeight;
    private final double slopeRegularizationWeight;
    private final int embeddingParameterCount;
    private final int interceptParameterIndex;
    private final int slopeParameterIndex;
    private final int parameterCount;
    private final int miniBatchSize;
    private final long miniBatchSeed;
    private final boolean useMiniBatches;
    private final double surrogateConstantTerm;

    private int[] batchOrder;
    private int batchCursor = 0;
    private int activeBatchStart = 0;
    private int activeBatchCount;
    private Random miniBatchRandom;

    private double[] bestParameters;
    private double bestLoss = Double.POSITIVE_INFINITY;
    private double bestFitLoss = Double.POSITIVE_INFINITY;
    private double bestSlopePenalty = 0.0;
    private double bestPcaPenalty = 0.0;
    private double bestSlope = 1.0;
    private double bestIntercept = 0.0;
    private double lastSlope = 1.0;
    private double lastIntercept = 0.0;
    private double lastFitLoss = Double.NaN;
    private double lastSlopePenalty = Double.NaN;
    private double lastPcaPenalty = Double.NaN;
    private int totalEvaluationCount = 0;
    private int runEvaluationCount = 0;
    private int currentIteration = 0;
    private int configuredMaxIterations = -1;
    private String progressLabel = "LearnedWeights";
    private long runStartNanos = -1L;
    private long lastProgressLogNanos = -1L;
    private double lastLoss = Double.NaN;

    public LearnedWeightsTreeObjectiveFunction(
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples,
            int taxonCount,
            int dim,
            double brownianRate) {
        this(trainingSamples, taxonCount, dim, brownianRate, null, 0.0, 0.0);
    }

    public LearnedWeightsTreeObjectiveFunction(
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples,
            int taxonCount,
            int dim,
            double brownianRate,
            double[] referenceEmbeddings,
            double pcaRegularizationWeight,
            double slopeRegularizationWeight) {
        this(trainingSamples, taxonCount, dim, brownianRate, referenceEmbeddings, pcaRegularizationWeight, slopeRegularizationWeight, 0, 1L);
    }

    public LearnedWeightsTreeObjectiveFunction(
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples,
            int taxonCount,
            int dim,
            double brownianRate,
            double[] referenceEmbeddings,
            double pcaRegularizationWeight,
            double slopeRegularizationWeight,
            int miniBatchSize,
            long miniBatchSeed) {
        if (trainingSamples == null || trainingSamples.isEmpty()) {
            throw new IllegalArgumentException("trainingSamples must not be null or empty");
        }

        this.trainingSamples = trainingSamples;
        this.taxonCount = taxonCount;
        this.dim = dim;
        this.brownianRate = brownianRate;
        this.embeddingParameterCount = taxonCount * dim;
        this.interceptParameterIndex = embeddingParameterCount;
        this.slopeParameterIndex = interceptParameterIndex + 1;
        this.parameterCount = embeddingParameterCount + 2;
        if (referenceEmbeddings != null && referenceEmbeddings.length != embeddingParameterCount) {
            throw new IllegalArgumentException("referenceEmbeddings length mismatch: expected "
                    + embeddingParameterCount + " but got " + referenceEmbeddings.length);
        }
        if (pcaRegularizationWeight > 0.0 && referenceEmbeddings == null) {
            throw new IllegalArgumentException("referenceEmbeddings must be provided when pcaRegularizationWeight > 0");
        }
        this.referenceEmbeddings = referenceEmbeddings == null ? null : Arrays.copyOf(referenceEmbeddings, referenceEmbeddings.length);
        this.pcaRegularizationWeight = pcaRegularizationWeight;
        this.slopeRegularizationWeight = slopeRegularizationWeight;
        this.useMiniBatches = miniBatchSize > 0 && miniBatchSize < trainingSamples.size();
        this.miniBatchSize = useMiniBatches ? miniBatchSize : trainingSamples.size();
        this.miniBatchSeed = miniBatchSeed;
        this.activeBatchCount = trainingSamples.size();
        this.surrogateConstantTerm = -0.5 * dim * taxonCount * (LOG_TWO_PI + Math.log(brownianRate));
    }

    @Override
    public void beginIteration(int iteration) {
        currentIteration = iteration;
        if (!useMiniBatches) {
            activeBatchStart = 0;
            activeBatchCount = trainingSamples.size();
            return;
        }
        if (batchOrder == null || batchOrder.length != trainingSamples.size()) {
            resetMiniBatchSchedule();
        }
        if (batchCursor >= batchOrder.length) {
            shuffleBatchOrder();
            batchCursor = 0;
        }
        activeBatchStart = batchCursor;
        activeBatchCount = Math.min(miniBatchSize, batchOrder.length - batchCursor);
        batchCursor += activeBatchCount;
    }

    @Override
    public double value(double[] parameters) {
        return evaluate(parameters, null);
    }

    @Override
    public double valueAndGradient(double[] parameters, double[] gradient) {
        return evaluate(parameters, gradient);
    }

    public void beginOptimizationRun(String label) {
        beginOptimizationRun(label, -1);
    }

    public void beginOptimizationRun(String label, int maxIterations) {
        if (label != null && !label.isBlank()) {
            progressLabel = label;
        }
        configuredMaxIterations = maxIterations;
        bestParameters = null;
        bestLoss = Double.POSITIVE_INFINITY;
        bestFitLoss = Double.POSITIVE_INFINITY;
        bestSlopePenalty = 0.0;
        bestPcaPenalty = 0.0;
        bestSlope = 1.0;
        bestIntercept = 0.0;
        currentIteration = 0;
        runEvaluationCount = 0;
        runStartNanos = -1L;
        lastProgressLogNanos = -1L;
        lastLoss = Double.NaN;
        lastFitLoss = Double.NaN;
        lastSlopePenalty = Double.NaN;
        lastPcaPenalty = Double.NaN;
        resetMiniBatchSchedule();
    }

    public boolean usesMiniBatches() {
        return useMiniBatches;
    }

    public int getMiniBatchSize() {
        return miniBatchSize;
    }

    public int getEmbeddingParameterCount() {
        return embeddingParameterCount;
    }

    public int getParameterCount() {
        return parameterCount;
    }

    public double gradientNorm(double[] gradient) {
        return l2Norm(gradient, 0, gradient.length);
    }

    public double embeddingGradientNorm(double[] gradient) {
        return l2Norm(gradient, 0, embeddingParameterCount);
    }

    public double calibrationGradientNorm(double[] gradient) {
        return l2Norm(gradient, embeddingParameterCount, parameterCount);
    }

    public double[] createInitialParameters(double[] initialEmbeddings) {
        if (initialEmbeddings == null || initialEmbeddings.length != embeddingParameterCount) {
            throw new IllegalArgumentException("initialEmbeddings length mismatch: expected "
                    + embeddingParameterCount + " but got " + (initialEmbeddings == null ? 0 : initialEmbeddings.length));
        }

        double[] surrogate = evaluateSurrogates(initialEmbeddings);
        double[] target = new double[trainingSamples.size()];
        for (int i = 0; i < trainingSamples.size(); i++) {
            target[i] = trainingSamples.get(i).targetLogLikelihood;
        }

        double[] affine = fitAffineCalibration(surrogate, target);
        double slope = hasNonZeroVariance(surrogate) ? affine[0] : 1.0;
        double intercept = affine[1];

        double[] initialParameters = new double[parameterCount];
        System.arraycopy(initialEmbeddings, 0, initialParameters, 0, embeddingParameterCount);
        initialParameters[interceptParameterIndex] = intercept;
        initialParameters[slopeParameterIndex] = slope;
        return initialParameters;
    }

    public double[] getBestParameters() {
        return bestParameters == null ? null : Arrays.copyOf(bestParameters, bestParameters.length);
    }

    public double getBestLoss() {
        return bestLoss;
    }

    public double getBestFitLoss() {
        return bestFitLoss;
    }

    public double getBestSlopePenalty() {
        return bestSlopePenalty;
    }

    public double getBestPcaPenalty() {
        return bestPcaPenalty;
    }

    public double getBestSlope() {
        return bestSlope;
    }

    public double getBestIntercept() {
        return bestIntercept;
    }

    public double getLastSlope() {
        return lastSlope;
    }

    public double getLastIntercept() {
        return lastIntercept;
    }

    public double getLastFitLoss() {
        return lastFitLoss;
    }

    public double getLastSlopePenalty() {
        return lastSlopePenalty;
    }

    public double getLastPcaPenalty() {
        return lastPcaPenalty;
    }

    public int getEvaluationCount() {
        return totalEvaluationCount;
    }

    public int getRunEvaluationCount() {
        return runEvaluationCount;
    }

    public double[] extractEmbeddings(double[] parameters) {
        validateParameters(parameters);
        return Arrays.copyOf(parameters, embeddingParameterCount);
    }

    public double extractIntercept(double[] parameters) {
        validateParameters(parameters);
        return parameters[interceptParameterIndex];
    }

    public double extractSlope(double[] parameters) {
        validateParameters(parameters);
        return parameters[slopeParameterIndex];
    }

    public double[] evaluateSurrogates(double[] embeddings) {
        if (embeddings == null || embeddings.length != embeddingParameterCount) {
            throw new IllegalArgumentException("embeddings length mismatch: expected "
                    + embeddingParameterCount + " but got " + (embeddings == null ? 0 : embeddings.length));
        }
        double[] surrogate = new double[trainingSamples.size()];
        for (int sampleIndex = 0; sampleIndex < trainingSamples.size(); sampleIndex++) {
            surrogate[sampleIndex] = computeSurrogateLogLikelihood(trainingSamples.get(sampleIndex), embeddings, dim, brownianRate);
        }
        return surrogate;
    }

    public double[][] evaluateSurrogateContributions(double[] embeddings) {
        if (embeddings == null || embeddings.length != embeddingParameterCount) {
            throw new IllegalArgumentException("embeddings length mismatch: expected "
                    + embeddingParameterCount + " but got " + (embeddings == null ? 0 : embeddings.length));
        }
        double[][] surrogate = new double[trainingSamples.size()][dim];
        for (int sampleIndex = 0; sampleIndex < trainingSamples.size(); sampleIndex++) {
            surrogate[sampleIndex] = computeSurrogateLogLikelihoodContributions(
                    trainingSamples.get(sampleIndex), embeddings, dim, brownianRate);
        }
        return surrogate;
    }

    /**
     * Returns [totalStd, logDetStd, quadFormStd, logDetFraction] measuring how much of the
     * surrogate cross-sample variation comes from log-determinant terms (embedding-independent)
     * vs the quadratic form (embedding-dependent).
     */
    public double[] computeSurrogateVarianceDecomposition(double[] embeddings) {
        if (embeddings == null || embeddings.length != embeddingParameterCount) {
            throw new IllegalArgumentException("embeddings length mismatch");
        }
        int n = trainingSamples.size();
        double[] logDetParts = new double[n];
        double[] quadFormParts = new double[n];
        double[] totals = new double[n];

        for (int i = 0; i < n; i++) {
            LearnedWeightsTreeTrainer.TrainingTreeSample sample = trainingSamples.get(i);
            double logDetPart = -0.5 * dim * sample.logDeterminant;
            double quadraticForm = 0.0;
            double[][] precision = sample.precisionMatrix;
            for (int row = 0; row < taxonCount; row++) {
                for (int col = 0; col < taxonCount; col++) {
                    double entry = precision[row][col];
                    if (entry == 0.0) continue;
                    int rOff = row * dim;
                    int cOff = col * dim;
                    for (int axis = 0; axis < dim; axis++) {
                        quadraticForm += embeddings[rOff + axis] * entry * embeddings[cOff + axis];
                    }
                }
            }
            double quadFormPart = -0.5 * quadraticForm / brownianRate;
            logDetParts[i] = logDetPart;
            quadFormParts[i] = quadFormPart;
            totals[i] = logDetPart + quadFormPart; // ignoring global constants (same for all samples)
        }

        return new double[] {
            standardDeviation(totals),
            standardDeviation(logDetParts),
            standardDeviation(quadFormParts),
            safeFraction(standardDeviation(logDetParts), standardDeviation(totals))
        };
    }

    private static double standardDeviation(double[] values) {
        double mean = 0.0;
        for (double v : values) mean += v;
        mean /= values.length;
        double sumSq = 0.0;
        for (double v : values) {
            double d = v - mean;
            sumSq += d * d;
        }
        return Math.sqrt(sumSq / values.length);
    }

    private static double safeFraction(double numerator, double denominator) {
        return denominator > 0.0 ? numerator / denominator : Double.NaN;
    }

    public static double[] fitAffineCalibration(double[] surrogate, double[] target) {
        if (surrogate.length != target.length || surrogate.length == 0) {
            throw new IllegalArgumentException("surrogate and target must be non-empty and have the same length");
        }

        double meanSurrogate = 0.0;
        double meanTarget = 0.0;
        for (int i = 0; i < surrogate.length; i++) {
            meanSurrogate += surrogate[i];
            meanTarget += target[i];
        }
        meanSurrogate /= surrogate.length;
        meanTarget /= target.length;

        double covariance = 0.0;
        double variance = 0.0;
        for (int i = 0; i < surrogate.length; i++) {
            double centeredSurrogate = surrogate[i] - meanSurrogate;
            covariance += centeredSurrogate * (target[i] - meanTarget);
            variance += centeredSurrogate * centeredSurrogate;
        }

        double slope = variance <= VARIANCE_EPS ? 0.0 : covariance / variance;
        double intercept = meanTarget - slope * meanSurrogate;
        return new double[] {slope, intercept};
    }

    public static double[] fitAffineCalibrationPerDimension(double[][] surrogateByDimension, double[] target) {
        if (surrogateByDimension.length != target.length || surrogateByDimension.length == 0) {
            throw new IllegalArgumentException("surrogateByDimension and target must be non-empty and have the same number of samples");
        }
        int dim = surrogateByDimension[0].length;
        if (dim == 0) {
            throw new IllegalArgumentException("surrogateByDimension must have at least one dimension");
        }
        for (double[] row : surrogateByDimension) {
            if (row.length != dim) {
                throw new IllegalArgumentException("All surrogate rows must have the same dimensionality");
            }
        }

        double[] meanSurrogate = new double[dim];
        double meanTarget = 0.0;
        for (int i = 0; i < surrogateByDimension.length; i++) {
            meanTarget += target[i];
            for (int axis = 0; axis < dim; axis++) {
                meanSurrogate[axis] += surrogateByDimension[i][axis];
            }
        }
        meanTarget /= target.length;
        for (int axis = 0; axis < dim; axis++) {
            meanSurrogate[axis] /= surrogateByDimension.length;
        }

        double[][] covariance = new double[dim][dim];
        double[] targetCovariance = new double[dim];
        for (int i = 0; i < surrogateByDimension.length; i++) {
            for (int axis = 0; axis < dim; axis++) {
                double centeredAxis = surrogateByDimension[i][axis] - meanSurrogate[axis];
                targetCovariance[axis] += centeredAxis * (target[i] - meanTarget);
                for (int otherAxis = 0; otherAxis < dim; otherAxis++) {
                    covariance[axis][otherAxis] += centeredAxis * (surrogateByDimension[i][otherAxis] - meanSurrogate[otherAxis]);
                }
            }
        }

        double[] slopes = solveLinearSystem(covariance, targetCovariance);
        if (slopes == null) {
            slopes = new double[dim];
            Arrays.fill(slopes, 1.0);
        }

        double intercept = meanTarget;
        for (int axis = 0; axis < dim; axis++) {
            intercept -= slopes[axis] * meanSurrogate[axis];
        }

        double[] parameters = new double[dim + 1];
        parameters[0] = intercept;
        System.arraycopy(slopes, 0, parameters, 1, dim);
        return parameters;
    }

    private static double[] solveLinearSystem(double[][] matrix, double[] rhs) {
        int size = rhs.length;
        double[][] augmented = new double[size][size + 1];
        for (int row = 0; row < size; row++) {
            System.arraycopy(matrix[row], 0, augmented[row], 0, size);
            augmented[row][size] = rhs[row];
        }

        for (int pivot = 0; pivot < size; pivot++) {
            int pivotRow = pivot;
            for (int row = pivot + 1; row < size; row++) {
                if (Math.abs(augmented[row][pivot]) > Math.abs(augmented[pivotRow][pivot])) {
                    pivotRow = row;
                }
            }
            if (Math.abs(augmented[pivotRow][pivot]) <= VARIANCE_EPS) {
                return null;
            }
            if (pivotRow != pivot) {
                double[] tmp = augmented[pivot];
                augmented[pivot] = augmented[pivotRow];
                augmented[pivotRow] = tmp;
            }

            double pivotValue = augmented[pivot][pivot];
            for (int col = pivot; col <= size; col++) {
                augmented[pivot][col] /= pivotValue;
            }
            for (int row = 0; row < size; row++) {
                if (row == pivot) {
                    continue;
                }
                double factor = augmented[row][pivot];
                if (factor == 0.0) {
                    continue;
                }
                for (int col = pivot; col <= size; col++) {
                    augmented[row][col] -= factor * augmented[pivot][col];
                }
            }
        }

        double[] solution = new double[size];
        for (int row = 0; row < size; row++) {
            solution[row] = augmented[row][size];
        }
        return solution;
    }

    public static double[] computeSurrogateLogLikelihoodContributions(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] embeddings,
            int dim,
            double brownianRate) {
        int taxonCount = sample.getTaxonCount();
        if (embeddings.length != taxonCount * dim) {
            throw new IllegalArgumentException("embeddings length mismatch for sample: expected "
                    + (taxonCount * dim) + " but got " + embeddings.length);
        }

        double[] quadraticForms = new double[dim];
        double[][] precision = sample.precisionMatrix;
        for (int i = 0; i < taxonCount; i++) {
            for (int j = 0; j < taxonCount; j++) {
                double precisionEntry = precision[i][j];
                if (precisionEntry == 0.0) {
                    continue;
                }
                int rowOffset = i * dim;
                int columnOffset = j * dim;
                for (int axis = 0; axis < dim; axis++) {
                    quadraticForms[axis] += embeddings[rowOffset + axis] * precisionEntry * embeddings[columnOffset + axis];
                }
            }
        }

        double[] contributions = new double[dim];
        double baseContribution = -0.5 * taxonCount * LOG_TWO_PI
                - 0.5 * sample.logDeterminant
                - 0.5 * taxonCount * Math.log(brownianRate);
        for (int axis = 0; axis < dim; axis++) {
            contributions[axis] = baseContribution - 0.5 * quadraticForms[axis] / brownianRate;
        }
        return contributions;
    }

    public static double computeSurrogateLogLikelihood(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] embeddings,
            int dim,
            double brownianRate) {
        double[] contributions = computeSurrogateLogLikelihoodContributions(sample, embeddings, dim, brownianRate);
        double total = 0.0;
        for (double contribution : contributions) {
            total += contribution;
        }
        return total;
    }

    private double evaluate(double[] parameters, double[] gradient) {
        validateParameters(parameters);
        totalEvaluationCount++;
        runEvaluationCount++;
        if (runStartNanos < 0L) {
            runStartNanos = System.nanoTime();
        }

        if (gradient != null) {
            if (gradient.length != parameterCount) {
                throw new IllegalArgumentException("gradient length mismatch: expected "
                        + parameterCount + " but got " + gradient.length);
            }
            Arrays.fill(gradient, 0.0);
        }

        double intercept = parameters[interceptParameterIndex];
        double slope = parameters[slopeParameterIndex];
        double fitLoss = 0.0;
        double gradientIntercept = 0.0;
        double gradientSlope = 0.0;
        double sampleWeight = 1.0 / activeBatchCount;

        for (int batchIndex = 0; batchIndex < activeBatchCount; batchIndex++) {
            LearnedWeightsTreeTrainer.TrainingTreeSample sample = trainingSamples.get(resolveSampleIndex(batchIndex));
            double[] precisionTimesEmbeddings = multiplyPrecisionByEmbeddings(sample.precisionMatrix, parameters);
            double surrogate = computeSurrogateLogLikelihood(sample, parameters, precisionTimesEmbeddings);
            double prediction = intercept + slope * surrogate;

            double residual = prediction - sample.targetLogLikelihood;
            fitLoss += sampleWeight * residual * residual;

            if (gradient != null) {
                for (int axis = 0; axis < dim; axis++) {
                    double embeddingFactor = -2.0 * sampleWeight * residual * slope / brownianRate;
                    for (int taxon = 0; taxon < taxonCount; taxon++) {
                        int index = taxon * dim + axis;
                        gradient[index] += embeddingFactor * precisionTimesEmbeddings[index];
                    }
                }
                gradientSlope += 2.0 * sampleWeight * residual * surrogate;
                gradientIntercept += 2.0 * sampleWeight * residual;
            }
        }

        double slopePenalty = 0.0;
        if (slopeRegularizationWeight > 0.0) {
            double centeredSlope = slope - 1.0;
            slopePenalty += slopeRegularizationWeight * centeredSlope * centeredSlope;
            if (gradient != null) {
                gradientSlope += 2.0 * slopeRegularizationWeight * centeredSlope;
            }
        }

        double pcaPenalty = 0.0;
        if (pcaRegularizationWeight > 0.0) {
            double scale = pcaRegularizationWeight / embeddingParameterCount;
            for (int index = 0; index < embeddingParameterCount; index++) {
                double delta = parameters[index] - referenceEmbeddings[index];
                pcaPenalty += scale * delta * delta;
                if (gradient != null) {
                    gradient[index] += 2.0 * scale * delta;
                }
            }
        }

        double loss = fitLoss + slopePenalty + pcaPenalty;

        lastSlope = slope;
        lastIntercept = intercept;
        lastFitLoss = fitLoss;
        lastSlopePenalty = slopePenalty;
        lastPcaPenalty = pcaPenalty;
        lastLoss = loss;
        if (gradient != null) {
            gradient[interceptParameterIndex] = gradientIntercept;
            gradient[slopeParameterIndex] = gradientSlope;
        }

        if (loss < bestLoss) {
            bestLoss = loss;
            bestFitLoss = fitLoss;
            bestSlopePenalty = slopePenalty;
            bestPcaPenalty = pcaPenalty;
            bestParameters = Arrays.copyOf(parameters, parameterCount);
            bestSlope = slope;
            bestIntercept = intercept;
        }
        maybeLogProgress();
        return loss;
    }

    private double computeSurrogateLogLikelihood(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] parameters,
            double[] precisionTimesEmbeddings) {
        double quadraticForm = 0.0;
        for (int index = 0; index < embeddingParameterCount; index++) {
            quadraticForm += parameters[index] * precisionTimesEmbeddings[index];
        }
        return surrogateConstantTerm - 0.5 * dim * sample.logDeterminant - 0.5 * quadraticForm / brownianRate;
    }

    private void resetMiniBatchSchedule() {
        if (!useMiniBatches) {
            activeBatchStart = 0;
            activeBatchCount = trainingSamples.size();
            batchOrder = null;
            batchCursor = 0;
            miniBatchRandom = null;
            return;
        }
        batchOrder = new int[trainingSamples.size()];
        for (int index = 0; index < batchOrder.length; index++) {
            batchOrder[index] = index;
        }
        batchCursor = 0;
        activeBatchStart = 0;
        activeBatchCount = Math.min(miniBatchSize, batchOrder.length);
        miniBatchRandom = new Random(miniBatchSeed);
        shuffleBatchOrder();
    }

    private void shuffleBatchOrder() {
        for (int index = batchOrder.length - 1; index > 0; index--) {
            int swapIndex = miniBatchRandom.nextInt(index + 1);
            int tmp = batchOrder[index];
            batchOrder[index] = batchOrder[swapIndex];
            batchOrder[swapIndex] = tmp;
        }
    }

    private int resolveSampleIndex(int batchIndex) {
        if (!useMiniBatches) {
            return batchIndex;
        }
        return batchOrder[activeBatchStart + batchIndex];
    }

    private double[] multiplyPrecisionByEmbeddings(double[][] precisionMatrix, double[] parameters) {
        double[] output = new double[embeddingParameterCount];
        for (int row = 0; row < taxonCount; row++) {
            int rowOffset = row * dim;
            for (int column = 0; column < taxonCount; column++) {
                double precisionEntry = precisionMatrix[row][column];
                if (precisionEntry == 0.0) {
                    continue;
                }
                int columnOffset = column * dim;
                for (int axis = 0; axis < dim; axis++) {
                    output[rowOffset + axis] += precisionEntry * parameters[columnOffset + axis];
                }
            }
        }
        return output;
    }

    private void validateParameters(double[] parameters) {
        if (parameters == null || parameters.length != parameterCount) {
            throw new IllegalArgumentException("parameter length mismatch: expected "
                    + parameterCount + " but got " + (parameters == null ? 0 : parameters.length));
        }
    }

    private double l2Norm(double[] values, int startInclusive, int endExclusive) {
        double sumSquares = 0.0;
        for (int index = startInclusive; index < endExclusive; index++) {
            double value = values[index];
            sumSquares += value * value;
        }
        return Math.sqrt(sumSquares);
    }

    private static boolean hasNonZeroVariance(double[] values) {
        double mean = 0.0;
        for (double value : values) {
            mean += value;
        }
        mean /= values.length;

        double variance = 0.0;
        for (double value : values) {
            double centered = value - mean;
            variance += centered * centered;
        }
        return variance > VARIANCE_EPS;
    }

    private void maybeLogProgress() {
        long now = System.nanoTime();
        boolean shouldLog = runEvaluationCount == 1;
        if (!shouldLog && lastProgressLogNanos >= 0L) {
            shouldLog = now - lastProgressLogNanos >= 10_000_000_000L;
        }
        if (!shouldLog) {
            return;
        }

        double elapsedSec = (now - runStartNanos) / 1_000_000_000.0;
        Log.info.println(String.format(
            "%s: objective iteration %d/%s, elapsed=%.1fs, trialLoss=%.6f, bestSeenLoss=%.6f, fitLoss=%.6f, slopePenalty=%.6f, pcaPenalty=%.6f, slope=%.6f, intercept=%.6f",
            progressLabel,
            currentIteration,
            configuredMaxIterations > 0 ? configuredMaxIterations : "?",
            elapsedSec,
            lastLoss,
            bestLoss,
            lastFitLoss,
            lastSlopePenalty,
            lastPcaPenalty,
            lastSlope,
            lastIntercept));
        lastProgressLogNanos = now;
    }
}
