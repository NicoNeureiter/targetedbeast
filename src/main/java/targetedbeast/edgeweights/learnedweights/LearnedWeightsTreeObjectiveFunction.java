package targetedbeast.edgeweights.learnedweights;

import java.util.Arrays;
import java.util.Random;
import java.util.List;

import beast.base.core.Log;

public class LearnedWeightsTreeObjectiveFunction implements LearnedWeightsOptimizer.DifferentiableFunction {

    private static final double VARIANCE_EPS = 1e-12;
    private static final double LOG_TWO_PI = Math.log(2.0 * Math.PI);
    private static final double ROOT_VARIANCE_FLOOR = 1e-12;

    private final List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples;
    private final int taxonCount;
    private final int dim;
    private final double[] brownianRates;
    private boolean optimizeBrownianRates = true;
    private final double[] initialRootPriorVariances;
    private final boolean optimizeRootPriorVariances;
    private final double[] referenceEmbeddings;
    private final double pcaRegularizationWeight;
    private final double slopeRegularizationWeight;
    private final boolean useLogSlopeParameterization;
    private final boolean separateBaseAndQuadraticCalibration;
    private final boolean scaleQuadraticPartOnly;
    private final boolean trainQuadraticOnly;
    private final int embeddingParameterCount;
    private final int interceptParameterIndex;
    private final int rateParameterStartIndex;
    private final int rootVarianceParameterStartIndex;
    private final int baseScaleParameterStartIndex;
    private final int slopeParameterStartIndex;
    private final int parameterCount;
    private final int miniBatchSize;
    private final long miniBatchSeed;
    private final boolean useMiniBatches;
    private int[] batchOrder;
    private int batchCursor = 0;
    private int activeBatchStart = 0;
    private int activeBatchCount;
    private Random miniBatchRandom;

    private double validationFraction = 0.0;
    private int earlyStoppingPatience = 0;
    private int validationInterval = 1;
    private int[] trainIndices;
    private int[] validationIndices = new int[0];
    private double bestValidationLoss = Double.POSITIVE_INFINITY;
    private double lastValidationLoss = Double.NaN;
    private int consecutiveValidationNonImprovements = 0;
    private int validationCheckCounter = 0;

    private double[] bestParameters;
    private double bestLoss = Double.POSITIVE_INFINITY;
    private double bestFitLoss = Double.POSITIVE_INFINITY;
    private double bestSlopePenalty = 0.0;
    private double bestPcaPenalty = 0.0;
    private double[] bestBaseScales;
    private double[] bestSlopes;
    private double[] bestRootPriorVariances;
    private double bestIntercept = 0.0;
    private double[] lastBaseScales;
    private double[] lastSlopes;
    private double[] lastRootPriorVariances;
    private double lastIntercept = 0.0;
    private double lastFitLoss = Double.NaN;
    private double lastSlopePenalty = Double.NaN;
    private double lastPcaPenalty = Double.NaN;
    private int totalEvaluationCount = 0;
    private int runEvaluationCount = 0;
    private int currentIteration = 0;
    private int configuredMaxIterations = -1;
    private String progressLabel = "LearnedWeights";
    private boolean progressLoggingEnabled = true;
    private long runStartNanos = -1L;
    private long lastProgressLogNanos = -1L;
    private double lastLoss = Double.NaN;

    public static final class SurrogateDecomposition {
        public final double total;
        public final double logDetPart;
        public final double quadraticPart;

        SurrogateDecomposition(double total, double logDetPart, double quadraticPart) {
            this.total = total;
            this.logDetPart = logDetPart;
            this.quadraticPart = quadraticPart;
        }
    }

    private static final class RootVarianceAdjustedSample {
        final double[] precisionTimesEmbeddings;
        final double[] quadraticForms;
        final double[] logDeterminants;
        final double[] logDetVarianceDerivatives;
        final double[] quadraticVarianceDerivativeFactors;
        final int logNormalizerDimension;

        RootVarianceAdjustedSample(
                double[] precisionTimesEmbeddings,
                double[] quadraticForms,
                double[] logDeterminants,
                double[] logDetVarianceDerivatives,
                double[] quadraticVarianceDerivativeFactors,
                int logNormalizerDimension) {
            this.precisionTimesEmbeddings = precisionTimesEmbeddings;
            this.quadraticForms = quadraticForms;
            this.logDeterminants = logDeterminants;
            this.logDetVarianceDerivatives = logDetVarianceDerivatives;
            this.quadraticVarianceDerivativeFactors = quadraticVarianceDerivativeFactors;
            this.logNormalizerDimension = logNormalizerDimension;
        }
    }
    public LearnedWeightsTreeObjectiveFunction(
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples,
            int taxonCount,
            int dim,
            double brownianRate) {
        this(trainingSamples, taxonCount, dim, uniformBrownianRates(dim, brownianRate), null, 0.0, 0.0);
    }

    public LearnedWeightsTreeObjectiveFunction(
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples,
            int taxonCount,
            int dim,
            double[] brownianRates) {
        this(trainingSamples, taxonCount, dim, brownianRates, null, 0.0, 0.0);
    }

    public LearnedWeightsTreeObjectiveFunction(
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples,
            int taxonCount,
            int dim,
            double brownianRate,
            double[] referenceEmbeddings,
            double pcaRegularizationWeight,
            double slopeRegularizationWeight) {
        this(trainingSamples, taxonCount, dim, uniformBrownianRates(dim, brownianRate), referenceEmbeddings, pcaRegularizationWeight, slopeRegularizationWeight, 0, 1L, false, false);
    }

    public LearnedWeightsTreeObjectiveFunction(
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples,
            int taxonCount,
            int dim,
            double[] brownianRates,
            double[] referenceEmbeddings,
            double pcaRegularizationWeight,
            double slopeRegularizationWeight) {
        this(trainingSamples, taxonCount, dim, brownianRates, referenceEmbeddings, pcaRegularizationWeight, slopeRegularizationWeight, 0, 1L, false, false);
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
        this(trainingSamples, taxonCount, dim, uniformBrownianRates(dim, brownianRate), referenceEmbeddings, pcaRegularizationWeight, slopeRegularizationWeight, miniBatchSize, miniBatchSeed, false, false);
    }

    public LearnedWeightsTreeObjectiveFunction(
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples,
            int taxonCount,
            int dim,
            double[] brownianRates,
            double[] referenceEmbeddings,
            double pcaRegularizationWeight,
            double slopeRegularizationWeight,
            int miniBatchSize,
            long miniBatchSeed) {
        this(trainingSamples, taxonCount, dim, brownianRates, referenceEmbeddings, pcaRegularizationWeight, slopeRegularizationWeight, miniBatchSize, miniBatchSeed, false, false);
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
            long miniBatchSeed,
            boolean useLogSlopeParameterization) {
        this(trainingSamples, taxonCount, dim, uniformBrownianRates(dim, brownianRate), referenceEmbeddings, pcaRegularizationWeight, slopeRegularizationWeight, miniBatchSize, miniBatchSeed, useLogSlopeParameterization, false);
    }

    public LearnedWeightsTreeObjectiveFunction(
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples,
            int taxonCount,
            int dim,
            double[] brownianRates,
            double[] referenceEmbeddings,
            double pcaRegularizationWeight,
            double slopeRegularizationWeight,
            int miniBatchSize,
            long miniBatchSeed,
            boolean useLogSlopeParameterization) {
        this(trainingSamples, taxonCount, dim, brownianRates, referenceEmbeddings, pcaRegularizationWeight, slopeRegularizationWeight, miniBatchSize, miniBatchSeed, useLogSlopeParameterization, false);
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
            long miniBatchSeed,
            boolean useLogSlopeParameterization,
            boolean scaleQuadraticPartOnly) {
            this(trainingSamples, taxonCount, dim, uniformBrownianRates(dim, brownianRate), referenceEmbeddings, pcaRegularizationWeight,
                slopeRegularizationWeight, miniBatchSize, miniBatchSeed, useLogSlopeParameterization,
                scaleQuadraticPartOnly, false);
    }

            public LearnedWeightsTreeObjectiveFunction(
                List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples,
                int taxonCount,
                int dim,
                double[] brownianRates,
                double[] referenceEmbeddings,
                double pcaRegularizationWeight,
                double slopeRegularizationWeight,
                int miniBatchSize,
                long miniBatchSeed,
                boolean useLogSlopeParameterization,
                boolean scaleQuadraticPartOnly) {
            this(trainingSamples, taxonCount, dim, brownianRates, referenceEmbeddings, pcaRegularizationWeight,
                slopeRegularizationWeight, miniBatchSize, miniBatchSeed, useLogSlopeParameterization,
                scaleQuadraticPartOnly, false);
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
            long miniBatchSeed,
            boolean useLogSlopeParameterization,
            boolean scaleQuadraticPartOnly,
            boolean trainQuadraticOnly) {
        this(trainingSamples, taxonCount, dim, uniformBrownianRates(dim, brownianRate), referenceEmbeddings,
                pcaRegularizationWeight, slopeRegularizationWeight, miniBatchSize, miniBatchSeed,
                useLogSlopeParameterization, false, scaleQuadraticPartOnly, trainQuadraticOnly);
    }

    public LearnedWeightsTreeObjectiveFunction(
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples,
            int taxonCount,
            int dim,
            double[] brownianRates,
            double[] referenceEmbeddings,
            double pcaRegularizationWeight,
            double slopeRegularizationWeight,
            int miniBatchSize,
            long miniBatchSeed,
            boolean useLogSlopeParameterization,
            boolean separateBaseAndQuadraticCalibration,
            boolean scaleQuadraticPartOnly,
            boolean trainQuadraticOnly) {
        this(trainingSamples, taxonCount, dim, brownianRates, null, referenceEmbeddings, pcaRegularizationWeight,
                slopeRegularizationWeight, miniBatchSize, miniBatchSeed, useLogSlopeParameterization,
                separateBaseAndQuadraticCalibration, scaleQuadraticPartOnly, trainQuadraticOnly);
    }

    public LearnedWeightsTreeObjectiveFunction(
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples,
            int taxonCount,
            int dim,
            double[] brownianRates,
            double[] initialRootPriorVariances,
            double[] referenceEmbeddings,
            double pcaRegularizationWeight,
            double slopeRegularizationWeight,
            int miniBatchSize,
            long miniBatchSeed,
            boolean useLogSlopeParameterization,
            boolean separateBaseAndQuadraticCalibration,
            boolean scaleQuadraticPartOnly,
            boolean trainQuadraticOnly) {
        if (trainingSamples == null || trainingSamples.isEmpty()) {
            throw new IllegalArgumentException("trainingSamples must not be null or empty");
        }
        if (scaleQuadraticPartOnly && trainQuadraticOnly) {
            throw new IllegalArgumentException("scaleQuadraticPartOnly and trainQuadraticOnly are mutually exclusive");
        }
        if (separateBaseAndQuadraticCalibration && (scaleQuadraticPartOnly || trainQuadraticOnly)) {
            throw new IllegalArgumentException(
                    "separateBaseAndQuadraticCalibration is mutually exclusive with scaleQuadraticPartOnly and trainQuadraticOnly");
        }

        this.trainingSamples = trainingSamples;
        this.taxonCount = taxonCount;
        this.dim = dim;
        this.brownianRates = copyAndValidateBrownianRates(dim, brownianRates);
        this.initialRootPriorVariances = resolveOptimizableRootPriorVariances(dim, initialRootPriorVariances);
        this.optimizeRootPriorVariances = this.initialRootPriorVariances != null;
        this.embeddingParameterCount = taxonCount * dim;
        this.interceptParameterIndex = embeddingParameterCount;
        this.rateParameterStartIndex = interceptParameterIndex + 1;
        this.rootVarianceParameterStartIndex = rateParameterStartIndex + dim;
        this.baseScaleParameterStartIndex = rootVarianceParameterStartIndex + (optimizeRootPriorVariances ? dim : 0);
        this.slopeParameterStartIndex = baseScaleParameterStartIndex + (separateBaseAndQuadraticCalibration ? dim : 0);
        this.parameterCount = embeddingParameterCount + 1 + dim + (optimizeRootPriorVariances ? dim : 0)
                + (separateBaseAndQuadraticCalibration ? 2 * dim : dim);
        if (referenceEmbeddings != null && referenceEmbeddings.length != embeddingParameterCount) {
            throw new IllegalArgumentException("referenceEmbeddings length mismatch: expected "
                    + embeddingParameterCount + " but got " + referenceEmbeddings.length);
        }
        if (optimizeRootPriorVariances) {
            for (LearnedWeightsTreeTrainer.TrainingTreeSample sample : trainingSamples) {
                if (!sample.hasRootFreeFactorization()) {
                    throw new IllegalArgumentException(
                            "trainingSamples must include root-free covariance data when optimizing rootPriorVariances; regenerate cached training samples with the current code");
                }
            }
        }
        if (pcaRegularizationWeight > 0.0 && referenceEmbeddings == null) {
            throw new IllegalArgumentException("referenceEmbeddings must be provided when pcaRegularizationWeight > 0");
        }
        this.referenceEmbeddings = referenceEmbeddings == null ? null : Arrays.copyOf(referenceEmbeddings, referenceEmbeddings.length);
        this.pcaRegularizationWeight = pcaRegularizationWeight;
        this.slopeRegularizationWeight = slopeRegularizationWeight;
        this.useLogSlopeParameterization = useLogSlopeParameterization;
        this.separateBaseAndQuadraticCalibration = separateBaseAndQuadraticCalibration;
        this.scaleQuadraticPartOnly = scaleQuadraticPartOnly;
        this.trainQuadraticOnly = trainQuadraticOnly;
        this.useMiniBatches = miniBatchSize > 0 && miniBatchSize < trainingSamples.size();
        this.miniBatchSize = useMiniBatches ? miniBatchSize : trainingSamples.size();
        this.miniBatchSeed = miniBatchSeed;
        this.trainIndices = identityIndices(trainingSamples.size());
        this.activeBatchCount = trainIndices.length;
    }

    public LearnedWeightsTreeObjectiveFunction(
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples,
            int taxonCount,
            int dim,
            double[] brownianRates,
            double[] referenceEmbeddings,
            double pcaRegularizationWeight,
            double slopeRegularizationWeight,
            int miniBatchSize,
            long miniBatchSeed,
            boolean useLogSlopeParameterization,
            boolean scaleQuadraticPartOnly,
            boolean trainQuadraticOnly) {
        this(trainingSamples, taxonCount, dim, brownianRates, referenceEmbeddings, pcaRegularizationWeight,
                slopeRegularizationWeight, miniBatchSize, miniBatchSeed, useLogSlopeParameterization, false,
                scaleQuadraticPartOnly, trainQuadraticOnly);
    }

    /**
     * Configures a held-out validation split. Must be called before {@link #beginOptimizationRun}.
     * The split is deterministic given {@code miniBatchSeed} (passed at construction) and is
     * disjoint from / exhaustive over the training samples. validationFraction &lt;= 0 disables
     * validation and leaves the mini-batch/full-batch training pool unchanged.
     */
    public void configureValidation(double validationFraction, int earlyStoppingPatience, int validationInterval) {
        if (!(validationFraction >= 0.0) || !(validationFraction < 1.0)) {
            throw new IllegalArgumentException("validationFraction must be in [0, 1), but was " + validationFraction);
        }
        if (earlyStoppingPatience < 0) {
            throw new IllegalArgumentException("earlyStoppingPatience must be >= 0, but was " + earlyStoppingPatience);
        }
        if (validationInterval < 1) {
            throw new IllegalArgumentException("validationInterval must be >= 1, but was " + validationInterval);
        }
        this.validationFraction = validationFraction;
        this.earlyStoppingPatience = earlyStoppingPatience;
        this.validationInterval = validationInterval;
        computeValidationSplit();
    }

    private void computeValidationSplit() {
        int n = trainingSamples.size();
        int validationCount = validationFraction > 0.0 ? (int) Math.floor(validationFraction * n) : 0;
        if (validationCount <= 0 || validationCount >= n) {
            trainIndices = identityIndices(n);
            validationIndices = new int[0];
            return;
        }
        int[] shuffled = identityIndices(n);
        Random splitRandom = new Random(miniBatchSeed);
        for (int i = shuffled.length - 1; i > 0; i--) {
            int j = splitRandom.nextInt(i + 1);
            int tmp = shuffled[i];
            shuffled[i] = shuffled[j];
            shuffled[j] = tmp;
        }
        validationIndices = Arrays.copyOfRange(shuffled, 0, validationCount);
        trainIndices = Arrays.copyOfRange(shuffled, validationCount, n);
        Arrays.sort(validationIndices);
        Arrays.sort(trainIndices);
    }

    private static int[] identityIndices(int n) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        return indices;
    }

    public boolean isValidationEnabled() {
        return validationIndices.length > 0;
    }

    public int[] getTrainIndices() {
        return Arrays.copyOf(trainIndices, trainIndices.length);
    }

    public int[] getValidationIndices() {
        return Arrays.copyOf(validationIndices, validationIndices.length);
    }

    public double getLastValidationLoss() {
        return lastValidationLoss;
    }

    public double getBestValidationLoss() {
        return bestValidationLoss;
    }

    public int getConsecutiveValidationNonImprovements() {
        return consecutiveValidationNonImprovements;
    }

    @Override
    public boolean shouldStopEarly() {
        return isValidationEnabled() && earlyStoppingPatience > 0
                && consecutiveValidationNonImprovements >= earlyStoppingPatience;
    }

    @Override
    public double[] getEarlyStoppingBestParameters() {
        return isValidationEnabled() ? getBestParameters() : null;
    }

    @Override
    public double getEarlyStoppingBestLoss() {
        return isValidationEnabled() ? bestLoss : Double.NaN;
    }

    @Override
    public String getEarlyStoppingStopReason() {
        return String.format(
                "early stopping: validation loss did not improve for %d evaluations (patience=%d)",
                consecutiveValidationNonImprovements, earlyStoppingPatience);
    }

    /**
     * Mean loss over the held-out validation set at the given parameters, using the same
     * residual/prediction path as training. Excludes regularisation penalties: those are a
     * property of the parameters, not the data, so they are not part of a held-out loss estimate.
     */
    public double evaluateValidationLoss(double[] parameters) {
        if (!isValidationEnabled()) {
            return Double.NaN;
        }
        validateParameters(parameters);
        double intercept = parameters[interceptParameterIndex];
        double[] currentBrownianRates = extractBrownianRates(parameters);
        double[] currentRootPriorVariances = extractRootPriorVariances(parameters);
        double[] baseScales = extractBaseScales(parameters);
        double[] slopes = extractSlopes(parameters);
        double[] embeddings = extractEmbeddings(parameters);
        double sampleWeight = 1.0 / validationIndices.length;
        double loss = 0.0;
        for (int idx : validationIndices) {
            double residual = computeResidual(
                    trainingSamples.get(idx), embeddings, intercept, currentBrownianRates,
                    currentRootPriorVariances, baseScales, slopes);
            loss += sampleWeight * residual * residual;
        }
        return loss;
    }

    private double computeResidual(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] embeddings,
            double intercept,
            double[] currentBrownianRates,
            double[] currentRootPriorVariances,
            double[] baseScales,
            double[] slopes) {
        double[] baseContributions = computeBaseLogLikelihoodContributions(sample, dim, currentBrownianRates, currentRootPriorVariances);
        double[] quadraticContributions = computeQuadraticLogLikelihoodContributions(sample, embeddings, dim, currentBrownianRates, currentRootPriorVariances);
        if (separateBaseAndQuadraticCalibration) {
            double prediction = intercept + dot(baseScales, baseContributions) + dot(slopes, quadraticContributions);
            return prediction - sample.targetLogLikelihood;
        }
        double[] fullContributions = add(baseContributions, quadraticContributions);
        double[] regressionContributions = usesQuadraticRegressionContributions() ? quadraticContributions : fullContributions;
        double prediction = intercept + dot(slopes, regressionContributions);
        double regressionTarget = getRegressionTarget(sample, currentBrownianRates, currentRootPriorVariances);
        return prediction - regressionTarget;
    }

    private void maybeCheckValidation(double[] parameters) {
        if (!isValidationEnabled()) {
            return;
        }
        validationCheckCounter++;
        if (validationCheckCounter % validationInterval != 0) {
            return;
        }
        double validationLoss = evaluateValidationLoss(parameters);
        lastValidationLoss = validationLoss;
        if (validationLoss < bestValidationLoss) {
            bestValidationLoss = validationLoss;
            consecutiveValidationNonImprovements = 0;
            bestLoss = validationLoss;
            bestFitLoss = lastFitLoss;
            bestSlopePenalty = lastSlopePenalty;
            bestPcaPenalty = lastPcaPenalty;
            bestParameters = Arrays.copyOf(parameters, parameterCount);
            bestBaseScales = Arrays.copyOf(lastBaseScales, lastBaseScales.length);
            bestSlopes = Arrays.copyOf(lastSlopes, lastSlopes.length);
            bestRootPriorVariances = lastRootPriorVariances == null ? null : Arrays.copyOf(lastRootPriorVariances, lastRootPriorVariances.length);
            bestIntercept = lastIntercept;
        } else {
            consecutiveValidationNonImprovements++;
        }
    }

    @Override
    public void beginIteration(int iteration) {
        currentIteration = iteration;
        if (!useMiniBatches) {
            activeBatchStart = 0;
            activeBatchCount = trainIndices.length;
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
        double loss = evaluate(parameters, gradient);
        maybeCheckValidation(parameters);
        return loss;
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
        bestBaseScales = createUnitScaleVector();
        bestSlopes = createUnitSlopeVector();
        bestRootPriorVariances = initialRootPriorVariances == null ? null : Arrays.copyOf(initialRootPriorVariances, initialRootPriorVariances.length);
        bestIntercept = 0.0;
        currentIteration = 0;
        runEvaluationCount = 0;
        runStartNanos = -1L;
        lastProgressLogNanos = -1L;
        lastLoss = Double.NaN;
        lastFitLoss = Double.NaN;
        lastSlopePenalty = Double.NaN;
        lastPcaPenalty = Double.NaN;
        lastBaseScales = createUnitScaleVector();
        lastSlopes = createUnitSlopeVector();
        lastRootPriorVariances = initialRootPriorVariances == null ? null : Arrays.copyOf(initialRootPriorVariances, initialRootPriorVariances.length);
        bestValidationLoss = Double.POSITIVE_INFINITY;
        lastValidationLoss = Double.NaN;
        consecutiveValidationNonImprovements = 0;
        validationCheckCounter = 0;
        resetMiniBatchSchedule();
    }

    public void setProgressLoggingEnabled(boolean enabled) {
        progressLoggingEnabled = enabled;
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

        double[] target = new double[trainingSamples.size()];
        for (int i = 0; i < trainingSamples.size(); i++) {
            target[i] = getRegressionTarget(trainingSamples.get(i), brownianRates, initialRootPriorVariances);
        }

        double intercept;
        double[] baseScales = createUnitScaleVector();
        double[] slopes;
        if (separateBaseAndQuadraticCalibration) {
            double[][] baseByDimension = evaluateBaseContributions(initialEmbeddings, initialRootPriorVariances);
            double[][] quadraticByDimension = evaluateQuadraticContributions(initialEmbeddings, initialRootPriorVariances);
            double[] affine = fitSplitCalibrationPerDimension(baseByDimension, quadraticByDimension, target);
            intercept = affine[0];
            baseScales = Arrays.copyOfRange(affine, 1, 1 + dim);
            slopes = Arrays.copyOfRange(affine, 1 + dim, affine.length);
        } else {
            double[][] surrogateByDimension = evaluateRegressionContributions(initialEmbeddings, initialRootPriorVariances);
            double[] affine = fitAffineCalibrationPerDimension(surrogateByDimension, target);
            intercept = affine[0];
            slopes = Arrays.copyOfRange(affine, 1, affine.length);
        }

        double[] initialParameters = new double[parameterCount];
        System.arraycopy(initialEmbeddings, 0, initialParameters, 0, embeddingParameterCount);
        initialParameters[interceptParameterIndex] = intercept;
        for (int i = 0; i < dim; i++) {
            initialParameters[rateParameterStartIndex + i] = Math.log(brownianRates[i]);
        }
        if (optimizeRootPriorVariances) {
            for (int i = 0; i < dim; i++) {
                initialParameters[rootVarianceParameterStartIndex + i] = Math.log(Math.max(ROOT_VARIANCE_FLOOR, initialRootPriorVariances[i]));
            }
        }
        if (separateBaseAndQuadraticCalibration) {
            if (useLogSlopeParameterization) {
                for (int i = 0; i < dim; i++) {
                    initialParameters[baseScaleParameterStartIndex + i] = Math.log(Math.max(1e-10, baseScales[i]));
                }
            } else {
                System.arraycopy(baseScales, 0, initialParameters, baseScaleParameterStartIndex, dim);
            }
        }
        if (useLogSlopeParameterization) {
            for (int i = 0; i < dim; i++) {
                initialParameters[slopeParameterStartIndex + i] = Math.log(Math.max(1e-10, slopes[i]));
            }
        } else {
            System.arraycopy(slopes, 0, initialParameters, slopeParameterStartIndex, dim);
        }
        return initialParameters;
    }

    public double[] createParameters(
            double[] embeddings,
            double[] initialBrownianRates,
            double[] initialRootPriorVariances,
            double[] baseScales,
            double[] slopes,
            double intercept) {
        if (embeddings == null || embeddings.length != embeddingParameterCount) {
            throw new IllegalArgumentException("embeddings length mismatch: expected "
                    + embeddingParameterCount + " but got " + (embeddings == null ? 0 : embeddings.length));
        }
        if (initialBrownianRates == null || initialBrownianRates.length != dim) {
            throw new IllegalArgumentException("brownianRates length mismatch: expected "
                    + dim + " but got " + (initialBrownianRates == null ? 0 : initialBrownianRates.length));
        }
        if (slopes == null || slopes.length != dim) {
            throw new IllegalArgumentException("slopes length mismatch: expected "
                    + dim + " but got " + (slopes == null ? 0 : slopes.length));
        }
        if (optimizeRootPriorVariances && (initialRootPriorVariances == null || initialRootPriorVariances.length != dim)) {
            throw new IllegalArgumentException("rootPriorVariances length mismatch: expected "
                    + dim + " but got " + (initialRootPriorVariances == null ? 0 : initialRootPriorVariances.length));
        }
        if (separateBaseAndQuadraticCalibration && (baseScales == null || baseScales.length != dim)) {
            throw new IllegalArgumentException("baseScales length mismatch: expected "
                    + dim + " but got " + (baseScales == null ? 0 : baseScales.length));
        }

        double[] parameters = new double[parameterCount];
        System.arraycopy(embeddings, 0, parameters, 0, embeddingParameterCount);
        parameters[interceptParameterIndex] = intercept;
        for (int i = 0; i < dim; i++) {
            parameters[rateParameterStartIndex + i] = Math.log(Math.max(ROOT_VARIANCE_FLOOR, initialBrownianRates[i]));
        }
        if (optimizeRootPriorVariances) {
            for (int i = 0; i < dim; i++) {
                parameters[rootVarianceParameterStartIndex + i] = Math.log(
                        Math.max(ROOT_VARIANCE_FLOOR, initialRootPriorVariances[i]));
            }
        }
        if (separateBaseAndQuadraticCalibration) {
            if (useLogSlopeParameterization) {
                for (int i = 0; i < dim; i++) {
                    parameters[baseScaleParameterStartIndex + i] = Math.log(Math.max(1e-10, baseScales[i]));
                }
            } else {
                System.arraycopy(baseScales, 0, parameters, baseScaleParameterStartIndex, dim);
            }
        }
        if (useLogSlopeParameterization) {
            for (int i = 0; i < dim; i++) {
                parameters[slopeParameterStartIndex + i] = Math.log(Math.max(1e-10, slopes[i]));
            }
        } else {
            System.arraycopy(slopes, 0, parameters, slopeParameterStartIndex, dim);
        }
        return parameters;
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
        return mean(bestSlopes);
    }

    public double[] getBestBaseScales() {
        return bestBaseScales == null ? createUnitScaleVector() : Arrays.copyOf(bestBaseScales, bestBaseScales.length);
    }

    public double[] getBestSlopes() {
        return bestSlopes == null ? createUnitSlopeVector() : Arrays.copyOf(bestSlopes, bestSlopes.length);
    }

    public double[] getBestRootPriorVariances() {
        return bestRootPriorVariances == null ? null : Arrays.copyOf(bestRootPriorVariances, bestRootPriorVariances.length);
    }

    public double getBestIntercept() {
        return bestIntercept;
    }

    public double getLastSlope() {
        return mean(lastSlopes);
    }

    public double[] getLastBaseScales() {
        return lastBaseScales == null ? createUnitScaleVector() : Arrays.copyOf(lastBaseScales, lastBaseScales.length);
    }

    public double[] getLastSlopes() {
        return lastSlopes == null ? createUnitSlopeVector() : Arrays.copyOf(lastSlopes, lastSlopes.length);
    }

    public double[] getLastRootPriorVariances() {
        return lastRootPriorVariances == null ? null : Arrays.copyOf(lastRootPriorVariances, lastRootPriorVariances.length);
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
        return mean(extractSlopes(parameters));
    }

    public double extractBaseScale(double[] parameters) {
        return mean(extractBaseScales(parameters));
    }

    public double extractBrownianRate(double[] parameters) {
        return mean(extractBrownianRates(parameters));
    }

    public double[] extractRootPriorVariances(double[] parameters) {
        if (!optimizeRootPriorVariances) {
            return initialRootPriorVariances == null ? null : Arrays.copyOf(initialRootPriorVariances, initialRootPriorVariances.length);
        }
        validateParameters(parameters);
        double[] raw = Arrays.copyOfRange(parameters, rootVarianceParameterStartIndex, rootVarianceParameterStartIndex + dim);
        for (int i = 0; i < raw.length; i++) {
            raw[i] = Math.exp(raw[i]);
        }
        return raw;
    }

    public double[] extractBrownianRates(double[] parameters) {
        if (!optimizeBrownianRates) {
            return Arrays.copyOf(brownianRates, brownianRates.length);
        }
        validateParameters(parameters);
        double[] raw = Arrays.copyOfRange(parameters, rateParameterStartIndex, rateParameterStartIndex + dim);
        for (int i = 0; i < raw.length; i++) {
            raw[i] = Math.exp(raw[i]);
        }
        return raw;
    }

    /** When false, brownian trait rates are frozen at their initial values and not optimized. */
    public void setOptimizeBrownianRates(boolean optimize) {
        this.optimizeBrownianRates = optimize;
    }

    public double[] extractSlopes(double[] parameters) {
        validateParameters(parameters);
        double[] raw = Arrays.copyOfRange(parameters, slopeParameterStartIndex, slopeParameterStartIndex + dim);
        if (useLogSlopeParameterization) {
            for (int i = 0; i < raw.length; i++) {
                raw[i] = Math.exp(raw[i]);
            }
        }
        return raw;
    }

    public double[] extractBaseScales(double[] parameters) {
        validateParameters(parameters);
        if (!separateBaseAndQuadraticCalibration) {
            if (scaleQuadraticPartOnly) {
                return createUnitScaleVector();
            }
            return extractSlopes(parameters);
        }
        double[] raw = Arrays.copyOfRange(parameters, baseScaleParameterStartIndex, baseScaleParameterStartIndex + dim);
        if (useLogSlopeParameterization) {
            for (int i = 0; i < raw.length; i++) {
                raw[i] = Math.exp(raw[i]);
            }
        }
        return raw;
    }

    public double[][] evaluateBaseContributions(double[] embeddings) {
        return evaluateBaseContributions(embeddings, initialRootPriorVariances);
    }

    public double[][] evaluateBaseContributions(double[] embeddings, double[] rootPriorVariances) {
        if (embeddings == null || embeddings.length != embeddingParameterCount) {
            throw new IllegalArgumentException("embeddings length mismatch: expected "
                    + embeddingParameterCount + " but got " + (embeddings == null ? 0 : embeddings.length));
        }
        double[][] base = new double[trainingSamples.size()][dim];
        for (int sampleIndex = 0; sampleIndex < trainingSamples.size(); sampleIndex++) {
            base[sampleIndex] = computeBaseLogLikelihoodContributions(trainingSamples.get(sampleIndex), dim, brownianRates, rootPriorVariances);
        }
        return base;
    }

    public double[] evaluateSurrogates(double[] embeddings) {
        return evaluateSurrogates(embeddings, initialRootPriorVariances);
    }

    public double[] evaluateSurrogates(double[] embeddings, double[] rootPriorVariances) {
        if (embeddings == null || embeddings.length != embeddingParameterCount) {
            throw new IllegalArgumentException("embeddings length mismatch: expected "
                    + embeddingParameterCount + " but got " + (embeddings == null ? 0 : embeddings.length));
        }
        double[] surrogate = new double[trainingSamples.size()];
        for (int sampleIndex = 0; sampleIndex < trainingSamples.size(); sampleIndex++) {
            surrogate[sampleIndex] = computeSurrogateLogLikelihood(trainingSamples.get(sampleIndex), embeddings, dim, brownianRates, rootPriorVariances);
        }
        return surrogate;
    }

    public double[][] evaluateQuadraticContributions(double[] embeddings) {
        return evaluateQuadraticContributions(embeddings, initialRootPriorVariances);
    }

    public double[][] evaluateQuadraticContributions(double[] embeddings, double[] rootPriorVariances) {
        if (embeddings == null || embeddings.length != embeddingParameterCount) {
            throw new IllegalArgumentException("embeddings length mismatch: expected "
                    + embeddingParameterCount + " but got " + (embeddings == null ? 0 : embeddings.length));
        }
        double[][] surrogate = new double[trainingSamples.size()][dim];
        for (int sampleIndex = 0; sampleIndex < trainingSamples.size(); sampleIndex++) {
            surrogate[sampleIndex] = computeQuadraticLogLikelihoodContributions(
                    trainingSamples.get(sampleIndex), embeddings, dim, brownianRates, rootPriorVariances);
        }
        return surrogate;
    }

    public static SurrogateDecomposition computeSurrogateLogLikelihoodDecomposition(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] embeddings,
            int dim,
            double brownianRate) {
        return computeSurrogateLogLikelihoodDecomposition(sample, embeddings, dim, uniformBrownianRates(dim, brownianRate));
    }

    public static SurrogateDecomposition computeSurrogateLogLikelihoodDecomposition(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] embeddings,
            int dim,
            double[] brownianRates) {
        return computeSurrogateLogLikelihoodDecomposition(sample, embeddings, dim, brownianRates, null);
    }

    public static SurrogateDecomposition computeSurrogateLogLikelihoodDecomposition(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] embeddings,
            int dim,
            double[] brownianRates,
            double[] rootPriorVariances) {
        double[] validatedBrownianRates = copyAndValidateBrownianRates(dim, brownianRates);
        int taxonCount = sample.getTaxonCount();
        if (embeddings.length != taxonCount * dim) {
            throw new IllegalArgumentException("embeddings length mismatch for sample: expected "
                    + (taxonCount * dim) + " but got " + embeddings.length);
        }
        RootVarianceAdjustedSample adjustedSample = adjustSampleForRootPriorVariances(
                sample, embeddings, dim, rootPriorVariances, false);
        double quadraticPart = 0.0;
        for (int axis = 0; axis < dim; axis++) {
            quadraticPart -= 0.5 * adjustedSample.quadraticForms[axis] / validatedBrownianRates[axis];
        }

        double logDetPart = -0.5 * dim * adjustedSample.logNormalizerDimension * LOG_TWO_PI;
        for (int axis = 0; axis < dim; axis++) {
            logDetPart -= 0.5 * (adjustedSample.logDeterminants[axis]
                    + adjustedSample.logNormalizerDimension * Math.log(validatedBrownianRates[axis]));
        }
        return new SurrogateDecomposition(logDetPart + quadraticPart, logDetPart, quadraticPart);
    }
    public double[][] evaluateSurrogateContributions(double[] embeddings) {
        return evaluateSurrogateContributions(embeddings, initialRootPriorVariances);
    }

    public double[][] evaluateSurrogateContributions(double[] embeddings, double[] rootPriorVariances) {
        if (embeddings == null || embeddings.length != embeddingParameterCount) {
            throw new IllegalArgumentException("embeddings length mismatch: expected "
                    + embeddingParameterCount + " but got " + (embeddings == null ? 0 : embeddings.length));
        }
        double[][] surrogate = new double[trainingSamples.size()][dim];
        for (int sampleIndex = 0; sampleIndex < trainingSamples.size(); sampleIndex++) {
            surrogate[sampleIndex] = computeSurrogateLogLikelihoodContributions(
                    trainingSamples.get(sampleIndex), embeddings, dim, brownianRates, rootPriorVariances);
        }
        return surrogate;
    }

    private double[][] evaluateRegressionContributions(double[] embeddings, double[] rootPriorVariances) {
        return usesQuadraticRegressionContributions()
                ? evaluateQuadraticContributions(embeddings, rootPriorVariances)
                : evaluateSurrogateContributions(embeddings, rootPriorVariances);
    }

    /**
     * Returns [totalStd, logDetStd, quadFormStd, logDetFraction] measuring how much of the
     * surrogate cross-sample variation comes from log-determinant terms (embedding-independent)
     * vs the quadratic form (embedding-dependent).
     */
    public double[] computeSurrogateVarianceDecomposition(double[] embeddings) {
        return computeSurrogateVarianceDecomposition(embeddings, initialRootPriorVariances);
    }

    public double[] computeSurrogateVarianceDecomposition(double[] embeddings, double[] rootPriorVariances) {
        if (embeddings == null || embeddings.length != embeddingParameterCount) {
            throw new IllegalArgumentException("embeddings length mismatch");
        }
        int n = trainingSamples.size();
        double[] logDetParts = new double[n];
        double[] quadFormParts = new double[n];
        double[] totals = new double[n];

        for (int i = 0; i < n; i++) {
            LearnedWeightsTreeTrainer.TrainingTreeSample sample = trainingSamples.get(i);
            SurrogateDecomposition decomposition = computeSurrogateLogLikelihoodDecomposition(
                    sample,
                    embeddings,
                    dim,
                    brownianRates,
                    rootPriorVariances);
            logDetParts[i] = decomposition.logDetPart;
            quadFormParts[i] = decomposition.quadraticPart;
            totals[i] = decomposition.total;
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

    public static double[] fitSplitCalibrationPerDimension(
            double[][] baseByDimension,
            double[][] quadraticByDimension,
            double[] target) {
        if (baseByDimension.length != target.length || baseByDimension.length != quadraticByDimension.length
                || baseByDimension.length == 0) {
            throw new IllegalArgumentException(
                    "baseByDimension, quadraticByDimension, and target must be non-empty and aligned");
        }
        int dim = baseByDimension[0].length;
        if (dim == 0) {
            throw new IllegalArgumentException("calibration features must have at least one dimension");
        }
        for (int sampleIndex = 0; sampleIndex < baseByDimension.length; sampleIndex++) {
            if (baseByDimension[sampleIndex].length != dim || quadraticByDimension[sampleIndex].length != dim) {
                throw new IllegalArgumentException("All calibration feature rows must share the same dimensionality");
            }
        }

        int featureCount = 2 * dim;
        double[] meanFeatures = new double[featureCount];
        double meanTarget = 0.0;
        for (int sampleIndex = 0; sampleIndex < target.length; sampleIndex++) {
            meanTarget += target[sampleIndex];
            for (int axis = 0; axis < dim; axis++) {
                meanFeatures[axis] += baseByDimension[sampleIndex][axis];
                meanFeatures[dim + axis] += quadraticByDimension[sampleIndex][axis];
            }
        }
        meanTarget /= target.length;
        for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            meanFeatures[featureIndex] /= target.length;
        }

        double[][] covariance = new double[featureCount][featureCount];
        double[] targetCovariance = new double[featureCount];
        for (int sampleIndex = 0; sampleIndex < target.length; sampleIndex++) {
            double[] centered = new double[featureCount];
            for (int axis = 0; axis < dim; axis++) {
                centered[axis] = baseByDimension[sampleIndex][axis] - meanFeatures[axis];
                centered[dim + axis] = quadraticByDimension[sampleIndex][axis] - meanFeatures[dim + axis];
            }
            double centeredTarget = target[sampleIndex] - meanTarget;
            for (int row = 0; row < featureCount; row++) {
                targetCovariance[row] += centered[row] * centeredTarget;
                for (int col = 0; col < featureCount; col++) {
                    covariance[row][col] += centered[row] * centered[col];
                }
            }
        }

        double[] weights = solveLinearSystem(covariance, targetCovariance);
        if (weights == null) {
            weights = new double[featureCount];
        }

        double intercept = meanTarget;
        for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            intercept -= weights[featureIndex] * meanFeatures[featureIndex];
        }

        double[] parameters = new double[1 + featureCount];
        parameters[0] = intercept;
        System.arraycopy(weights, 0, parameters, 1, featureCount);
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
        return computeSurrogateLogLikelihoodContributions(sample, embeddings, dim, uniformBrownianRates(dim, brownianRate));
    }

    public static double[] computeSurrogateLogLikelihoodContributions(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] embeddings,
            int dim,
            double[] brownianRates) {
        return computeSurrogateLogLikelihoodContributions(sample, embeddings, dim, brownianRates, null);
    }

    public static double[] computeSurrogateLogLikelihoodContributions(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] embeddings,
            int dim,
            double[] brownianRates,
            double[] rootPriorVariances) {
        double[] validatedBrownianRates = copyAndValidateBrownianRates(dim, brownianRates);
        int taxonCount = sample.getTaxonCount();
        if (embeddings.length != taxonCount * dim) {
            throw new IllegalArgumentException("embeddings length mismatch for sample: expected "
                    + (taxonCount * dim) + " but got " + embeddings.length);
        }
        RootVarianceAdjustedSample adjustedSample = adjustSampleForRootPriorVariances(
                sample, embeddings, dim, rootPriorVariances, false);

        double[] contributions = new double[dim];
        for (int axis = 0; axis < dim; axis++) {
            double brownianRate = validatedBrownianRates[axis];
            double baseContribution = -0.5 * adjustedSample.logNormalizerDimension * LOG_TWO_PI
                    - 0.5 * adjustedSample.logDeterminants[axis]
                    - 0.5 * adjustedSample.logNormalizerDimension * Math.log(brownianRate);
            contributions[axis] = baseContribution - 0.5 * adjustedSample.quadraticForms[axis] / brownianRate;
        }
        return contributions;
    }

    public static double computeSurrogateLogLikelihood(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] embeddings,
            int dim,
            double brownianRate) {
        return computeSurrogateLogLikelihoodDecomposition(sample, embeddings, dim, brownianRate).total;
    }

    public static double computeSurrogateLogLikelihood(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] embeddings,
            int dim,
            double[] brownianRates) {
        return computeSurrogateLogLikelihoodDecomposition(sample, embeddings, dim, brownianRates, null).total;
    }

    public static double computeSurrogateLogLikelihood(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] embeddings,
            int dim,
            double[] brownianRates,
            double[] rootPriorVariances) {
        return computeSurrogateLogLikelihoodDecomposition(sample, embeddings, dim, brownianRates, rootPriorVariances).total;
    }

    public static double[] computeQuadraticLogLikelihoodContributions(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] embeddings,
            int dim,
            double brownianRate) {
        return computeQuadraticLogLikelihoodContributions(sample, embeddings, dim, uniformBrownianRates(dim, brownianRate));
    }

    public static double[] computeQuadraticLogLikelihoodContributions(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] embeddings,
            int dim,
            double[] brownianRates) {
        return computeQuadraticLogLikelihoodContributions(sample, embeddings, dim, brownianRates, null);
    }

    public static double[] computeQuadraticLogLikelihoodContributions(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] embeddings,
            int dim,
            double[] brownianRates,
            double[] rootPriorVariances) {
        double[] validatedBrownianRates = copyAndValidateBrownianRates(dim, brownianRates);
        int taxonCount = sample.getTaxonCount();
        if (embeddings.length != taxonCount * dim) {
            throw new IllegalArgumentException("embeddings length mismatch for sample: expected "
                    + (taxonCount * dim) + " but got " + embeddings.length);
        }
        RootVarianceAdjustedSample adjustedSample = adjustSampleForRootPriorVariances(
                sample, embeddings, dim, rootPriorVariances, false);
        double[] quadraticForms = new double[dim];
        for (int axis = 0; axis < dim; axis++) {
            quadraticForms[axis] = -0.5 * adjustedSample.quadraticForms[axis] / validatedBrownianRates[axis];
        }
        return quadraticForms;
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
        double[] currentBrownianRates = extractBrownianRates(parameters);
        double[] currentRootPriorVariances = extractRootPriorVariances(parameters);
        double[] baseScales = extractBaseScales(parameters);
        double[] slopes = extractSlopes(parameters);
        double fitLoss = 0.0;
        double gradientIntercept = 0.0;
        double[] gradientRate = new double[dim];
        double[] gradientRootVariance = new double[dim];
        double[] gradientBaseScale = new double[dim];
        double[] gradientSlope = new double[dim];
        double sampleWeight = 1.0 / activeBatchCount;

        for (int batchIndex = 0; batchIndex < activeBatchCount; batchIndex++) {
            LearnedWeightsTreeTrainer.TrainingTreeSample sample = trainingSamples.get(resolveSampleIndex(batchIndex));
            RootVarianceAdjustedSample adjustedSample = adjustSampleForRootPriorVariances(
                    sample,
                    parameters,
                    dim,
                    currentRootPriorVariances,
                    true);
            double[] precisionTimesEmbeddings = adjustedSample.precisionTimesEmbeddings;
            double[] baseContributions = computeBaseLogLikelihoodContributions(adjustedSample, currentBrownianRates);
            double[] quadraticContributions = computeQuadraticLogLikelihoodContributions(adjustedSample, currentBrownianRates);
            double[] regressionContributions;
            double prediction;
            double regressionTarget;
            if (separateBaseAndQuadraticCalibration) {
                regressionContributions = quadraticContributions;
                prediction = intercept + dot(baseScales, baseContributions) + dot(slopes, quadraticContributions);
                regressionTarget = sample.targetLogLikelihood;
            } else {
                double[] fullContributions = add(baseContributions, quadraticContributions);
                regressionContributions = usesQuadraticRegressionContributions()
                        ? quadraticContributions
                        : fullContributions;
                prediction = intercept + dot(slopes, regressionContributions);
                regressionTarget = getRegressionTarget(sample, currentBrownianRates, currentRootPriorVariances);
            }

            double residual = prediction - regressionTarget;
            fitLoss += sampleWeight * residual * residual;

            if (gradient != null) {
                for (int axis = 0; axis < dim; axis++) {
                    double embeddingFactor = -2.0 * sampleWeight * residual * slopes[axis] / currentBrownianRates[axis];
                    for (int taxon = 0; taxon < taxonCount; taxon++) {
                        int index = taxon * dim + axis;
                        gradient[index] += embeddingFactor * precisionTimesEmbeddings[index];
                    }

                    double logDetRateDerivative = -0.5 * adjustedSample.logNormalizerDimension;
                    double quadraticRateDerivative = -quadraticContributions[axis];
                    double residualRateDerivative;
                    if (separateBaseAndQuadraticCalibration) {
                        residualRateDerivative = baseScales[axis] * logDetRateDerivative
                                + slopes[axis] * quadraticRateDerivative;
                    } else if (scaleQuadraticPartOnly) {
                        residualRateDerivative = slopes[axis] * quadraticRateDerivative + logDetRateDerivative;
                    } else if (trainQuadraticOnly) {
                        residualRateDerivative = slopes[axis] * quadraticRateDerivative;
                    } else {
                        residualRateDerivative = slopes[axis] * (logDetRateDerivative + quadraticRateDerivative);
                    }
                    gradientRate[axis] += 2.0 * sampleWeight * residual * residualRateDerivative;
                    if (separateBaseAndQuadraticCalibration) {
                        gradientBaseScale[axis] += 2.0 * sampleWeight * residual * baseContributions[axis];
                    }
                    gradientSlope[axis] += 2.0 * sampleWeight * residual * regressionContributions[axis];

                    if (optimizeRootPriorVariances) {
                        double baseVarianceDerivative = -0.5 * adjustedSample.logDetVarianceDerivatives[axis];
                        double quadraticVarianceDerivative = 0.5 * adjustedSample.quadraticVarianceDerivativeFactors[axis]
                                / currentBrownianRates[axis];
                        double residualVarianceDerivative;
                        if (separateBaseAndQuadraticCalibration) {
                            residualVarianceDerivative = baseScales[axis] * baseVarianceDerivative
                                    + slopes[axis] * quadraticVarianceDerivative;
                        } else if (scaleQuadraticPartOnly) {
                            residualVarianceDerivative = slopes[axis] * quadraticVarianceDerivative + baseVarianceDerivative;
                        } else if (trainQuadraticOnly) {
                            residualVarianceDerivative = slopes[axis] * quadraticVarianceDerivative;
                        } else {
                            residualVarianceDerivative = slopes[axis] * (baseVarianceDerivative + quadraticVarianceDerivative);
                        }
                        gradientRootVariance[axis] += 2.0 * sampleWeight * residual * residualVarianceDerivative;
                    }
                }
                gradientIntercept += 2.0 * sampleWeight * residual;
            }
        }

        if (gradient != null) {
            if (optimizeRootPriorVariances) {
                for (int axis = 0; axis < dim; axis++) {
                    gradientRootVariance[axis] *= currentRootPriorVariances[axis];
                }
            }
        }

        if (useLogSlopeParameterization && gradient != null) {
            for (int axis = 0; axis < dim; axis++) {
                if (separateBaseAndQuadraticCalibration) {
                    gradientBaseScale[axis] *= baseScales[axis];
                }
                gradientSlope[axis] *= slopes[axis];
            }
        }

        double slopePenalty = 0.0;
        if (slopeRegularizationWeight > 0.0) {
            if (useLogSlopeParameterization) {
                for (int axis = 0; axis < dim; axis++) {
                    if (separateBaseAndQuadraticCalibration) {
                        double logBaseScale = parameters[baseScaleParameterStartIndex + axis];
                        slopePenalty += slopeRegularizationWeight * logBaseScale * logBaseScale;
                        if (gradient != null) {
                            gradientBaseScale[axis] += 2.0 * slopeRegularizationWeight * logBaseScale;
                        }
                    }
                    double logSlope = parameters[slopeParameterStartIndex + axis];
                    slopePenalty += slopeRegularizationWeight * logSlope * logSlope;
                    if (gradient != null) {
                        gradientSlope[axis] += 2.0 * slopeRegularizationWeight * logSlope;
                    }
                }
            } else {
                for (int axis = 0; axis < dim; axis++) {
                    if (separateBaseAndQuadraticCalibration) {
                        double centeredBaseScale = baseScales[axis] - 1.0;
                        slopePenalty += slopeRegularizationWeight * centeredBaseScale * centeredBaseScale;
                        if (gradient != null) {
                            gradientBaseScale[axis] += 2.0 * slopeRegularizationWeight * centeredBaseScale;
                        }
                    }
                    double centeredSlope = slopes[axis] - 1.0;
                    slopePenalty += slopeRegularizationWeight * centeredSlope * centeredSlope;
                    if (gradient != null) {
                        gradientSlope[axis] += 2.0 * slopeRegularizationWeight * centeredSlope;
                    }
                }
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

        lastBaseScales = Arrays.copyOf(baseScales, baseScales.length);
        lastSlopes = Arrays.copyOf(slopes, slopes.length);
        lastRootPriorVariances = currentRootPriorVariances == null ? null : Arrays.copyOf(currentRootPriorVariances, currentRootPriorVariances.length);
        lastIntercept = intercept;
        lastFitLoss = fitLoss;
        lastSlopePenalty = slopePenalty;
        lastPcaPenalty = pcaPenalty;
        lastLoss = loss;
        if (gradient != null) {
            gradient[interceptParameterIndex] = gradientIntercept;
            if (optimizeBrownianRates) {
                System.arraycopy(gradientRate, 0, gradient, rateParameterStartIndex, dim);
            }
            if (optimizeRootPriorVariances) {
                System.arraycopy(gradientRootVariance, 0, gradient, rootVarianceParameterStartIndex, dim);
            }
            if (separateBaseAndQuadraticCalibration) {
                System.arraycopy(gradientBaseScale, 0, gradient, baseScaleParameterStartIndex, dim);
            }
            System.arraycopy(gradientSlope, 0, gradient, slopeParameterStartIndex, dim);
        }

        // When validation is enabled, best-parameter selection is driven exclusively by
        // maybeCheckValidation() (called from valueAndGradient), not by this training loss.
        if (!isValidationEnabled() && loss < bestLoss) {
            bestLoss = loss;
            bestFitLoss = fitLoss;
            bestSlopePenalty = slopePenalty;
            bestPcaPenalty = pcaPenalty;
            bestParameters = Arrays.copyOf(parameters, parameterCount);
            bestBaseScales = Arrays.copyOf(baseScales, baseScales.length);
            bestSlopes = Arrays.copyOf(slopes, slopes.length);
            bestRootPriorVariances = currentRootPriorVariances == null ? null : Arrays.copyOf(currentRootPriorVariances, currentRootPriorVariances.length);
            bestIntercept = intercept;
        }
        maybeLogProgress();
        return loss;
    }

    private static double[] computeBaseLogLikelihoodContributions(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            int dim,
            double[] currentBrownianRates,
            double[] rootPriorVariances) {
        RootVarianceAdjustedSample adjustedSample = adjustSampleForRootPriorVariances(
                sample, null, dim, rootPriorVariances, false);
        return computeBaseLogLikelihoodContributions(adjustedSample, currentBrownianRates);
    }

    private static double[] computeBaseLogLikelihoodContributions(
            RootVarianceAdjustedSample adjustedSample,
            double[] currentBrownianRates) {
        double[] contributions = new double[currentBrownianRates.length];
        for (int axis = 0; axis < currentBrownianRates.length; axis++) {
            contributions[axis] = -0.5 * (
                adjustedSample.logNormalizerDimension * (LOG_TWO_PI + Math.log(currentBrownianRates[axis]))
                        + adjustedSample.logDeterminants[axis]);
        }
        return contributions;
    }

    private double[] computeQuadraticLogLikelihoodContributions(
            RootVarianceAdjustedSample adjustedSample,
            double[] currentBrownianRates) {
        double[] contributions = new double[dim];
        for (int axis = 0; axis < dim; axis++) {
            contributions[axis] = -0.5 * adjustedSample.quadraticForms[axis] / currentBrownianRates[axis];
        }
        return contributions;
    }

    private double getRegressionTarget(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] currentBrownianRates,
            double[] currentRootPriorVariances) {
        return scaleQuadraticPartOnly
                ? sample.targetLogLikelihood - computeLogDetPart(sample, currentBrownianRates, currentRootPriorVariances)
                : sample.targetLogLikelihood;
    }

    private boolean usesQuadraticRegressionContributions() {
        return scaleQuadraticPartOnly || trainQuadraticOnly;
    }

    private double computeLogDetPart(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] currentBrownianRates,
            double[] currentRootPriorVariances) {
        RootVarianceAdjustedSample adjustedSample = adjustSampleForRootPriorVariances(
                sample, null, dim, currentRootPriorVariances, false);
        double total = -0.5 * dim * adjustedSample.logNormalizerDimension * LOG_TWO_PI;
        for (int axis = 0; axis < dim; axis++) {
            total -= 0.5 * adjustedSample.logDeterminants[axis];
            total -= 0.5 * adjustedSample.logNormalizerDimension * Math.log(currentBrownianRates[axis]);
        }
        return total;
    }

    private static double[] uniformBrownianRates(int dim, double brownianRate) {
        if (!(brownianRate > 0.0) || !Double.isFinite(brownianRate)) {
            throw new IllegalArgumentException("brownianRate must be finite and > 0, but was " + brownianRate);
        }
        double[] rates = new double[dim];
        Arrays.fill(rates, brownianRate);
        return rates;
    }

    private static double[] copyAndValidateBrownianRates(int dim, double[] brownianRates) {
        if (brownianRates == null || brownianRates.length != dim) {
            throw new IllegalArgumentException("brownianRates length mismatch: expected "
                    + dim + " but got " + (brownianRates == null ? 0 : brownianRates.length));
        }
        double[] copy = Arrays.copyOf(brownianRates, brownianRates.length);
        for (int axis = 0; axis < copy.length; axis++) {
            if (!(copy[axis] > 0.0) || !Double.isFinite(copy[axis])) {
                throw new IllegalArgumentException("brownianRates must all be finite and > 0, but axis "
                        + axis + " was " + copy[axis]);
            }
        }
        return copy;
    }

    private static double[] resolveOptimizableRootPriorVariances(int dim, double[] rootPriorVariances) {
        if (rootPriorVariances == null) {
            return null;
        }
        if (rootPriorVariances.length != dim) {
            throw new IllegalArgumentException("rootPriorVariances length mismatch: expected "
                    + dim + " but got " + rootPriorVariances.length);
        }
        double[] copy = Arrays.copyOf(rootPriorVariances, rootPriorVariances.length);
        boolean anyInfinite = false;
        for (int axis = 0; axis < copy.length; axis++) {
            if (copy[axis] < 0.0 || Double.isNaN(copy[axis])) {
                throw new IllegalArgumentException("rootPriorVariances must all be >= 0 or Infinity, but axis "
                        + axis + " was " + copy[axis]);
            }
            if (!Double.isFinite(copy[axis])) {
                anyInfinite = true;
            }
        }
        return anyInfinite ? null : copy;
    }

    private static double[] copyAndValidateFiniteRootPriorVariances(int dim, double[] rootPriorVariances) {
        if (rootPriorVariances == null) {
            return null;
        }
        if (rootPriorVariances.length != dim) {
            throw new IllegalArgumentException("rootPriorVariances length mismatch: expected "
                    + dim + " but got " + rootPriorVariances.length);
        }
        double[] copy = Arrays.copyOf(rootPriorVariances, rootPriorVariances.length);
        for (int axis = 0; axis < copy.length; axis++) {
            if (!(copy[axis] >= 0.0) || !Double.isFinite(copy[axis])) {
                throw new IllegalArgumentException("rootPriorVariances must all be finite and >= 0, but axis "
                        + axis + " was " + copy[axis]);
            }
        }
        return copy;
    }

    private double[] createUnitSlopeVector() {
        double[] slopes = new double[dim];
        Arrays.fill(slopes, 1.0);
        return slopes;
    }

    private double[] createUnitScaleVector() {
        double[] scales = new double[dim];
        Arrays.fill(scales, 1.0);
        return scales;
    }

    private static double[] add(double[] left, double[] right) {
        double[] total = new double[left.length];
        for (int i = 0; i < left.length; i++) {
            total[i] = left[i] + right[i];
        }
        return total;
    }

    private static double dot(double[] left, double[] right) {
        double total = 0.0;
        for (int i = 0; i < left.length; i++) {
            total += left[i] * right[i];
        }
        return total;
    }

    private static double mean(double[] values) {
        if (values == null || values.length == 0) {
            return Double.NaN;
        }
        double total = 0.0;
        for (double value : values) {
            total += value;
        }
        return total / values.length;
    }

    private static RootVarianceAdjustedSample adjustSampleForRootPriorVariances(
            LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            double[] embeddings,
            int dim,
            double[] rootPriorVariances,
            boolean computePrecisionTimesEmbeddings) {
        if (rootPriorVariances == null) {
            double[] precisionTimesEmbeddings = multiplyPrecisionByEmbeddings(sample.precisionMatrix, embeddings, dim);
            double[] quadraticForms = computeQuadraticForms(embeddings, precisionTimesEmbeddings, sample.getTaxonCount(), dim);
            double[] logDeterminants = new double[dim];
            Arrays.fill(logDeterminants, sample.logDeterminant);
            return new RootVarianceAdjustedSample(
                    precisionTimesEmbeddings,
                    quadraticForms,
                    logDeterminants,
                    new double[dim],
                    new double[dim],
                    sample.logNormalizerDimension);
        }

        if (!sample.hasRootFreeFactorization()) {
            throw new IllegalArgumentException(
                    "Training sample is missing root-free covariance data required for trait-specific rootPriorVariances");
        }
        double[] validatedRootPriorVariances = copyAndValidateFiniteRootPriorVariances(dim, rootPriorVariances);
        double precisionOnesSumForLogDet = sample.rootFreePrecisionOnesSum;

        if (embeddings == null) {
            // Caller (computeBaseLogLikelihoodContributions) only needs the per-axis
            // logDeterminants to form base contributions; skip the embedding-dependent
            // qf / projection / derivative work entirely.
            double[] logDeterminantsOnly = new double[dim];
            for (int axis = 0; axis < dim; axis++) {
                double variance = validatedRootPriorVariances[axis];
                double denominator = 1.0 + variance * precisionOnesSumForLogDet;
                if (!(denominator > 0.0) || !Double.isFinite(denominator)) {
                    throw new IllegalArgumentException("Invalid rootPriorVariance update denominator: " + denominator);
                }
                logDeterminantsOnly[axis] = sample.rootFreeLogDeterminant + Math.log(denominator);
            }
            return new RootVarianceAdjustedSample(
                    null,
                    null,
                    logDeterminantsOnly,
                    new double[dim],
                    new double[dim],
                    sample.rootFreeLogNormalizerDimension);
        }

        double[] basePrecisionTimesEmbeddings = sample.multiplyRootFreePrecisionByEmbeddings(embeddings, dim);
        double[] adjustedPrecisionTimesEmbeddings = computePrecisionTimesEmbeddings
                ? new double[sample.getTaxonCount() * dim]
                : basePrecisionTimesEmbeddings;
        double[] quadraticForms = new double[dim];
        double[] logDeterminants = new double[dim];
        double[] logDetVarianceDerivatives = new double[dim];
        double[] quadraticVarianceDerivativeFactors = new double[dim];
        double precisionOnesSum = sample.rootFreePrecisionOnesSum;

        for (int axis = 0; axis < dim; axis++) {
            double variance = validatedRootPriorVariances[axis];
            double denominator = 1.0 + variance * precisionOnesSum;
            if (!(denominator > 0.0) || !Double.isFinite(denominator)) {
                throw new IllegalArgumentException("Invalid rootPriorVariance update denominator: " + denominator);
            }
            double alpha = variance / denominator;
            double axisProjection = 0.0;
            double baseQuadraticForm = 0.0;
            for (int taxon = 0; taxon < sample.getTaxonCount(); taxon++) {
                int index = taxon * dim + axis;
                double embedding = embeddings[index];
                axisProjection += embedding * sample.rootFreePrecisionTimesOnes[taxon];
                baseQuadraticForm += embedding * basePrecisionTimesEmbeddings[index];
            }
            quadraticForms[axis] = baseQuadraticForm - alpha * axisProjection * axisProjection;
            logDeterminants[axis] = sample.rootFreeLogDeterminant + Math.log(denominator);
            logDetVarianceDerivatives[axis] = precisionOnesSum / denominator;
            quadraticVarianceDerivativeFactors[axis] = (axisProjection * axisProjection) / (denominator * denominator);

            if (computePrecisionTimesEmbeddings) {
                for (int taxon = 0; taxon < sample.getTaxonCount(); taxon++) {
                    int index = taxon * dim + axis;
                    adjustedPrecisionTimesEmbeddings[index] = basePrecisionTimesEmbeddings[index]
                            - alpha * sample.rootFreePrecisionTimesOnes[taxon] * axisProjection;
                }
            }
        }

        return new RootVarianceAdjustedSample(
                adjustedPrecisionTimesEmbeddings,
                quadraticForms,
                logDeterminants,
                logDetVarianceDerivatives,
                quadraticVarianceDerivativeFactors,
                sample.rootFreeLogNormalizerDimension);
    }

    private static double[] multiplyPrecisionByEmbeddings(double[][] precisionMatrix, double[] embeddings, int dim) {
        int taxonCount = precisionMatrix.length;
        double[] output = new double[taxonCount * dim];
        if (embeddings == null) {
            return output;
        }
        for (int row = 0; row < taxonCount; row++) {
            int rowOffset = row * dim;
            for (int column = 0; column < taxonCount; column++) {
                double precisionEntry = precisionMatrix[row][column];
                if (precisionEntry == 0.0) {
                    continue;
                }
                int columnOffset = column * dim;
                for (int axis = 0; axis < dim; axis++) {
                    output[rowOffset + axis] += precisionEntry * embeddings[columnOffset + axis];
                }
            }
        }
        return output;
    }

    private static double[] computeQuadraticForms(
            double[] embeddings,
            double[] precisionTimesEmbeddings,
            int taxonCount,
            int dim) {
        double[] quadraticForms = new double[dim];
        if (embeddings == null) {
            return quadraticForms;
        }
        for (int taxon = 0; taxon < taxonCount; taxon++) {
            int offset = taxon * dim;
            for (int axis = 0; axis < dim; axis++) {
                quadraticForms[axis] += embeddings[offset + axis] * precisionTimesEmbeddings[offset + axis];
            }
        }
        return quadraticForms;
    }

    private void resetMiniBatchSchedule() {
        if (!useMiniBatches) {
            activeBatchStart = 0;
            activeBatchCount = trainIndices.length;
            batchOrder = null;
            batchCursor = 0;
            miniBatchRandom = null;
            return;
        }
        batchOrder = Arrays.copyOf(trainIndices, trainIndices.length);
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
            return trainIndices[batchIndex];
        }
        return batchOrder[activeBatchStart + batchIndex];
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

    private void maybeLogProgress() {
        if (!progressLoggingEnabled) {
            return;
        }
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
            "%s: objective iteration %d/%s, elapsed=%.1fs, trialLoss=%.6f, bestSeenLoss=%.6f, fitLoss=%.6f, slopePenalty=%.6f, pcaPenalty=%.6f, baseScale=%.6f, quadraticScale=%.6f, intercept=%.6f, validationLoss=%.6f",
            progressLabel,
            currentIteration,
            configuredMaxIterations > 0 ? configuredMaxIterations : "?",
            elapsedSec,
            lastLoss,
            bestLoss,
            lastFitLoss,
            lastSlopePenalty,
            lastPcaPenalty,
            mean(lastBaseScales),
            mean(lastSlopes),
            lastIntercept,
            lastValidationLoss));
        lastProgressLogNanos = now;
    }
}
