package targetedbeast.edgeweights;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.core.Log;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.branchratemodel.BranchRateModel;
import beast.base.evolution.likelihood.GenericTreeLikelihood;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Distribution;
import beast.base.inference.State;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsOptimizer;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsTreeObjectiveFunction;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsTrainingSampleIO;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsTreeTrainer;
import targetedbeast.util.Alignment2PCA;

@Description("Learns leaf embeddings so a Brownian surrogate tree likelihood tracks the full sequence tree likelihood. "
    + "Runtime delegates edge and target weights to PCAWeightsBrownianFullLikelihood.")
public class LearnedWeights extends Distribution implements EdgeWeights {

    private static final int INITIAL_SCALE_SEARCH_EXPANSION_STEPS = 8;
    private static final int INITIAL_SCALE_SEARCH_REFINEMENT_STEPS = 8;
    private static final double INITIAL_SCALE_SEARCH_MULTIPLIER = 2.0;
    private static final double BROWNIAN_RATE = 1.0;

    public static final String SKIP_LEARNING_SYSTEM_PROPERTY = "targetedbeast.learnedweights.skipLearning";

    final public Input<Alignment> dataInput = new Input<>("data", "sequence data for the beast.tree", Validate.REQUIRED);

    final public Input<TreeInterface> treeInput = new Input<>("tree", "phylogenetic beast.tree with sequence data in the leafs", Validate.REQUIRED);

    final public Input<Double> maxWeightInput = new Input<>("maxWeight", "maximum weight for an edge", 10.0);
    final public Input<Double> minWeightInput = new Input<>("minWeight", "maximum weight for an edge", 0.01);

    final public Input<Integer> dimensionInput = new Input<>("dimension", "dimension of learned points", 4);

    final public Input<String> valueInput = new Input<>("value", "comma separated list of taxon=x1 x2 x3 pairs, "
            + "where <dimension> number of dimensions are specified, e.g. taxon1=0.2 0.4, taxon2=0.4 0.3, taxon3=0.9 0.1. "
            + "If value is specified, learned initialization is ignored");

    final public Input<Boolean> useInverseMeanDistanceProposalInput = new Input<>(
            "useInverseMeanDistanceProposal",
            "whether runtime targeted proposals should use inverse-distance weights instead of scaled surrogate log-likelihood scores",
            false);
    final public Input<Double> offsetInput = new Input<>("offset", "offset in weight", 0.01);

        final public Input<Integer> nTrainingTreesInput = new Input<>("nTrainingTrees", "number of training trees to use when generating samples from the prior or a posterior tree log", 50);
        final public Input<List<File>> trainingTreeFileInput = new Input<>(
            "trainingTreeFile",
            "optional posterior tree logs to use directly as the LearnedWeights training set; specify multiple trainingTreeFile entries in XML to combine multiple logs",
            new ArrayList<>());
        final public Input<List<File>> trainingLogFileInput = new Input<>(
            "trainingLogFile",
            "optional posterior parameter log files paired with trainingTreeFile entries; used to restore sampled clock-rate state when rates are not encoded in the trees",
            new ArrayList<>());
        final public Input<Integer> trainingTreeBurninPercentageInput = new Input<>(
            "trainingTreeBurninPercentage",
            "burn-in percentage applied when reading trainingTreeFile",
            10);
        final public Input<Integer> trainingTreeThinIntervalInput = new Input<>(
            "trainingTreeThinInterval",
            "thinning interval applied when reading trainingTreeFile",
            1);
        final public Input<String> trainingSamplesFileInput = new Input<>(
            "trainingSamplesFile",
            "optional path to cached training samples generated from prior or posterior trees; if set, these are loaded instead of generating prior-sampled trees");
    final public Input<Integer> maxIterationsInput = new Input<>("maxIterations", "maximum number of optimizer iterations", 500);
    final public Input<Double> learningRateInput = new Input<>("learningRate", "optimizer step-size parameter for covariance-matrix training", 0.01);
    final public Input<Double> gradientToleranceInput = new Input<>("gradientTolerance", "optimizer stop threshold on the gradient norm", 1e-16);
        final public Input<Integer> miniBatchSizeInput = new Input<>(
            "miniBatchSize",
            "mini-batch size for covariance training; values <= 0 or >= number of training samples use full-batch optimization",
            0);
        final public Input<Long> miniBatchSeedInput = new Input<>(
            "miniBatchSeed",
            "random seed used to shuffle training samples between mini-batches",
            1L);
        final public Input<Double> pcaRegularizationWeightInput = new Input<>(
            "pcaRegularizationWeight",
            "L2 regularization strength that keeps learned embeddings close to the PCA initialization",
            0.0);
        final public Input<Double> slopeRegularizationWeightInput = new Input<>(
            "slopeRegularizationWeight",
            "regularization strength on slope, centered at slope=1",
            0.0);
    final public Input<GenericTreeLikelihood> likelihoodInput = new Input<>("likelihood", "tree likelihood template used to configure per-tree training likelihoods", Validate.REQUIRED);

    private double offset;
    private int dim;
    private double learnedSlope = 1.0;
    private double learnedIntercept = 0.0;
    private PCAWeightsBrownianFullLikelihood runtimeModel;
    private List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples = Collections.emptyList();

    @Override
    public void initAndValidate() {
        final TreeInterface tree = treeInput.get();
        final Alignment data = dataInput.get();
        final String value = valueInput.get();

        dim = dimensionInput.get();
        offset = offsetInput.get();

        if (dim <= 0) {
            throw new IllegalArgumentException("dimension must be > 0, but was " + dim);
        }
        if (offset < 0.0) {
            throw new IllegalArgumentException("offset must be >= 0, but was " + offset);
        }
        if (!(learningRateInput.get() > 0.0)) {
            throw new IllegalArgumentException("learningRate must be > 0, but was " + learningRateInput.get());
        }
        if (!(gradientToleranceInput.get() > 0.0)) {
            throw new IllegalArgumentException("gradientTolerance must be > 0, but was " + gradientToleranceInput.get());
        }
        if (miniBatchSizeInput.get() < 0) {
            throw new IllegalArgumentException("miniBatchSize must be >= 0, but was " + miniBatchSizeInput.get());
        }
        if (pcaRegularizationWeightInput.get() < 0.0) {
            throw new IllegalArgumentException("pcaRegularizationWeight must be >= 0, but was " + pcaRegularizationWeightInput.get());
        }
        if (slopeRegularizationWeightInput.get() < 0.0) {
            throw new IllegalArgumentException("slopeRegularizationWeight must be >= 0, but was " + slopeRegularizationWeightInput.get());
        }

        if (data.getTaxonCount() != tree.getLeafNodeCount()) {
            String leaves = "?";
            if (tree instanceof Tree) {
                leaves = String.join(", ", ((Tree) tree).getTaxaNames());
            }
            throw new IllegalArgumentException(String.format(
                    "The number of leaves in the tree (%d) does not match the number of sequences (%d). "
                            + "The tree has leaves [%s], while the data refers to taxa [%s].",
                    tree.getLeafNodeCount(), data.getTaxonCount(), leaves,
                    String.join(", ", data.getTaxaNames())));
        }

        runtimeModel = createRuntimeModel(tree, value);
        if (value == null) {
            runLearningPhase();
        }
        runtimeModel.updateByOperator();
    }

    private PCAWeightsBrownianFullLikelihood createRuntimeModel(TreeInterface tree, String value) {
        BranchRateModel.Base branchRateModel = likelihoodInput.get().branchRateModelInput.get();
        PCAWeightsBrownianFullLikelihood model = new PCAWeightsBrownianFullLikelihood();
        model.initByName(
            "data", dataInput.get(),
            "tree", tree,
            "maxWeight", maxWeightInput.get(),
            "minWeight", minWeightInput.get(),
            "dimension", dim,
            "brownianRate", BROWNIAN_RATE,
            "useInverseMeanDistanceProposal", useInverseMeanDistanceProposalInput.get(),
            "offset", offset,
            "value", value,
            "branchRateModel", branchRateModel
        );
        return model;
    }

    private void runLearningPhase() {
        if (Boolean.getBoolean(SKIP_LEARNING_SYSTEM_PROPERTY)) {
            Log.info.println("LearnedWeights: skipping learning phase because system property "
                    + SKIP_LEARNING_SYSTEM_PROPERTY + "=true");
            return;
        }

        final long startNanos = System.nanoTime();

        trainingSamples = loadOrGenerateTrainingSamples();
        Log.info.println("LearnedWeights: training mode = covariance regression with affine calibration");

        if (trainingSamples.isEmpty()) {
            Log.warning("LearnedWeights: no training samples generated; keeping PCA initialization.");
            return;
        }

        String[] canonicalTaxa = dataInput.get().getTaxaNames().toArray(new String[0]);
        int[] runtimeLeafOrder = getLeafOrderForTaxa(treeInput.get(), canonicalTaxa);
        double[] directPcaEmbeddings = getPcaLeafEmbeddingsForTaxa(canonicalTaxa);
        InitialScaleSearchResult initialScaleSearch = optimizeInitialEmbeddingScale(canonicalTaxa.length, directPcaEmbeddings);
        double[] initialEmbeddings = initialScaleSearch.scaledEmbeddings;

        LearnedWeightsTreeObjectiveFunction reportingObjective = new LearnedWeightsTreeObjectiveFunction(
            trainingSamples,
            canonicalTaxa.length,
            dim,
            BROWNIAN_RATE,
            initialEmbeddings,
            pcaRegularizationWeightInput.get(),
            slopeRegularizationWeightInput.get());
        LearnedWeightsTreeObjectiveFunction optimizationObjective = new LearnedWeightsTreeObjectiveFunction(
            trainingSamples,
            canonicalTaxa.length,
            dim,
            BROWNIAN_RATE,
            initialEmbeddings,
            pcaRegularizationWeightInput.get(),
            slopeRegularizationWeightInput.get(),
            miniBatchSizeInput.get(),
            miniBatchSeedInput.get());
        double[] initialParameters = reportingObjective.createInitialParameters(initialEmbeddings);
        double initialSlope = reportingObjective.extractSlope(initialParameters);
        double initialIntercept = reportingObjective.extractIntercept(initialParameters);
        double[] initialGradient = new double[reportingObjective.getParameterCount()];
        double initialPreviewLoss = reportingObjective.valueAndGradient(initialParameters, initialGradient);
        double initialGradientNorm = reportingObjective.gradientNorm(initialGradient);
        double initialEmbeddingGradientNorm = reportingObjective.embeddingGradientNorm(initialGradient);
        double initialCalibrationGradientNorm = reportingObjective.calibrationGradientNorm(initialGradient);
        double initialEmbeddingNorm = l2Norm(initialEmbeddings);
        double initialEmbeddingMaxAbs = maxAbs(initialEmbeddings);
        double directPcaEmbeddingNorm = l2Norm(directPcaEmbeddings);
        double directPcaEmbeddingMaxAbs = maxAbs(directPcaEmbeddings);
        double runtimeVsDirectPcaDiffNorm = diffNorm(initialEmbeddings, directPcaEmbeddings);
        int initialEmbeddingNonZeroCount = countNonZeroEntries(initialEmbeddings, 1e-12);
        int directPcaEmbeddingNonZeroCount = countNonZeroEntries(directPcaEmbeddings, 1e-12);
        double[] surrogateStats = reportingObjective.computeSurrogateVarianceDecomposition(initialEmbeddings);

        LearnedWeightsOptimizer optimizer = new LearnedWeightsOptimizer(
                new LearnedWeightsOptimizer.Config()
                        .withMaxIter(maxIterationsInput.get())
                        .withLearningRate(learningRateInput.get())
                        .withGradientTolerance(gradientToleranceInput.get()));

        Log.info.println(String.format(
            "LearnedWeights: starting covariance optimization with %d embedding parameters (+ intercept + 1 slope), %d training trees, maxIter=%d, learningRate=%.4g, gradientTolerance=%.3g, pcaRegularizationWeight=%.4g, slopeRegularizationWeight=%.4g",
                reportingObjective.getEmbeddingParameterCount(),
                trainingSamples.size(),
                maxIterationsInput.get(),
                learningRateInput.get(),
                gradientToleranceInput.get(),
            pcaRegularizationWeightInput.get(),
            slopeRegularizationWeightInput.get()));
        Log.info.println("LearnedWeights: optimizer starts from PCA embeddings with affine calibration fitted on the initial surrogate.");
        Log.info.println(String.format(
            "LearnedWeights: initial objective preview loss=%.6f, fullGradientNorm=%.6g, embeddingGradientNorm=%.6g, calibrationGradientNorm=%.6g, embeddingNorm=%.6g, embeddingMaxAbs=%.6g, embeddingNonZero=%d, initialSlope=%.6g, initialIntercept=%.6g",
                initialPreviewLoss,
                initialGradientNorm,
                initialEmbeddingGradientNorm,
            initialCalibrationGradientNorm,
            initialEmbeddingNorm,
            initialEmbeddingMaxAbs,
            initialEmbeddingNonZeroCount,
            initialSlope,
            initialIntercept));
        Log.info.println(String.format(
            "LearnedWeights: surrogate variance decomposition: totalStd=%.6g, logDetStd=%.6g, quadFormStd=%.6g, logDetFraction=%.4f",
            surrogateStats[0], surrogateStats[1], surrogateStats[2], surrogateStats[3]));
        Log.info.println(String.format(
            "LearnedWeights: PCA initialization scale search: scale=%.6g, baselineLoss=%.6f, scaledLoss=%.6f",
            initialScaleSearch.scale,
            initialScaleSearch.baselineLoss,
            initialScaleSearch.scaledLoss));
        Log.info.println(String.format(
            "LearnedWeights: direct PCA diagnostic: directPcaNorm=%.6g, directPcaMaxAbs=%.6g, directPcaNonZero=%d, runtimeVsDirectPcaDiffNorm=%.6g",
            directPcaEmbeddingNorm,
            directPcaEmbeddingMaxAbs,
            directPcaEmbeddingNonZeroCount,
            runtimeVsDirectPcaDiffNorm));
        Log.info.println(String.format(
            "LearnedWeights: optimizer batch mode = %s (miniBatchSize=%d, configuredMiniBatchSize=%d)",
            optimizationObjective.usesMiniBatches() ? "mini-batch" : "full-batch",
            optimizationObjective.getMiniBatchSize(),
            miniBatchSizeInput.get()));
        Log.info.println(
            optimizationObjective.usesMiniBatches()
                ? "LearnedWeights: progress lines report batch-specific objective evaluations; trialLoss values are not directly comparable across batches."
                : "LearnedWeights: progress lines report raw line-search objective evaluations; trialLoss can spike even when bestSeenLoss improves.");

        optimizationObjective.beginOptimizationRun("LearnedWeights", maxIterationsInput.get());
        LearnedWeightsOptimizer.Result result = optimizer.optimize(initialParameters, optimizationObjective);

        double[] finalGradient = new double[reportingObjective.getParameterCount()];
        double fullFinalLoss = reportingObjective.valueAndGradient(result.optimizedTheta, finalGradient);
        double fullFinalGradientNorm = reportingObjective.gradientNorm(finalGradient);

        learnedSlope = reportingObjective.extractSlope(result.optimizedTheta);
        learnedIntercept = reportingObjective.extractIntercept(result.optimizedTheta);
        double[] optimizedEmbeddings = reportingObjective.extractEmbeddings(result.optimizedTheta);
        double optimizedEmbeddingNorm = l2Norm(optimizedEmbeddings);
        double optimizedEmbeddingMaxAbs = maxAbs(optimizedEmbeddings);
        int optimizedEmbeddingNonZeroCount = countNonZeroEntries(optimizedEmbeddings, 1e-12);
        double optimizedVsInitialDiffNorm = diffNorm(optimizedEmbeddings, initialEmbeddings);
        double[] optimizedSurrogateStats = reportingObjective.computeSurrogateVarianceDecomposition(optimizedEmbeddings);
        runtimeModel.setProposalLogLikelihoodScale(resolveRuntimeProposalScale(learnedSlope));
        runtimeModel.setLeafEmbeddings(optimizedEmbeddings, runtimeLeafOrder);

        double elapsedSec = (System.nanoTime() - startNanos) / 1_000_000_000.0;
        Log.info.println(String.format(
            "LearnedWeights: optimization %s in %.3fs (stopReason=%s, iter=%d, eval=%d, objectiveEvalCalls=%d, initial=%.6f, final=%.6f, delta=%.6f, initialGradientNorm=%.6g, finalGradientNorm=%.6g, fitLoss=%.6f, slopePenalty=%.6f, pcaPenalty=%.6f, slope=%.6f, runtimeScale=%.6f, intercept=%.6f)",
                result.status,
                elapsedSec,
                result.stopReason,
                result.iterations,
                result.evaluations,
                optimizationObjective.getRunEvaluationCount(),
                result.initialLoss,
                fullFinalLoss,
                result.initialLoss - fullFinalLoss,
            result.initialGradientNorm,
            fullFinalGradientNorm,
            reportingObjective.getLastFitLoss(),
            reportingObjective.getLastSlopePenalty(),
            reportingObjective.getLastPcaPenalty(),
            reportingObjective.extractSlope(result.optimizedTheta),
            resolveRuntimeProposalScale(learnedSlope),
            learnedIntercept));
        Log.info.println(String.format(
            "LearnedWeights: optimized embedding diagnostic: embeddingNorm=%.6g, embeddingMaxAbs=%.6g, embeddingNonZero=%d, deltaFromInitialNorm=%.6g",
                optimizedEmbeddingNorm,
                optimizedEmbeddingMaxAbs,
                optimizedEmbeddingNonZeroCount,
                optimizedVsInitialDiffNorm));
        Log.info.println(String.format(
            "LearnedWeights: optimized surrogate variance decomposition: totalStd=%.6g, logDetStd=%.6g, quadFormStd=%.6g, logDetFraction=%.4f",
                optimizedSurrogateStats[0],
                optimizedSurrogateStats[1],
                optimizedSurrogateStats[2],
                optimizedSurrogateStats[3]));
    }

    private List<LearnedWeightsTreeTrainer.TrainingTreeSample> loadOrGenerateTrainingSamples() {
        List<File> treeFiles = trainingTreeFileInput.get();
        if (!treeFiles.isEmpty()) {
            List<File> logFiles = trainingLogFileInput.get();
            LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                    likelihoodInput.get(),
                    dataInput.get(),
                    nTrainingTreesInput.get());
            try {
                List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = trainer.generateTrainingSamplesFromTreeFiles(
                    treeFiles,
                    logFiles,
                    trainingTreeBurninPercentageInput.get(),
                    trainingTreeThinIntervalInput.get(),
                    nTrainingTreesInput.get()
                );
                Log.info.println("LearnedWeights: loaded " + samples.size() + " training samples from posterior tree file"
                    + (treeFiles.size() == 1 ? " " : "s ")
                    + joinTrainingTreeFiles(treeFiles));
                    return samples;
            } catch (IOException e) {
            throw new IllegalArgumentException("Failed to load training trees from " + joinTrainingTreeFiles(treeFiles), e);
            }
        }

        String trainingSamplesFile = trainingSamplesFileInput.get();
        String[] canonicalTaxa = dataInput.get().getTaxaNames().toArray(new String[0]);
        if (trainingSamplesFile != null && !trainingSamplesFile.isBlank()) {
            Path path = Paths.get(trainingSamplesFile);
            try {
                List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = LearnedWeightsTrainingSampleIO.readSamples(path, canonicalTaxa);
                Log.info.println("LearnedWeights: loaded " + samples.size() + " cached training samples from " + path.toAbsolutePath());
                return samples;
            } catch (IOException e) {
                throw new IllegalArgumentException("Failed to load training samples from " + path.toAbsolutePath(), e);
            }
        }

        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                likelihoodInput.get(),
                dataInput.get(),
                nTrainingTreesInput.get());
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = trainer.generateTrainingSamples();
        Log.info.println("LearnedWeights: generated " + samples.size() + " prior-sampled training trees.");
        return samples;
    }

    private double resolveRuntimeProposalScale(double slope) {
        if (Double.isFinite(slope)) {
            return -slope;
        }
        Log.warning(String.format(
                "LearnedWeights: learned slope %.6f is not finite; falling back to 1.0",
                slope));
        return 1.0;
    }

    private InitialScaleSearchResult optimizeInitialEmbeddingScale(int taxonCount, double[] directPcaEmbeddings) {
        LearnedWeightsTreeObjectiveFunction baselineObjective = new LearnedWeightsTreeObjectiveFunction(
            trainingSamples,
            taxonCount,
            dim,
            BROWNIAN_RATE,
            directPcaEmbeddings,
            pcaRegularizationWeightInput.get(),
            slopeRegularizationWeightInput.get());
        double baselineLoss = evaluateInitializationLoss(baselineObjective, directPcaEmbeddings);
        double bestScale = 1.0;
        double bestLoss = baselineLoss;
        double lowerScale = 1.0;
        double upperScale = 1.0;

        double smallerScale = 1.0;
        for (int step = 0; step < INITIAL_SCALE_SEARCH_EXPANSION_STEPS; step++) {
            smallerScale /= INITIAL_SCALE_SEARCH_MULTIPLIER;
            double trialLoss = evaluateInitializationLossForScale(taxonCount, directPcaEmbeddings, smallerScale);
            lowerScale = smallerScale;
            if (trialLoss < bestLoss) {
                bestLoss = trialLoss;
                bestScale = smallerScale;
            } else {
                break;
            }
        }

        double largerScale = 1.0;
        for (int step = 0; step < INITIAL_SCALE_SEARCH_EXPANSION_STEPS; step++) {
            largerScale *= INITIAL_SCALE_SEARCH_MULTIPLIER;
            double trialLoss = evaluateInitializationLossForScale(taxonCount, directPcaEmbeddings, largerScale);
            upperScale = largerScale;
            if (trialLoss < bestLoss) {
                bestLoss = trialLoss;
                bestScale = largerScale;
            } else {
                break;
            }
        }

        if (lowerScale == upperScale) {
            lowerScale = 1.0 / INITIAL_SCALE_SEARCH_MULTIPLIER;
            upperScale = INITIAL_SCALE_SEARCH_MULTIPLIER;
        }

        double bestLogScale = Math.log(bestScale);
        double logLower = Math.log(lowerScale);
        double logUpper = Math.log(upperScale);
        for (int step = 0; step < INITIAL_SCALE_SEARCH_REFINEMENT_STEPS; step++) {
            double leftLog = 0.5 * (logLower + bestLogScale);
            double rightLog = 0.5 * (bestLogScale + logUpper);
            double leftScale = Math.exp(leftLog);
            double rightScale = Math.exp(rightLog);
            double leftLoss = evaluateInitializationLossForScale(taxonCount, directPcaEmbeddings, leftScale);
            double rightLoss = evaluateInitializationLossForScale(taxonCount, directPcaEmbeddings, rightScale);

            if (leftLoss < bestLoss && leftLoss <= rightLoss) {
                logUpper = bestLogScale;
                bestLogScale = leftLog;
                bestScale = leftScale;
                bestLoss = leftLoss;
            } else if (rightLoss < bestLoss) {
                logLower = bestLogScale;
                bestLogScale = rightLog;
                bestScale = rightScale;
                bestLoss = rightLoss;
            } else if (leftLoss <= rightLoss) {
                logUpper = rightLog;
            } else {
                logLower = leftLog;
            }
        }

        return new InitialScaleSearchResult(bestScale, baselineLoss, bestLoss, scaleEmbeddings(directPcaEmbeddings, bestScale));
    }

    private double evaluateInitializationLossForScale(int taxonCount, double[] directPcaEmbeddings, double scale) {
        double[] scaledEmbeddings = scaleEmbeddings(directPcaEmbeddings, scale);
        LearnedWeightsTreeObjectiveFunction objective = new LearnedWeightsTreeObjectiveFunction(
            trainingSamples,
            taxonCount,
            dim,
            BROWNIAN_RATE,
            scaledEmbeddings,
            pcaRegularizationWeightInput.get(),
            slopeRegularizationWeightInput.get());
        return evaluateInitializationLoss(objective, scaledEmbeddings);
    }

    private double evaluateInitializationLoss(LearnedWeightsTreeObjectiveFunction objective, double[] embeddings) {
        double[] initialParameters = objective.createInitialParameters(embeddings);
        return objective.value(initialParameters);
    }

    private double[] scaleEmbeddings(double[] embeddings, double scale) {
        double[] scaled = new double[embeddings.length];
        for (int i = 0; i < embeddings.length; i++) {
            scaled[i] = scale * embeddings[i];
        }
        return scaled;
    }

    private record InitialScaleSearchResult(double scale, double baselineLoss, double scaledLoss, double[] scaledEmbeddings) {
    }

    public List<LearnedWeightsTreeTrainer.TrainingTreeSample> getTrainingSamples() {
        return trainingSamples;
    }

    public int getEmbeddingDimension() {
        return dim;
    }

    public double getDistanceOffset() {
        return offset;
    }

    public double getLearnedSlope() {
        return learnedSlope;
    }

    public double[] getLearnedSlopes() {
        double[] slopes = new double[dim];
        Arrays.fill(slopes, learnedSlope);
        return slopes;
    }

    public double getLearnedIntercept() {
        return learnedIntercept;
    }

    private String joinTrainingTreeFiles(List<File> treeFiles) {
        List<String> paths = new ArrayList<>(treeFiles.size());
        for (File treeFile : treeFiles) {
            paths.add(treeFile.getAbsolutePath());
        }
        return String.join(", ", paths);
    }

    private double l2Norm(double[] values) {
        double sumSquares = 0.0;
        for (double value : values) {
            sumSquares += value * value;
        }
        return Math.sqrt(sumSquares);
    }

    private double maxAbs(double[] values) {
        double max = 0.0;
        for (double value : values) {
            max = Math.max(max, Math.abs(value));
        }
        return max;
    }

    private double diffNorm(double[] left, double[] right) {
        if (left.length != right.length) {
            throw new IllegalArgumentException("vector length mismatch");
        }
        double sumSquares = 0.0;
        for (int i = 0; i < left.length; i++) {
            double delta = left[i] - right[i];
            sumSquares += delta * delta;
        }
        return Math.sqrt(sumSquares);
    }

    private int countNonZeroEntries(double[] values, double threshold) {
        int count = 0;
        for (double value : values) {
            if (Math.abs(value) > threshold) {
                count++;
            }
        }
        return count;
    }

    public boolean usesInverseMeanDistanceProposal() {
        return useInverseMeanDistanceProposalInput.get();
    }

    public double[] getCurrentLeafEmbeddingsForTaxa(String[] taxa) {
        return runtimeModel.getLeafEmbeddings(getLeafOrderForTaxa(treeInput.get(), taxa));
    }

    public double[] getPcaLeafEmbeddingsForTaxa(String[] taxa) {
        Map<String, double[]> map = Alignment2PCA.getPoints(dataInput.get(), dim, true, true);
        double[] theta = new double[taxa.length * dim];

        for (int i = 0; i < taxa.length; i++) {
            double[] point = map.get(taxa[i]);
            if (point == null) {
                throw new IllegalArgumentException("No PCA embedding found for taxon: " + taxa[i]);
            }
            System.arraycopy(point, 0, theta, i * dim, dim);
        }
        return theta;
    }

    private int[] getLeafOrderForTaxa(TreeInterface tree, String[] taxa) {
        int[] leafOrder = new int[taxa.length];
        for (int i = 0; i < taxa.length; i++) {
            leafOrder[i] = getLeafNodeNrByTaxon(tree, taxa[i]);
        }
        return leafOrder;
    }

    private int getLeafNodeNrByTaxon(TreeInterface tree, String taxon) {
        for (int i = 0; i < tree.getNodeCount(); i++) {
            Node node = tree.getNode(i);
            if (node.isLeaf() && taxon.equals(node.getID())) {
                return node.getNr();
            }
        }
        throw new IllegalArgumentException("Taxon not found among tree leaves: " + taxon);
    }

    @Override
    public void updateByOperator() {
        runtimeModel.updateByOperator();
    }

    @Override
    public void updateByOperatorWithoutNode(int ignore, List<Integer> nodes) {
        runtimeModel.updateByOperatorWithoutNode(ignore, nodes);
    }

    @Override
    public void fakeUpdateByOperator() {
        runtimeModel.fakeUpdateByOperator();
    }

    @Override
    protected boolean requiresRecalculation() {
        return dataInput.get().isDirtyCalculation() || treeInput.get().somethingIsDirty();
    }

    @Override
    public void store() {
        super.store();
        runtimeModel.store();
    }

    @Override
    public void prestore() {
        runtimeModel.prestore();
    }

    @Override
    public void reset() {
        runtimeModel.reset();
    }

    @Override
    public void restore() {
        super.restore();
        runtimeModel.restore();
    }

    public double getEdgeMutations(int i) {
        return runtimeModel.getEdgeWeights(i);
    }

    public boolean getChanged(int i) {
        return false;
    }

    public double[] getPoints(int nr) {
        return runtimeModel.getPoints(nr);
    }

    @Override
    public double getEdgeWeights(int nodeNr) {
        return runtimeModel.getEdgeWeights(nodeNr);
    }

    public double[] getTargetWeights(int fromNodeNr, List<Node> toNodeNrs) {
        return runtimeModel.getTargetWeights(fromNodeNr, toNodeNrs);
    }

    @Override
    public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
        return runtimeModel.getTargetWeightsInteger(fromNodeNr, toNodeNrs);
    }

    public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, double toHeight) {
        return runtimeModel.getTargetWeightsInteger(fromNodeNr, toNodeNrs, toHeight);
    }

    @Override
    public List<String> getArguments() {
        return Collections.emptyList();
    }

    @Override
    public List<String> getConditions() {
        return Collections.emptyList();
    }

    @Override
    public void sample(State state, Random random) {
    }

    @Override
    public void init(PrintStream out) {
        out.println("#NEXUS\n");
        out.println("Begin trees;");
    }

    @Override
    public void log(long sample, PrintStream out) {
        Tree tree = (Tree) treeInput.get();
        out.print("tree STATE_" + sample + " = ");
        out.print(toNewick(tree.getRoot()));
        out.print(";");
    }

    public String getTree() {
        Tree tree = (Tree) treeInput.get();
        return toNewick(tree.getRoot());
    }

    public String toNewick(Node n) {
        StringBuilder buf = new StringBuilder();
        if (!n.isLeaf()) {
            buf.append("(");
            boolean isFirst = true;
            for (Node child : n.getChildren()) {
                if (isFirst) {
                    isFirst = false;
                } else {
                    buf.append(",");
                }
                buf.append(toNewick(child));
            }
            buf.append(")");

            if (n.getID() != null) {
                buf.append(n.getID());
            }
        } else if (n.getID() != null) {
            buf.append(n.getID());
        }

        buf.append("[&sum=").append(getEdgeWeights(n.getNr())).append("]");
        buf.append(":").append(n.getLength());
        return buf.toString();
    }

    @Override
    public void close(PrintStream out) {
        out.print("End;");
    }

    @Override
    public double minEdgeWeight() {
        return runtimeModel.minEdgeWeight();
    }

    public double getBrownianRate() {
        return BROWNIAN_RATE;
    }
}
