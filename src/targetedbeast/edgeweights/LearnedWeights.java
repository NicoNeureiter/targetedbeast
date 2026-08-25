package targetedbeast.edgeweights;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
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
import targetedbeast.edgeweights.learnedweights.EmbeddingMLPSurrogate;
import targetedbeast.edgeweights.learnedweights.PCAWeightsMLPLikelihood;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsOptimizer;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsModelIO;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsStochasticOptimizer;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsTreeObjectiveFunction;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsTrainingSampleIO;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsTreeTrainer;
import targetedbeast.edgeweights.learnedweights.ProposalExporter;
import targetedbeast.util.Alignment2PCA;

@Description("Learns leaf embeddings so a Brownian surrogate tree likelihood tracks the full sequence tree likelihood. "
    + "Runtime delegates edge and target weights to PCAWeightsBrownianFullLikelihood.")
public class LearnedWeights extends Distribution implements EdgeWeights {

    private static final int INITIAL_SCALE_SEARCH_EXPANSION_STEPS = 8;
    private static final int INITIAL_SCALE_SEARCH_REFINEMENT_STEPS = 8;
    private static final double INITIAL_SCALE_SEARCH_MULTIPLIER = 2.0;
    private static final double BROWNIAN_RATE = 1.0;

    public static final String SKIP_LEARNING_SYSTEM_PROPERTY = "targetedbeast.learnedweights.skipLearning";
    public static final String TRAIN_QUADRATIC_ONLY_SYSTEM_PROPERTY = "targetedbeast.learnedweights.trainQuadraticOnly";
    public static final String PATH_BASE_DIR_SYSTEM_PROPERTY = "targetedbeast.learnedweights.pathBaseDir";
    public static final String LOAD_TRAINING_SAMPLES_SYSTEM_PROPERTY = "targetedbeast.learnedweights.loadTrainingSamples";

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
        final public Input<String> brownianTraitRatesInput = new Input<>(
            "brownianTraitRates",
            "comma or whitespace separated initial per-trait Brownian diffusion rates; when learning runs these seed the optimizer and default to 1.0 for every learned trait dimension",
            "");
        final public Input<Boolean> optimizeBrownianRatesInput = new Input<>(
            "optimizeBrownianRates",
            "if true (default), per-trait Brownian rates are tuned by the optimizer; if false they stay fixed at brownianTraitRates (or 1.0 if unset)",
            true);
    final public Input<String> rootPriorVariancesInput = new Input<>(
            "rootPriorVariances",
            "comma or whitespace separated initial per-trait root-state prior variances in Brownian diffusion units; when blank the scalar rootPriorVariance input is reused for every trait",
            "");
    final public Input<Double> offsetInput = new Input<>("offset", "offset in weight", 0.01);
        final public Input<Double> rootPriorVarianceInput = new Input<>(
            "rootPriorVariance",
            "root-state prior variance in Brownian diffusion units; use Infinity for a flat root prior and 0 for a root fixed at 0",
            Double.POSITIVE_INFINITY);

        final public Input<Integer> nTrainingTreesInput = new Input<>("nTrainingTrees", "number of training trees to use when generating samples from the prior or a posterior tree log", 50);
        final public Input<List<File>> trainingTreeFileInput = new Input<>(
            "trainingTreeFile",
            "optional posterior tree logs to use directly as the LearnedWeights training set; specify multiple trainingTreeFile entries in XML to combine multiple logs",
            new ArrayList<>());
        final public Input<List<File>> trainingLogFileInput = new Input<>(
            "trainingLogFile",
            "posterior log files paired with trainingTreeFile entries; these must contain the target tree-likelihood column used for training and any logged state needed when rates are not encoded in the trees",
            new ArrayList<>());
        final public Input<String> targetLogLikelihoodColumnInput = new Input<>(
            "targetLogLikelihoodColumn",
            "name of the posterior log column containing the target tree likelihood for LearnedWeights training; defaults to the referenced likelihood id",
            "");
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
        final public Input<String> trainedWeightsFileInput = new Input<>(
            "trainedWeightsFile",
            "optional path to cached trained LearnedWeights parameters; if the file exists it is loaded instead of retraining, otherwise learned parameters are written there after optimization",
            "");
        final public Input<Boolean> continueTrainingFromTrainedWeightsInput = new Input<>(
            "continueTrainingFromTrainedWeights",
            "if true and trainedWeightsFile exists, use the cached LearnedWeights parameters as the optimizer starting point instead of short-circuiting learning",
            false);
    final public Input<Integer> maxIterationsInput = new Input<>("maxIterations", "maximum number of optimizer iterations", 500);
    final public Input<Double> learningRateInput = new Input<>("learningRate", "optimizer step-size parameter for covariance-matrix training", 0.01);
    final public Input<Double> gradientToleranceInput = new Input<>("gradientTolerance", "optimizer stop threshold on the gradient norm", 1e-16);
        final public Input<Boolean> useStochasticOptimizerInput = new Input<>(
                "useStochasticOptimizer",
                "whether to use Adam-style stochastic optimization instead of nonlinear conjugate gradient; recommended for small mini-batches",
                true);
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
        final public Input<Boolean> separateBaseAndQuadraticCalibrationInput = new Input<>(
            "separateBaseAndQuadraticCalibration",
            "if true, learn separate per-axis calibration scales for Gaussian normalization/log-det terms and quadratic data-fit terms",
            false);
        final public Input<Boolean> scaleQuadraticPartOnlyInput = new Input<>(
            "scaleQuadraticPartOnly",
            "if true, regress on target minus the sample-specific log-det term and apply learned scales only to the quadratic term at runtime",
            false);
        final public Input<Boolean> trainQuadraticOnlyInput = new Input<>(
            "trainQuadraticOnly",
            "if true, training regresses the full target likelihood on quadratic surrogate contributions only; this is a training-only determinant ablation and does not change runtime scoring semantics",
            false);
        final public Input<Boolean> useLogSlopeParameterizationInput = new Input<>(
            "useLogSlopeParameterization",
            "if true, the affine calibration slope is parameterized as exp(logSlope) during optimization, "
                    + "enforcing positivity structurally with smooth gradients; "
                    + "when true the runtime proposal scale equals the learned slope directly (no sign flip); "
                    + "set to false (default) to use the current unconstrained-slope / sign-flip behavior",
            false);
    final public Input<GenericTreeLikelihood> likelihoodInput = new Input<>("likelihood", "tree likelihood template used to configure per-tree training likelihoods", Validate.REQUIRED);

    final public Input<String> surrogateWeightsFileInput = new Input<>(
        "surrogateWeightsFile",
        "path to a JSON weight file written by EmbeddingMLP.export_weights() in Python; "
        + "when provided, uses the pre-trained MLP surrogate for targeted operator proposal scoring");

    private double offset;
    private int dim;
    private double learnedBaseScale = 1.0;
    private double[] learnedBaseScales;
    private double learnedSlope = 1.0;
    private double[] learnedSlopes;
    private double runtimeProposalBaseScale = 1.0;
    private double[] runtimeProposalBaseScales;
    private double runtimeProposalScale = -1.0;
    private double[] runtimeProposalScales;
    private double learnedIntercept = 0.0;
    private PCAWeightsBrownianFullLikelihood runtimeModel;
    private List<LearnedWeightsTreeTrainer.TrainingTreeSample> trainingSamples = Collections.emptyList();
    private double[] brownianTraitRates;
    private double[] rootPriorVariances;
    private boolean optimizeRootPriorVariances;
    private Path trainedWeightsFile;
    private LearnedWeightsModelIO.ModelState warmStartModelState;
    @Override
    public void initAndValidate() {
        final TreeInterface tree = treeInput.get();
        final Alignment data = dataInput.get();
        final String value = valueInput.get();

        dim = dimensionInput.get();
        offset = offsetInput.get();
        learnedBaseScales = new double[dim];
        Arrays.fill(learnedBaseScales, 1.0);
        learnedSlopes = new double[dim];
        Arrays.fill(learnedSlopes, 1.0);
        runtimeProposalBaseScales = new double[dim];
        Arrays.fill(runtimeProposalBaseScales, 1.0);
        runtimeProposalScales = new double[dim];
        Arrays.fill(runtimeProposalScales, -1.0);
        brownianTraitRates = resolveBrownianTraitRates();
        rootPriorVariances = resolveRootPriorVariances();
        optimizeRootPriorVariances = areAllFinite(rootPriorVariances);

        if (dim <= 0) {
            throw new IllegalArgumentException("dimension must be > 0, but was " + dim);
        }
        if (offset < 0.0) {
            throw new IllegalArgumentException("offset must be >= 0, but was " + offset);
        }
        if (rootPriorVarianceInput.get() < 0.0 || Double.isNaN(rootPriorVarianceInput.get())) {
            throw new IllegalArgumentException("rootPriorVariance must be >= 0 or Infinity, but was " + rootPriorVarianceInput.get());
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
        if (scaleQuadraticPartOnlyInput.get() && trainQuadraticOnlyInput.get()) {
            throw new IllegalArgumentException("scaleQuadraticPartOnly and trainQuadraticOnly cannot both be enabled in XML");
        }
        if (separateBaseAndQuadraticCalibrationInput.get()
                && (scaleQuadraticPartOnlyInput.get() || trainQuadraticOnlyInput.get())) {
            throw new IllegalArgumentException(
                    "separateBaseAndQuadraticCalibration cannot be combined with scaleQuadraticPartOnly or trainQuadraticOnly");
        }
        if (!trainingTreeFileInput.get().isEmpty() && trainingLogFileInput.get().isEmpty()) {
            throw new IllegalArgumentException(
                    "LearnedWeights posterior-tree training requires paired trainingLogFile entries containing column '"
                            + getTargetLogLikelihoodColumn() + "'.");
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
        trainedWeightsFile = resolveTrainedWeightsFile();
        warmStartModelState = loadTrainedWeightsStateIfAvailable();
        if (warmStartModelState != null) {
            validateLoadedCalibrationMode(warmStartModelState);
            if (continueTrainingFromTrainedWeightsInput.get()) {
                if (value != null) {
                    throw new IllegalArgumentException(
                            "continueTrainingFromTrainedWeights requires LearnedWeights to run in learning mode, but value was provided explicitly.");
                }
                validateWarmStartCompatibility(warmStartModelState, data.getTaxonCount());
            } else {
                if (Boolean.getBoolean(LOAD_TRAINING_SAMPLES_SYSTEM_PROPERTY)) {
                    trainingSamples = loadOrGenerateTrainingSamples();
                }
                applyTrainedWeightsState(tree, warmStartModelState, true);
                runtimeModel.updateByOperator();
                loadMlpSurrogateIfConfigured(tree);
                return;
            }
        }
        String surrogateConfigured = surrogateWeightsFileInput.get();
        boolean hasSurrogate = surrogateConfigured != null && !surrogateConfigured.isBlank();
        if (value == null && !hasSurrogate) {
            runLearningPhase();
        } else if (value == null) {
            Log.info.println("LearnedWeights: surrogateWeightsFile provided — skipping learning phase, using default Brownian rates.");
        }
        runtimeModel.updateByOperator();
        loadMlpSurrogateIfConfigured(tree);
    }

    private PCAWeightsBrownianFullLikelihood createRuntimeModel(TreeInterface tree, String value) {
        BranchRateModel.Base branchRateModel = likelihoodInput.get().branchRateModelInput.get();
        String surrogateConfigured = surrogateWeightsFileInput.get();
        boolean hasSurrogate = surrogateConfigured != null && !surrogateConfigured.isBlank();
        PCAWeightsBrownianFullLikelihood model = hasSurrogate
            ? new PCAWeightsMLPLikelihood()
            : new PCAWeightsBrownianFullLikelihood();
        model.initByName(
            "data", dataInput.get(),
            "tree", tree,
            "maxWeight", maxWeightInput.get(),
            "minWeight", minWeightInput.get(),
            "dimension", dim,
            "brownianRate", BROWNIAN_RATE,
            "brownianTraitRates", formatBrownianTraitRates(brownianTraitRates),
            "rootPriorVariances", formatRootPriorVariances(rootPriorVariances),
            "rootPriorVariance", rootPriorVarianceInput.get(),
            "useInverseMeanDistanceProposal", useInverseMeanDistanceProposalInput.get(),
            "separateBaseAndQuadraticCalibration", separateBaseAndQuadraticCalibrationInput.get(),
            "scaleQuadraticPartOnly", scaleQuadraticPartOnlyInput.get(),
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
        Log.info.println("LearnedWeights: training mode = covariance regression with affine calibration"
            + (separateBaseAndQuadraticCalibrationInput.get()
                ? " (separate base/log-det and quadratic calibration scales)"
                : "")
            + (scaleQuadraticPartOnlyInput.get() ? " (log-det moved to target side; slopes scale quadratic term only)" : "")
            + (trainsQuadraticOnly() ? " (quadratic-only training ablation; target kept unchanged while log-det is removed from regression features)" : ""));

        if (trainingSamples.isEmpty()) {
            Log.warning("LearnedWeights: no training samples generated; keeping PCA initialization.");
            return;
        }

        String[] canonicalTaxa = dataInput.get().getTaxaNames().toArray(new String[0]);
        int[] runtimeLeafOrder = getLeafOrderForTaxa(treeInput.get(), canonicalTaxa);
        double[] directPcaEmbeddings = getPcaLeafEmbeddingsForTaxa(canonicalTaxa);
        InitialScaleSearchResult initialScaleSearch = optimizeInitialEmbeddingScale(canonicalTaxa.length, directPcaEmbeddings);
        double[] referenceEmbeddings = initialScaleSearch.scaledEmbeddings;
        double[] initialEmbeddings = warmStartModelState == null ? referenceEmbeddings : warmStartModelState.leafEmbeddings();
        double[] objectiveBrownianTraitRates = warmStartModelState == null
            ? brownianTraitRates
            : warmStartModelState.brownianTraitRates();
        double[] objectiveRootPriorVariances = optimizeRootPriorVariances
            ? (warmStartModelState == null ? rootPriorVariances : warmStartModelState.rootPriorVariances())
            : rootPriorVariances;

        LearnedWeightsTreeObjectiveFunction reportingObjective = new LearnedWeightsTreeObjectiveFunction(
            trainingSamples,
            canonicalTaxa.length,
            dim,
            objectiveBrownianTraitRates,
            optimizeRootPriorVariances ? objectiveRootPriorVariances : null,
            referenceEmbeddings,
            pcaRegularizationWeightInput.get(),
            slopeRegularizationWeightInput.get(),
            0,
            1L,
            useLogSlopeParameterizationInput.get(),
            separateBaseAndQuadraticCalibrationInput.get(),
            scaleQuadraticPartOnlyInput.get(),
            trainsQuadraticOnly());
        LearnedWeightsTreeObjectiveFunction optimizationObjective = new LearnedWeightsTreeObjectiveFunction(
            trainingSamples,
            canonicalTaxa.length,
            dim,
            objectiveBrownianTraitRates,
            optimizeRootPriorVariances ? objectiveRootPriorVariances : null,
            referenceEmbeddings,
            pcaRegularizationWeightInput.get(),
            slopeRegularizationWeightInput.get(),
            miniBatchSizeInput.get(),
            miniBatchSeedInput.get(),
            useLogSlopeParameterizationInput.get(),
            separateBaseAndQuadraticCalibrationInput.get(),
            scaleQuadraticPartOnlyInput.get(),
            trainsQuadraticOnly());
        boolean optimizeBrownianRates = optimizeBrownianRatesInput.get();
        reportingObjective.setOptimizeBrownianRates(optimizeBrownianRates);
        optimizationObjective.setOptimizeBrownianRates(optimizeBrownianRates);
            double[] initialParameters = warmStartModelState == null
                ? reportingObjective.createInitialParameters(initialEmbeddings)
                : reportingObjective.createParameters(
                    warmStartModelState.leafEmbeddings(),
                    warmStartModelState.brownianTraitRates(),
                    optimizeRootPriorVariances ? warmStartModelState.rootPriorVariances() : null,
                    resolveWarmStartBaseScales(warmStartModelState),
                    warmStartModelState.quadraticScales(),
                    warmStartModelState.intercept());
        double[] initialBrownianTraitRates = reportingObjective.extractBrownianRates(initialParameters);
        double[] initialRootPriorVariances = reportingObjective.extractRootPriorVariances(initialParameters);
        double initialBaseScale = reportingObjective.extractBaseScale(initialParameters);
        double[] initialBaseScales = reportingObjective.extractBaseScales(initialParameters);
        double initialSlope = reportingObjective.extractSlope(initialParameters);
        double[] initialSlopes = reportingObjective.extractSlopes(initialParameters);
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
        double[] surrogateStats = reportingObjective.computeSurrogateVarianceDecomposition(initialEmbeddings, initialRootPriorVariances);

        boolean useStochasticOptimizer = useStochasticOptimizerInput.get();
        String optimizerName = useStochasticOptimizer ? "adam" : "nonlinear-conjugate-gradient";

        Log.info.println(String.format(
            "LearnedWeights: starting covariance optimization with optimizer=%s, %d embedding parameters (+ intercept + %d log-rates%s + %d calibration parameters), %d training trees, maxIter=%d, learningRate=%.4g, gradientTolerance=%.3g, pcaRegularizationWeight=%.4g, slopeRegularizationWeight=%.4g",
                optimizerName,
                reportingObjective.getEmbeddingParameterCount(),
                dim,
                optimizeRootPriorVariances ? " + " + dim + " log-root-variances" : "",
                separateBaseAndQuadraticCalibrationInput.get() ? 2 * dim : dim,
                trainingSamples.size(),
                maxIterationsInput.get(),
                learningRateInput.get(),
                gradientToleranceInput.get(),
            pcaRegularizationWeightInput.get(),
            slopeRegularizationWeightInput.get()));
        if (warmStartModelState == null) {
            Log.info.println("LearnedWeights: optimizer starts from PCA embeddings with affine calibration fitted on the initial surrogate.");
        } else {
            Log.info.println("LearnedWeights: optimizer warm-starts from trainedWeightsFile parameters while keeping the current XML training objective.");
        }
        Log.info.println(String.format(
            "LearnedWeights: initial objective preview loss=%.6f, fullGradientNorm=%.6g, embeddingGradientNorm=%.6g, calibrationGradientNorm=%.6g, embeddingNorm=%.6g, embeddingMaxAbs=%.6g, embeddingNonZero=%d, initialBaseScale=%.6g, initialQuadraticScale=%.6g, initialIntercept=%.6g",
                initialPreviewLoss,
                initialGradientNorm,
                initialEmbeddingGradientNorm,
            initialCalibrationGradientNorm,
            initialEmbeddingNorm,
            initialEmbeddingMaxAbs,
            initialEmbeddingNonZeroCount,
            initialBaseScale,
            initialSlope,
            initialIntercept));
        Log.info.println("LearnedWeights: initial per-dimension Brownian rates=" + Arrays.toString(initialBrownianTraitRates));
        if (initialRootPriorVariances != null) {
            Log.info.println("LearnedWeights: initial per-dimension root prior variances=" + Arrays.toString(initialRootPriorVariances));
        }
        Log.info.println("LearnedWeights: initial per-dimension base scales=" + Arrays.toString(initialBaseScales));
        Log.info.println("LearnedWeights: initial per-dimension slopes=" + Arrays.toString(initialSlopes));
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
        if (useStochasticOptimizer) {
            Log.info.println(
                "LearnedWeights: progress lines report periodic full-objective checkpoints.");
        } else {
            Log.info.println(
                optimizationObjective.usesMiniBatches()
                    ? "LearnedWeights: progress lines report batch-specific objective evaluations; trialLoss values are not directly comparable across batches."
                    : "LearnedWeights: progress lines report raw line-search objective evaluations; trialLoss can spike even when bestSeenLoss improves.");
        }

        reportingObjective.beginOptimizationRun("LearnedWeights", maxIterationsInput.get());
        optimizationObjective.setProgressLoggingEnabled(!useStochasticOptimizer);
        optimizationObjective.beginOptimizationRun("LearnedWeights", maxIterationsInput.get());
        LearnedWeightsOptimizer.Result result;
        if (useStochasticOptimizer) {
            LearnedWeightsStochasticOptimizer optimizer = new LearnedWeightsStochasticOptimizer(
                    new LearnedWeightsStochasticOptimizer.Config()
                            .withMaxIter(maxIterationsInput.get())
                            .withLearningRate(learningRateInput.get())
                            .withGradientTolerance(gradientToleranceInput.get()));
            result = optimizer.optimize(initialParameters, optimizationObjective, reportingObjective);
        } else {
            LearnedWeightsOptimizer optimizer = new LearnedWeightsOptimizer(
                    new LearnedWeightsOptimizer.Config()
                            .withMaxIter(maxIterationsInput.get())
                            .withLearningRate(learningRateInput.get())
                            .withGradientTolerance(gradientToleranceInput.get()));
            result = optimizer.optimize(initialParameters, optimizationObjective);
        }

        double[] finalGradient = new double[reportingObjective.getParameterCount()];
        double fullFinalLoss = reportingObjective.valueAndGradient(result.optimizedTheta, finalGradient);
        double fullFinalGradientNorm = reportingObjective.gradientNorm(finalGradient);

        brownianTraitRates = reportingObjective.extractBrownianRates(result.optimizedTheta);
        double[] optimizedRootPriorVariances = reportingObjective.extractRootPriorVariances(result.optimizedTheta);
        if (optimizedRootPriorVariances != null) {
            rootPriorVariances = optimizedRootPriorVariances;
        }
        learnedBaseScales = reportingObjective.extractBaseScales(result.optimizedTheta);
        learnedBaseScale = mean(learnedBaseScales);
        runtimeProposalBaseScales = resolveRuntimeProposalScales(learnedBaseScales);
        runtimeProposalBaseScale = mean(runtimeProposalBaseScales);
        learnedSlopes = reportingObjective.extractSlopes(result.optimizedTheta);
        learnedSlope = mean(learnedSlopes);
        runtimeProposalScales = resolveRuntimeProposalScales(learnedSlopes);
        runtimeProposalScale = mean(runtimeProposalScales);
        learnedIntercept = reportingObjective.extractIntercept(result.optimizedTheta);
        double[] optimizedEmbeddings = reportingObjective.extractEmbeddings(result.optimizedTheta);
        double optimizedEmbeddingNorm = l2Norm(optimizedEmbeddings);
        double optimizedEmbeddingMaxAbs = maxAbs(optimizedEmbeddings);
        int optimizedEmbeddingNonZeroCount = countNonZeroEntries(optimizedEmbeddings, 1e-12);
        double optimizedVsInitialDiffNorm = diffNorm(optimizedEmbeddings, initialEmbeddings);
        double[] optimizedSurrogateStats = reportingObjective.computeSurrogateVarianceDecomposition(optimizedEmbeddings, optimizedRootPriorVariances);
        runtimeModel.setBrownianTraitRates(brownianTraitRates);
        runtimeModel.setRootPriorVariances(rootPriorVariances);
        runtimeModel.setProposalBaseLogLikelihoodScales(runtimeProposalBaseScales);
        runtimeModel.setProposalLogLikelihoodScales(runtimeProposalScales);
        runtimeModel.setLeafEmbeddings(optimizedEmbeddings, runtimeLeafOrder);
        saveTrainedWeightsIfRequested(optimizedEmbeddings);

        double elapsedSec = (System.nanoTime() - startNanos) / 1_000_000_000.0;
        Log.info.println(String.format(
            "LearnedWeights: optimization %s in %.3fs (stopReason=%s, iter=%d, eval=%d, objectiveEvalCalls=%d, initial=%.6f, final=%.6f, delta=%.6f, initialGradientNorm=%.6g, finalGradientNorm=%.6g, fitLoss=%.6f, slopePenalty=%.6f, pcaPenalty=%.6f, baseScale=%.6f, quadraticScale=%.6f, baseRuntimeScale=%.6f, runtimeScale=%.6f, intercept=%.6f)",
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
            mean(learnedBaseScales),
            mean(learnedSlopes),
            runtimeProposalBaseScale,
            runtimeProposalScale,
            learnedIntercept));
        Log.info.println("LearnedWeights: optimized per-dimension Brownian rates=" + Arrays.toString(brownianTraitRates));
        if (optimizedRootPriorVariances != null) {
            Log.info.println("LearnedWeights: optimized per-dimension root prior variances=" + Arrays.toString(rootPriorVariances));
        }
        Log.info.println("LearnedWeights: optimized per-dimension base scales=" + Arrays.toString(learnedBaseScales));
        Log.info.println("LearnedWeights: optimized per-dimension slopes=" + Arrays.toString(learnedSlopes));
        Log.info.println("LearnedWeights: runtime per-dimension base proposal scales=" + Arrays.toString(runtimeProposalBaseScales));
        Log.info.println("LearnedWeights: runtime per-dimension proposal scales=" + Arrays.toString(runtimeProposalScales));
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
        List<File> treeFiles = resolveConfiguredFiles(trainingTreeFileInput.get());
        if (!treeFiles.isEmpty()) {
            List<File> logFiles = resolveConfiguredFiles(trainingLogFileInput.get());
            LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                    likelihoodInput.get(),
                    dataInput.get(),
                    nTrainingTreesInput.get(),
                    rootPriorVarianceInput.get(),
                    getTargetLogLikelihoodColumn());
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
            Path path = resolveConfiguredPath(trainingSamplesFile);
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
                nTrainingTreesInput.get(),
            rootPriorVarianceInput.get(),
            getTargetLogLikelihoodColumn());
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = trainer.generateTrainingSamples();
        Log.info.println("LearnedWeights: generated " + samples.size() + " prior-sampled training trees.");
        return samples;
    }

    private double resolveRuntimeProposalScale(double slope) {
        if (!Double.isFinite(slope)) {
            throw new IllegalStateException(String.format(
                    "LearnedWeights: learned slope %.6f is not finite; cannot derive runtime proposal scale",
                    slope));
        }
        if (useLogSlopeParameterizationInput.get() ) {
            // Log-slope parameterization enforces slope > 0 during optimization;
            // the learned slope is positive and maps directly to the proposal scale.
            return slope;
        }
        return slope;
    }

    private double[] resolveRuntimeProposalScales(double[] slopes) {
        double[] runtimeScales = new double[slopes.length];
        for (int i = 0; i < slopes.length; i++) {
            runtimeScales[i] = resolveRuntimeProposalScale(slopes[i]);
        }
        return runtimeScales;
    }

    private InitialScaleSearchResult optimizeInitialEmbeddingScale(int taxonCount, double[] directPcaEmbeddings) {
        LearnedWeightsTreeObjectiveFunction baselineObjective = new LearnedWeightsTreeObjectiveFunction(
            trainingSamples,
            taxonCount,
            dim,
            brownianTraitRates,
            directPcaEmbeddings,
            pcaRegularizationWeightInput.get(),
            slopeRegularizationWeightInput.get(),
            0,
            1L,
            useLogSlopeParameterizationInput.get(),
            scaleQuadraticPartOnlyInput.get(),
            trainsQuadraticOnly());
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
            brownianTraitRates,
            scaledEmbeddings,
            pcaRegularizationWeightInput.get(),
            slopeRegularizationWeightInput.get(),
            0,
            1L,
            useLogSlopeParameterizationInput.get(),
            scaleQuadraticPartOnlyInput.get(),
            trainsQuadraticOnly());
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

    public double getBrownianRate() {
        return mean(brownianTraitRates);
    }

    public double[] getBrownianTraitRates() {
        return Arrays.copyOf(brownianTraitRates, brownianTraitRates.length);
    }

    public double getPcaRegularizationWeight() {
        return pcaRegularizationWeightInput.get();
    }

    public double getSlopeRegularizationWeight() {
        return slopeRegularizationWeightInput.get();
    }

    public String getTargetLogLikelihoodColumn() {
        String configured = targetLogLikelihoodColumnInput.get();
        if (configured != null && !configured.isBlank()) {
            return configured;
        }
        String likelihoodId = likelihoodInput.get().getID();
        return likelihoodId != null && !likelihoodId.isBlank() ? likelihoodId : "treeLikelihood";
    }

    public double getLearnedSlope() {
        return learnedSlope;
    }

    public double getLearnedBaseScale() {
        return learnedBaseScale;
    }

    public double[] getLearnedBaseScales() {
        return Arrays.copyOf(learnedBaseScales, learnedBaseScales.length);
    }

    public double[] getLearnedSlopes() {
        return Arrays.copyOf(learnedSlopes, learnedSlopes.length);
    }

    public double getRuntimeProposalScale() {
        return runtimeProposalScale;
    }

    public double getRuntimeProposalBaseScale() {
        return runtimeProposalBaseScale;
    }

    public double[] getRuntimeProposalBaseScales() {
        return Arrays.copyOf(runtimeProposalBaseScales, runtimeProposalBaseScales.length);
    }

    public double[] getRuntimeProposalScales() {
        return Arrays.copyOf(runtimeProposalScales, runtimeProposalScales.length);
    }

    public boolean usesSeparateBaseAndQuadraticCalibration() {
        return separateBaseAndQuadraticCalibrationInput.get();
    }

    public boolean scalesQuadraticPartOnly() {
        return scaleQuadraticPartOnlyInput.get();
    }

    public boolean trainsQuadraticOnly() {
        return trainQuadraticOnlyInput.get() || Boolean.getBoolean(TRAIN_QUADRATIC_ONLY_SYSTEM_PROPERTY);
    }

    public boolean usesLogSlopeParameterization() {
        return useLogSlopeParameterizationInput.get();
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

    private double[] resolveBrownianTraitRates() {
        String configured = brownianTraitRatesInput.get();
        if (configured == null || configured.isBlank()) {
            double[] rates = new double[dim];
            Arrays.fill(rates, BROWNIAN_RATE);
            return rates;
        }

        String[] tokens = configured.trim().split("[\\s,]+");
        if (tokens.length != dim) {
            throw new IllegalArgumentException("brownianTraitRates length mismatch: expected "
                    + dim + " entries but got " + tokens.length);
        }

        double[] rates = new double[dim];
        for (int axis = 0; axis < dim; axis++) {
            rates[axis] = Double.parseDouble(tokens[axis]);
            if (!(rates[axis] > 0.0) || !Double.isFinite(rates[axis])) {
                throw new IllegalArgumentException("brownianTraitRates must all be finite and > 0, but axis "
                        + axis + " was " + rates[axis]);
            }
        }
        return rates;
    }

    private double[] resolveRootPriorVariances() {
        String configured = rootPriorVariancesInput.get();
        double[] variances = new double[dim];
        if (configured == null || configured.isBlank()) {
            double rootPriorVariance = rootPriorVarianceInput.get();
            if (rootPriorVariance < 0.0 || Double.isNaN(rootPriorVariance)) {
                throw new IllegalArgumentException("rootPriorVariance must be >= 0 or Infinity, but was " + rootPriorVariance);
            }
            Arrays.fill(variances, rootPriorVariance);
            return variances;
        }

        String[] tokens = configured.trim().split("[\\s,]+");
        if (tokens.length != dim) {
            throw new IllegalArgumentException("rootPriorVariances length mismatch: expected "
                    + dim + " entries but got " + tokens.length);
        }

        boolean anyFinite = false;
        boolean anyInfinite = false;
        for (int axis = 0; axis < dim; axis++) {
            variances[axis] = Double.parseDouble(tokens[axis]);
            if ((variances[axis] < 0.0 && !Double.isInfinite(variances[axis])) || Double.isNaN(variances[axis])) {
                throw new IllegalArgumentException("rootPriorVariances must all be >= 0 or Infinity, but axis "
                        + axis + " was " + variances[axis]);
            }
            if (Double.isInfinite(variances[axis])) {
                anyInfinite = true;
            } else {
                anyFinite = true;
            }
        }
        if (anyFinite && anyInfinite) {
            throw new IllegalArgumentException("rootPriorVariances must currently be either all finite or all Infinity");
        }
        return variances;
    }

    private boolean areAllFinite(double[] values) {
        for (double value : values) {
            if (!Double.isFinite(value)) {
                return false;
            }
        }
        return true;
    }

    private String formatBrownianTraitRates(double[] rates) {
        StringBuilder builder = new StringBuilder();
        for (int axis = 0; axis < rates.length; axis++) {
            if (axis > 0) {
                builder.append(' ');
            }
            builder.append(rates[axis]);
        }
        return builder.toString();
    }

    private String formatRootPriorVariances(double[] variances) {
        StringBuilder builder = new StringBuilder();
        for (int axis = 0; axis < variances.length; axis++) {
            if (axis > 0) {
                builder.append(' ');
            }
            builder.append(variances[axis]);
        }
        return builder.toString();
    }

    private List<File> resolveConfiguredFiles(List<File> files) {
        if (files == null || files.isEmpty()) {
            return Collections.emptyList();
        }

        List<File> resolved = new ArrayList<>(files.size());
        for (File file : files) {
            resolved.add(resolveConfiguredFile(file));
        }
        return resolved;
    }

    private File resolveConfiguredFile(File file) {
        if (file == null) {
            return null;
        }
        if (file.isAbsolute()) {
            return file.getAbsoluteFile();
        }

        Path baseDir = getConfiguredPathBaseDir();
        if (baseDir == null) {
            return file.getAbsoluteFile();
        }
        return baseDir.resolve(file.getPath()).normalize().toFile();
    }

    private Path resolveConfiguredPath(String configuredPath) {
        Path path = Paths.get(configuredPath);
        if (path.isAbsolute()) {
            return path.normalize();
        }

        Path baseDir = getConfiguredPathBaseDir();
        if (baseDir == null) {
            return path;
        }
        return baseDir.resolve(path).normalize();
    }

    private Path getConfiguredPathBaseDir() {
        String configuredBaseDir = System.getProperty(PATH_BASE_DIR_SYSTEM_PROPERTY);
        if (configuredBaseDir == null || configuredBaseDir.isBlank()) {
            return null;
        }
        return Paths.get(configuredBaseDir).toAbsolutePath().normalize();
    }

    private Path resolveTrainedWeightsFile() {
        String configured = trainedWeightsFileInput.get();
        if (configured == null || configured.isBlank()) {
            return null;
        }
        return resolveConfiguredPath(configured);
    }

    // ------------------------------------------------------------------
    // MLP surrogate loading
    // ------------------------------------------------------------------

    /**
     * If a surrogate weights JSON path is configured, load it, validate taxa
     * and dimension against the BEAST tree, then hand the surrogate plus the
     * taxon-index mapping to the runtime model (which must be a
     * {@link PCAWeightsMLPLikelihood} — created by {@link #createRuntimeModel}).
     * Also injects the surrogate's leaf embeddings into the pruning model so
     * that {@code getEdgeWeights} and {@code getTargetWeights*} share E.
     */
    private void loadMlpSurrogateIfConfigured(TreeInterface tree) {
        String configured = surrogateWeightsFileInput.get();
        if (configured == null || configured.isBlank()) {
            return;
        }
        if (!(runtimeModel instanceof PCAWeightsMLPLikelihood)) {
            throw new IllegalStateException(
                "surrogateWeightsFile is set but runtimeModel is not a PCAWeightsMLPLikelihood. "
                + "This is a programming error in createRuntimeModel().");
        }
        Path path = resolveConfiguredPath(configured);
        EmbeddingMLPSurrogate surrogate;
        try {
            surrogate = EmbeddingMLPSurrogate.fromFile(path);
        } catch (IOException e) {
            throw new IllegalArgumentException(
                "Failed to load MLP surrogate weights from " + path.toAbsolutePath(), e);
        }

        // MLP dim and Java dimension index the same E — they must agree.
        if (surrogate.dim() != dim) {
            throw new IllegalArgumentException(
                "MLP surrogate has dim=" + surrogate.dim()
                + " but LearnedWeights has dimension=" + dim
                + ". Set dimension=\"" + surrogate.dim() + "\" in the XML.");
        }

        // Map MLP taxon names to BEAST leaf node numbers.  Only used for
        // injecting E into the pruning model via setLeafEmbeddings — the
        // runtime MLP override no longer needs a taxon-index lookup of its own.
        Map<String, Integer> taxonToNodeNr = new HashMap<>();
        for (Node node : ((Tree) tree).getNodesAsArray()) {
            if (node.isLeaf()) {
                taxonToNodeNr.put(node.getID(), node.getNr());
            }
        }
        String[] mlpTaxa = surrogate.taxa();
        int[] mlpLeafNodeNrs = new int[mlpTaxa.length];
        for (int i = 0; i < mlpTaxa.length; i++) {
            Integer nr = taxonToNodeNr.get(mlpTaxa[i]);
            if (nr == null) {
                throw new IllegalArgumentException(
                    "MLP surrogate taxon '" + mlpTaxa[i] + "' not found in the BEAST tree");
            }
            mlpLeafNodeNrs[i] = nr;
        }

        // Wire the surrogate into the runtime model and inject E into the
        // pruning model so getEdgeWeights and getTargetWeights* share E.
        ((PCAWeightsMLPLikelihood) runtimeModel).setSurrogate(surrogate);
        runtimeModel.setLeafEmbeddings(surrogate.eFlat(), mlpLeafNodeNrs);
        runtimeModel.updateByOperator();

        Log.info.println("LearnedWeights: loaded MLP surrogate from " + path.toAbsolutePath()
            + " (dim=" + surrogate.dim() + ", nTaxa=" + surrogate.nTaxa() + ")"
            + "; surrogate embeddings injected into pruning model.");

        configureProposalExporterIfRequested(tree);
    }

    /**
     * If {@code -Dtargetedbeast.surrogate.proposalexport.tsv=<path>} is set,
     * wire a ProposalExporter into the runtime MLP model so every scored
     * proposal is dumped (precision matrix, true likelihood) into a V4
     * training-sample TSV.  Off by default.
     */
    private void configureProposalExporterIfRequested(TreeInterface tree) {
        String path = System.getProperty(ProposalExporter.EXPORT_PATH_PROP);
        if (path == null || path.isBlank()) return;
        if (!(tree instanceof Tree mutableTree)) {
            Log.warning.println("ProposalExporter: tree is not a mutable Tree — exporter disabled.");
            return;
        }
        Alignment alignment = dataInput.get();
        LearnedWeightsTreeTrainer exporterTrainer = new LearnedWeightsTreeTrainer(
                likelihoodInput.get(),
                alignment,
                0,
                getRootPriorVariance(),
                getTargetLogLikelihoodColumn());
        String[] canonicalTaxa = alignment.getTaxaNames().toArray(new String[0]);
        ProposalExporter exporter = ProposalExporter.createFromSystemProperties(
                mutableTree, exporterTrainer, canonicalTaxa);
        if (exporter != null) {
            ((PCAWeightsMLPLikelihood) runtimeModel).setProposalExporter(exporter);
        }
    }

    private LearnedWeightsModelIO.ModelState loadTrainedWeightsStateIfAvailable() {
        if (trainedWeightsFile == null || !Files.exists(trainedWeightsFile)) {
            return null;
        }

        try {
            String[] canonicalTaxa = dataInput.get().getTaxaNames().toArray(new String[0]);
            return LearnedWeightsModelIO.readModel(trainedWeightsFile, canonicalTaxa);
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to load trained weights from " + trainedWeightsFile.toAbsolutePath(), e);
        }
    }

    private void validateLoadedCalibrationMode(LearnedWeightsModelIO.ModelState learnedState) {
        if (learnedState.hasExplicitCalibrationMode()
                && learnedState.separateBaseAndQuadraticCalibration() != separateBaseAndQuadraticCalibrationInput.get()) {
            throw new IllegalArgumentException(
                    "LearnedWeights model file was trained with separateBaseAndQuadraticCalibration="
                            + learnedState.separateBaseAndQuadraticCalibration()
                            + " but current XML sets it to " + separateBaseAndQuadraticCalibrationInput.get());
        }
    }

    private void validateWarmStartCompatibility(LearnedWeightsModelIO.ModelState learnedState, int taxonCount) {
        if (learnedState.brownianTraitRates().length != dim) {
            throw new IllegalArgumentException(
                    "LearnedWeights trained model dimension mismatch: XML dimension=" + dim
                            + " but trainedWeightsFile stores " + learnedState.brownianTraitRates().length + " Brownian trait rates.");
        }
        if (learnedState.leafEmbeddings().length != taxonCount * dim) {
            throw new IllegalArgumentException(
                    "LearnedWeights trained model embedding size mismatch: expected " + (taxonCount * dim)
                            + " values for " + taxonCount + " taxa x dimension " + dim
                            + " but trainedWeightsFile stores " + learnedState.leafEmbeddings().length + ".");
        }
        if (learnedState.quadraticScales().length != dim) {
            throw new IllegalArgumentException(
                    "LearnedWeights trained model quadratic scale length mismatch: expected " + dim
                            + " but trainedWeightsFile stores " + learnedState.quadraticScales().length + ".");
        }
        if (separateBaseAndQuadraticCalibrationInput.get()) {
            double[] baseScales = resolveWarmStartBaseScales(learnedState);
            if (baseScales.length != dim) {
                throw new IllegalArgumentException(
                        "LearnedWeights trained model base scale length mismatch: expected " + dim
                                + " but trainedWeightsFile stores " + baseScales.length + ".");
            }
        }
        boolean trainedModelOptimizesRootPriorVariances = areAllFinite(learnedState.rootPriorVariances());
        if (optimizeRootPriorVariances != trainedModelOptimizesRootPriorVariances) {
            throw new IllegalArgumentException(
                    "LearnedWeights training objective mismatch for rootPriorVariances: XML optimizeRootPriorVariances="
                            + optimizeRootPriorVariances
                            + " but trainedWeightsFile implies " + trainedModelOptimizesRootPriorVariances + ".");
        }
        if (learnedState.hasExplicitTrainingObjectiveMetadata()) {
            if (learnedState.scaleQuadraticPartOnly() != scaleQuadraticPartOnlyInput.get()) {
                throw new IllegalArgumentException(
                        "LearnedWeights model file was trained with scaleQuadraticPartOnly="
                                + learnedState.scaleQuadraticPartOnly()
                                + " but current XML sets it to " + scaleQuadraticPartOnlyInput.get());
            }
            if (learnedState.trainQuadraticOnly() != trainsQuadraticOnly()) {
                throw new IllegalArgumentException(
                        "LearnedWeights model file was trained with trainQuadraticOnly="
                                + learnedState.trainQuadraticOnly()
                                + " but current XML sets it to " + trainsQuadraticOnly());
            }
            if (Double.compare(learnedState.pcaRegularizationWeight(), pcaRegularizationWeightInput.get()) != 0) {
                throw new IllegalArgumentException(
                        "LearnedWeights model file was trained with pcaRegularizationWeight="
                                + learnedState.pcaRegularizationWeight()
                                + " but current XML sets it to " + pcaRegularizationWeightInput.get());
            }
            if (Double.compare(learnedState.slopeRegularizationWeight(), slopeRegularizationWeightInput.get()) != 0) {
                throw new IllegalArgumentException(
                        "LearnedWeights model file was trained with slopeRegularizationWeight="
                                + learnedState.slopeRegularizationWeight()
                                + " but current XML sets it to " + slopeRegularizationWeightInput.get());
            }
        }
    }

    private void applyTrainedWeightsState(TreeInterface tree, LearnedWeightsModelIO.ModelState learnedState, boolean logLoadedState) {
        String[] canonicalTaxa = dataInput.get().getTaxaNames().toArray(new String[0]);
        brownianTraitRates = learnedState.brownianTraitRates();
        if (learnedState.rootPriorVariances() != null) {
            rootPriorVariances = learnedState.rootPriorVariances();
            optimizeRootPriorVariances = areAllFinite(rootPriorVariances);
        }
        learnedBaseScales = resolveWarmStartBaseScales(learnedState);
        learnedBaseScale = mean(learnedBaseScales);
        runtimeProposalBaseScales = resolveRuntimeProposalScales(learnedBaseScales);
        runtimeProposalBaseScale = mean(runtimeProposalBaseScales);
        learnedSlopes = learnedState.quadraticScales();
        learnedSlope = mean(learnedSlopes);
        runtimeProposalScales = resolveRuntimeProposalScales(learnedSlopes);
        runtimeProposalScale = mean(runtimeProposalScales);
        learnedIntercept = learnedState.intercept();
        runtimeModel.setBrownianTraitRates(brownianTraitRates);
        runtimeModel.setRootPriorVariances(rootPriorVariances);
        runtimeModel.setProposalBaseLogLikelihoodScales(runtimeProposalBaseScales);
        runtimeModel.setProposalLogLikelihoodScales(runtimeProposalScales);
        runtimeModel.setLeafEmbeddings(learnedState.leafEmbeddings(), getLeafOrderForTaxa(tree, canonicalTaxa));
        if (logLoadedState) {
            Log.info.println("LearnedWeights: loaded trained weights from " + trainedWeightsFile.toAbsolutePath());
            Log.info.println("LearnedWeights: loaded per-dimension Brownian rates=" + Arrays.toString(brownianTraitRates));
            if (learnedState.rootPriorVariances() != null) {
                Log.info.println("LearnedWeights: loaded per-dimension root prior variances=" + Arrays.toString(rootPriorVariances));
            }
            Log.info.println("LearnedWeights: loaded per-dimension base scales=" + Arrays.toString(learnedBaseScales));
            Log.info.println("LearnedWeights: loaded per-dimension slopes=" + Arrays.toString(learnedSlopes));
            Log.info.println("LearnedWeights: loaded intercept=" + learnedIntercept);
        }
    }

    private void saveTrainedWeightsIfRequested(double[] optimizedEmbeddings) {
        if (trainedWeightsFile == null) {
            return;
        }

        try {
            String[] canonicalTaxa = dataInput.get().getTaxaNames().toArray(new String[0]);
            LearnedWeightsModelIO.writeModel(
                    trainedWeightsFile,
                    canonicalTaxa,
                    optimizedEmbeddings,
                    brownianTraitRates,
                    rootPriorVariances,
                    learnedBaseScales,
                    learnedSlopes,
                    learnedIntercept,
                    separateBaseAndQuadraticCalibrationInput.get(),
                    scaleQuadraticPartOnlyInput.get(),
                    trainsQuadraticOnly(),
                    pcaRegularizationWeightInput.get(),
                    slopeRegularizationWeightInput.get());
            Log.info.println("LearnedWeights: wrote trained weights to " + trainedWeightsFile.toAbsolutePath());
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to write trained weights to " + trainedWeightsFile.toAbsolutePath(), e);
        }
    }

    private double[] resolveWarmStartBaseScales(LearnedWeightsModelIO.ModelState learnedState) {
        if (learnedState.baseScales() != null) {
            return learnedState.baseScales();
        }
        return defaultLoadedBaseScales(learnedState.quadraticScales());
    }

    private double[] defaultLoadedBaseScales(double[] loadedQuadraticScales) {
        double[] baseScales = new double[dim];
        if (scaleQuadraticPartOnlyInput.get()) {
            Arrays.fill(baseScales, 1.0);
            return baseScales;
        }
        System.arraycopy(loadedQuadraticScales, 0, baseScales, 0, dim);
        return baseScales;
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

    private double mean(double[] values) {
        double total = 0.0;
        for (double value : values) {
            total += value;
        }
        return total / values.length;
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

    @Override
    public double[] getTargetWeights(int fromNodeNr, List<Node> toNodeNrs) {
        return runtimeModel.getTargetWeights(fromNodeNr, toNodeNrs);
    }

    @Override
    public double[] getTargetWeights(int fromNodeNr, List<Node> toNodeNrs, double toHeight) {
        return runtimeModel.getTargetWeights(fromNodeNr, toNodeNrs, toHeight);
    }

    @Override
    public double[] getTargetWeights(int fromNodeNr, List<Node> toNodeNrs, List<Double> toHeights) {
        return runtimeModel.getTargetWeights(fromNodeNr, toNodeNrs, toHeights);
    }

    @Override
    public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
        return runtimeModel.getTargetWeightsInteger(fromNodeNr, toNodeNrs);
    }

    @Override
    public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, double toHeight) {
        return runtimeModel.getTargetWeightsInteger(fromNodeNr, toNodeNrs, toHeight);
    }

    @Override
    public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, List<Double> toHeights) {
        return runtimeModel.getTargetWeightsInteger(fromNodeNr, toNodeNrs, toHeights);
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

    public double getRootPriorVariance() {
        return rootPriorVarianceInput.get();
    }

    public double[] getRootPriorVariances() {
        return Arrays.copyOf(rootPriorVariances, rootPriorVariances.length);
    }
}
