package targetedbeast.edgeweights.learnedweights;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.BufferedReader;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import beast.base.core.BEASTInterface;
import beast.base.core.Input;
import beast.base.core.Log;
import beast.base.evolution.alignment.Alignment;
import beast.base.spec.evolution.branchratemodel.Base;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.likelihood.GenericTreeLikelihood;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.StateNode;
import beast.base.inference.parameter.Parameter;
import beast.base.inference.parameter.RealParameter;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.inference.parameter.IntVectorParam;
import beast.base.spec.inference.parameter.RealVectorParam;
import beast.base.spec.evolution.tree.coalescent.ConstantPopulation;
import beast.base.spec.evolution.tree.coalescent.RandomTree;
import beast.base.parser.NexusParser;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.NonPositiveDefiniteMatrixException;
import org.apache.commons.math3.linear.RealMatrix;

public class LearnedWeightsTreeTrainer {

    private static final double CHOLESKY_THRESHOLD = 1e-12;
    private static final double[] JITTER_SCHEDULE = new double[] {0.0, 1e-10, 1e-8, 1e-6};
    private static final String TREE_RATE_METADATA_KEY = "rate";
    private static final Pattern TRAILING_INTEGER_PATTERN = Pattern.compile("(\\d+)$");

    public static final class TrainingTreeSample {
        /** Sentinel groupId for samples that don't come from a competing-candidate set. */
        public static final long NO_GROUP = -1L;

        public final double targetLogLikelihood;
        // Stored as the inverse Brownian covariance so valid off-diagonal entries can be negative.
        public final double[][] precisionMatrix;
        public final double logDeterminant;
        public final int logNormalizerDimension;
        public final double[][] rootFreePrecisionMatrix;
        public final double[] rootFreeCovarianceCholeskyLower;
        public final double rootFreeLogDeterminant;
        public final int rootFreeLogNormalizerDimension;
        public final double[] rootFreePrecisionTimesOnes;
        public final double rootFreePrecisionOnesSum;
        public final double rootHeight;
        public final double totalBranchLength;
        public final double totalDiffusionLength;
        // Identifies which competing-candidate regraft set (same fromNode, same toNodeNrs batch)
        // this proposal was scored in; NO_GROUP for samples that aren't proposal-export rows.
        public final long groupId;

        public TrainingTreeSample(double targetLogLikelihood, double[][] precisionMatrix, double logDeterminant) {
            this(targetLogLikelihood, precisionMatrix, logDeterminant, precisionMatrix.length,
                    (double[][]) null, Double.NaN, 0,
                    Double.NaN, Double.NaN, Double.NaN);
        }

        public TrainingTreeSample(double targetLogLikelihood, double[][] precisionMatrix, double logDeterminant,
                int logNormalizerDimension) {
            this(targetLogLikelihood, precisionMatrix, logDeterminant, logNormalizerDimension,
                (double[][]) null, Double.NaN, 0,
                    Double.NaN, Double.NaN, Double.NaN);
        }

        public TrainingTreeSample(double targetLogLikelihood, double[][] precisionMatrix, double logDeterminant,
                int logNormalizerDimension, double rootHeight, double totalBranchLength,
                double totalDiffusionLength) {
            this(targetLogLikelihood, precisionMatrix, logDeterminant, logNormalizerDimension,
                (double[][]) null, Double.NaN, 0,
                    rootHeight, totalBranchLength, totalDiffusionLength);
        }

        public TrainingTreeSample(double targetLogLikelihood, double[][] precisionMatrix, double logDeterminant,
                int logNormalizerDimension, double[][] rootFreePrecisionMatrix,
                double rootFreeLogDeterminant, int rootFreeLogNormalizerDimension,
                double rootHeight, double totalBranchLength, double totalDiffusionLength) {
            this(targetLogLikelihood, precisionMatrix, logDeterminant, logNormalizerDimension,
                    rootFreePrecisionMatrix, null, rootFreeLogDeterminant, rootFreeLogNormalizerDimension,
                    rootHeight, totalBranchLength, totalDiffusionLength, NO_GROUP);
        }

        public TrainingTreeSample(double targetLogLikelihood, double[][] precisionMatrix, double logDeterminant,
                int logNormalizerDimension, double[] rootFreeCovarianceCholeskyLower,
                double rootFreeLogDeterminant, int rootFreeLogNormalizerDimension,
                double rootHeight, double totalBranchLength, double totalDiffusionLength) {
            this(targetLogLikelihood, precisionMatrix, logDeterminant, logNormalizerDimension,
                    null, rootFreeCovarianceCholeskyLower, rootFreeLogDeterminant, rootFreeLogNormalizerDimension,
                    rootHeight, totalBranchLength, totalDiffusionLength, NO_GROUP);
        }

        /** Overload for callers (e.g. proposal export) that know which competing-candidate group a sample belongs to. */
        public TrainingTreeSample(double targetLogLikelihood, double[][] precisionMatrix, double logDeterminant,
                int logNormalizerDimension, double[] rootFreeCovarianceCholeskyLower,
                double rootFreeLogDeterminant, int rootFreeLogNormalizerDimension,
                double rootHeight, double totalBranchLength, double totalDiffusionLength, long groupId) {
            this(targetLogLikelihood, precisionMatrix, logDeterminant, logNormalizerDimension,
                    null, rootFreeCovarianceCholeskyLower, rootFreeLogDeterminant, rootFreeLogNormalizerDimension,
                    rootHeight, totalBranchLength, totalDiffusionLength, groupId);
        }

        private TrainingTreeSample(double targetLogLikelihood, double[][] precisionMatrix, double logDeterminant,
                int logNormalizerDimension, double[][] rootFreePrecisionMatrix,
                double[] rootFreeCovarianceCholeskyLower,
                double rootFreeLogDeterminant, int rootFreeLogNormalizerDimension,
                double rootHeight, double totalBranchLength, double totalDiffusionLength, long groupId) {
            this.targetLogLikelihood = targetLogLikelihood;
            this.precisionMatrix = precisionMatrix;
            this.logDeterminant = logDeterminant;
            this.logNormalizerDimension = logNormalizerDimension;
            this.rootFreePrecisionMatrix = copyMatrix(rootFreePrecisionMatrix);
            this.rootFreeCovarianceCholeskyLower = copyVector(rootFreeCovarianceCholeskyLower);
            this.rootFreeLogDeterminant = rootFreeLogDeterminant;
            this.rootFreeLogNormalizerDimension = rootFreeLogNormalizerDimension;
            if (this.rootFreePrecisionMatrix == null && this.rootFreeCovarianceCholeskyLower == null) {
                this.rootFreePrecisionTimesOnes = null;
                this.rootFreePrecisionOnesSum = Double.NaN;
            } else if (this.rootFreePrecisionMatrix != null) {
                this.rootFreePrecisionTimesOnes = multiplyByOnes(this.rootFreePrecisionMatrix);
                this.rootFreePrecisionOnesSum = sum(this.rootFreePrecisionTimesOnes);
            } else {
                this.rootFreePrecisionTimesOnes = multiplyPrecisionByVector(
                        this.rootFreeCovarianceCholeskyLower,
                        getTaxonCount(),
                        createOnesVector(getTaxonCount()));
                this.rootFreePrecisionOnesSum = sum(this.rootFreePrecisionTimesOnes);
            }
            this.rootHeight = rootHeight;
            this.totalBranchLength = totalBranchLength;
            this.totalDiffusionLength = totalDiffusionLength;
            this.groupId = groupId;
        }

        public int getTaxonCount() {
            return precisionMatrix.length;
        }

        public boolean hasRootFreeFactorization() {
            return rootFreePrecisionMatrix != null || rootFreeCovarianceCholeskyLower != null;
        }

        public double[] multiplyRootFreePrecisionByEmbeddings(double[] embeddings, int dim) {
            if (rootFreePrecisionMatrix != null) {
                return multiplyPrecisionMatrixByEmbeddings(rootFreePrecisionMatrix, embeddings, dim);
            }
            if (rootFreeCovarianceCholeskyLower != null) {
                return multiplyPrecisionByEmbeddings(rootFreeCovarianceCholeskyLower, getTaxonCount(), embeddings, dim);
            }
            throw new IllegalStateException("Training sample is missing root-free covariance data");
        }

        private static double[][] copyMatrix(double[][] matrix) {
            if (matrix == null) {
                return null;
            }
            double[][] copy = new double[matrix.length][];
            for (int row = 0; row < matrix.length; row++) {
                copy[row] = Arrays.copyOf(matrix[row], matrix[row].length);
            }
            return copy;
        }

        private static double[] copyVector(double[] values) {
            return values == null ? null : Arrays.copyOf(values, values.length);
        }

        private static double[] multiplyByOnes(double[][] matrix) {
            double[] result = new double[matrix.length];
            for (int row = 0; row < matrix.length; row++) {
                double total = 0.0;
                for (int column = 0; column < matrix[row].length; column++) {
                    total += matrix[row][column];
                }
                result[row] = total;
            }
            return result;
        }

        private static double sum(double[] values) {
            double total = 0.0;
            for (double value : values) {
                total += value;
            }
            return total;
        }

        private static double[] multiplyPrecisionMatrixByEmbeddings(double[][] precisionMatrix, double[] embeddings, int dim) {
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

        private static double[] createOnesVector(int length) {
            double[] ones = new double[length];
            Arrays.fill(ones, 1.0);
            return ones;
        }

        private static double[] multiplyPrecisionByVector(double[] packedLowerTriangular, int dimension, double[] vector) {
            if (vector == null) {
                return new double[dimension];
            }
            double[] forward = solvePackedLowerTriangular(packedLowerTriangular, dimension, vector);
            return solvePackedUpperTriangular(packedLowerTriangular, dimension, forward);
        }

        private static double[] multiplyPrecisionByEmbeddings(
                double[] packedLowerTriangular,
                int taxonCount,
                double[] embeddings,
                int dim) {
            double[] output = new double[taxonCount * dim];
            if (embeddings == null) {
                return output;
            }
            double[] axisValues = new double[taxonCount];
            for (int axis = 0; axis < dim; axis++) {
                for (int taxon = 0; taxon < taxonCount; taxon++) {
                    axisValues[taxon] = embeddings[taxon * dim + axis];
                }
                double[] solved = multiplyPrecisionByVector(packedLowerTriangular, taxonCount, axisValues);
                for (int taxon = 0; taxon < taxonCount; taxon++) {
                    output[taxon * dim + axis] = solved[taxon];
                }
            }
            return output;
        }

        private static double[] solvePackedLowerTriangular(double[] packedLowerTriangular, int dimension, double[] rhs) {
            double[] solution = Arrays.copyOf(rhs, rhs.length);
            int index = 0;
            for (int row = 0; row < dimension; row++) {
                double diagonal = Double.NaN;
                for (int column = 0; column <= row; column++) {
                    double entry = packedLowerTriangular[index++];
                    if (column < row) {
                        solution[row] -= entry * solution[column];
                    } else {
                        diagonal = entry;
                    }
                }
                solution[row] /= diagonal;
            }
            return solution;
        }

        private static double[] solvePackedUpperTriangular(double[] packedLowerTriangular, int dimension, double[] rhs) {
            double[] solution = Arrays.copyOf(rhs, rhs.length);
            for (int row = dimension - 1; row >= 0; row--) {
                for (int column = row + 1; column < dimension; column++) {
                    solution[row] -= packedLowerTriangular[packedLowerIndex(column, row)] * solution[column];
                }
                solution[row] /= packedLowerTriangular[packedLowerIndex(row, row)];
            }
            return solution;
        }

        private static int packedLowerIndex(int row, int column) {
            return row * (row + 1) / 2 + column;
        }
    }

    private final Alignment alignment;
    private final GenericTreeLikelihood likelihoodTemplate;
    private final GenericTreeLikelihood likelihoodEvaluator;
    private final int nTrainingTrees;
    private final String[] canonicalTaxa;
    private final SampledBranchRateModel sampledBranchRateModel;
    private final Tree runtimeLikelihoodTree;
    private final Map<String, StateNode> loggedParametersById;
    private final double rootPriorVariance;
    private final String targetLogLikelihoodColumn;
    private SampledRateContext currentSampledRateContext = SampledRateContext.empty();

    public LearnedWeightsTreeTrainer(
            GenericTreeLikelihood likelihoodTemplate,
            Alignment alignment,
            int nTrainingTrees) {
        this(likelihoodTemplate, alignment, nTrainingTrees, 0.0, null);
    }

    public LearnedWeightsTreeTrainer(
            GenericTreeLikelihood likelihoodTemplate,
            Alignment alignment,
            int nTrainingTrees,
            double rootPriorVariance) {
        this(likelihoodTemplate, alignment, nTrainingTrees, rootPriorVariance, null);
    }

    public LearnedWeightsTreeTrainer(
            GenericTreeLikelihood likelihoodTemplate,
            Alignment alignment,
            int nTrainingTrees,
            double rootPriorVariance,
            String targetLogLikelihoodColumn) {
        this.likelihoodTemplate = likelihoodTemplate;
        this.alignment = alignment;
        this.nTrainingTrees = nTrainingTrees;
        this.rootPriorVariance = rootPriorVariance;
        this.targetLogLikelihoodColumn = resolveTargetLogLikelihoodColumnName(likelihoodTemplate, targetLogLikelihoodColumn);
        this.canonicalTaxa = alignment.getTaxaNames().toArray(new String[0]);
        if (likelihoodTemplate == null) {
            this.sampledBranchRateModel = new SampledBranchRateModel(null);
            this.likelihoodEvaluator = null;
            this.runtimeLikelihoodTree = null;
            this.loggedParametersById = Collections.emptyMap();
        } else {
            this.sampledBranchRateModel = new SampledBranchRateModel(likelihoodTemplate.branchRateModelInput.get());
            this.likelihoodEvaluator = createTrainingLikelihoodEvaluator(likelihoodTemplate, sampledBranchRateModel);
            this.runtimeLikelihoodTree = getRuntimeLikelihoodTree(likelihoodEvaluator);
            this.loggedParametersById = collectLoggedParameters(likelihoodEvaluator, sampledBranchRateModel.getDelegate());
        }
    }

    public List<TrainingTreeSample> generateTrainingSamples() {
        final long startNanos = System.nanoTime();
        List<TrainingTreeSample> samples = new ArrayList<>();
        int logEvery = Math.max(1, nTrainingTrees / 10);
        Log.info.println("LearnedWeights: generating " + nTrainingTrees + " prior-sampled training trees...");
        for (int treeIdx = 0; treeIdx < nTrainingTrees; treeIdx++) {
            Tree tree = sampleRandomTree(alignment);
            double treeLogLikelihood = computeCurrentTreeLogLikelihood(tree);
            samples.add(buildTrainingSample(tree, treeLogLikelihood));
            int completed = treeIdx + 1;
            if (completed % logEvery == 0 || completed == nTrainingTrees) {
                double elapsedSec = (System.nanoTime() - startNanos) / 1_000_000_000.0;
                Log.info.println(String.format(
                        "LearnedWeights: generated %d/%d training trees (usable=%d, elapsed=%.1fs)",
                        completed,
                        nTrainingTrees,
                        samples.size(),
                        elapsedSec));
            }
        }
        return Collections.unmodifiableList(samples);
    }

    public List<TrainingTreeSample> generateTrainingSamplesFromTreeFiles(
            List<File> treeFiles,
            List<File> logFiles,
            int burninPercentage,
            int thinInterval,
            int maxSamples) throws IOException {
        if (treeFiles == null || treeFiles.isEmpty()) {
            throw new IllegalArgumentException("treeFiles must not be null or empty");
        }
        List<File> normalizedFiles = new ArrayList<>();
        for (File treeFile : treeFiles) {
            if (treeFile == null) {
                throw new IllegalArgumentException("treeFiles must not contain null entries");
            }
            if (!treeFile.exists()) {
                throw new IllegalArgumentException("Tree file does not exist: " + treeFile.getAbsolutePath());
            }
            normalizedFiles.add(treeFile);
        }
        List<File> normalizedLogFiles = normalizeLogFiles(logFiles, normalizedFiles.size());
        if (burninPercentage < 0 || burninPercentage >= 100) {
            throw new IllegalArgumentException("burninPercentage must be in [0, 100), but was " + burninPercentage);
        }
        if (thinInterval <= 0) {
            throw new IllegalArgumentException("thinInterval must be > 0, but was " + thinInterval);
        }

        final long startNanos = System.nanoTime();
        List<FileTreeSelectionPlan> treeSelectionPlans = new ArrayList<>(normalizedFiles.size());
        int parsedTreeCount = 0;
        int totalSelectedTreeCount = 0;
        for (File treeFile : normalizedFiles) {
            int fileParsedTreeCount = countTreesInFile(treeFile);
            int fileSelectedTreeCount = countSelectedEntries(fileParsedTreeCount, burninPercentage, thinInterval);
            treeSelectionPlans.add(new FileTreeSelectionPlan(treeFile, fileParsedTreeCount, fileSelectedTreeCount));
            parsedTreeCount += fileParsedTreeCount;
            totalSelectedTreeCount += fileSelectedTreeCount;
        }

        int[] retainedGlobalSelectedIndices = computeRetainedSelectedIndices(totalSelectedTreeCount, maxSamples);
        assignRetainedSelectedIndices(treeSelectionPlans, retainedGlobalSelectedIndices);

        List<TrainingTreeSample> samples = new ArrayList<>(retainedGlobalSelectedIndices.length);
        int totalRetainedTreeCount = retainedGlobalSelectedIndices.length;
        int logEvery = Math.max(1, Math.max(1, totalRetainedTreeCount) / 10);
        int[] processedTreeCount = new int[] {0};
        for (int fileIndex = 0; fileIndex < normalizedFiles.size(); fileIndex++) {
            File treeFile = normalizedFiles.get(fileIndex);
            File logFile = normalizedLogFiles.get(fileIndex);
            FileTreeSelectionPlan selectionPlan = treeSelectionPlans.get(fileIndex);
            List<Long> retainedTreeSampleNumbers = loadRetainedTreeSampleNumbers(
                    selectionPlan,
                    burninPercentage,
                    thinInterval);
                LoggedPosteriorSamples loggedPosteriorSamples = loadRetainedLoggedPosteriorSamples(
                    logFile,
                    burninPercentage,
                    thinInterval,
                    selectionPlan.retainedSelectedIndices,
                    retainedTreeSampleNumbers);
            streamRetainedPosteriorTrees(
                    selectionPlan,
                    burninPercentage,
                    thinInterval,
                    (tree, retainedIndex) -> {
                        LoggedPosteriorSample loggedRate = requireLoggedPosteriorSample(
                                tree,
                                loggedPosteriorSamples.bySampleNumber,
                                logFile);
                        withSampledRateContext(SampledRateContext.fromTree(tree, loggedRate), () -> {
                            LoggedStateContext loggedStateContext = LoggedStateContext.fromLoggedPosteriorSample(loggedRate);
                            withLoggedStateContext(loggedStateContext, () -> {
                                samples.add(buildTrainingSample(tree, loggedStateContext.getTargetLogLikelihood()));
                            });
                        });
                        processedTreeCount[0]++;
                        if (processedTreeCount[0] % logEvery == 0 || processedTreeCount[0] == totalRetainedTreeCount) {
                            double elapsedSec = (System.nanoTime() - startNanos) / 1_000_000_000.0;
                            Log.info.println(String.format(
                                    "LearnedWeights: processed %d/%d posterior trees (usable=%d, elapsed=%.1fs)",
                                    processedTreeCount[0],
                                    totalRetainedTreeCount,
                                    samples.size(),
                                    elapsedSec));
                        }
                    });
            Log.info.println(String.format(
                    "LearnedWeights: retained %d/%d selected posterior trees from %s (parsed=%d, burnin=%d%%, thin=%d)",
                    selectionPlan.retainedSelectedIndices.size(),
                    selectionPlan.selectedCount,
                    treeFile.getAbsolutePath(),
                    selectionPlan.parsedCount,
                    burninPercentage,
                    thinInterval));
        }

        Log.info.println(String.format(
                "LearnedWeights: generating training samples from posterior tree file%s %s (parsed=%d, selected=%d, burnin=%d%%, thin=%d, maxSamples=%s)",
                normalizedFiles.size() == 1 ? "" : "s",
                joinFilePaths(normalizedFiles),
                parsedTreeCount,
                totalRetainedTreeCount,
                burninPercentage,
                thinInterval,
                maxSamples > 0 ? Integer.toString(maxSamples) : "all"));
        return Collections.unmodifiableList(samples);
    }

    private List<Long> loadRetainedTreeSampleNumbers(
            FileTreeSelectionPlan selectionPlan,
            int burninPercentage,
            int thinInterval) throws IOException {
        if (selectionPlan.retainedSelectedIndices.isEmpty()) {
            return Collections.emptyList();
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(selectionPlan.treeFile))) {
            TreeBlockContext treeBlockContext = moveToTreesBlock(reader);
            List<Long> sampleNumbers = new ArrayList<>(selectionPlan.retainedSelectedIndices.size());
            int burninCount = selectionPlan.parsedCount * burninPercentage / 100;
            int parsedIndex = 0;
            int selectedIndex = 0;
            NexusTreeCommand nextCommand = readNextTreeCommand(reader);
            while (nextCommand != null && !nextCommand.isEndOfBlock()) {
                if (nextCommand.isCommand("translate")) {
                    treeBlockContext = treeBlockContext.withTranslation(parseTranslateCommand(nextCommand.arguments));
                } else if (nextCommand.isCommand("tree")) {
                    boolean isSelectedTree = parsedIndex >= burninCount && ((parsedIndex - burninCount) % thinInterval == 0);
                    if (isSelectedTree) {
                        if (selectionPlan.retainedSelectedIndices.contains(selectedIndex)) {
                            ParsedTreeCommand parsedTree = parseTreeCommand(nextCommand.arguments, treeBlockContext);
                            sampleNumbers.add(extractTreeSampleNumber(parsedTree.tree()));
                        }
                        selectedIndex++;
                    }
                    parsedIndex++;
                }
                nextCommand = readNextTreeCommand(reader);
            }
            return sampleNumbers;
        }
    }

    private void streamRetainedPosteriorTrees(
            FileTreeSelectionPlan selectionPlan,
            int burninPercentage,
            int thinInterval,
            RetainedTreeProcessor treeProcessor) throws IOException {
        if (selectionPlan.retainedSelectedIndices.isEmpty()) {
            return;
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(selectionPlan.treeFile))) {
            TreeBlockContext treeBlockContext = moveToTreesBlock(reader);
            int burninCount = selectionPlan.parsedCount * burninPercentage / 100;
            int parsedIndex = 0;
            int selectedIndex = 0;
            int retainedIndex = 0;
            NexusTreeCommand nextCommand = readNextTreeCommand(reader);
            while (nextCommand != null && !nextCommand.isEndOfBlock()) {
                if (nextCommand.isCommand("translate")) {
                    treeBlockContext = treeBlockContext.withTranslation(parseTranslateCommand(nextCommand.arguments));
                } else if (nextCommand.isCommand("tree")) {
                    boolean isSelectedTree = parsedIndex >= burninCount && ((parsedIndex - burninCount) % thinInterval == 0);
                    if (isSelectedTree) {
                        if (selectionPlan.retainedSelectedIndices.contains(selectedIndex)) {
                            ParsedTreeCommand parsedTree = parseTreeCommand(nextCommand.arguments, treeBlockContext);
                            treeProcessor.process(parsedTree.tree(), retainedIndex++);
                        }
                        selectedIndex++;
                    }
                    parsedIndex++;
                }
                nextCommand = readNextTreeCommand(reader);
            }
        }
    }

    private LoggedPosteriorSamples loadRetainedLoggedPosteriorSamples(
            File logFile,
            int burninPercentage,
            int thinInterval,
            Set<Integer> retainedSelectedIndices,
            List<Long> requiredSampleNumbers) throws IOException {
        if (retainedSelectedIndices.isEmpty()) {
            return LoggedPosteriorSamples.empty();
        }

        Set<Long> requiredSampleNumberSet = new HashSet<>();
        for (Long sampleNumber : requiredSampleNumbers) {
            if (sampleNumber != null) {
                requiredSampleNumberSet.add(sampleNumber);
            }
        }

        Map<Long, LoggedPosteriorSample> bySample = new LinkedHashMap<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(logFile))) {
            String line;
            String[] headers = null;
            int sampleIndex = -1;
            int targetLogLikelihoodIndex = -1;
            List<LoggedParameterBinding> parameterBindings = Collections.emptyList();
            while ((line = reader.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                    continue;
                }
                String[] fields = trimmed.split("\\t");
                if (headers == null) {
                    headers = fields;
                    sampleIndex = findColumnIndex(headers, "Sample");
                    if (sampleIndex < 0) {
                        throw new IllegalArgumentException("Could not find Sample column in training log "
                                + logFile.getAbsolutePath());
                    }
                    targetLogLikelihoodIndex = resolveTargetLogLikelihoodColumn(headers, logFile);
                    parameterBindings = resolveLoggedParameterBindings(headers, logFile);
                    continue;
                }
                if (fields.length <= sampleIndex) {
                    continue;
                }
                long sampleNumber = Long.parseLong(fields[sampleIndex]);
                boolean retainByExactSampleNumber = requiredSampleNumberSet.contains(sampleNumber);
                if (retainByExactSampleNumber) {
                    Double targetLogLikelihood = extractLoggedTargetLogLikelihood(fields, targetLogLikelihoodIndex);
                    List<LoggedParameterSample> parameterSamples = extractLoggedParameterSamples(fields, parameterBindings);
                    double strictClockMeanRate = extractStrictClockMeanRate(parameterSamples);
                    LoggedPosteriorSample sample = new LoggedPosteriorSample(
                            sampleNumber,
                            strictClockMeanRate,
                            targetLogLikelihood,
                            parameterSamples);
                    bySample.put(sampleNumber, sample);
                }
            }
        }

        if (bySample.size() != requiredSampleNumberSet.size()) {
            Set<Long> missingSampleNumbers = new HashSet<>(requiredSampleNumberSet);
            missingSampleNumbers.removeAll(bySample.keySet());
            throw new IllegalArgumentException(
                    "Training log " + logFile.getAbsolutePath()
                            + " is missing rows for retained tree sample numbers " + missingSampleNumbers + ".");
        }

        return new LoggedPosteriorSamples(
            bySample.isEmpty() ? Collections.emptyMap() : Collections.unmodifiableMap(bySample));
    }

    private int countTreesInFile(File treeFile) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(treeFile))) {
            moveToTreesBlock(reader);
            int parsedCount = 0;
            NexusTreeCommand nextCommand = readNextTreeCommand(reader);
            while (nextCommand != null && !nextCommand.isEndOfBlock()) {
                if (nextCommand.isCommand("tree")) {
                    parsedCount++;
                }
                nextCommand = readNextTreeCommand(reader);
            }
            return parsedCount;
        }
    }

    private int countSelectedEntries(int parsedCount, int burninPercentage, int thinInterval) {
        if (parsedCount <= 0) {
            return 0;
        }
        int burninCount = parsedCount * burninPercentage / 100;
        if (burninCount >= parsedCount) {
            return 0;
        }
        return ((parsedCount - 1 - burninCount) / thinInterval) + 1;
    }

    private int[] computeRetainedSelectedIndices(int totalSelectedCount, int maxSamples) {
        if (totalSelectedCount <= 0) {
            return new int[0];
        }
        int retainedCount = maxSamples > 0 ? Math.min(totalSelectedCount, maxSamples) : totalSelectedCount;
        int[] retainedIndices = new int[retainedCount];
        if (retainedCount == totalSelectedCount) {
            for (int i = 0; i < retainedCount; i++) {
                retainedIndices[i] = i;
            }
            return retainedIndices;
        }
        double step = (double) totalSelectedCount / retainedCount;
        for (int i = 0; i < retainedCount; i++) {
            retainedIndices[i] = Math.min(totalSelectedCount - 1, (int) Math.floor(i * step));
        }
        return retainedIndices;
    }

    private void assignRetainedSelectedIndices(
            List<FileTreeSelectionPlan> selectionPlans,
            int[] retainedGlobalSelectedIndices) {
        int retainedPointer = 0;
        int selectedOffset = 0;
        for (int fileIndex = 0; fileIndex < selectionPlans.size(); fileIndex++) {
            FileTreeSelectionPlan selectionPlan = selectionPlans.get(fileIndex);
            Set<Integer> retainedLocalIndices = new HashSet<>();
            int selectedUpperBound = selectedOffset + selectionPlan.selectedCount;
            while (retainedPointer < retainedGlobalSelectedIndices.length
                    && retainedGlobalSelectedIndices[retainedPointer] < selectedUpperBound) {
                retainedLocalIndices.add(retainedGlobalSelectedIndices[retainedPointer] - selectedOffset);
                retainedPointer++;
            }
            selectionPlans.set(fileIndex, selectionPlan.withRetainedSelectedIndices(retainedLocalIndices));
            selectedOffset = selectedUpperBound;
        }
    }

    private TreeBlockContext moveToTreesBlock(BufferedReader reader) throws IOException {
        String line;
        while ((line = reader.readLine()) != null) {
            if (line.trim().toLowerCase().matches("^begin\\s+trees;\\s*$")) {
                return new TreeBlockContext(null, -1, null);
            }
        }
        throw new IOException("Could not find begin trees block in posterior tree file");
    }

    private NexusTreeCommand readNextTreeCommand(BufferedReader reader) throws IOException {
        StringBuilder commandBuilder = new StringBuilder();
        while (true) {
            int nextValue = reader.read();
            if (nextValue < 0) {
                break;
            }
            char nextChar = (char) nextValue;
            if (nextChar == ';') {
                break;
            }
            commandBuilder.append(nextChar);
            switch (nextChar) {
                case '[':
                    readNexusComment(reader, commandBuilder);
                    break;
                case '"':
                case '\'':
                    readNexusString(reader, commandBuilder, nextChar);
                    break;
                default:
                    break;
            }
        }
        if (commandBuilder.toString().isEmpty()) {
            return null;
        }
        return new NexusTreeCommand(commandBuilder.toString());
    }

    private void readNexusComment(BufferedReader reader, StringBuilder builder) throws IOException {
        int depth = 1;
        while (depth > 0) {
            int nextValue = reader.read();
            if (nextValue < 0) {
                throw new IOException("Unterminated nexus comment in tree file");
            }
            char nextChar = (char) nextValue;
            builder.append(nextChar);
            if (nextChar == '[') {
                depth++;
            } else if (nextChar == ']') {
                depth--;
            } else if (nextChar == '"' || nextChar == '\'') {
                readNexusString(reader, builder, nextChar);
            }
        }
    }

    private void readNexusString(BufferedReader reader, StringBuilder builder, char delimiter) throws IOException {
        while (true) {
            int nextValue = reader.read();
            if (nextValue < 0) {
                throw new IOException("Unterminated nexus string in tree file");
            }
            char nextChar = (char) nextValue;
            builder.append(nextChar);
            if (nextChar == delimiter) {
                return;
            }
        }
    }

    private ParsedTreeCommand parseTreeCommand(String treeString, TreeBlockContext treeBlockContext) {
        final int startOfNewick = treeString.indexOf('(');
        String id = "0";
        try {
            id = treeString.substring(5, startOfNewick).split("=")[0].trim();
        } catch (Exception ignored) {
            // keep default id
        }
        String newick = startOfNewick > 0 ? treeString.substring(startOfNewick) : treeString;
        TreeParser treeParser;
        if (treeBlockContext.origin != -1) {
            treeParser = new TreeParser(treeBlockContext.taxa, newick, treeBlockContext.origin, false);
        } else {
            try {
                treeParser = new TreeParser(treeBlockContext.taxa, newick, 0, false);
            } catch (ArrayIndexOutOfBoundsException e) {
                treeParser = new TreeParser(treeBlockContext.taxa, newick, 1, false);
            }
        }
        treeParser.setID(id);
        return new ParsedTreeCommand(treeParser, id);
    }

    private Map<String, String> parseTranslateCommand(String translateArgs) throws IOException {
        Map<String, String> translationMap = new HashMap<>();
        String[] taxaTranslations = translateArgs.split(",");
        for (String taxaTranslation : taxaTranslations) {
            taxaTranslation = taxaTranslation.trim();
            int splitIndex = 0;
            while (splitIndex < taxaTranslation.length() && !Character.isWhitespace(taxaTranslation.charAt(splitIndex))) {
                splitIndex++;
            }
            if (splitIndex > 0) {
                String key = taxaTranslation.substring(0, splitIndex);
                String value = taxaTranslation.substring(splitIndex).trim();
                translationMap.put(key, value);
            }
        }
        return translationMap;
    }

    private static int getIndexedTranslationMapOrigin(Map<String, String> translationMap) {
        SortedSet<Integer> indices = new TreeSet<>();
        int count = 0;
        for (String key : translationMap.keySet()) {
            int index = Integer.parseInt(key);
            indices.add(index);
            count++;
        }
        if (!indices.isEmpty()
                && (indices.last() - indices.first() == count - 1)
                && (indices.first() == 0 || indices.first() == 1)) {
            return indices.first();
        }
        return -1;
    }

    private static List<String> getIndexedTranslationMap(Map<String, String> translationMap, int origin) {
        String[] taxa = new String[translationMap.size()];
        for (String key : translationMap.keySet()) {
            taxa[Integer.parseInt(key) - origin] = translationMap.get(key);
        }
        return Arrays.asList(taxa);
    }

    protected Tree sampleRandomTree(Alignment data) {
        ConstantPopulation popFunc = new ConstantPopulation();
        popFunc.initByName("popSize", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

        RandomTree tree = new RandomTree();
        tree.initByName(
                "taxa", data,
                "populationModel", popFunc);
        return tree;
    }

    protected TrainingTreeSample buildTrainingSample(Tree tree, double targetLogLikelihood) {
        return buildTrainingSample(tree, targetLogLikelihood, TrainingTreeSample.NO_GROUP);
    }

    /** Overload for callers (e.g. proposal export) that can attribute the sample to a competing-candidate group. */
    protected TrainingTreeSample buildTrainingSample(Tree tree, double targetLogLikelihood, long groupId) {
        double[][] covariance = buildCovarianceMatrix(tree);
        MatrixFactorization factorization = factorizeCovariance(covariance);
        MatrixFactorization rootFreeFactorization = factorizeRootFreeCovariance(covariance, covariance.length);
        TreeScaleStatistics treeScaleStatistics = computeTreeScaleStatistics(tree);
        return new TrainingTreeSample(
            targetLogLikelihood,
            factorization.precisionMatrix,
            factorization.logDeterminant,
            factorization.logNormalizerDimension,
            rootFreeFactorization.covarianceCholeskyLower,
            rootFreeFactorization.logDeterminant,
            rootFreeFactorization.logNormalizerDimension,
            treeScaleStatistics.rootHeight,
            treeScaleStatistics.totalBranchLength,
            treeScaleStatistics.totalDiffusionLength,
            groupId);
    }

    private TreeScaleStatistics computeTreeScaleStatistics(Tree tree) {
        double rootHeight = tree.getRoot().getHeight();
        double totalBranchLength = 0.0;
        double totalDiffusionLength = 0.0;
        for (Node node : tree.getNodesAsArray()) {
            if (node.isRoot()) {
                continue;
            }
            double branchLength = node.getLength();
            if (!Double.isFinite(branchLength)) {
                throw new IllegalStateException("Non-finite branch length for node " + describeNode(node) + ": "
                        + branchLength);
            }
            if (branchLength < 0.0) {
                throw new IllegalStateException("Negative branch length for node " + describeNode(node) + ": "
                        + branchLength);
            }
            totalBranchLength += branchLength;
            double branchRate = getBranchRate(node);
            if (!Double.isFinite(branchRate)) {
                throw new IllegalStateException("Non-finite branch rate for node " + describeNode(node) + ": "
                        + branchRate);
            }
            if (branchRate < 0.0) {
                throw new IllegalStateException("Negative branch rate for node " + describeNode(node) + ": "
                        + branchRate);
            }
            totalDiffusionLength += branchRate * branchLength;
        }
        return new TreeScaleStatistics(rootHeight, totalBranchLength, totalDiffusionLength);
    }

    protected double[][] buildCovarianceMatrix(Tree tree) {
        int n = canonicalTaxa.length;
        double[][] covariance = new double[n][n];
        Map<String, Node> leafByTaxon = new HashMap<>();
        double[] cumulativeDiffusionLength = new double[tree.getNodeCount()];
        populateCumulativeDiffusionLengths(tree.getRoot(), cumulativeDiffusionLength);
        for (Node node : tree.getNodesAsArray()) {
            if (node.isLeaf()) {
                leafByTaxon.put(node.getID(), node);
            }
        }

        for (int i = 0; i < n; i++) {
            Node left = leafByTaxon.get(canonicalTaxa[i]);
            if (left == null) {
                throw new IllegalStateException("Taxon not found in training tree: " + canonicalTaxa[i]);
            }
            covariance[i][i] = cumulativeDiffusionLength[left.getNr()];
            if (covariance[i][i] < 0.0 && covariance[i][i] > -1e-12) {
                covariance[i][i] = 0.0;
            }
            if (covariance[i][i] < 0.0) {
                throw new IllegalStateException("Negative marginal Brownian path length for taxon "
                        + canonicalTaxa[i] + ": " + covariance[i][i]);
            }
            for (int j = i + 1; j < n; j++) {
                Node right = leafByTaxon.get(canonicalTaxa[j]);
                if (right == null) {
                    throw new IllegalStateException("Taxon not found in training tree: " + canonicalTaxa[j]);
                }
                Node mrca = findMrca(left, right);
                double sharedPathLength = cumulativeDiffusionLength[mrca.getNr()];
                if (sharedPathLength < 0.0 && sharedPathLength > -1e-12) {
                    sharedPathLength = 0.0;
                }
                if (sharedPathLength < 0.0) {
                    throw new IllegalStateException("Negative shared Brownian path length: " + sharedPathLength);
                }
                covariance[i][j] = sharedPathLength;
                covariance[j][i] = sharedPathLength;
            }
        }
        return covariance;
    }

    private void populateCumulativeDiffusionLengths(Node node, double[] cumulativeDiffusionLength) {
        if (node.isRoot()) {
            cumulativeDiffusionLength[node.getNr()] = 0.0;
        } else {
            double branchLength = node.getLength();
            if (!Double.isFinite(branchLength)) {
                throw new IllegalStateException("Non-finite branch length for node " + describeNode(node) + ": " + branchLength);
            }
            if (branchLength < -1e-12) {
                throw new IllegalStateException("Negative branch length for node " + describeNode(node) + ": " + branchLength);
            }
            if (branchLength < 0.0) {
                branchLength = 0.0;
            }

            double branchRate = getBranchRate(node);
            if (!Double.isFinite(branchRate)) {
                throw new IllegalStateException("Non-finite branch rate for node " + describeNode(node) + ": " + branchRate);
            }
            if (branchRate < 0.0) {
                throw new IllegalStateException("Negative branch rate for node " + describeNode(node) + ": " + branchRate);
            }

            double branchDiffusionLength = branchRate * branchLength;
            cumulativeDiffusionLength[node.getNr()] = cumulativeDiffusionLength[node.getParent().getNr()] + branchDiffusionLength;
        }

        for (Node child : node.getChildren()) {
            populateCumulativeDiffusionLengths(child, cumulativeDiffusionLength);
        }
    }

    private double getBranchRate(Node node) {
        return sampledBranchRateModel.getRateForContext(node, currentSampledRateContext);
    }

    private record TreeScaleStatistics(double rootHeight, double totalBranchLength, double totalDiffusionLength) {
    }

    private static String describeNode(Node node) {
        if (node == null) {
            return "<null>";
        }
        if (node.getID() != null && !node.getID().isBlank()) {
            return node.getID() + " (#" + node.getNr() + ")";
        }
        return "#" + node.getNr();
    }

    private MatrixFactorization factorizeCovariance(double[][] covariance) {
        if (Double.isInfinite(rootPriorVariance)) {
            return factorizeImproperRootCovariance(covariance);
        }
        return factorizeProperRootCovariance(withRootPriorVariance(covariance, rootPriorVariance), covariance.length);
    }

    private MatrixFactorization factorizeProperRootCovariance(double[][] covariance, int logNormalizerDimension) {
        for (double jitter : JITTER_SCHEDULE) {
            try {
                RealMatrix matrix = new Array2DRowRealMatrix(addJitter(covariance, jitter), false);
                CholeskyDecomposition decomposition = new CholeskyDecomposition(
                        matrix,
                        CHOLESKY_THRESHOLD,
                        CHOLESKY_THRESHOLD);
                double logDeterminant = getLogDeterminantFromCholesky(decomposition);
                if (!Double.isFinite(logDeterminant)) {
                    throw new IllegalStateException("Non-finite covariance log determinant: " + logDeterminant);
                }
                return new MatrixFactorization(
                        decomposition.getSolver().getInverse().getData(),
                        null,
                        logDeterminant,
                        logNormalizerDimension);
            } catch (NonPositiveDefiniteMatrixException | IllegalStateException ignored) {
                // try larger diagonal jitter
            }
        }
        throw new IllegalStateException("Failed to factorize Brownian covariance matrix for LearnedWeights training sample.");
    }

    private MatrixFactorization factorizeRootFreeCovariance(double[][] covariance, int logNormalizerDimension) {
        for (double jitter : JITTER_SCHEDULE) {
            try {
                RealMatrix matrix = new Array2DRowRealMatrix(addJitter(covariance, jitter), false);
                CholeskyDecomposition decomposition = new CholeskyDecomposition(
                        matrix,
                        CHOLESKY_THRESHOLD,
                        CHOLESKY_THRESHOLD);
                double logDeterminant = getLogDeterminantFromCholesky(decomposition);
                if (!Double.isFinite(logDeterminant)) {
                    throw new IllegalStateException("Non-finite covariance log determinant: " + logDeterminant);
                }
                return new MatrixFactorization(
                        null,
                        packLowerTriangular(decomposition.getL()),
                        logDeterminant,
                        logNormalizerDimension);
            } catch (NonPositiveDefiniteMatrixException | IllegalStateException ignored) {
                // try larger diagonal jitter
            }
        }
        throw new IllegalStateException("Failed to factorize root-free Brownian covariance matrix for LearnedWeights training sample.");
    }

    private double getLogDeterminantFromCholesky(CholeskyDecomposition decomposition) {
        RealMatrix lMatrix = decomposition.getL();
        double logDeterminant = 0.0;
        for (int i = 0; i < lMatrix.getRowDimension(); i++) {
            double diagonal = lMatrix.getEntry(i, i);
            if (!(diagonal > 0.0) || !Double.isFinite(diagonal)) {
                throw new IllegalStateException("Invalid Cholesky diagonal entry: " + diagonal);
            }
            logDeterminant += 2.0 * Math.log(diagonal);
        }
        return logDeterminant;
    }

    private MatrixFactorization factorizeImproperRootCovariance(double[][] covariance) {
        MatrixFactorization fixedRoot = factorizeProperRootCovariance(covariance, covariance.length);
        double[][] inverse = fixedRoot.precisionMatrix;
        int n = inverse.length;
        double[] inverseOnes = new double[n];
        double onesInverseOnes = 0.0;
        for (int row = 0; row < n; row++) {
            double sum = 0.0;
            for (int column = 0; column < n; column++) {
                sum += inverse[row][column];
            }
            inverseOnes[row] = sum;
            onesInverseOnes += sum;
        }
        if (!(onesInverseOnes > 0.0) || !Double.isFinite(onesInverseOnes)) {
            throw new IllegalStateException("Invalid improper-root normalization term: " + onesInverseOnes);
        }

        double[][] projectedPrecision = new double[n][n];
        for (int row = 0; row < n; row++) {
            for (int column = 0; column < n; column++) {
                projectedPrecision[row][column] = inverse[row][column]
                        - (inverseOnes[row] * inverseOnes[column]) / onesInverseOnes;
            }
        }
        return new MatrixFactorization(
                projectedPrecision,
            null,
                fixedRoot.logDeterminant + Math.log(onesInverseOnes),
                Math.max(0, n - 1));
    }

    private double[][] withRootPriorVariance(double[][] covariance, double variance) {
        if (variance == 0.0) {
            return covariance;
        }
        double[][] adjusted = new double[covariance.length][covariance.length];
        for (int row = 0; row < covariance.length; row++) {
            for (int column = 0; column < covariance[row].length; column++) {
                adjusted[row][column] = covariance[row][column] + variance;
            }
        }
        return adjusted;
    }

    private double[][] addJitter(double[][] covariance, double jitter) {
        int n = covariance.length;
        double[][] copy = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(covariance[i], 0, copy[i], 0, n);
            copy[i][i] += jitter;
        }
        return copy;
    }

    private double[] packLowerTriangular(RealMatrix lowerTriangular) {
        int dimension = lowerTriangular.getRowDimension();
        double[] packed = new double[dimension * (dimension + 1) / 2];
        int index = 0;
        for (int row = 0; row < dimension; row++) {
            for (int column = 0; column <= row; column++) {
                packed[index++] = lowerTriangular.getEntry(row, column);
            }
        }
        return packed;
    }

    private Node findMrca(Node left, Node right) {
        Set<Integer> ancestors = new HashSet<>();
        Node current = left;
        while (current != null) {
            ancestors.add(current.getNr());
            current = current.getParent();
        }

        current = right;
        while (current != null) {
            if (ancestors.contains(current.getNr())) {
                return current;
            }
            current = current.getParent();
        }
        throw new IllegalStateException("No MRCA found for taxa " + left.getID() + " and " + right.getID());
    }

    protected double computeCurrentTreeLogLikelihood(Tree tree) {
        if (likelihoodEvaluator == null || runtimeLikelihoodTree == null) {
            throw new IllegalStateException("A likelihood template is required to evaluate tree log likelihoods.");
        }
        Tree backupTree = runtimeLikelihoodTree.copy();
        try {
            runtimeLikelihoodTree.assignFrom(tree);
            runtimeLikelihoodTree.setEverythingDirty(true);
            SampledRateContext evaluationRateContext = currentSampledRateContext.alignToTree(runtimeLikelihoodTree);
            sampledBranchRateModel.setCurrentSampledRateContext(evaluationRateContext);
            double logP = likelihoodEvaluator.calculateLogP();
            if (!Double.isFinite(logP)) {
                throw new IllegalStateException("Non-finite tree log likelihood for sampled tree " + tree.getID() + ": " + logP);
            }
            return logP;
        } finally {
            runtimeLikelihoodTree.assignFrom(backupTree);
            runtimeLikelihoodTree.setEverythingDirty(true);
            sampledBranchRateModel.setCurrentSampledRateContext(currentSampledRateContext);
        }
    }

    protected Tree getRuntimeLikelihoodTree() {
        return getRuntimeLikelihoodTree(likelihoodTemplate);
    }

    private Tree getRuntimeLikelihoodTree(GenericTreeLikelihood likelihood) {
        TreeInterface tree = likelihood.treeInput.get();
        if (!(tree instanceof Tree runtimeTree)) {
            throw new IllegalArgumentException("likelihood template must use a mutable Tree state node");
        }
        return runtimeTree;
    }

    private GenericTreeLikelihood createTrainingLikelihoodEvaluator(
            GenericTreeLikelihood template,
            Base branchRateModel) {
        try {
            if (!(template.treeInput.get() instanceof Tree templateTree)) {
                throw new IllegalArgumentException("likelihood template must use a mutable Tree state node");
            }
            Tree copiedTree = new Tree(templateTree.getRoot().copy());
            copiedTree.setID(templateTree.getID());
            GenericTreeLikelihood evaluator = template.getClass().getDeclaredConstructor().newInstance();
            List<Object> initArguments = new ArrayList<>();
            Collections.addAll(
                    initArguments,
                    "data", template.dataInput.get(),
                    "tree", copiedTree,
                    "siteModel", template.siteModelInput.get());
            if (branchRateModel != null) {
                initArguments.add("branchRateModel");
                initArguments.add(branchRateModel);
            }
            addOptionalTrainingLikelihoodInput(initArguments, template, "useAmbiguities");
            addOptionalTrainingLikelihoodInput(initArguments, template, "useTipLikelihoods");
            addOptionalTrainingLikelihoodInput(initArguments, template, "scaling");
            addOptionalTrainingLikelihoodInput(initArguments, template, "rootFrequencies");
            addOptionalTrainingLikelihoodInput(initArguments, template, "implementation");
            addOptionalTrainingLikelihoodInput(initArguments, template, "threads");
            addOptionalTrainingLikelihoodInput(initArguments, template, "proportions");
            evaluator.initByName(initArguments.toArray());
            return evaluator;
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException
                | NoSuchMethodException e) {
            throw new IllegalArgumentException(
                    "Failed to construct training likelihood evaluator from template class "
                            + template.getClass().getName(),
                    e);
        }
    }

    private static void addOptionalTrainingLikelihoodInput(
            List<Object> initArguments,
            GenericTreeLikelihood template,
            String inputName) {
        Input<?> input = findInputByName(template, inputName);
        if (input == null) {
            return;
        }
        Object value = input.get();
        if (value == null) {
            return;
        }
        initArguments.add(inputName);
        initArguments.add("scaling".equals(inputName) ? value.toString() : value);
    }

    private static Input<?> findInputByName(GenericTreeLikelihood template, String inputName) {
        for (Input<?> input : template.listInputs()) {
            if (inputName.equals(input.getName())) {
                return input;
            }
        }
        return null;
    }

    private List<File> normalizeLogFiles(List<File> logFiles, int expectedSize) {
        if (logFiles == null || logFiles.isEmpty()) {
            throw new IllegalArgumentException(
                    "Posterior-tree training requires paired trainingLogFile entries containing column '"
                            + targetLogLikelihoodColumn + "'.");
        }
        if (logFiles.size() != expectedSize) {
            throw new IllegalArgumentException(
                    "trainingLogFile count must match trainingTreeFile count when provided; expected "
                            + expectedSize + " but was " + logFiles.size());
        }
        List<File> normalized = new ArrayList<>(logFiles.size());
        for (File logFile : logFiles) {
            if (logFile == null) {
                throw new IllegalArgumentException("trainingLogFile must not contain null entries");
            }
            if (!logFile.exists()) {
                throw new IllegalArgumentException("Training log file does not exist: " + logFile.getAbsolutePath());
            }
            normalized.add(logFile);
        }
        return normalized;
    }

    private record FileTreeSelectionPlan(
            File treeFile,
            int parsedCount,
            int selectedCount,
            Set<Integer> retainedSelectedIndices) {
        private FileTreeSelectionPlan(File treeFile, int parsedCount, int selectedCount) {
            this(treeFile, parsedCount, selectedCount, Collections.emptySet());
        }

        private FileTreeSelectionPlan withRetainedSelectedIndices(Set<Integer> retainedSelectedIndices) {
            return new FileTreeSelectionPlan(
                    treeFile,
                    parsedCount,
                    selectedCount,
                    Collections.unmodifiableSet(retainedSelectedIndices));
        }
    }

    private record TreeBlockContext(Map<String, String> translationMap, int origin, List<String> taxa) {
        private TreeBlockContext withTranslation(Map<String, String> translationMap) {
            int indexedOrigin = getIndexedTranslationMapOrigin(translationMap);
            List<String> indexedTaxa = indexedOrigin == -1 ? null : getIndexedTranslationMap(translationMap, indexedOrigin);
            return new TreeBlockContext(translationMap, indexedOrigin, indexedTaxa);
        }
    }

    private record NexusTreeCommand(String command, String arguments) {
        private NexusTreeCommand(String commandString) {
            this(
                    commandString.trim().replaceAll("\\s+", " ").split(" ")[0].toLowerCase(),
                    extractArguments(commandString));
        }

        private static String extractArguments(String commandString) {
            String normalized = commandString.trim().replaceAll("\\s+", " ");
            String command = normalized.split(" ")[0].toLowerCase();
            try {
                return normalized.substring(command.length() + 1);
            } catch (IndexOutOfBoundsException ex) {
                return "";
            }
        }

        private boolean isCommand(String commandName) {
            return command.equals(commandName.toLowerCase());
        }

        private boolean isEndOfBlock() {
            return command.equals("end");
        }
    }

    private record ParsedTreeCommand(Tree tree, String id) {
    }

    @FunctionalInterface
    private interface RetainedTreeProcessor {
        void process(Tree tree, int retainedIndex);
    }

    private LoggedPosteriorSample requireLoggedPosteriorSample(
            Tree tree,
            Map<Long, LoggedPosteriorSample> loggedRatesBySample,
            File logFile) {
        Long sampleNumber = extractTreeSampleNumber(tree);
        if (sampleNumber == null) {
            throw new IllegalArgumentException(
                    "Retained posterior tree is missing a numeric sample number in its id; cannot match it to training log "
                            + logFile.getAbsolutePath() + ".");
        }
        LoggedPosteriorSample matched = loggedRatesBySample.get(sampleNumber);
        if (matched == null) {
            throw new IllegalArgumentException(
                    "Could not find training-log row for posterior tree sample " + sampleNumber
                            + " in " + logFile.getAbsolutePath() + ". Exact Sample-based matching is required.");
        }
        return matched;
    }

    private int resolveTargetLogLikelihoodColumn(String[] headers, File logFile) {
        int exactIndex = findColumnIndex(headers, targetLogLikelihoodColumn);
        if (exactIndex < 0) {
            throw new IllegalArgumentException(
                    "Could not find required target log-likelihood column '" + targetLogLikelihoodColumn
                            + "' in training log " + logFile.getAbsolutePath() + ".");
        }
        return exactIndex;
    }

    private Double extractLoggedTargetLogLikelihood(String[] fields, int targetLogLikelihoodIndex) {
        if (targetLogLikelihoodIndex < 0 || fields.length <= targetLogLikelihoodIndex) {
            throw new IllegalArgumentException("Training log row is missing target column '" + targetLogLikelihoodColumn + "'.");
        }
        String rawValue = fields[targetLogLikelihoodIndex].trim();
        if (rawValue.isEmpty() || "NaN".equalsIgnoreCase(rawValue)) {
            throw new IllegalArgumentException(
                    "Training log contains an empty or NaN target likelihood in column '"
                            + targetLogLikelihoodColumn + "'.");
        }
        double parsed = Double.parseDouble(rawValue);
        if (!Double.isFinite(parsed)) {
            throw new IllegalArgumentException(
                    "Training log contains a non-finite target likelihood in column '"
                            + targetLogLikelihoodColumn + "': " + parsed);
        }
        return parsed;
    }

    private List<LoggedParameterBinding> resolveLoggedParameterBindings(String[] headers, File logFile) {
        if (loggedParametersById.isEmpty()) {
            return Collections.emptyList();
        }
        List<LoggedParameterBinding> bindings = new ArrayList<>();
        for (StateNode parameter : loggedParametersById.values()) {
            if (parameter.getID() == null || parameter.getID().isBlank()) {
                continue;
            }
            int[] indices = resolveParameterColumnIndices(headers, parameter);
            if (indices == null) {
                continue;
            }
            if (indices.length != dimensionOf(parameter)) {
                throw new IllegalArgumentException(
                        "Incomplete logged state for parameter " + parameter.getID()
                                + " in training log " + logFile.getAbsolutePath() + ".");
            }
            bindings.add(new LoggedParameterBinding(parameter, indices));
        }
        return bindings;
    }


    /**
     * BEAST3 spec parameters are StateNodes but share no Parameter.Base supertype, so the
     * logged-parameter machinery dispatches on the concrete spec types alongside the legacy ones.
     */
    private static boolean isLoggableParameter(BEASTInterface o) {
        return o instanceof Parameter.Base<?> || o instanceof RealScalarParam
                || o instanceof RealVectorParam || o instanceof IntVectorParam;
    }

    private static int dimensionOf(StateNode parameter) {
        if (parameter instanceof Parameter.Base<?> p) return p.getDimension();
        if (parameter instanceof RealScalarParam) return 1;
        if (parameter instanceof RealVectorParam<?> v) return v.size();
        if (parameter instanceof IntVectorParam<?> v) return v.size();
        throw new IllegalArgumentException("Unsupported logged parameter type: " + parameter.getClass().getName());
    }

    private int[] resolveParameterColumnIndices(String[] headers, StateNode parameter) {
        if (dimensionOf(parameter) == 1) {
            int exactIndex = findColumnIndex(headers, parameter.getID());
            return exactIndex >= 0 ? new int[] {exactIndex} : null;
        }

        int[] indices = new int[dimensionOf(parameter)];
        for (int valueIndex = 0; valueIndex < dimensionOf(parameter); valueIndex++) {
            String columnName = parameter.getID() + "." + (valueIndex + 1);
            int columnIndex = findColumnIndex(headers, columnName);
            if (columnIndex < 0) {
                return null;
            }
            indices[valueIndex] = columnIndex;
        }
        return indices;
    }

    private List<LoggedParameterSample> extractLoggedParameterSamples(
            String[] fields,
            List<LoggedParameterBinding> parameterBindings) {
        if (parameterBindings.isEmpty()) {
            return Collections.emptyList();
        }
        List<LoggedParameterSample> samples = new ArrayList<>(parameterBindings.size());
        for (LoggedParameterBinding binding : parameterBindings) {
            LoggedParameterSample sample = LoggedParameterSample.fromLogFields(binding.parameter, binding.columnIndices, fields);
            if (sample != null) {
                samples.add(sample);
            }
        }
        return samples;
    }

    private double extractStrictClockMeanRate(List<LoggedParameterSample> parameterSamples) {
        if (!(likelihoodTemplate.branchRateModelInput.get() instanceof StrictClockModel strictClockModel)) {
            return Double.NaN;
        }
        Object meanRateInput = strictClockModel.meanRateInput.get();
        String meanRateId = null;
        if (meanRateInput instanceof BEASTInterface beastObject && beastObject.getID() != null && !beastObject.getID().isBlank()) {
            meanRateId = beastObject.getID();
        } else if (meanRateInput instanceof RealParameter realParameter && realParameter.getID() != null && !realParameter.getID().isBlank()) {
            meanRateId = realParameter.getID();
        }
        if (meanRateId == null) {
            return Double.NaN;
        }
        for (LoggedParameterSample parameterSample : parameterSamples) {
            if (meanRateId.equals(parameterSample.getParameterId()) && parameterSample.getRealValues() != null
                    && parameterSample.getRealValues().length == 1) {
                return parameterSample.getRealValues()[0];
            }
        }
        return Double.NaN;
    }

    private int findColumnIndex(String[] headers, String columnName) {
        for (int i = 0; i < headers.length; i++) {
            if (columnName.equals(headers[i])) {
                return i;
            }
        }
        return -1;
    }

    private Long extractTreeSampleNumber(Tree tree) {
        if (tree.getID() == null) {
            return null;
        }
        Matcher matcher = TRAILING_INTEGER_PATTERN.matcher(tree.getID());
        if (!matcher.find()) {
            return null;
        }
        return Long.parseLong(matcher.group(1));
    }

    private static String resolveTargetLogLikelihoodColumnName(
            GenericTreeLikelihood likelihoodTemplate,
            String configuredTargetLogLikelihoodColumn) {
        if (configuredTargetLogLikelihoodColumn != null && !configuredTargetLogLikelihoodColumn.isBlank()) {
            return configuredTargetLogLikelihoodColumn;
        }
        if (likelihoodTemplate != null && likelihoodTemplate.getID() != null && !likelihoodTemplate.getID().isBlank()) {
            return likelihoodTemplate.getID();
        }
        return "treeLikelihood";
    }

    private void withSampledRateContext(SampledRateContext rateContext, Runnable task) {
        SampledRateContext previousContext = currentSampledRateContext;
        currentSampledRateContext = rateContext == null ? SampledRateContext.empty() : rateContext;
        sampledBranchRateModel.setCurrentSampledRateContext(currentSampledRateContext);
        try {
            task.run();
        } finally {
            currentSampledRateContext = previousContext;
            sampledBranchRateModel.setCurrentSampledRateContext(previousContext);
        }
    }

    private void withLoggedStateContext(LoggedStateContext loggedStateContext, Runnable task) {
        LoggedStateContext normalizedContext = loggedStateContext == null ? LoggedStateContext.empty() : loggedStateContext;
        if (normalizedContext.parameterSamples.isEmpty()) {
            task.run();
            return;
        }

        List<LoggedParameterSample> backups = new ArrayList<>(normalizedContext.parameterSamples.size());
        try {
            for (LoggedParameterSample parameterSample : normalizedContext.parameterSamples) {
                backups.add(LoggedParameterSample.captureCurrentValue(parameterSample.parameter));
                parameterSample.apply();
            }
            task.run();
        } finally {
            for (int i = backups.size() - 1; i >= 0; i--) {
                backups.get(i).apply();
            }
        }
    }

    private Map<String, StateNode> collectLoggedParameters(BEASTInterface... roots) {
        Map<String, StateNode> parameters = new LinkedHashMap<>();
        Set<BEASTInterface> visited = Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        for (BEASTInterface root : roots) {
            collectLoggedParameters(root, parameters, visited);
        }
        return Collections.unmodifiableMap(parameters);
    }

    private void collectLoggedParameters(
            BEASTInterface root,
            Map<String, StateNode> parameters,
            Set<BEASTInterface> visited) {
        if (root == null || !visited.add(root)) {
            return;
        }
        if (isLoggableParameter(root) && root.getID() != null && !root.getID().isBlank()) {
            StateNode parameter = (StateNode) root;
            parameters.putIfAbsent(parameter.getID(), parameter);
        }
        for (BEASTInterface child : root.listActiveBEASTObjects()) {
            collectLoggedParameters(child, parameters, visited);
        }
    }

    private String joinFilePaths(List<File> treeFiles) {
        List<String> paths = new ArrayList<>(treeFiles.size());
        for (File treeFile : treeFiles) {
            paths.add(treeFile.getAbsolutePath());
        }
        return String.join(", ", paths);
    }

        private record MatrixFactorization(
            double[][] precisionMatrix,
            double[] covarianceCholeskyLower,
            double logDeterminant,
            int logNormalizerDimension) {
    }

    private record LoggedParameterBinding(StateNode parameter, int[] columnIndices) {
    }

    private record LoggedPosteriorSample(
            long sampleNumber,
            double strictClockMeanRate,
            Double targetLogLikelihood,
            List<LoggedParameterSample> parameterSamples) {
    }

    private record LoggedPosteriorSamples(
            Map<Long, LoggedPosteriorSample> bySampleNumber) {
        private static LoggedPosteriorSamples empty() {
            return new LoggedPosteriorSamples(Collections.emptyMap());
        }
    }

    private static final class LoggedStateContext {
        private static final LoggedStateContext EMPTY = new LoggedStateContext(Collections.emptyList(), null);

        private final List<LoggedParameterSample> parameterSamples;
        private final Double targetLogLikelihood;

        private LoggedStateContext(List<LoggedParameterSample> parameterSamples, Double targetLogLikelihood) {
            this.parameterSamples = parameterSamples;
            this.targetLogLikelihood = targetLogLikelihood;
        }

        private static LoggedStateContext empty() {
            return EMPTY;
        }

        private static LoggedStateContext fromLoggedPosteriorSample(LoggedPosteriorSample loggedPosteriorSample) {
            if (loggedPosteriorSample == null) {
                return EMPTY;
            }
            List<LoggedParameterSample> parameterSamples = loggedPosteriorSample.parameterSamples == null
                    ? Collections.emptyList()
                    : loggedPosteriorSample.parameterSamples;
            if (parameterSamples.isEmpty() && loggedPosteriorSample.targetLogLikelihood == null) {
                return EMPTY;
            }
            return new LoggedStateContext(parameterSamples, loggedPosteriorSample.targetLogLikelihood);
        }

        private double getTargetLogLikelihood() {
            if (targetLogLikelihood == null) {
                throw new IllegalStateException("Missing target likelihood in logged state context.");
            }
            return targetLogLikelihood;
        }
    }

    private static final class LoggedParameterSample {
        private final StateNode parameter;
        private final double[] realValues;
        private final int[] integerValues;

        private LoggedParameterSample(StateNode parameter, double[] realValues, int[] integerValues) {
            this.parameter = parameter;
            this.realValues = realValues;
            this.integerValues = integerValues;
        }

        private static LoggedParameterSample fromLogFields(
                StateNode parameter,
                int[] columnIndices,
                String[] fields) {
            if (parameter instanceof RealParameter) {
                double[] values = new double[columnIndices.length];
                for (int i = 0; i < columnIndices.length; i++) {
                    if (fields.length <= columnIndices[i]) {
                        throw new IllegalArgumentException(
                                "Training log row is missing real-valued field for parameter " + parameter.getID() + ".");
                    }
                    values[i] = Double.parseDouble(fields[columnIndices[i]]);
                }
                return new LoggedParameterSample(parameter, values, null);
            }
            if (parameter instanceof IntegerParameter) {
                int[] values = new int[columnIndices.length];
                for (int i = 0; i < columnIndices.length; i++) {
                    if (fields.length <= columnIndices[i]) {
                        throw new IllegalArgumentException(
                                "Training log row is missing integer-valued field for parameter " + parameter.getID() + ".");
                    }
                    values[i] = Integer.parseInt(fields[columnIndices[i]]);
                }
                return new LoggedParameterSample(parameter, null, values);
            }
            if (parameter instanceof RealScalarParam || parameter instanceof RealVectorParam) {
                double[] values = new double[columnIndices.length];
                for (int i = 0; i < columnIndices.length; i++) {
                    if (fields.length <= columnIndices[i]) {
                        throw new IllegalArgumentException(
                                "Training log row is missing real-valued field for parameter " + parameter.getID() + ".");
                    }
                    values[i] = Double.parseDouble(fields[columnIndices[i]]);
                }
                return new LoggedParameterSample(parameter, values, null);
            }
            if (parameter instanceof IntVectorParam) {
                int[] values = new int[columnIndices.length];
                for (int i = 0; i < columnIndices.length; i++) {
                    if (fields.length <= columnIndices[i]) {
                        throw new IllegalArgumentException(
                                "Training log row is missing integer-valued field for parameter " + parameter.getID() + ".");
                    }
                    values[i] = Integer.parseInt(fields[columnIndices[i]]);
                }
                return new LoggedParameterSample(parameter, null, values);
            }
            throw new IllegalArgumentException("Unsupported logged parameter type: " + parameter.getClass().getName());
        }

        private static LoggedParameterSample captureCurrentValue(StateNode parameter) {
            if (parameter instanceof RealParameter realParameter) {
                double[] values = new double[realParameter.getDimension()];
                for (int i = 0; i < values.length; i++) {
                    values[i] = realParameter.getValue(i);
                }
                return new LoggedParameterSample(parameter, values, null);
            }
            if (parameter instanceof IntegerParameter integerParameter) {
                int[] values = new int[integerParameter.getDimension()];
                for (int i = 0; i < values.length; i++) {
                    values[i] = integerParameter.getValue(i);
                }
                return new LoggedParameterSample(parameter, null, values);
            }
            if (parameter instanceof RealScalarParam<?> p) {
                return new LoggedParameterSample(parameter, new double[] {p.get()}, null);
            }
            if (parameter instanceof RealVectorParam<?> p) {
                double[] values = new double[p.size()];
                for (int i = 0; i < values.length; i++) {
                    values[i] = p.get(i);
                }
                return new LoggedParameterSample(parameter, values, null);
            }
            if (parameter instanceof IntVectorParam<?> p) {
                int[] values = new int[p.size()];
                for (int i = 0; i < values.length; i++) {
                    values[i] = p.get(i);
                }
                return new LoggedParameterSample(parameter, null, values);
            }
            throw new IllegalArgumentException("Unsupported logged parameter type: " + parameter.getClass().getName());
        }

        private void apply() {
            if (parameter instanceof RealParameter realParameter) {
                for (int i = 0; i < realValues.length; i++) {
                    realParameter.setValue(i, realValues[i]);
                }
                return;
            }
            if (parameter instanceof IntegerParameter integerParameter) {
                for (int i = 0; i < integerValues.length; i++) {
                    integerParameter.setValue(i, integerValues[i]);
                }
                return;
            }
            if (parameter instanceof RealScalarParam<?> p) {
                p.set(realValues[0]);
                return;
            }
            if (parameter instanceof RealVectorParam<?> p) {
                for (int i = 0; i < realValues.length; i++) {
                    p.set(i, realValues[i]);
                }
                return;
            }
            if (parameter instanceof IntVectorParam<?> p) {
                for (int i = 0; i < integerValues.length; i++) {
                    p.set(i, integerValues[i]);
                }
                return;
            }
            throw new IllegalArgumentException("Unsupported logged parameter type: " + parameter.getClass().getName());
        }

        private String getParameterId() {
            return parameter.getID();
        }

        private double[] getRealValues() {
            return realValues;
        }
    }

    private static final class SampledRateContext {
        private static final SampledRateContext EMPTY = new SampledRateContext(Double.NaN, null);

        private final double strictClockMeanRate;
        private final double[] explicitBranchRates;

        private SampledRateContext(double strictClockMeanRate, double[] explicitBranchRates) {
            this.strictClockMeanRate = strictClockMeanRate;
            this.explicitBranchRates = explicitBranchRates;
        }

        private static SampledRateContext empty() {
            return EMPTY;
        }

        private static SampledRateContext fromTree(Tree tree, LoggedPosteriorSample loggedRate) {
            double[] explicitBranchRates = extractExplicitBranchRates(tree);
            double strictClockMeanRate = loggedRate == null ? Double.NaN : loggedRate.strictClockMeanRate;
            if (explicitBranchRates == null && !Double.isFinite(strictClockMeanRate)) {
                return EMPTY;
            }
            return new SampledRateContext(strictClockMeanRate, explicitBranchRates);
        }

        private static double[] extractExplicitBranchRates(Tree tree) {
            double[] rates = new double[tree.getNodeCount()];
            Arrays.fill(rates, Double.NaN);
            boolean foundRate = false;
            for (Node node : tree.getNodesAsArray()) {
                if (node.isRoot()) {
                    continue;
                }
                Double parsedRate = extractNodeRate(node);
                if (parsedRate != null) {
                    rates[node.getNr()] = parsedRate;
                    foundRate = true;
                }
            }
            return foundRate ? rates : null;
        }

        private static Double extractNodeRate(Node node) {
            Object metadataValue = node.getMetaData(TREE_RATE_METADATA_KEY);
            if (metadataValue == null) {
                metadataValue = node.getLengthMetaData(TREE_RATE_METADATA_KEY);
            }
            if (metadataValue instanceof Number number) {
                return number.doubleValue();
            }
            if (metadataValue instanceof String text && !text.isBlank()) {
                return Double.parseDouble(text);
            }
            return null;
        }

        private Double getExplicitRate(Node node) {
            if (explicitBranchRates == null) {
                return null;
            }
            double rate = explicitBranchRates[node.getNr()];
            return Double.isFinite(rate) ? rate : null;
        }

        private boolean hasStrictClockMeanRate() {
            return Double.isFinite(strictClockMeanRate);
        }

        private SampledRateContext alignToTree(Tree tree) {
            if (tree == null || explicitBranchRates == null) {
                return this;
            }
            double[] alignedRates = extractExplicitBranchRates(tree);
            if (alignedRates == null) {
                return new SampledRateContext(strictClockMeanRate, null);
            }
            return new SampledRateContext(strictClockMeanRate, alignedRates);
        }
    }

    // The simple name `Base` must be fully qualified inside this class. Spec Base implements
    // beast.base.evolution.branchratemodel.BranchRateModel, which has its own nested class
    // named Base; that inherited member type shadows the single-type import in this scope.
    private static final class SampledBranchRateModel
            extends beast.base.spec.evolution.branchratemodel.Base {
        private final beast.base.spec.evolution.branchratemodel.Base delegate;
        private SampledRateContext currentSampledRateContext = SampledRateContext.empty();

        private SampledBranchRateModel(beast.base.spec.evolution.branchratemodel.Base delegate) {
            this.delegate = delegate;
        }

        private beast.base.spec.evolution.branchratemodel.Base getDelegate() {
            return delegate;
        }

        @Override
        public void initAndValidate() {
            // No-op: this model delegates to the existing branch-rate model and overlays sampled rates.
        }

        @Override
        public double getRateForBranch(Node node) {
            return getRateForContext(node, currentSampledRateContext);
        }

        private double getRateForContext(Node node, SampledRateContext rateContext) {
            if (node != null && rateContext != null) {
                Double explicitRate = rateContext.getExplicitRate(node);
                if (explicitRate != null) {
                    return explicitRate;
                }
                if (rateContext.hasStrictClockMeanRate()) {
                    return rateContext.strictClockMeanRate;
                }
            }
            if (delegate != null) {
                return delegate.getRateForBranch(node);
            }
            return 1.0;
        }

        private void setCurrentSampledRateContext(SampledRateContext sampledRateContext) {
            currentSampledRateContext = sampledRateContext == null ? SampledRateContext.empty() : sampledRateContext;
        }

        @Override
        protected boolean requiresRecalculation() {
            return true;
        }
    }
}
