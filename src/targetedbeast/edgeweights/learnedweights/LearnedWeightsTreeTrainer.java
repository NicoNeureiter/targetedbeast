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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import beast.base.core.BEASTInterface;
import beast.base.core.Input;
import beast.base.core.Log;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.branchratemodel.BranchRateModel;
import beast.base.evolution.branchratemodel.StrictClockModel;
import beast.base.evolution.likelihood.GenericTreeLikelihood;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.Parameter;
import beast.base.inference.parameter.RealParameter;
import beast.base.evolution.tree.coalescent.ConstantPopulation;
import beast.base.evolution.tree.coalescent.RandomTree;
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
        public final double targetLogLikelihood;
        public final double[][] precisionMatrix;
        public final double logDeterminant;

        public TrainingTreeSample(double targetLogLikelihood, double[][] precisionMatrix, double logDeterminant) {
            this.targetLogLikelihood = targetLogLikelihood;
            this.precisionMatrix = precisionMatrix;
            this.logDeterminant = logDeterminant;
        }

        public int getTaxonCount() {
            return precisionMatrix.length;
        }
    }

    private final Alignment alignment;
    private final GenericTreeLikelihood likelihoodTemplate;
    private final GenericTreeLikelihood likelihoodEvaluator;
    private final int nTrainingTrees;
    private final String[] canonicalTaxa;
    private final SampledBranchRateModel sampledBranchRateModel;
    private final Tree runtimeLikelihoodTree;
    private final Map<String, Parameter.Base<?>> loggedParametersById;
    private SampledRateContext currentSampledRateContext = SampledRateContext.empty();

    public LearnedWeightsTreeTrainer(
            GenericTreeLikelihood likelihoodTemplate,
            Alignment alignment,
            int nTrainingTrees) {
        this.likelihoodTemplate = likelihoodTemplate;
        this.alignment = alignment;
        this.nTrainingTrees = nTrainingTrees;
        this.canonicalTaxa = alignment.getTaxaNames().toArray(new String[0]);
        this.sampledBranchRateModel = new SampledBranchRateModel(likelihoodTemplate.branchRateModelInput.get());
        this.likelihoodEvaluator = createTrainingLikelihoodEvaluator(likelihoodTemplate, sampledBranchRateModel);
        this.runtimeLikelihoodTree = getRuntimeLikelihoodTree(likelihoodEvaluator);
        this.loggedParametersById = collectLoggedParameters(likelihoodEvaluator, sampledBranchRateModel.getDelegate());
    }

    public List<TrainingTreeSample> generateTrainingSamples() {
        final long startNanos = System.nanoTime();
        List<TrainingTreeSample> samples = new ArrayList<>();
        int logEvery = Math.max(1, nTrainingTrees / 10);
        Log.info.println("LearnedWeights: generating " + nTrainingTrees + " prior-sampled training trees...");
        for (int treeIdx = 0; treeIdx < nTrainingTrees; treeIdx++) {
            Tree tree = sampleRandomTree(alignment);
            Double treeLogLikelihood = computeCurrentTreeLogLikelihood(tree);
            if (treeLogLikelihood != null && Double.isFinite(treeLogLikelihood)) {
                samples.add(buildTrainingSample(tree, treeLogLikelihood));
            }
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

    public List<TrainingTreeSample> generateTrainingSamplesFromTreeFile(
            File treeFile,
            int burninPercentage,
            int thinInterval,
            int maxSamples) throws IOException {
        return generateTrainingSamplesFromTreeFiles(
                List.of(treeFile),
                Collections.emptyList(),
                burninPercentage,
                thinInterval,
                maxSamples);
    }

    public List<TrainingTreeSample> generateTrainingSamplesFromTreeFiles(
            List<File> treeFiles,
            int burninPercentage,
            int thinInterval,
            int maxSamples) throws IOException {
        return generateTrainingSamplesFromTreeFiles(
            treeFiles,
            Collections.emptyList(),
            burninPercentage,
            thinInterval,
            maxSamples);
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
        List<PosteriorTreeSample> selectedTrees = new ArrayList<>();
        int parsedTreeCount = 0;
        for (int fileIndex = 0; fileIndex < normalizedFiles.size(); fileIndex++) {
            File treeFile = normalizedFiles.get(fileIndex);
            File logFile = normalizedLogFiles.isEmpty() ? null : normalizedLogFiles.get(fileIndex);
            List<Tree> parsedTrees = loadTreesFromFile(treeFile);
            parsedTreeCount += parsedTrees.size();
                LoggedPosteriorSamples loggedPosteriorSamples = logFile == null
                    ? LoggedPosteriorSamples.empty()
                    : loadLoggedPosteriorSamples(logFile);
            List<LoggedPosteriorSample> orderedLoggedRates = logFile == null
                ? Collections.emptyList()
                    : selectPosteriorLogRows(loggedPosteriorSamples.orderedSamples, burninPercentage, thinInterval, 0);
            List<PosteriorTreeSample> fileSelection = selectPosteriorTreeSamples(
                parsedTrees,
                    loggedPosteriorSamples.bySampleNumber,
                orderedLoggedRates,
                burninPercentage,
                thinInterval,
                0);
            selectedTrees.addAll(fileSelection);
            Log.info.println(String.format(
                    "LearnedWeights: selected %d/%d posterior trees from %s (burnin=%d%%, thin=%d)",
                    fileSelection.size(),
                    parsedTrees.size(),
                    treeFile.getAbsolutePath(),
                    burninPercentage,
                    thinInterval));
        }
        selectedTrees = capSelectedTrees(selectedTrees, maxSamples);

        Log.info.println(String.format(
                "LearnedWeights: generating training samples from posterior tree file%s %s (parsed=%d, selected=%d, burnin=%d%%, thin=%d, maxSamples=%s)",
                normalizedFiles.size() == 1 ? "" : "s",
                joinFilePaths(normalizedFiles),
                parsedTreeCount,
                selectedTrees.size(),
                burninPercentage,
                thinInterval,
                maxSamples > 0 ? Integer.toString(maxSamples) : "all"));

        List<TrainingTreeSample> samples = new ArrayList<>();
        int logEvery = Math.max(1, Math.max(1, selectedTrees.size()) / 10);
        for (int treeIdx = 0; treeIdx < selectedTrees.size(); treeIdx++) {
            PosteriorTreeSample posteriorTreeSample = selectedTrees.get(treeIdx);
            withSampledRateContext(posteriorTreeSample.rateContext, () -> {
                withLoggedStateContext(posteriorTreeSample.loggedStateContext, () -> {
                    Double treeLogLikelihood = posteriorTreeSample.loggedStateContext.getTargetLogLikelihood();
                    if (treeLogLikelihood == null) {
                        treeLogLikelihood = computeCurrentTreeLogLikelihood(posteriorTreeSample.tree);
                    }
                    if (treeLogLikelihood != null && Double.isFinite(treeLogLikelihood)) {
                        samples.add(buildTrainingSample(posteriorTreeSample.tree, treeLogLikelihood));
                    }
                });
            });
            int completed = treeIdx + 1;
            if (completed % logEvery == 0 || completed == selectedTrees.size()) {
                double elapsedSec = (System.nanoTime() - startNanos) / 1_000_000_000.0;
                Log.info.println(String.format(
                        "LearnedWeights: processed %d/%d posterior trees (usable=%d, elapsed=%.1fs)",
                        completed,
                        selectedTrees.size(),
                        samples.size(),
                        elapsedSec));
            }
        }
        return Collections.unmodifiableList(samples);
    }

    protected Tree sampleRandomTree(Alignment data) {
        ConstantPopulation popFunc = new ConstantPopulation();
        popFunc.initByName("popSize", new RealParameter("1.0"));

        RandomTree tree = new RandomTree();
        tree.initByName(
                "taxa", data,
                "populationModel", popFunc);
        return tree;
    }

    protected TrainingTreeSample buildTrainingSample(Tree tree, double targetLogLikelihood) {
        double[][] covariance = buildCovarianceMatrix(tree);
        MatrixFactorization factorization = factorizeCovariance(covariance);
        return new TrainingTreeSample(targetLogLikelihood, factorization.precisionMatrix, factorization.logDeterminant);
    }

    protected List<Tree> loadTreesFromFile(File treeFile) throws IOException {
        NexusParser parser = new NexusParser();
        parser.parseFile(treeFile);
        if (parser.trees == null || parser.trees.isEmpty()) {
            throw new IllegalStateException("No trees found in posterior tree file: " + treeFile.getAbsolutePath());
        }
        return parser.trees;
    }

    protected List<Tree> selectPosteriorTrees(
            List<Tree> parsedTrees,
            int burninPercentage,
            int thinInterval,
            int maxSamples) {
        if (parsedTrees.isEmpty()) {
            return Collections.emptyList();
        }

        int burninCount = parsedTrees.size() * burninPercentage / 100;
        List<Tree> selected = new ArrayList<>();

        for (int treeIndex = burninCount; treeIndex < parsedTrees.size(); treeIndex += thinInterval) {
            Tree tree = parsedTrees.get(treeIndex);
            selected.add(tree);
            if (maxSamples > 0 && selected.size() >= maxSamples) {
                break;
            }
        }
        return selected;
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
            assert left != null : "Taxon not found in training tree: " + canonicalTaxa[i];
            covariance[i][i] = cumulativeDiffusionLength[left.getNr()];
            if (covariance[i][i] < 0.0 && covariance[i][i] > -1e-12) {
                covariance[i][i] = 0.0;
            }
            assert covariance[i][i] >= 0.0 : "Negative marginal Brownian path length for taxon "
                    + canonicalTaxa[i] + ": " + covariance[i][i];
            for (int j = i + 1; j < n; j++) {
                Node right = leafByTaxon.get(canonicalTaxa[j]);
                assert right != null : "Taxon not found in training tree: " + canonicalTaxa[j];
                Node mrca = findMrca(left, right);
                double sharedPathLength = cumulativeDiffusionLength[mrca.getNr()];
                if (sharedPathLength < 0.0 && sharedPathLength > -1e-12) {
                    sharedPathLength = 0.0;
                }
                assert sharedPathLength >= 0.0 : "Negative shared Brownian path length: " + sharedPathLength;
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
            double branchDiffusionLength = getBranchRate(node) * Math.max(node.getLength(), 0.0);
            cumulativeDiffusionLength[node.getNr()] = cumulativeDiffusionLength[node.getParent().getNr()] + branchDiffusionLength;
        }

        for (Node child : node.getChildren()) {
            populateCumulativeDiffusionLengths(child, cumulativeDiffusionLength);
        }
    }

    private double getBranchRate(Node node) {
        return sampledBranchRateModel.getRateForContext(node, currentSampledRateContext);
    }

    private MatrixFactorization factorizeCovariance(double[][] covariance) {
        for (double jitter : JITTER_SCHEDULE) {
            try {
                RealMatrix matrix = new Array2DRowRealMatrix(addJitter(covariance, jitter), false);
                CholeskyDecomposition decomposition = new CholeskyDecomposition(
                        matrix,
                        CHOLESKY_THRESHOLD,
                        CHOLESKY_THRESHOLD);
                double determinant = decomposition.getDeterminant();
                if (!(determinant > 0.0) || !Double.isFinite(determinant)) {
                    throw new IllegalStateException("Non-positive or non-finite covariance determinant: " + determinant);
                }
                return new MatrixFactorization(
                        decomposition.getSolver().getInverse().getData(),
                        Math.log(determinant));
            } catch (NonPositiveDefiniteMatrixException | IllegalStateException ignored) {
                // try larger diagonal jitter
            }
        }
        throw new IllegalStateException("Failed to factorize Brownian covariance matrix for LearnedWeights training sample.");
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

    protected Double computeCurrentTreeLogLikelihood(Tree tree) {
        Tree backupTree = runtimeLikelihoodTree.copy();
        try {
            runtimeLikelihoodTree.assignFrom(tree);
            runtimeLikelihoodTree.setEverythingDirty(true);
            SampledRateContext evaluationRateContext = currentSampledRateContext.alignToTree(runtimeLikelihoodTree);
            sampledBranchRateModel.setCurrentSampledRateContext(evaluationRateContext);
            double logP = likelihoodEvaluator.calculateLogP();
            return Double.isFinite(logP) ? logP : null;
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

    private List<PosteriorTreeSample> capSelectedTrees(List<PosteriorTreeSample> trees, int maxSamples) {
        if (maxSamples <= 0 || trees.size() <= maxSamples) {
            return trees;
        }
        List<PosteriorTreeSample> capped = new ArrayList<>(maxSamples);
        double step = (double) trees.size() / maxSamples;
        for (int sampleIndex = 0; sampleIndex < maxSamples; sampleIndex++) {
            int sourceIndex = Math.min(trees.size() - 1, (int) Math.floor(sampleIndex * step));
            capped.add(trees.get(sourceIndex));
        }
        return capped;
    }

    private GenericTreeLikelihood createTrainingLikelihoodEvaluator(
            GenericTreeLikelihood template,
            BranchRateModel.Base branchRateModel) {
        try {
            if (!(template.treeInput.get() instanceof Tree templateTree)) {
                throw new IllegalArgumentException("likelihood template must use a mutable Tree state node");
            }
            Tree copiedTree = new Tree(templateTree.getRoot().copy());
            copiedTree.setID(templateTree.getID());
            GenericTreeLikelihood evaluator = template.getClass().getDeclaredConstructor().newInstance();
            List<Object> initArguments = new ArrayList<>();
            for (Input<?> input : template.listInputs()) {
                String inputName = input.getName();
                Object inputValue;
                if ("tree".equals(inputName)) {
                    inputValue = copiedTree;
                } else if ("branchRateModel".equals(inputName)) {
                    inputValue = branchRateModel;
                } else {
                    inputValue = input.get();
                }
                if (inputValue != null) {
                    initArguments.add(inputName);
                    initArguments.add(inputValue);
                }
            }
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

    private List<File> normalizeLogFiles(List<File> logFiles, int expectedSize) {
        if (logFiles == null || logFiles.isEmpty()) {
            return Collections.emptyList();
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

    private List<PosteriorTreeSample> selectPosteriorTreeSamples(
            List<Tree> parsedTrees,
            Map<Long, LoggedPosteriorSample> loggedRatesBySample,
            List<LoggedPosteriorSample> orderedLoggedRates,
            int burninPercentage,
            int thinInterval,
            int maxSamples) {
        if (parsedTrees.isEmpty()) {
            return Collections.emptyList();
        }

        int burninCount = parsedTrees.size() * burninPercentage / 100;
        List<PosteriorTreeSample> selected = new ArrayList<>();

        for (int treeIndex = burninCount; treeIndex < parsedTrees.size(); treeIndex += thinInterval) {
            Tree tree = parsedTrees.get(treeIndex);
            LoggedPosteriorSample loggedRate = resolveLoggedRate(tree, loggedRatesBySample, orderedLoggedRates, selected.size());
            selected.add(new PosteriorTreeSample(
                    tree,
                    SampledRateContext.fromTree(tree, loggedRate),
                    LoggedStateContext.fromLoggedPosteriorSample(loggedRate)));
            if (maxSamples > 0 && selected.size() >= maxSamples) {
                break;
            }
        }
        return selected;
    }

        private List<LoggedPosteriorSample> selectPosteriorLogRows(
            List<LoggedPosteriorSample> parsedLogRows,
            int burninPercentage,
            int thinInterval,
            int maxSamples) {
        if (parsedLogRows.isEmpty()) {
            return Collections.emptyList();
        }
        int burninCount = parsedLogRows.size() * burninPercentage / 100;
        List<LoggedPosteriorSample> selected = new ArrayList<>();
        for (int rowIndex = burninCount; rowIndex < parsedLogRows.size(); rowIndex += thinInterval) {
            selected.add(parsedLogRows.get(rowIndex));
            if (maxSamples > 0 && selected.size() >= maxSamples) {
                break;
            }
        }
        return selected;
    }

    private LoggedPosteriorSample resolveLoggedRate(
            Tree tree,
            Map<Long, LoggedPosteriorSample> loggedRatesBySample,
            List<LoggedPosteriorSample> orderedLoggedRates,
            int selectedIndex) {
        if (!loggedRatesBySample.isEmpty()) {
            Long sampleNumber = extractTreeSampleNumber(tree);
            if (sampleNumber != null) {
                LoggedPosteriorSample matched = loggedRatesBySample.get(sampleNumber);
                if (matched != null) {
                    return matched;
                }
            }
        }
        if (selectedIndex < orderedLoggedRates.size()) {
            return orderedLoggedRates.get(selectedIndex);
        }
        return null;
    }

    private LoggedPosteriorSamples loadLoggedPosteriorSamples(File logFile) throws IOException {
        List<LoggedPosteriorSample> orderedSamples = new ArrayList<>();
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
                Double targetLogLikelihood = extractLoggedTargetLogLikelihood(fields, targetLogLikelihoodIndex);
                List<LoggedParameterSample> parameterSamples = extractLoggedParameterSamples(fields, parameterBindings);
                double strictClockMeanRate = extractStrictClockMeanRate(parameterSamples);
                orderedSamples.add(new LoggedPosteriorSample(sampleNumber, strictClockMeanRate, targetLogLikelihood, parameterSamples));
            }
        }

        Map<Long, LoggedPosteriorSample> bySample = new LinkedHashMap<>();
        for (LoggedPosteriorSample sample : orderedSamples) {
            bySample.put(sample.sampleNumber, sample);
        }
        return new LoggedPosteriorSamples(orderedSamples, bySample);
    }

    private int resolveTargetLogLikelihoodColumn(String[] headers, File logFile) {
        if (likelihoodTemplate.getID() != null && !likelihoodTemplate.getID().isBlank()) {
            int exactIndex = findColumnIndex(headers, likelihoodTemplate.getID());
            if (exactIndex >= 0) {
                return exactIndex;
            }
            if (findColumnIndex(headers, "likelihood") >= 0) {
                Log.warning.println("LearnedWeights: ignoring generic likelihood column in training log "
                        + logFile.getAbsolutePath()
                        + " because it does not match training likelihood id " + likelihoodTemplate.getID()
                        + "; target likelihoods will be recomputed for training samples.");
            }
            return -1;
        }

        return findColumnIndex(headers, "likelihood");
    }

    private Double extractLoggedTargetLogLikelihood(String[] fields, int targetLogLikelihoodIndex) {
        if (targetLogLikelihoodIndex < 0 || fields.length <= targetLogLikelihoodIndex) {
            return null;
        }
        String rawValue = fields[targetLogLikelihoodIndex].trim();
        if (rawValue.isEmpty() || "NaN".equalsIgnoreCase(rawValue)) {
            return null;
        }
        double parsed = Double.parseDouble(rawValue);
        return Double.isFinite(parsed) ? parsed : null;
    }

    private List<LoggedParameterBinding> resolveLoggedParameterBindings(String[] headers, File logFile) {
        if (loggedParametersById.isEmpty()) {
            return Collections.emptyList();
        }
        List<LoggedParameterBinding> bindings = new ArrayList<>();
        for (Parameter.Base<?> parameter : loggedParametersById.values()) {
            if (parameter.getID() == null || parameter.getID().isBlank()) {
                continue;
            }
            int[] indices = resolveParameterColumnIndices(headers, parameter);
            if (indices == null) {
                continue;
            }
            if (indices.length != parameter.getDimension()) {
                Log.warning.println("LearnedWeights: ignoring incomplete logged state for parameter "
                        + parameter.getID() + " in training log " + logFile.getAbsolutePath() + ".");
                continue;
            }
            bindings.add(new LoggedParameterBinding(parameter, indices));
        }
        return bindings;
    }

    private int[] resolveParameterColumnIndices(String[] headers, Parameter.Base<?> parameter) {
        if (parameter.getDimension() == 1) {
            int exactIndex = findColumnIndex(headers, parameter.getID());
            return exactIndex >= 0 ? new int[] {exactIndex} : null;
        }

        int[] indices = new int[parameter.getDimension()];
        for (int valueIndex = 0; valueIndex < parameter.getDimension(); valueIndex++) {
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

    private Map<String, Parameter.Base<?>> collectLoggedParameters(BEASTInterface... roots) {
        Map<String, Parameter.Base<?>> parameters = new LinkedHashMap<>();
        Set<BEASTInterface> visited = Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        for (BEASTInterface root : roots) {
            collectLoggedParameters(root, parameters, visited);
        }
        return Collections.unmodifiableMap(parameters);
    }

    private void collectLoggedParameters(
            BEASTInterface root,
            Map<String, Parameter.Base<?>> parameters,
            Set<BEASTInterface> visited) {
        if (root == null || !visited.add(root)) {
            return;
        }
        if (root instanceof Parameter.Base<?> parameter && parameter.getID() != null && !parameter.getID().isBlank()) {
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

    private record MatrixFactorization(double[][] precisionMatrix, double logDeterminant) {
    }

    private record PosteriorTreeSample(Tree tree, SampledRateContext rateContext, LoggedStateContext loggedStateContext) {
    }

    private record LoggedParameterBinding(Parameter.Base<?> parameter, int[] columnIndices) {
    }

    private record LoggedPosteriorSample(
            long sampleNumber,
            double strictClockMeanRate,
            Double targetLogLikelihood,
            List<LoggedParameterSample> parameterSamples) {
    }

    private record LoggedPosteriorSamples(
            List<LoggedPosteriorSample> orderedSamples,
            Map<Long, LoggedPosteriorSample> bySampleNumber) {
        private static LoggedPosteriorSamples empty() {
            return new LoggedPosteriorSamples(Collections.emptyList(), Collections.emptyMap());
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

        private Double getTargetLogLikelihood() {
            return targetLogLikelihood;
        }
    }

    private static final class LoggedParameterSample {
        private final Parameter.Base<?> parameter;
        private final double[] realValues;
        private final int[] integerValues;

        private LoggedParameterSample(Parameter.Base<?> parameter, double[] realValues, int[] integerValues) {
            this.parameter = parameter;
            this.realValues = realValues;
            this.integerValues = integerValues;
        }

        private static LoggedParameterSample fromLogFields(
                Parameter.Base<?> parameter,
                int[] columnIndices,
                String[] fields) {
            if (parameter instanceof RealParameter) {
                double[] values = new double[columnIndices.length];
                for (int i = 0; i < columnIndices.length; i++) {
                    if (fields.length <= columnIndices[i]) {
                        return null;
                    }
                    values[i] = Double.parseDouble(fields[columnIndices[i]]);
                }
                return new LoggedParameterSample(parameter, values, null);
            }
            if (parameter instanceof IntegerParameter) {
                int[] values = new int[columnIndices.length];
                for (int i = 0; i < columnIndices.length; i++) {
                    if (fields.length <= columnIndices[i]) {
                        return null;
                    }
                    values[i] = Integer.parseInt(fields[columnIndices[i]]);
                }
                return new LoggedParameterSample(parameter, null, values);
            }
            return null;
        }

        private static LoggedParameterSample captureCurrentValue(Parameter.Base<?> parameter) {
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

    private static final class SampledBranchRateModel extends BranchRateModel.Base {
        private final BranchRateModel.Base delegate;
        private SampledRateContext currentSampledRateContext = SampledRateContext.empty();

        private SampledBranchRateModel(BranchRateModel.Base delegate) {
            this.delegate = delegate;
        }

        private BranchRateModel.Base getDelegate() {
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
