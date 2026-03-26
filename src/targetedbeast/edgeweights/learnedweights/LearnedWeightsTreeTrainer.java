package targetedbeast.edgeweights.learnedweights;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import beast.base.core.Log;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.likelihood.GenericTreeLikelihood;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.evolution.tree.coalescent.ConstantPopulation;
import beast.base.evolution.tree.coalescent.RandomTree;
import beast.base.inference.parameter.RealParameter;
import beast.base.parser.NexusParser;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.NonPositiveDefiniteMatrixException;
import org.apache.commons.math3.linear.RealMatrix;

public class LearnedWeightsTreeTrainer {

    private static final double CHOLESKY_THRESHOLD = 1e-12;
    private static final double[] JITTER_SCHEDULE = new double[] {0.0, 1e-10, 1e-8, 1e-6};

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
    private final int nTrainingTrees;
    private final String[] canonicalTaxa;

    public LearnedWeightsTreeTrainer(
            GenericTreeLikelihood likelihoodTemplate,
            Alignment alignment,
            int nTrainingTrees) {
        this.likelihoodTemplate = likelihoodTemplate;
        this.alignment = alignment;
        this.nTrainingTrees = nTrainingTrees;
        this.canonicalTaxa = alignment.getTaxaNames().toArray(new String[0]);
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
                burninPercentage,
                thinInterval,
                maxSamples);
    }

    public List<TrainingTreeSample> generateTrainingSamplesFromTreeFiles(
            List<File> treeFiles,
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
        if (burninPercentage < 0 || burninPercentage >= 100) {
            throw new IllegalArgumentException("burninPercentage must be in [0, 100), but was " + burninPercentage);
        }
        if (thinInterval <= 0) {
            throw new IllegalArgumentException("thinInterval must be > 0, but was " + thinInterval);
        }

        final long startNanos = System.nanoTime();
        List<Tree> selectedTrees = new ArrayList<>();
        int parsedTreeCount = 0;
        for (File treeFile : normalizedFiles) {
            List<Tree> parsedTrees = loadTreesFromFile(treeFile);
            parsedTreeCount += parsedTrees.size();
            List<Tree> fileSelection = selectPosteriorTrees(parsedTrees, burninPercentage, thinInterval, 0);
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
            Tree tree = selectedTrees.get(treeIdx);
            Double treeLogLikelihood = computeCurrentTreeLogLikelihood(tree);
            if (treeLogLikelihood != null && Double.isFinite(treeLogLikelihood)) {
                samples.add(buildTrainingSample(tree, treeLogLikelihood));
            }
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
        double rootHeight = tree.getRoot().getHeight();
        Map<String, Node> leafByTaxon = new HashMap<>();
        for (Node node : tree.getNodesAsArray()) {
            if (node.isLeaf()) {
                leafByTaxon.put(node.getID(), node);
            }
        }

        for (int i = 0; i < n; i++) {
            Node left = leafByTaxon.get(canonicalTaxa[i]);
            assert left != null : "Taxon not found in training tree: " + canonicalTaxa[i];
            covariance[i][i] = rootHeight - left.getHeight();
            if (covariance[i][i] < 0.0 && covariance[i][i] > -1e-12) {
                covariance[i][i] = 0.0;
            }
            assert covariance[i][i] >= 0.0 : "Negative marginal Brownian path length for taxon "
                    + canonicalTaxa[i] + ": " + covariance[i][i];
            for (int j = i + 1; j < n; j++) {
                Node right = leafByTaxon.get(canonicalTaxa[j]);
                assert right != null : "Taxon not found in training tree: " + canonicalTaxa[j];
                Node mrca = findMrca(left, right);
                double sharedPathLength = rootHeight - mrca.getHeight();
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
        Tree runtimeTree = getRuntimeLikelihoodTree();
        Tree backupTree = runtimeTree.copy();
        try {
            runtimeTree.assignFrom(tree);
            runtimeTree.setEverythingDirty(true);
            double logP = likelihoodTemplate.calculateLogP();
            return Double.isFinite(logP) ? logP : null;
        } finally {
            runtimeTree.assignFrom(backupTree);
            runtimeTree.setEverythingDirty(true);
        }
    }

    protected Tree getRuntimeLikelihoodTree() {
        TreeInterface tree = likelihoodTemplate.treeInput.get();
        if (!(tree instanceof Tree runtimeTree)) {
            throw new IllegalArgumentException("likelihood template must use a mutable Tree state node");
        }
        return runtimeTree;
    }

    private List<Tree> capSelectedTrees(List<Tree> trees, int maxSamples) {
        if (maxSamples <= 0 || trees.size() <= maxSamples) {
            return trees;
        }
        List<Tree> capped = new ArrayList<>(maxSamples);
        double step = (double) trees.size() / maxSamples;
        for (int sampleIndex = 0; sampleIndex < maxSamples; sampleIndex++) {
            int sourceIndex = Math.min(trees.size() - 1, (int) Math.floor(sampleIndex * step));
            capped.add(trees.get(sourceIndex));
        }
        return capped;
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
}
