package targetedbeast.edgeweights;

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
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Distribution;
import beast.base.inference.State;
import targetedbeast.util.Alignment2PCA;

@Description("Brownian-motion edge weights with upward and downward Gaussian messages, "
    + "approximating full-tree trait likelihoods for targeted proposals.")
public class PCAWeightsBrownianFullLikelihood extends Distribution implements EdgeWeights {
    private static final double VARIANCE_FLOOR = 1e-12;
    private static final double NEUTRAL_VARIANCE = Double.POSITIVE_INFINITY;

    final public Input<Alignment> dataInput = new Input<>("data", "sequence data for the tree", Validate.REQUIRED);
    final public Input<TreeInterface> treeInput = new Input<>("tree", "phylogenetic tree with sequence data in the leaves", Validate.REQUIRED);
    final public Input<Double> maxWeightInput = new Input<>("maxWeight", "maximum weight for an edge", 10.0);
    final public Input<Double> minWeightInput = new Input<>("minWeight", "minimum weight for an edge", 0.01);
    final public Input<Integer> dimensionInput = new Input<>("dimension", "dimension of PCA points", 2);
    final public Input<String> valueInput = new Input<>("value", "comma separated list of taxon=x1 x2 x3 pairs; if specified, data is ignored");
    final public Input<Boolean> distanceBasedInput = new Input<>("distanceBased", "use distance matrix for PCA", true);
    final public Input<Boolean> compressedInput = new Input<>("compressed", "remove duplicate entries for PCA", true);
    final public Input<Boolean> useOneNormInput = new Input<>("useOneNorm", "use one norm for edge weights", true);
    final public Input<BranchRateModel.Base> branchRateModelInput = new Input<>(
            "branchRateModel",
            "optional branch rate model used to scale Brownian variance propagation along branches");
    final public Input<Boolean> useInverseMeanDistanceProposalInput = new Input<>(
            "useInverseMeanDistanceProposal",
            "use inverse distance between propagated attachment means instead of approximate full-tree log likelihoods",
            false);
    final public Input<Double> brownianRateInput = new Input<>(
            "brownianRate",
            "Brownian diffusion rate used when propagating Gaussian trait messages along branches",
            1.0);
        final public Input<Double> rootPriorVarianceInput = new Input<>(
            "rootPriorVariance",
            "root-state prior variance in Brownian diffusion units; use Infinity for a flat root prior and 0 for a root fixed at 0",
            Double.POSITIVE_INFINITY);
    final public Input<Double> offsetInput = new Input<>("offset", "offset in weight", 0.01);

    private int dim;
    private double offset;
    private boolean useOneNorm;
    private boolean useInverseMeanDistanceProposal;
    private BranchRateModel.Base branchRateModel;
    private double brownianRate;
    private double rootPriorVariance;
    private double proposalLogLikelihoodScale = 1.0;
    private double maxWeight;
    private double minWeight;

    private double[][] leafPoints;
    private double[][] insideMeans;
    private double[][] insideVariances;
    private double[] insideLogNorm;
    private double[][] insideWithoutMeans;
    private double[][] insideWithoutVariances;
    private double[] insideWithoutLogNorm;
    private boolean[] insideWithoutValid;
    private double[][] outsideMeans;
    private double[][] outsideVariances;
    private double[] outsideLogNorm;
    private boolean[] outsideValid;
    private double[] edgeMutations;
    private double[] cachedBranchRates;

    private boolean initialized;
    private boolean insideDirty = true;
    private int currentIgnoredNode = -1;

    @Override
    public void initAndValidate() {
        dim = dimensionInput.get();
        offset = offsetInput.get();
        useOneNorm = useOneNormInput.get();
        branchRateModel = branchRateModelInput.get();
        useInverseMeanDistanceProposal = useInverseMeanDistanceProposalInput.get();
        brownianRate = brownianRateInput.get();
        rootPriorVariance = rootPriorVarianceInput.get();
        maxWeight = maxWeightInput.get();
        minWeight = minWeightInput.get();

        if (branchRateModel == null && (!(brownianRate > 0.0) || !Double.isFinite(brownianRate))) {
            throw new IllegalArgumentException("brownianRate must be finite and > 0, but was " + brownianRate);
        }
        if (rootPriorVariance < 0.0 || Double.isNaN(rootPriorVariance)) {
            throw new IllegalArgumentException("rootPriorVariance must be >= 0 or Infinity, but was " + rootPriorVariance);
        }

        if (dataInput.get().getTaxonCount() != treeInput.get().getLeafNodeCount()) {
            String leaves = "?";
            if (treeInput.get() instanceof Tree tree) {
                leaves = String.join(", ", tree.getTaxaNames());
            }
            throw new IllegalArgumentException(String.format(
                    "The number of leaves in the tree (%d) does not match the number of sequences (%d). "
                            + "The tree has leaves [%s], while the data refers to taxa [%s].",
                    treeInput.get().getLeafNodeCount(), dataInput.get().getTaxonCount(), leaves,
                    String.join(", ", dataInput.get().getTaxaNames())));
        }

        initializeLeafPoints();
        initializeCaches();
        recomputeInsideMessages();
    }

    private void initializeLeafPoints() {
        int nodeCount = treeInput.get().getNodeCount();
        leafPoints = new double[nodeCount][dim];
        if (valueInput.get() != null) {
            parseValue();
        } else {
            calcValue();
        }
    }

    private void initializeCaches() {
        int nodeCount = treeInput.get().getNodeCount();
        insideMeans = new double[nodeCount][dim];
        insideVariances = new double[nodeCount][dim];
        insideLogNorm = new double[nodeCount];
        insideWithoutMeans = new double[nodeCount][dim];
        insideWithoutVariances = new double[nodeCount][dim];
        insideWithoutLogNorm = new double[nodeCount];
        insideWithoutValid = new boolean[nodeCount];
        outsideMeans = new double[nodeCount][dim];
        outsideVariances = new double[nodeCount][dim];
        outsideLogNorm = new double[nodeCount];
        outsideValid = new boolean[nodeCount];
        edgeMutations = new double[nodeCount];
        cachedBranchRates = new double[nodeCount];
        Arrays.fill(cachedBranchRates, Double.NaN);
    }

    private void calcValue() {
        Map<String, double[]> map = Alignment2PCA.getPoints(dataInput.get(),
                dim,
                distanceBasedInput.get(),
                compressedInput.get());

        Tree tree = (Tree) treeInput.get();
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isLeaf()) {
                continue;
            }
            double[] point = map.get(node.getID());
            if (point == null) {
                throw new IllegalArgumentException("No PCA point found for taxon " + node.getID());
            }
            System.arraycopy(point, 0, leafPoints[node.getNr()], 0, dim);
        }
    }

    private void parseValue() {
        String[] traits = valueInput.get().split(",");
        List<String> labels = treeInput.get().getTaxonset().asStringList();
        if (traits.length != labels.size()) {
            Log.warning("Number of points specified (" + traits.length + " should equal number of taxa (" + labels.size() + ")");
        }

        for (String trait : traits) {
            trait = trait.replaceAll("\\s+", " ");
            String[] parts = trait.split("=");
            if (parts.length != 2) {
                throw new IllegalArgumentException("could not parse trait: " + trait);
            }
            String taxonId = normalize(parts[0]);
            int taxonNr = findLeafNodeNrByTaxonId(taxonId);
            String[] point = parts[1].split("\\s+");
            if (point.length != dim) {
                throw new IllegalArgumentException("could not parse trait: " + trait + " since dimension is not " + dim);
            }
            for (int i = 0; i < dim; i++) {
                leafPoints[taxonNr][i] = Double.parseDouble(point[i]);
            }
        }
    }

    private int findLeafNodeNrByTaxonId(String taxonId) {
        for (Node node : ((Tree) treeInput.get()).getNodesAsArray()) {
            if (node.isLeaf() && taxonId.equals(node.getID())) {
                return node.getNr();
            }
        }
        throw new IllegalArgumentException("Unknown taxon in PCA value input: " + taxonId);
    }

    protected String normalize(String str) {
        if (str.charAt(0) == ' ') {
            str = str.substring(1);
        }
        if (str.endsWith(" ")) {
            str = str.substring(0, str.length() - 1);
        }
        return str;
    }

    private void ensureInitialized() {
        if (!initialized) {
            initialized = true;
        }
        if (insideDirty) {
            recomputeInsideMessages();
        }
    }

    private void recomputeInsideMessages() {
        Arrays.fill(edgeMutations, minWeight);
        recomputeInsideRecursive(treeInput.get().getRoot());
        insideDirty = false;
        clearWithoutNodeState();
        refreshCachedBranchRates();
    }

    private GaussianMessage recomputeInsideRecursive(Node node) {
        int nodeNr = node.getNr();
        if (node.isLeaf()) {
            System.arraycopy(leafPoints[nodeNr], 0, insideMeans[nodeNr], 0, dim);
            Arrays.fill(insideVariances[nodeNr], 0.0);
            insideLogNorm[nodeNr] = 0.0;
            return new GaussianMessage(insideMeans[nodeNr], insideVariances[nodeNr], 0.0);
        }

        GaussianMessage left = recomputeInsideRecursive(node.getLeft());
        GaussianMessage right = recomputeInsideRecursive(node.getRight());
        GaussianMessage leftUp = propagateMessage(left, node.getLeft(), node.getLeft().getLength());
        GaussianMessage rightUp = propagateMessage(right, node.getRight(), node.getRight().getLength());
        GaussianMessage merged = combineMessages(leftUp, rightUp);
        if (node.isRoot()) {
            merged = combineMessages(merged, rootPriorMessage());
        }

        copyMessage(merged, insideMeans[nodeNr], insideVariances[nodeNr]);
        insideLogNorm[nodeNr] = merged.logNorm;
        edgeMutations[node.getLeft().getNr()] = boundedEdgeWeight(distance(node.getLeft().getNr(), nodeNr));
        edgeMutations[node.getRight().getNr()] = boundedEdgeWeight(distance(node.getRight().getNr(), nodeNr));
        return new GaussianMessage(insideMeans[nodeNr], insideVariances[nodeNr], merged.logNorm);
    }

    private void clearWithoutNodeState() {
        Arrays.fill(insideWithoutValid, false);
        Arrays.fill(outsideValid, false);
        currentIgnoredNode = -1;
    }

    @Override
    public void updateByOperator() {
        insideDirty = true;
        ensureInitialized();
    }

    @Override
    public void updateByOperatorWithoutNode(int ignore, List<Integer> nodes) {
        ensureInitialized();
        currentIgnoredNode = ignore;
        Arrays.fill(insideWithoutValid, false);
        Arrays.fill(outsideValid, false);

        Node root = treeInput.get().getRoot();
        Node effectiveRoot = becomesUnary(root, ignore) ? getIncludedChild(root, ignore) : root;
        recomputeInsideWithoutRecursive(effectiveRoot, ignore);
        setRootOutside(effectiveRoot.getNr());
        if (effectiveRoot != root) {
            setNeutralOutside(root.getNr());
        }
        recomputeOutsideRecursive(effectiveRoot, ignore);
    }

    private GaussianMessage recomputeInsideWithoutRecursive(Node node, int ignore) {
        int nodeNr = node.getNr();
        if (insideWithoutValid[nodeNr]) {
            return new GaussianMessage(insideWithoutMeans[nodeNr], insideWithoutVariances[nodeNr], 0.0);
        }
        if (node.isLeaf()) {
            System.arraycopy(insideMeans[nodeNr], 0, insideWithoutMeans[nodeNr], 0, dim);
            System.arraycopy(insideVariances[nodeNr], 0, insideWithoutVariances[nodeNr], 0, dim);
            insideWithoutLogNorm[nodeNr] = 0.0;
            insideWithoutValid[nodeNr] = true;
            return new GaussianMessage(insideWithoutMeans[nodeNr], insideWithoutVariances[nodeNr], 0.0);
        }

        EffectiveBranch branch1 = getEffectiveBranch(node, 0, ignore);
        EffectiveBranch branch2 = getEffectiveBranch(node, 1, ignore);
        GaussianMessage message;
        if (branch1 == null) {
            GaussianMessage child2Message = recomputeInsideWithoutRecursive(branch2.node, ignore);
            message = propagateMessage(child2Message, branch2.diffusionLength);
        } else if (branch2 == null) {
            GaussianMessage child1Message = recomputeInsideWithoutRecursive(branch1.node, ignore);
            message = propagateMessage(child1Message, branch1.diffusionLength);
        } else {
            GaussianMessage up1 = propagateMessage(recomputeInsideWithoutRecursive(branch1.node, ignore), branch1.diffusionLength);
            GaussianMessage up2 = propagateMessage(recomputeInsideWithoutRecursive(branch2.node, ignore), branch2.diffusionLength);
            message = combineMessages(up1, up2);
        }

        copyMessage(message, insideWithoutMeans[nodeNr], insideWithoutVariances[nodeNr]);
    insideWithoutLogNorm[nodeNr] = message.logNorm;
        insideWithoutValid[nodeNr] = true;
        return new GaussianMessage(insideWithoutMeans[nodeNr], insideWithoutVariances[nodeNr], message.logNorm);
    }

    private void recomputeOutsideRecursive(Node node, int ignore) {
        if (node.isLeaf()) {
            return;
        }

        GaussianMessage nodeOutside = getOutsideMessage(node.getNr());
        EffectiveBranch leftBranch = getEffectiveBranch(node, 0, ignore);
        EffectiveBranch rightBranch = getEffectiveBranch(node, 1, ignore);

        if (leftBranch != null) {
            GaussianMessage siblingContribution = rightBranch == null
                    ? neutralMessage()
                    : propagateMessage(getInsideWithoutMessage(rightBranch.node.getNr()), rightBranch.diffusionLength);
            GaussianMessage combined = combineMessages(nodeOutside, siblingContribution);
            GaussianMessage childOutside = propagateMessage(combined, leftBranch.diffusionLength);
            setOutside(leftBranch.node.getNr(), childOutside);
            recomputeOutsideRecursive(leftBranch.node, ignore);
        }

        if (rightBranch != null) {
            GaussianMessage siblingContribution = leftBranch == null
                    ? neutralMessage()
                    : propagateMessage(getInsideWithoutMessage(leftBranch.node.getNr()), leftBranch.diffusionLength);
            GaussianMessage combined = combineMessages(nodeOutside, siblingContribution);
            GaussianMessage childOutside = propagateMessage(combined, rightBranch.diffusionLength);
            setOutside(rightBranch.node.getNr(), childOutside);
            recomputeOutsideRecursive(rightBranch.node, ignore);
        }
    }

    public double[] getInsideWithoutMeans(int nodeNr) {
        return insideWithoutMeans[nodeNr];
    }

    public double[] getInsideWithoutVariances(int nodeNr) {
        return insideWithoutVariances[nodeNr];
    }

    public double[] getOutsideMeans(int nodeNr) {
        return outsideMeans[nodeNr];
    }

    public double[] getOutsideVariances(int nodeNr) {
        return outsideVariances[nodeNr];
    }

    public double[] getPoints(int nodeNr) {
        ensureInitialized();
        return insideMeans[nodeNr];
    }

    public double[] getVariances(int nodeNr) {
        ensureInitialized();
        return insideVariances[nodeNr];
    }

    public double[] getLeafEmbeddings(int[] leafOrder) {
        ensureInitialized();
        double[] flattened = new double[leafOrder.length * dim];
        for (int i = 0; i < leafOrder.length; i++) {
            int nodeNr = leafOrder[i];
            if (!treeInput.get().getNode(nodeNr).isLeaf()) {
                throw new IllegalArgumentException("Leaf embedding requested for non-leaf node " + nodeNr);
            }
            System.arraycopy(leafPoints[nodeNr], 0, flattened, i * dim, dim);
        }
        return flattened;
    }

    void printDistanceMatrix(double[][] points) {
        int n = points.length;
        double[][] dists = new double[n][n];
        for (int i=0; i < n; i++) {

            System.out.printf("[%d] ", i);
            for (int j=0; j < n; j++) {
                dists[i][j] = distance(points[i], points[j]);
                System.out.printf("%.0f ", 10 * (dists[i][j]));
            }
            System.out.println();
        }
    }

    double distance(double[] p1, double[] p2) {
        double sum = 0.0;
        for (int i = 0; i < p1.length; i++) {
            double d = p1[i] - p2[i];
            sum += useOneNorm ? Math.abs(d) : d * d;
        }
        return useOneNorm ? sum : Math.sqrt(sum);
    }

    public void setLeafEmbeddings(double[] flattenedEmbeddings, int[] leafOrder) {
        // System.out.println();
        // System.out.println();
        // System.out.println();
        // printDistanceMatrix(leafPoints);
        // System.out.println();
        // System.out.println();
        if (flattenedEmbeddings.length != leafOrder.length * dim) {
            throw new IllegalArgumentException("Leaf embedding length mismatch: expected "
                    + (leafOrder.length * dim) + " but got " + flattenedEmbeddings.length);
        }
        for (int i = 0; i < leafOrder.length; i++) {
            int nodeNr = leafOrder[i];
            if (!treeInput.get().getNode(nodeNr).isLeaf()) {
                throw new IllegalArgumentException("Leaf embeddings can only be set for leaf nodes, got " + nodeNr);
            }
            System.arraycopy(flattenedEmbeddings, i * dim, leafPoints[nodeNr], 0, dim);
        }
        insideDirty = true;
        clearWithoutNodeState();
        // printDistanceMatrix(leafPoints);
        // System.out.println();
    }

    public GaussianMessage getFullConditional(int nodeNr) {
        ensureInitialized();
        if (currentIgnoredNode < 0) {
            updateByOperatorWithoutNode(-1, Collections.emptyList());
        }
        return combineMessages(getInsideWithoutMessage(nodeNr), getOutsideMessage(nodeNr));
    }

    public double getApproximateTreeLogLikelihood() {
        ensureInitialized();
        return insideLogNorm[treeInput.get().getRoot().getNr()];
    }

    public double getApproximateTreeLogLikelihoodAtNode(int nodeNr) {
        return getFullConditional(nodeNr).logNorm;
    }

    @Override
    public void fakeUpdateByOperator() {
        insideDirty = true;
    }

    @Override
    protected boolean requiresRecalculation() {
        if (dataInput.get().isDirtyCalculation()) {
            initializeLeafPoints();
            insideDirty = true;
        }
        if (branchRateModel != null && branchRateModel.isDirtyCalculation()) {
            insideDirty = true;
        }
        if (refreshCachedBranchRates()) {
            insideDirty = true;
        }
        if (treeInput.get().somethingIsDirty() || insideDirty) {
            ensureInitialized();
            return true;
        }
        return false;
    }

    @Override
    public void store() {
        super.store();
    }

    @Override
    public void prestore() {
    }

    @Override
    public void reset() {
        clearWithoutNodeState();
    }

    @Override
    public void restore() {
        super.restore();
        insideDirty = true;
    }

    @Override
    public double getEdgeWeights(int nodeNr) {
        ensureInitialized();
        return nodeNr == treeInput.get().getRoot().getNr() ? minWeight : edgeMutations[nodeNr];
    }

    @Override
    public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes) {
        return getTargetWeightsInteger(fromNodeNr, toNodes.stream().map(Node::getNr).toList());
    }

    @Override
    public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
        ensureInitialized();
        ensureWithoutNodeMessages(fromNodeNr);
        Node fromNode = getNode(fromNodeNr);
        GaussianMessage fromInside = getInsideMessage(fromNodeNr);
        if (useInverseMeanDistanceProposal) {
            double[] weights = new double[toNodeNrs.size()];
            for (int i = 0; i < toNodeNrs.size(); i++) {
                Node toNode = getNode(toNodeNrs.get(i));
                double toHeight = getToHeightWithoutNode(toNode, fromNode);
                weights[i] = toOperatorReadyWeight(1.0 / Math.max(
                        getMeanDist(fromNode, toNode, fromInside, getInsideWithoutMessage(toNodeNrs.get(i)), toHeight),
                        VARIANCE_FLOOR));
            }
            return weights;
        }

        double[] logScores = new double[toNodeNrs.size()];
        for (int i = 0; i < toNodeNrs.size(); i++) {
            Node toNode = getNode(toNodeNrs.get(i));
            double toHeight = getToHeightWithoutNode(toNode, fromNode);
            logScores[i] = computeScaledProposedTreeLogLikelihood(fromNode, toNode,
                fromInside, getInsideWithoutMessage(toNodeNrs.get(i)), toHeight);
        }
        return toOperatorReadyWeights(logScores);
    }

    @Override
    public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes, double toHeight) {
        return getTargetWeightsInteger(fromNodeNr, toNodes.stream().map(Node::getNr).toList(), toHeight);
    }

    public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, double toHeight) {
        ensureInitialized();
        ensureWithoutNodeMessages(fromNodeNr);
        Node fromNode = getNode(fromNodeNr);
        GaussianMessage fromInside = getInsideMessage(fromNodeNr);

        double[] weights = new double[toNodeNrs.size()];
        for (int i = 0; i < toNodeNrs.size(); i++) {
            Node toNode = getNode(toNodeNrs.get(i));
            GaussianMessage toInside = getInsideWithoutMessage(toNodeNrs.get(i));

            if (useInverseMeanDistanceProposal) {
                double dist = getMeanDist(fromNode, toNode, fromInside, toInside, toHeight);
                weights[i] = toOperatorReadyWeight(1.0 / Math.max(dist, VARIANCE_FLOOR));
            } else {
                weights[i] = computeScaledProposedTreeLogLikelihood(fromNode, toNode, fromInside, toInside, toHeight);
            }
        }
        if (!useInverseMeanDistanceProposal) {
            weights = toOperatorReadyWeights(weights);
        }

        return weights;
    }

    @Override
    public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes, List<Double> toHeights) {
        return getTargetWeightsInteger(fromNodeNr, toNodes.stream().map(Node::getNr).toList(), toHeights);
    }

    public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, List<Double> toHeights) {
        ensureInitialized();
        ensureWithoutNodeMessages(fromNodeNr);
        Node fromNode = getNode(fromNodeNr);
        GaussianMessage fromInside = getInsideMessage(fromNodeNr);

        double[] weights = new double[toNodeNrs.size()];
        for (int i = 0; i < toNodeNrs.size(); i++) {
            Node toNode = getNode(toNodeNrs.get(i));
            GaussianMessage toInside = getInsideWithoutMessage(toNodeNrs.get(i));

            if (useInverseMeanDistanceProposal) {
                double dist = getMeanDist(fromNode, toNode, fromInside, toInside, toHeights.get(i));
                weights[i] = toOperatorReadyWeight(1.0 / Math.max(dist, VARIANCE_FLOOR));
            } else {
                weights[i] = computeScaledProposedTreeLogLikelihood(fromNode, toNode, fromInside, toInside, toHeights.get(i));
            }
        }
        if (!useInverseMeanDistanceProposal) {
            weights = toOperatorReadyWeights(weights);
        }

        return weights;
    }

    public void setProposalLogLikelihoodScale(double scale) {
        if (!Double.isFinite(scale)) {
            throw new IllegalArgumentException("proposalLogLikelihoodScale must be finite, but was " + scale);
        }
        proposalLogLikelihoodScale = scale;
    }

    public double getProposalLogLikelihoodScale() {
        return proposalLogLikelihoodScale;
    }

    public double getToHeightWithoutNode(Node toNode, Node fromNode) {
        double minHeight = Math.max(toNode.getHeight(), fromNode.getHeight());
        Node effectiveParent = getEffectiveParent(toNode, currentIgnoredNode);
        if (effectiveParent == null) {
            return minHeight + 0.1;
        }
        return (minHeight + effectiveParent.getHeight()) / 2.0;
    }

    public double computeFullConditionalAgreement(Node fromNode, Node toNode, GaussianMessage fromFullConditional,
            GaussianMessage toFullConditional) {
        double toHeight = getToHeightWithoutNode(toNode, fromNode);
        return computeFullConditionalAgreement(fromNode, toNode, fromFullConditional, toFullConditional, toHeight);
    }

    public double computeFullConditionalAgreement(Node fromNode, Node toNode, GaussianMessage fromFullConditional,
            GaussianMessage toFullConditional, double toHeight) {
        GaussianMessage fromAtHeight = propagateMessage(fromFullConditional, fromNode, Math.max(0.0, toHeight - fromNode.getHeight()));
        GaussianMessage toAtHeight = propagateMessage(toFullConditional, toNode, Math.max(0.0, toHeight - toNode.getHeight()));
        return combineMessages(fromAtHeight, toAtHeight).logNorm;
    }

    public double computeProposedTreeLogLikelihood(Node fromNode, Node toNode, GaussianMessage fromInside, GaussianMessage toInsideWithout, double toHeight) {
        double total = 0.0;
        double[] contributions = computeProposedTreeLogLikelihoodContributions(fromNode, toNode, fromInside, toInsideWithout, toHeight);
        for (double contribution : contributions) {
            total += contribution;
        }
        return total;
    }

    public double[] computeProposedTreeLogLikelihoodContributions(Node fromNode, Node toNode, GaussianMessage fromInside,
            GaussianMessage toInsideWithout, double toHeight) {
        assert toHeight > fromNode.getHeight();
        assert toHeight > toNode.getHeight();

        GaussianMessage fromAtHeight = propagateMessage(fromInside, fromNode, toHeight - fromNode.getHeight());
        GaussianMessage toAtHeight = propagateMessage(toInsideWithout, toNode, toHeight - toNode.getHeight());
        double[] contributions = new double[dim];
        GaussianMessage parentMessage = combineMessagesWithContributions(fromAtHeight, toAtHeight, contributions);

        Node grandparent = getEffectiveParent(toNode, currentIgnoredNode);
        if (grandparent == null) {
            combineMessagesWithContributions(parentMessage, getOutsideMessage(toNode.getNr()), contributions);
            return contributions;
        }

        GaussianMessage gpMessage = propagateMessage(parentMessage, toNode, Math.max(0.0, grandparent.getHeight() - toHeight));
        EffectiveBranch siblingBranch = getEffectiveSiblingBranch(toNode, currentIgnoredNode);
        if (siblingBranch != null) {
            GaussianMessage siblingUp = propagateMessage(getInsideWithoutMessage(siblingBranch.node.getNr()), siblingBranch.diffusionLength);
            gpMessage = combineMessagesWithContributions(gpMessage, siblingUp, contributions);
        }

        if (outsideValid[grandparent.getNr()]) {
            combineMessagesWithContributions(gpMessage, getOutsideMessage(grandparent.getNr()), contributions);
            return contributions;
        }
        return contributions;
    }

    private double computeScaledProposedTreeLogLikelihood(Node fromNode, Node toNode, GaussianMessage fromInside,
            GaussianMessage toInsideWithout, double toHeight) {
        return proposalLogLikelihoodScale * computeProposedTreeLogLikelihood(fromNode, toNode, fromInside, toInsideWithout, toHeight);
    }

	private double getMeanDist(Node fromNode, Node toNode, GaussianMessage fromInside, GaussianMessage toInsideWithout, double toHeight) {
        assert toHeight > fromNode.getHeight();
        assert toHeight > toNode.getHeight();

        GaussianMessage toFullConditional = combineMessages(toInsideWithout, getOutsideMessage(toNode.getNr()));
        double squaredErrorSum = 0.0;
        for (int i = 0; i < dim; i++) {
            double diff = fromInside.mean[i] - toFullConditional.mean[i];
            double expectedSquaredError = diff * diff;  // + fromInside.variance[i] + toFullConditional.variance[i];
            squaredErrorSum += expectedSquaredError;
        }
		return Math.sqrt(squaredErrorSum);  //  divide by dim?
	}

    public double sqr(double x) {
        return x * x;
    }

    public Node getEffectiveParent(Node node, int ignore) {
        if (ignore < 0) {
            return node.getParent();
        }
        Node parent = node.getParent();
        while (parent != null && becomesUnary(parent, ignore)) {
            parent = parent.getParent();
        }
        return parent;
    }

    boolean becomesUnary(Node node, int ignore) {
        if (ignore < 0 || node == null || node.isLeaf()) {
            return false;
        }
        return node.getLeft().getNr() == ignore || node.getRight().getNr() == ignore;
    }

    Node getIncludedChild(Node node, int ignore) {
        if (node.getLeft().getNr() == ignore) {
            return node.getRight();
        }
        if (node.getRight().getNr() == ignore) {
            return node.getLeft();
        }
        throw new IllegalArgumentException("Node " + node.getID() + " does not have ignored child " + ignore);
    }

    Node getEffectiveChild(Node node, int childIndex, int ignore) {
        EffectiveBranch branch = getEffectiveBranch(node, childIndex, ignore);
        return branch == null ? null : branch.node;
    }

    EffectiveBranch getEffectiveBranch(Node node, int childIndex, int ignore) {
        Node child = node.getChild(childIndex);
        if (child.getNr() == ignore) {
            return null;
        }
        double diffusionLength = getDiffusionLength(child, child.getLength());
        while (becomesUnary(child, ignore)) {
            child = getIncludedChild(child, ignore);
            diffusionLength += getDiffusionLength(child, child.getLength());
        }
        return new EffectiveBranch(child, diffusionLength);
    }

    Node getEffectiveSibling(Node node, int ignore) {
        Node parent = getEffectiveParent(node, ignore);
        if (parent == null) {
            return null;
        }
        Node left = getEffectiveChild(parent, 0, ignore);
        Node right = getEffectiveChild(parent, 1, ignore);
        if (left == node) {
            return right;
        }
        if (right == node) {
            return left;
        }
        return null;
    }

    EffectiveBranch getEffectiveSiblingBranch(Node node, int ignore) {
        Node parent = getEffectiveParent(node, ignore);
        if (parent == null) {
            return null;
        }
        EffectiveBranch left = getEffectiveBranch(parent, 0, ignore);
        EffectiveBranch right = getEffectiveBranch(parent, 1, ignore);
        if (left != null && left.node == node) {
            return right;
        }
        if (right != null && right.node == node) {
            return left;
        }
        return null;
    }

    private GaussianMessage getInsideMessage(int nodeNr) {
        return new GaussianMessage(insideMeans[nodeNr], insideVariances[nodeNr], insideLogNorm[nodeNr]);
    }

    private GaussianMessage getInsideWithoutMessage(int nodeNr) {
        if (currentIgnoredNode < 0 || !insideWithoutValid[nodeNr]) {
            return getInsideMessage(nodeNr);
        }
        return new GaussianMessage(insideWithoutMeans[nodeNr], insideWithoutVariances[nodeNr], insideWithoutLogNorm[nodeNr]);
    }

    private GaussianMessage getOutsideMessage(int nodeNr) {
        if (!outsideValid[nodeNr]) {
            return neutralMessage();
        }
        return new GaussianMessage(outsideMeans[nodeNr], outsideVariances[nodeNr], outsideLogNorm[nodeNr]);
    }

    private void ensureWithoutNodeMessages(int ignoreNodeNr) {
        if (currentIgnoredNode != ignoreNodeNr) {
            updateByOperatorWithoutNode(ignoreNodeNr, Collections.emptyList());
        }
    }

    private void setNeutralOutside(int nodeNr) {
        Arrays.fill(outsideMeans[nodeNr], 0.0);
        Arrays.fill(outsideVariances[nodeNr], NEUTRAL_VARIANCE);
        outsideLogNorm[nodeNr] = 0.0;
        outsideValid[nodeNr] = true;
    }

    private void setRootOutside(int nodeNr) {
        GaussianMessage prior = rootPriorMessage();
        copyMessage(prior, outsideMeans[nodeNr], outsideVariances[nodeNr]);
        outsideLogNorm[nodeNr] = prior.logNorm;
        outsideValid[nodeNr] = true;
    }

    private void setOutside(int nodeNr, GaussianMessage message) {
        copyMessage(message, outsideMeans[nodeNr], outsideVariances[nodeNr]);
        outsideLogNorm[nodeNr] = message.logNorm;
        outsideValid[nodeNr] = true;
    }

    private GaussianMessage neutralMessage() {
        double[] mean = new double[dim];
        double[] variance = new double[dim];
        Arrays.fill(variance, NEUTRAL_VARIANCE);
        return new GaussianMessage(mean, variance, 0.0);
    }

    private GaussianMessage rootPriorMessage() {
        if (Double.isInfinite(rootPriorVariance)) {
            return neutralMessage();
        }
        double[] mean = new double[dim];
        double[] variance = new double[dim];
        Arrays.fill(variance, Math.max(rootPriorVariance, 0.0));
        return new GaussianMessage(mean, variance, 0.0);
    }

    private GaussianMessage propagateMessage(GaussianMessage message, Node branchNode, double branchLength) {
        return propagateMessage(message, getDiffusionLength(branchNode, branchLength));
    }

    private GaussianMessage propagateMessage(GaussianMessage message, double diffusionLength) {
        double[] mean = Arrays.copyOf(message.mean, dim);
        double[] variance = Arrays.copyOf(message.variance, dim);
        for (int i = 0; i < dim; i++) {
            variance[i] = propagatedVariance(variance[i], diffusionLength);
        }
        return new GaussianMessage(mean, variance, message.logNorm);
    }

    private GaussianMessage combineMessages(GaussianMessage first, GaussianMessage second) {
        return combineMessagesWithContributions(first, second, null);
    }

    private GaussianMessage combineMessagesWithContributions(GaussianMessage first, GaussianMessage second, double[] contributions) {
        double[] mean = new double[dim];
        double[] variance = new double[dim];
        double logNorm = first.logNorm + second.logNorm;
        for (int i = 0; i < dim; i++) {
            double v1 = sanitizeVariance(first.variance[i]);
            double v2 = sanitizeVariance(second.variance[i]);
            boolean neutralFirst = Double.isInfinite(v1);
            boolean neutralSecond = Double.isInfinite(v2);
            if (neutralFirst && neutralSecond) {
                mean[i] = 0.0;
                variance[i] = NEUTRAL_VARIANCE;
                continue;
            }
            if (neutralFirst) {
                mean[i] = second.mean[i];
                variance[i] = v2;
                continue;
            }
            if (neutralSecond) {
                mean[i] = first.mean[i];
                variance[i] = v1;
                continue;
            }

            double precision1 = 1.0 / v1;
            double precision2 = 1.0 / v2;
            double precisionSum = precision1 + precision2;
            variance[i] = Math.max(1.0 / precisionSum, VARIANCE_FLOOR);
            mean[i] = variance[i] * (first.mean[i] * precision1 + second.mean[i] * precision2);
            double sumVariance = Math.max(v1 + v2, VARIANCE_FLOOR);
            double diff = first.mean[i] - second.mean[i];
            double axisContribution = -0.5 * (Math.log(sumVariance) + (diff * diff) / sumVariance);
            logNorm += axisContribution;
            if (contributions != null) {
                contributions[i] += axisContribution;
            }
        }
        return new GaussianMessage(mean, variance, logNorm);
    }

    private void copyMessage(GaussianMessage source, double[] targetMean, double[] targetVariance) {
        System.arraycopy(source.mean, 0, targetMean, 0, dim);
        System.arraycopy(source.variance, 0, targetVariance, 0, dim);
    }

    private boolean refreshCachedBranchRates() {
        if (branchRateModel == null || cachedBranchRates == null) {
            return false;
        }
        boolean changed = false;
        for (Node node : getTree().getNodesAsArray()) {
            if (node.isRoot()) {
                continue;
            }
            double rate = getBranchRate(node);
            if (cachedBranchRates[node.getNr()] != rate) {
                cachedBranchRates[node.getNr()] = rate;
                changed = true;
            }
        }
        return changed;
    }

    private double getDiffusionLength(Node node, double branchLength) {
        return getBranchRate(node) * Math.max(branchLength, 0.0);
    }

    private double getBranchRate(Node node) {
        if (branchRateModel == null)
            return brownianRate;
        else
            return branchRateModel.getRateForBranch(node);
    }

    private double propagatedVariance(double variance, double diffusionLength) {
        if (Double.isInfinite(variance)) {
            return variance;
        }
        return Math.max(variance + diffusionLength, VARIANCE_FLOOR);
    }

    private double sanitizeVariance(double variance) {
        if (Double.isInfinite(variance)) {
            return variance;
        }
        return Math.max(variance, VARIANCE_FLOOR);
    }

    private double boundedEdgeWeight(double rawWeight) {
        return Math.max(minWeight, Math.min(maxWeight, rawWeight));
    }

    private double distance(int childNodeNr, int parentNodeNr) {
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            sum += norm(insideMeans[childNodeNr][i] - insideMeans[parentNodeNr][i]);
        }
        return Math.sqrt(sum);
    }

    private double[] exponentiateLogScores(double[] logScores) {
        double[] scores = new double[logScores.length];
        double maxLogScore = Double.NEGATIVE_INFINITY;
        for (double logScore : logScores) {
            maxLogScore = Math.max(maxLogScore, logScore);
        }
        if (!Double.isFinite(maxLogScore)) {
            Arrays.fill(scores, 1.0);
            return scores;
        }
        for (int i = 0; i < logScores.length; i++) {
            scores[i] = Math.max(Math.exp(logScores[i] - maxLogScore), VARIANCE_FLOOR) + offset;
        }
        return scores;
    }

    private double[] toOperatorReadyWeights(double[] logScores) {
        double[] scores = exponentiateLogScores(logScores);
        for (int i = 0; i < scores.length; i++) {
            scores[i] = toOperatorReadyWeight(scores[i]);
        }
        return scores;
    }

    private double toOperatorReadyWeight(double finalWeight) {
        return Math.sqrt(Math.max(finalWeight, VARIANCE_FLOOR));
    }

    private double norm(double x) {
        return useOneNorm ? Math.abs(x) : x * x;
    }

    public Node getNode(int i) {
        return treeInput.get().getNode(i);
    }

    public Tree getTree() {
        return (Tree) treeInput.get();
    }

    @Override
    public List<String> getArguments() {
        return null;
    }

    @Override
    public List<String> getConditions() {
        return null;
    }

    @Override
    public void sample(State state, Random random) {
    }

    @Override
    public double minEdgeWeight() {
        return minWeight;
    }

    public static final class GaussianMessage {
        public final double[] mean;
        public final double[] variance;
        public final double logNorm;

        GaussianMessage(double[] mean, double[] variance, double logNorm) {
            this.mean = mean;
            this.variance = variance;
            this.logNorm = logNorm;
        }
    }

    static final class EffectiveBranch {
        private final Node node;
        private final double diffusionLength;

        private EffectiveBranch(Node node, double diffusionLength) {
            this.node = node;
            this.diffusionLength = diffusionLength;
        }
    }
}
