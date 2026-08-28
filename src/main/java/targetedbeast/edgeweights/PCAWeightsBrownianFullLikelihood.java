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
import beast.base.spec.evolution.branchratemodel.Base;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Distribution;
import beast.base.inference.State;
import targetedbeast.util.Alignment2PCA;

@Description("Brownian-motion edge weights with upward and downward Gaussian messages, "
    + "approximating full-tree trait likelihoods for targeted proposals.")
public class PCAWeightsBrownianFullLikelihood extends Distribution implements EdgeWeights {
    private static final double VARIANCE_FLOOR = 1e-16;
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
    final public Input<Base> branchRateModelInput = new Input<>(
            "branchRateModel",
            "optional branch rate model used to scale Brownian variance propagation along branches");
    final public Input<Boolean> useInverseMeanDistanceProposalInput = new Input<>(
            "useInverseMeanDistanceProposal",
            "use inverse distance between propagated attachment means instead of approximate full-tree log likelihoods",
            false);
        final public Input<Boolean> scaleQuadraticPartOnlyInput = new Input<>(
            "scaleQuadraticPartOnly",
            "if true, proposal scales are applied only to the quadratic data-fit terms while Gaussian normalization/log-det-like terms remain unscaled",
            false);
            final public Input<Boolean> separateBaseAndQuadraticCalibrationInput = new Input<>(
                "separateBaseAndQuadraticCalibration",
                "if true, proposal base/log-normalization terms and quadratic data-fit terms use separate per-axis calibration scales",
                false);
    final public Input<Double> brownianRateInput = new Input<>(
            "brownianRate",
            "Brownian diffusion rate used when propagating Gaussian trait messages along branches",
            1.0);
    final public Input<String> brownianTraitRatesInput = new Input<>(
            "brownianTraitRates",
            "comma or whitespace separated per-trait Brownian diffusion rates; when provided these scale each trait axis independently and multiply any branchRateModel branch rates",
            "");
    final public Input<String> rootPriorVariancesInput = new Input<>(
            "rootPriorVariances",
            "comma or whitespace separated per-trait root-state prior variances in Brownian diffusion units; when blank the scalar rootPriorVariance input is used for every trait",
            "");
        final public Input<Double> rootPriorVarianceInput = new Input<>(
            "rootPriorVariance",
            "root-state prior variance in Brownian diffusion units; use Infinity for a flat root prior and 0 for a root fixed at 0",
            Double.POSITIVE_INFINITY);
    final public Input<Double> offsetInput = new Input<>("offset", "offset in weight", 0.01);

    private int dim;
    private double offset;
    private boolean useOneNorm;
    private boolean useInverseMeanDistanceProposal;
    private boolean scaleQuadraticPartOnly;
    private boolean separateBaseAndQuadraticCalibration;
    private Base branchRateModel;
    private double brownianRate;
    private double[] brownianTraitRates;
    private double[] rootPriorVariances;
    private double proposalBaseLogLikelihoodScale = 1.0;
    private double[] proposalBaseLogLikelihoodScales;
    private double proposalLogLikelihoodScale = 1.0;
    private double[] proposalLogLikelihoodScales;
    private double maxWeight;
    private double minWeight;

    private double[][] leafPoints;
    private double[][] insideMeans;
    private double[][] insideVariances;
    private double[] insideLogNorm;
    private double[][] insideBaseContributions;           // per-axis base accumulated in the subtree below this node
    private double[][] insideQuadraticContributions;      // per-axis quadratic accumulated in the subtree below this node
    private double[][] insideWithoutMeans;
    private double[][] insideWithoutVariances;
    private double[] insideWithoutLogNorm;
    private boolean[] insideWithoutValid;
    private double[][] insideWithoutBaseContributions;        // per-axis base, fromNode excluded
    private double[][] insideWithoutQuadraticContributions;   // per-axis quadratic, fromNode excluded
    private double[][] outsideMeans;
    private double[][] outsideVariances;
    private double[] outsideLogNorm;
    private boolean[] outsideValid;
    private double[][] outsideBaseContributions;          // per-axis base accumulated outside this node
    private double[][] outsideQuadraticContributions;     // per-axis quadratic accumulated outside this node
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
        scaleQuadraticPartOnly = scaleQuadraticPartOnlyInput.get();
        separateBaseAndQuadraticCalibration = separateBaseAndQuadraticCalibrationInput.get();
        brownianRate = brownianRateInput.get();
        brownianTraitRates = resolveBrownianTraitRates();
        rootPriorVariances = resolveRootPriorVariances();
        maxWeight = maxWeightInput.get();
        minWeight = minWeightInput.get();
        proposalBaseLogLikelihoodScales = new double[dim];
        Arrays.fill(proposalBaseLogLikelihoodScales, proposalBaseLogLikelihoodScale);
        proposalLogLikelihoodScales = new double[dim];
        Arrays.fill(proposalLogLikelihoodScales, proposalLogLikelihoodScale);

        if (separateBaseAndQuadraticCalibration && scaleQuadraticPartOnly) {
            throw new IllegalArgumentException(
                "separateBaseAndQuadraticCalibration and scaleQuadraticPartOnly cannot both be enabled");
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
        insideBaseContributions = new double[nodeCount][dim];
        insideQuadraticContributions = new double[nodeCount][dim];
        insideWithoutMeans = new double[nodeCount][dim];
        insideWithoutVariances = new double[nodeCount][dim];
        insideWithoutLogNorm = new double[nodeCount];
        insideWithoutValid = new boolean[nodeCount];
        insideWithoutBaseContributions = new double[nodeCount][dim];
        insideWithoutQuadraticContributions = new double[nodeCount][dim];
        outsideMeans = new double[nodeCount][dim];
        outsideVariances = new double[nodeCount][dim];
        outsideLogNorm = new double[nodeCount];
        outsideValid = new boolean[nodeCount];
        outsideBaseContributions = new double[nodeCount][dim];
        outsideQuadraticContributions = new double[nodeCount][dim];
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
            Arrays.fill(insideBaseContributions[nodeNr], 0.0);
            Arrays.fill(insideQuadraticContributions[nodeNr], 0.0);
            return new GaussianMessage(insideMeans[nodeNr], insideVariances[nodeNr], 0.0);
        }

        GaussianMessage left = recomputeInsideRecursive(node.getLeft());
        GaussianMessage right = recomputeInsideRecursive(node.getRight());
        GaussianMessage leftUp = propagateMessage(left, node.getLeft(), node.getLeft().getLength());
        GaussianMessage rightUp = propagateMessage(right, node.getRight(), node.getRight().getLength());

        // Per-axis accumulator: start with the children's accumulated contributions,
        // then have the combine append its own per-axis contribution.
        int leftNr  = node.getLeft().getNr();
        int rightNr = node.getRight().getNr();
        double[] base = insideBaseContributions[nodeNr];
        double[] quad = insideQuadraticContributions[nodeNr];
        for (int d = 0; d < dim; d++) {
            base[d] = insideBaseContributions[leftNr][d] + insideBaseContributions[rightNr][d];
            quad[d] = insideQuadraticContributions[leftNr][d] + insideQuadraticContributions[rightNr][d];
        }
        GaussianMessage merged = combineMessagesWithContributionParts(leftUp, rightUp, base, quad);
        if (node.isRoot()) {
            merged = combineMessagesWithContributionParts(merged, rootPriorMessage(), base, quad);
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
            Arrays.fill(insideWithoutBaseContributions[nodeNr], 0.0);
            Arrays.fill(insideWithoutQuadraticContributions[nodeNr], 0.0);
            insideWithoutValid[nodeNr] = true;
            return new GaussianMessage(insideWithoutMeans[nodeNr], insideWithoutVariances[nodeNr], 0.0);
        }

        EffectiveBranch branch1 = getEffectiveBranch(node, 0, ignore);
        EffectiveBranch branch2 = getEffectiveBranch(node, 1, ignore);
        GaussianMessage message;
        double[] base = insideWithoutBaseContributions[nodeNr];
        double[] quad = insideWithoutQuadraticContributions[nodeNr];
        if (branch1 == null) {
            GaussianMessage child2Message = recomputeInsideWithoutRecursive(branch2.node, ignore);
            message = propagateMessage(child2Message, branch2.diffusionLength);
            // No combine at this node — contributions pass through from the included child.
            int childNr = branch2.node.getNr();
            for (int d = 0; d < dim; d++) {
                base[d] = insideWithoutBaseContributions[childNr][d];
                quad[d] = insideWithoutQuadraticContributions[childNr][d];
            }
        } else if (branch2 == null) {
            GaussianMessage child1Message = recomputeInsideWithoutRecursive(branch1.node, ignore);
            message = propagateMessage(child1Message, branch1.diffusionLength);
            int childNr = branch1.node.getNr();
            for (int d = 0; d < dim; d++) {
                base[d] = insideWithoutBaseContributions[childNr][d];
                quad[d] = insideWithoutQuadraticContributions[childNr][d];
            }
        } else {
            GaussianMessage up1 = propagateMessage(recomputeInsideWithoutRecursive(branch1.node, ignore), branch1.diffusionLength);
            GaussianMessage up2 = propagateMessage(recomputeInsideWithoutRecursive(branch2.node, ignore), branch2.diffusionLength);
            int c1Nr = branch1.node.getNr();
            int c2Nr = branch2.node.getNr();
            for (int d = 0; d < dim; d++) {
                base[d] = insideWithoutBaseContributions[c1Nr][d] + insideWithoutBaseContributions[c2Nr][d];
                quad[d] = insideWithoutQuadraticContributions[c1Nr][d] + insideWithoutQuadraticContributions[c2Nr][d];
            }
            message = combineMessagesWithContributionParts(up1, up2, base, quad);
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

        int nodeNr = node.getNr();
        GaussianMessage nodeOutside = getOutsideMessage(nodeNr);
        EffectiveBranch leftBranch = getEffectiveBranch(node, 0, ignore);
        EffectiveBranch rightBranch = getEffectiveBranch(node, 1, ignore);

        if (leftBranch != null) {
            int childNr = leftBranch.node.getNr();
            GaussianMessage siblingContribution = rightBranch == null
                    ? neutralMessage()
                    : propagateMessage(getInsideWithoutMessage(rightBranch.node.getNr()), rightBranch.diffusionLength);
            // Seed child's outside accumulators with parent's outside + sibling inside-without;
            // the combine then appends its own per-axis contribution.
            double[] base = outsideBaseContributions[childNr];
            double[] quad = outsideQuadraticContributions[childNr];
            for (int d = 0; d < dim; d++) {
                base[d] = outsideBaseContributions[nodeNr][d];
                quad[d] = outsideQuadraticContributions[nodeNr][d];
                if (rightBranch != null) {
                    int sibNr = rightBranch.node.getNr();
                    base[d] += insideWithoutBaseContributions[sibNr][d];
                    quad[d] += insideWithoutQuadraticContributions[sibNr][d];
                }
            }
            GaussianMessage combined = combineMessagesWithContributionParts(nodeOutside, siblingContribution, base, quad);
            GaussianMessage childOutside = propagateMessage(combined, leftBranch.diffusionLength);
            setOutside(childNr, childOutside);
            recomputeOutsideRecursive(leftBranch.node, ignore);
        }

        if (rightBranch != null) {
            int childNr = rightBranch.node.getNr();
            GaussianMessage siblingContribution = leftBranch == null
                    ? neutralMessage()
                    : propagateMessage(getInsideWithoutMessage(leftBranch.node.getNr()), leftBranch.diffusionLength);
            double[] base = outsideBaseContributions[childNr];
            double[] quad = outsideQuadraticContributions[childNr];
            for (int d = 0; d < dim; d++) {
                base[d] = outsideBaseContributions[nodeNr][d];
                quad[d] = outsideQuadraticContributions[nodeNr][d];
                if (leftBranch != null) {
                    int sibNr = leftBranch.node.getNr();
                    base[d] += insideWithoutBaseContributions[sibNr][d];
                    quad[d] += insideWithoutQuadraticContributions[sibNr][d];
                }
            }
            GaussianMessage combined = combineMessagesWithContributionParts(nodeOutside, siblingContribution, base, quad);
            GaussianMessage childOutside = propagateMessage(combined, rightBranch.diffusionLength);
            setOutside(childNr, childOutside);
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

    /**
     * Run the inside (Felsenstein-style) message recursion on the current tree
     * with per-axis contribution tracking.  Mirrors {@link #recomputeInsideRecursive}
     * but funnels every message-combine through
     * {@link #combineMessagesWithContributionParts}, so that on return
     * <pre>
     *   sum baseContributions[d]      = −½·logDet(V_axis_d)
     *   sum quadraticContributions[d] = −½·qf[d]
     * </pre>
     * for each axis {@code d} under the current Brownian rates / root prior.
     *
     * <p>Test-only: side-effect free relative to the cached fields except for
     * resetting the input accumulator arrays.
     *
     * @param baseContributions     length-dim accumulator (zeroed and filled)
     * @param quadraticContributions length-dim accumulator (zeroed and filled)
     */
    public void computeFullTreeContributionParts(double[] baseContributions, double[] quadraticContributions) {
        ensureInitialized();
        if (baseContributions.length != dim || quadraticContributions.length != dim) {
            throw new IllegalArgumentException(
                "Contribution arrays must have length dim=" + dim);
        }
        Arrays.fill(baseContributions, 0.0);
        Arrays.fill(quadraticContributions, 0.0);
        computeFullTreeContributionsRecursive(treeInput.get().getRoot(),
            baseContributions, quadraticContributions);
    }

    private GaussianMessage computeFullTreeContributionsRecursive(Node node,
            double[] base, double[] quad) {
        if (node.isLeaf()) {
            double[] zeroVar = new double[dim];   // leaf message has zero variance
            return new GaussianMessage(leafPoints[node.getNr()], zeroVar, 0.0);
        }
        GaussianMessage left  = computeFullTreeContributionsRecursive(node.getLeft(),  base, quad);
        GaussianMessage right = computeFullTreeContributionsRecursive(node.getRight(), base, quad);
        GaussianMessage leftUp  = propagateMessage(left,  node.getLeft(),  node.getLeft().getLength());
        GaussianMessage rightUp = propagateMessage(right, node.getRight(), node.getRight().getLength());
        GaussianMessage merged = combineMessagesWithContributionParts(leftUp, rightUp, base, quad);
        if (node.isRoot()) {
            merged = combineMessagesWithContributionParts(merged, rootPriorMessage(), base, quad);
        }
        return merged;
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
        return treeInput.get().getNode(nodeNr).getLength();
        // ensureInitialized();
        // if (nodeNr == treeInput.get().getRoot().getNr())
        //     return minWeight;
        // return Math.sqrt((0.5 + edgeMutations[nodeNr]) * (2.0 + treeInput.get().getNode(nodeNr).getLength()));
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
        if (false) { //useInverseMeanDistanceProposal) {
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

        double[] heights = new double[toNodeNrs.size()];
        for (int i = 0; i < toNodeNrs.size(); i++) {
            heights[i] = getToHeightWithoutNode(getNode(toNodeNrs.get(i)), fromNode);
        }
        double[] logScores = computeProposalLogScores(fromNode, fromInside, toNodeNrs, heights);
        // System.out.println(Arrays.toString(logScores));
        return toOperatorReadyWeights(logScores);
    }

    @Override
    public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes, double toHeight) {
        return getTargetWeightsInteger(fromNodeNr, toNodes.stream().map(Node::getNr).toList(), toHeight);
    }

    public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, double toHeight) {
        double[] toHeights = new double[toNodeNrs.size()];
        Arrays.fill(toHeights, toHeight);
        return getTargetWeightsAtHeights(fromNodeNr, toNodeNrs, toHeights);
    }

    @Override
    public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes, List<Double> toHeights) {
        return getTargetWeightsInteger(fromNodeNr, toNodes.stream().map(Node::getNr).toList(), toHeights);
    }

    public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, List<Double> toHeights) {
        assert toNodeNrs.size() == toHeights.size();

        double[] heights = new double[toHeights.size()];
        for (int i = 0; i < toHeights.size(); i++) {
            heights[i] = toHeights.get(i);
        }
        return getTargetWeightsAtHeights(fromNodeNr, toNodeNrs, heights);
    }

    private double[] getTargetWeightsAtHeights(int fromNodeNr, List<Integer> toNodeNrs, double[] toHeights) {
        ensureInitialized();
        ensureWithoutNodeMessages(fromNodeNr);
        Node fromNode = getNode(fromNodeNr);
        GaussianMessage fromInside = getInsideMessage(fromNodeNr);

        if (false) { //useInverseMeanDistanceProposal) {
            double[] weights = new double[toNodeNrs.size()];
            for (int i = 0; i < toNodeNrs.size(); i++) {
                Node toNode = getNode(toNodeNrs.get(i));
                double toHeight = sanitizeAttachmentHeight(fromNode, toNode, toHeights[i]);
                weights[i] = toOperatorReadyWeight(1.0 / Math.max(
                        getMeanDist(fromNode, toNode, fromInside, getInsideWithoutMessage(toNodeNrs.get(i)), toHeight),
                        VARIANCE_FLOOR));
            }
            return weights;
        }

        double[] heights = new double[toNodeNrs.size()];
        for (int i = 0; i < toNodeNrs.size(); i++) {
            heights[i] = sanitizeAttachmentHeight(fromNode, getNode(toNodeNrs.get(i)), toHeights[i]);
        }
        double[] logScores = computeProposalLogScores(fromNode, fromInside, toNodeNrs, heights);
        // System.out.println(Arrays.toString(logScores));
        return toOperatorReadyWeights(logScores);
    }

    /**
     * Compute per-proposal log-scores for the candidate {@code toNodeNrs} at the
     * already-resolved {@code resolvedHeights}.  Default implementation runs the
     * Brownian-motion pruning likelihood with affine calibration; subclasses may
     * override to use a different scoring model (e.g. an MLP surrogate).
     */
    protected double[] computeProposalLogScores(Node fromNode, GaussianMessage fromInside,
                                                 List<Integer> toNodeNrs, double[] resolvedHeights) {
        double[] logScores = new double[toNodeNrs.size()];
        for (int i = 0; i < toNodeNrs.size(); i++) {
            Node toNode = getNode(toNodeNrs.get(i));
            logScores[i] = computeScaledProposedTreeLogLikelihood(fromNode, toNode,
                    fromInside, getInsideWithoutMessage(toNodeNrs.get(i)), resolvedHeights[i]);
        }
        return logScores;
    }

    private double sanitizeAttachmentHeight(Node fromNode, Node toNode, double toHeight) {
        double minHeight = Math.max(fromNode.getHeight(), toNode.getHeight());
        Node effectiveParent = getEffectiveParent(toNode, currentIgnoredNode);
        if (effectiveParent == null) {
            assert toHeight > minHeight : "Invalid attachment height without effective parent: toHeight=" + toHeight
                    + ", minHeight=" + minHeight
                    + ", fromNode=" + fromNode.getNr()
                    + ", toNode=" + toNode.getNr()
                    + ", ignoredNode=" + currentIgnoredNode;
            return Math.max(toHeight, minHeight + 1e-8);
        }

        double maxHeight = effectiveParent.getHeight();
        if (maxHeight <= minHeight) {
            assert false : "Degenerate attachment interval: maxHeight <= minHeight"
                    + " (maxHeight=" + maxHeight
                    + ", minHeight=" + minHeight
                    + ", fromNode=" + fromNode.getNr()
                    + ", toNode=" + toNode.getNr()
                    + ", parentNode=" + effectiveParent.getNr()
                    + ", ignoredNode=" + currentIgnoredNode + ")";
            return minHeight;
        }

            assert toHeight >= minHeight && toHeight <= maxHeight : "Attachment height outside legal interval"
                + " (toHeight=" + toHeight
                + ", minHeight=" + minHeight
                + ", maxHeight=" + maxHeight
                + ", fromNode=" + fromNode.getNr()
                + ", toNode=" + toNode.getNr()
                + ", parentNode=" + effectiveParent.getNr()
                + ", ignoredNode=" + currentIgnoredNode + ")";

        double span = maxHeight - minHeight;
        double epsilon = Math.max(1e-10, 1e-6 * span);
        double lower = minHeight + epsilon;
        double upper = maxHeight - epsilon;
        if (lower >= upper) {
            assert false : "Degenerate interior attachment interval: lower >= upper"
                    + " (lower=" + lower
                    + ", upper=" + upper
                    + ", minHeight=" + minHeight
                    + ", maxHeight=" + maxHeight
                    + ", fromNode=" + fromNode.getNr()
                    + ", toNode=" + toNode.getNr()
                    + ", parentNode=" + effectiveParent.getNr()
                    + ", ignoredNode=" + currentIgnoredNode + ")";
            return 0.5 * (minHeight + maxHeight);
        }
        return Math.min(Math.max(toHeight, lower), upper);
    }

    public void setProposalLogLikelihoodScale(double scale) {
        if (!Double.isFinite(scale)) {
            throw new IllegalArgumentException("proposalLogLikelihoodScale must be finite, but was " + scale);
        }
        proposalLogLikelihoodScale = scale;
        Arrays.fill(proposalLogLikelihoodScales, scale);
    }

    public double getProposalLogLikelihoodScale() {
        return mean(proposalLogLikelihoodScales);
    }

    public void setProposalBaseLogLikelihoodScales(double[] scales) {
        if (scales == null || scales.length != dim) {
            throw new IllegalArgumentException("proposalBaseLogLikelihoodScales length mismatch: expected "
                    + dim + " but got " + (scales == null ? 0 : scales.length));
        }
        proposalBaseLogLikelihoodScales = Arrays.copyOf(scales, scales.length);
        for (double scale : proposalBaseLogLikelihoodScales) {
            if (!Double.isFinite(scale)) {
                throw new IllegalArgumentException("proposalBaseLogLikelihoodScales must all be finite");
            }
        }
        proposalBaseLogLikelihoodScale = mean(proposalBaseLogLikelihoodScales);
    }

    public double[] getProposalBaseLogLikelihoodScales() {
        return Arrays.copyOf(proposalBaseLogLikelihoodScales, proposalBaseLogLikelihoodScales.length);
    }

    public void setProposalLogLikelihoodScales(double[] scales) {
        if (scales == null || scales.length != dim) {
            throw new IllegalArgumentException("proposalLogLikelihoodScales length mismatch: expected "
                    + dim + " but got " + (scales == null ? 0 : scales.length));
        }
        proposalLogLikelihoodScales = Arrays.copyOf(scales, scales.length);
        for (double scale : proposalLogLikelihoodScales) {
            if (!Double.isFinite(scale)) {
                throw new IllegalArgumentException("proposalLogLikelihoodScales must all be finite");
            }
        }
        proposalLogLikelihoodScale = mean(proposalLogLikelihoodScales);
    }

    public double[] getProposalLogLikelihoodScales() {
        return Arrays.copyOf(proposalLogLikelihoodScales, proposalLogLikelihoodScales.length);
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
        double[] base = new double[dim];
        double[] quad = new double[dim];
        computeProposedTreeLogLikelihoodContributionParts(
                fromNode, toNode, fromInside, toInsideWithout, toHeight, base, quad);
        double[] contributions = new double[dim];
        for (int d = 0; d < dim; d++) {
            contributions[d] = base[d] + quad[d];
        }
        return contributions;
    }

    public ProposalScoreBreakdown getProposalScoreBreakdown(int fromNodeNr, int toNodeNr, double toHeight) {
        ensureInitialized();
        if (useInverseMeanDistanceProposal) {
            throw new IllegalStateException("Proposal score breakdown is not available in inverse-distance proposal mode.");
        }

        ensureWithoutNodeMessages(fromNodeNr);
        Node fromNode = getNode(fromNodeNr);
        Node toNode = getNode(toNodeNr);
        double sanitizedToHeight = sanitizeAttachmentHeight(fromNode, toNode, toHeight);
        GaussianMessage fromInside = getInsideMessage(fromNodeNr);
        GaussianMessage toInsideWithout = getInsideWithoutMessage(toNodeNr);
        double[] baseContributions = new double[dim];
        double[] quadraticContributions = new double[dim];
        computeProposedTreeLogLikelihoodContributionParts(
                fromNode, toNode, fromInside, toInsideWithout, sanitizedToHeight, baseContributions, quadraticContributions);

        double rawBaseContribution = 0.0;
        double rawQuadraticContribution = 0.0;
        double scaledBaseContribution = 0.0;
        double scaledQuadraticContribution = 0.0;
        for (int axis = 0; axis < dim; axis++) {
            rawBaseContribution += baseContributions[axis];
            rawQuadraticContribution += quadraticContributions[axis];
            if (separateBaseAndQuadraticCalibration) {
                scaledBaseContribution += proposalBaseLogLikelihoodScales[axis] * baseContributions[axis];
                scaledQuadraticContribution += proposalLogLikelihoodScales[axis] * quadraticContributions[axis];
            } else if (scaleQuadraticPartOnly) {
                scaledBaseContribution += baseContributions[axis];
                scaledQuadraticContribution += proposalLogLikelihoodScales[axis] * quadraticContributions[axis];
            } else {
                scaledBaseContribution += proposalLogLikelihoodScales[axis] * baseContributions[axis];
                scaledQuadraticContribution += proposalLogLikelihoodScales[axis] * quadraticContributions[axis];
            }
        }

        return new ProposalScoreBreakdown(
                sanitizedToHeight,
                rawBaseContribution,
                rawQuadraticContribution,
                scaledBaseContribution,
                scaledQuadraticContribution);
    }

    protected double computeScaledProposedTreeLogLikelihood(Node fromNode, Node toNode, GaussianMessage fromInside,
            GaussianMessage toInsideWithout, double toHeight) {
        if (separateBaseAndQuadraticCalibration || scaleQuadraticPartOnly) {
            double[] baseContributions = new double[dim];
            double[] quadraticContributions = new double[dim];
            computeProposedTreeLogLikelihoodContributionParts(
                    fromNode, toNode, fromInside, toInsideWithout, toHeight, baseContributions, quadraticContributions);
            double total = 0.0;
            for (int axis = 0; axis < dim; axis++) {
                if (separateBaseAndQuadraticCalibration) {
                    total += proposalBaseLogLikelihoodScales[axis] * baseContributions[axis]
                            + proposalLogLikelihoodScales[axis] * quadraticContributions[axis];
                } else {
                    total += baseContributions[axis] + proposalLogLikelihoodScales[axis] * quadraticContributions[axis];
                }
            }
            return total;
        }
        double[] contributions = computeProposedTreeLogLikelihoodContributions(fromNode, toNode, fromInside, toInsideWithout, toHeight);
        double total = 0.0;
        for (int axis = 0; axis < dim; axis++) {
            total += proposalLogLikelihoodScales[axis] * contributions[axis];
        }
        return total;
    }

    protected void computeProposedTreeLogLikelihoodContributionParts(Node fromNode, Node toNode, GaussianMessage fromInside,
            GaussianMessage toInsideWithout, double toHeight, double[] baseContributions, double[] quadraticContributions) {
        assert toHeight > fromNode.getHeight();
        assert toHeight > toNode.getHeight();

        // Seed the per-axis accumulators with the contributions already cached in
        // the messages we're about to consume.  Each combineMessagesWithContributionParts
        // call only adds the contribution from that single combine; the cached
        // children's contributions live in the scalar logNorm of the GaussianMessage,
        // but the per-axis decomposition is stored on the node-level cache arrays,
        // and needs to be added explicitly here so the output (base, quad) describes
        // the whole proposed tree (not just the local combines).
        int fromNr = fromNode.getNr();
        int toNr   = toNode.getNr();
        boolean useInsideWithoutTo = (currentIgnoredNode >= 0) && insideWithoutValid[toNr];
        double[] toBaseSrc = useInsideWithoutTo ? insideWithoutBaseContributions[toNr]
                                                : insideBaseContributions[toNr];
        double[] toQuadSrc = useInsideWithoutTo ? insideWithoutQuadraticContributions[toNr]
                                                : insideQuadraticContributions[toNr];
        for (int d = 0; d < dim; d++) {
            baseContributions[d]      += insideBaseContributions[fromNr][d]      + toBaseSrc[d];
            quadraticContributions[d] += insideQuadraticContributions[fromNr][d] + toQuadSrc[d];
        }

        GaussianMessage fromAtHeight = propagateMessage(fromInside, fromNode, toHeight - fromNode.getHeight());
        GaussianMessage toAtHeight = propagateMessage(toInsideWithout, toNode, toHeight - toNode.getHeight());
        GaussianMessage parentMessage = combineMessagesWithContributionParts(fromAtHeight, toAtHeight, baseContributions, quadraticContributions);

        Node grandparent = getEffectiveParent(toNode, currentIgnoredNode);
        if (grandparent == null) {
            // outside(toNode) here is the root-prior / neutral message — its cached
            // contribution arrays are zero by construction; just do the combine.
            combineMessagesWithContributionParts(parentMessage, getOutsideMessage(toNr), baseContributions, quadraticContributions);
            return;
        }

        GaussianMessage gpMessage = propagateMessage(parentMessage, toNode, Math.max(0.0, grandparent.getHeight() - toHeight));
        EffectiveBranch siblingBranch = getEffectiveSiblingBranch(toNode, currentIgnoredNode);
        if (siblingBranch != null) {
            int sibNr = siblingBranch.node.getNr();
            boolean useInsideWithoutSib = (currentIgnoredNode >= 0) && insideWithoutValid[sibNr];
            double[] sibBaseSrc = useInsideWithoutSib ? insideWithoutBaseContributions[sibNr]
                                                      : insideBaseContributions[sibNr];
            double[] sibQuadSrc = useInsideWithoutSib ? insideWithoutQuadraticContributions[sibNr]
                                                      : insideQuadraticContributions[sibNr];
            for (int d = 0; d < dim; d++) {
                baseContributions[d]      += sibBaseSrc[d];
                quadraticContributions[d] += sibQuadSrc[d];
            }
            GaussianMessage siblingUp = propagateMessage(getInsideWithoutMessage(sibNr), siblingBranch.diffusionLength);
            gpMessage = combineMessagesWithContributionParts(gpMessage, siblingUp, baseContributions, quadraticContributions);
        }

        int gpNr = grandparent.getNr();
        if (outsideValid[gpNr]) {
            for (int d = 0; d < dim; d++) {
                baseContributions[d]      += outsideBaseContributions[gpNr][d];
                quadraticContributions[d] += outsideQuadraticContributions[gpNr][d];
            }
            combineMessagesWithContributionParts(gpMessage, getOutsideMessage(gpNr), baseContributions, quadraticContributions);
        }
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
        double diffusionLength = getBaseDiffusionLength(child, child.getLength());
        while (becomesUnary(child, ignore)) {
            child = getIncludedChild(child, ignore);
            diffusionLength += getBaseDiffusionLength(child, child.getLength());
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
        Arrays.fill(outsideBaseContributions[nodeNr], 0.0);
        Arrays.fill(outsideQuadraticContributions[nodeNr], 0.0);
        outsideValid[nodeNr] = true;
    }

    private void setRootOutside(int nodeNr) {
        GaussianMessage prior = rootPriorMessage();
        copyMessage(prior, outsideMeans[nodeNr], outsideVariances[nodeNr]);
        outsideLogNorm[nodeNr] = prior.logNorm;
        // Root prior message has logNorm = 0 by construction, so no contributions to seed.
        Arrays.fill(outsideBaseContributions[nodeNr], 0.0);
        Arrays.fill(outsideQuadraticContributions[nodeNr], 0.0);
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
        double[] mean = new double[dim];
        double[] variance = new double[dim];
        boolean allInfinite = true;
        for (int axis = 0; axis < dim; axis++) {
            double rootPriorVariance = rootPriorVariances[axis];
            if (Double.isInfinite(rootPriorVariance)) {
                variance[axis] = NEUTRAL_VARIANCE;
            } else {
                variance[axis] = Math.max(rootPriorVariance, 0.0);
                allInfinite = false;
            }
        }
        if (allInfinite) {
            return neutralMessage();
        }
        return new GaussianMessage(mean, variance, 0.0);
    }

    private GaussianMessage propagateMessage(GaussianMessage message, Node branchNode, double branchLength) {
        return propagateMessage(message, getBaseDiffusionLength(branchNode, branchLength));
    }

    private GaussianMessage propagateMessage(GaussianMessage message, double baseDiffusionLength) {
        double[] mean = Arrays.copyOf(message.mean, dim);
        double[] variance = Arrays.copyOf(message.variance, dim);
        for (int i = 0; i < dim; i++) {
            variance[i] = propagatedVariance(variance[i], brownianTraitRates[i] * baseDiffusionLength);
        }
        return new GaussianMessage(mean, variance, message.logNorm);
    }

    private GaussianMessage combineMessages(GaussianMessage first, GaussianMessage second) {
        return combineMessagesWithContributions(first, second, null);
    }

    private GaussianMessage combineMessagesWithContributions(GaussianMessage first, GaussianMessage second, double[] contributions) {
        return combineMessagesWithContributionParts(first, second, contributions, contributions);
    }

    private GaussianMessage combineMessagesWithContributionParts(
            GaussianMessage first,
            GaussianMessage second,
            double[] baseContributions,
            double[] quadraticContributions) {
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
            double baseContribution = -0.5 * Math.log(sumVariance);
            double quadraticContribution = -0.5 * (diff * diff) / sumVariance;
            double axisContribution = baseContribution + quadraticContribution;
            logNorm += axisContribution;
            if (baseContributions != null) {
                baseContributions[i] += baseContribution;
            }
            if (quadraticContributions != null) {
                quadraticContributions[i] += quadraticContribution;
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

    public double[] getBrownianTraitRates() {
        return Arrays.copyOf(brownianTraitRates, brownianTraitRates.length);
    }

    public double[] getRootPriorVariances() {
        return Arrays.copyOf(rootPriorVariances, rootPriorVariances.length);
    }

    public void setBrownianTraitRates(double[] rates) {
        if (rates == null || rates.length != dim) {
            throw new IllegalArgumentException("brownianTraitRates length mismatch: expected "
                    + dim + " but got " + (rates == null ? 0 : rates.length));
        }
        brownianTraitRates = Arrays.copyOf(rates, rates.length);
        for (int axis = 0; axis < brownianTraitRates.length; axis++) {
            if (!(brownianTraitRates[axis] > 0.0) || !Double.isFinite(brownianTraitRates[axis])) {
                throw new IllegalArgumentException("brownianTraitRates must all be finite and > 0, but axis "
                        + axis + " was " + brownianTraitRates[axis]);
            }
        }
        insideDirty = true;
        clearWithoutNodeState();
    }

    public void setRootPriorVariances(double[] variances) {
        if (variances == null || variances.length != dim) {
            throw new IllegalArgumentException("rootPriorVariances length mismatch: expected "
                    + dim + " but got " + (variances == null ? 0 : variances.length));
        }
        rootPriorVariances = Arrays.copyOf(variances, variances.length);
        for (int axis = 0; axis < rootPriorVariances.length; axis++) {
            double variance = rootPriorVariances[axis];
            if ((variance < 0.0 && !Double.isInfinite(variance)) || Double.isNaN(variance)) {
                throw new IllegalArgumentException("rootPriorVariances must all be >= 0 or Infinity, but axis "
                        + axis + " was " + variance);
            }
        }
        insideDirty = true;
        clearWithoutNodeState();
    }

    private double[] resolveBrownianTraitRates() {
        String configured = brownianTraitRatesInput.get();
        if (configured == null || configured.isBlank()) {
            double fallbackRate = branchRateModel == null ? brownianRate : 1.0;
            if (!(fallbackRate > 0.0) || !Double.isFinite(fallbackRate)) {
                throw new IllegalArgumentException("brownianRate must be finite and > 0, but was " + brownianRate);
            }
            double[] rates = new double[dim];
            Arrays.fill(rates, fallbackRate);
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
        if (configured == null || configured.isBlank()) {
            double rootPriorVariance = rootPriorVarianceInput.get();
            if (rootPriorVariance < 0.0 || Double.isNaN(rootPriorVariance)) {
                throw new IllegalArgumentException("rootPriorVariance must be >= 0 or Infinity, but was " + rootPriorVariance);
            }
            double[] variances = new double[dim];
            Arrays.fill(variances, rootPriorVariance);
            return variances;
        }

        String[] tokens = configured.trim().split("[\\s,]+");
        if (tokens.length != dim) {
            throw new IllegalArgumentException("rootPriorVariances length mismatch: expected "
                    + dim + " entries but got " + tokens.length);
        }

        double[] variances = new double[dim];
        for (int axis = 0; axis < dim; axis++) {
            variances[axis] = Double.parseDouble(tokens[axis]);
            if ((variances[axis] < 0.0 && !Double.isInfinite(variances[axis])) || Double.isNaN(variances[axis])) {
                throw new IllegalArgumentException("rootPriorVariances must all be >= 0 or Infinity, but axis "
                        + axis + " was " + variances[axis]);
            }
        }
        return variances;
    }

    private double getBaseDiffusionLength(Node node, double branchLength) {
        return getBranchRate(node) * Math.max(branchLength, 0.0);
    }

    private double getBranchRate(Node node) {
        if (branchRateModel == null) {
            return 1.0;
        }
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
            if (Double.isFinite(logScore)) maxLogScore = Math.max(maxLogScore, logScore);
        }
        if (!Double.isFinite(maxLogScore)) {
            // All proposals produced non-finite scores; return uniform floor weights.
            Arrays.fill(scores, VARIANCE_FLOOR + offset);
            return scores;
        }
        for (int i = 0; i < logScores.length; i++) {
            scores[i] = Math.max(Math.exp(logScores[i] - maxLogScore), VARIANCE_FLOOR) + offset;
        }
        return scores;
    }

    static public double[] clipMaxToSecondMaxRatio(double[] weights, double maxRatio) {
		if (weights == null || weights.length < 2) {
			return weights;
		}

		int firstIdx = -1;
		int secondIdx = -1;
		double first = Double.NEGATIVE_INFINITY;
		double second = Double.NEGATIVE_INFINITY;

		for (int i = 0; i < weights.length; i++) {
			double w = weights[i];
			if (w > first) {
				second = first;
				secondIdx = firstIdx;
				first = w;
				firstIdx = i;
			} else if (w > second) {
				second = w;
				secondIdx = i;
			}
		}

		if (firstIdx < 0 || secondIdx < 0 || first <= maxRatio * second) {
			return weights;
		}

		double[] clipped = Arrays.copyOf(weights, weights.length);
		clipped[firstIdx] = maxRatio * second;
		return clipped;
    }

    static public double[] clipMaxToSecondMaxDiff(final double[] weights, final double maxDiff) {
		if (weights == null || weights.length < 2) {
			return weights;
		}

		int firstIdx = -1;
		int secondIdx = -1;
		double first = Double.NEGATIVE_INFINITY;
		double second = Double.NEGATIVE_INFINITY;

		for (int i = 0; i < weights.length; i++) {
			double w = weights[i];
			if (w > first) {
				second = first;
				secondIdx = firstIdx;
				first = w;
				firstIdx = i;
			} else if (w > second) {
				second = w;
				secondIdx = i;
			}
		}

		if (firstIdx < 0 || secondIdx < 0 || first <= second + maxDiff) {
			return weights;
		}

		double[] clipped = Arrays.copyOf(weights, weights.length);
		clipped[firstIdx] = second + maxDiff;
		return clipped;
    }


    protected double[] toOperatorReadyWeights(double[] logScores) {
        logScores = clipMaxToSecondMaxDiff(logScores, 2.5);
        double[] scores = exponentiateLogScores(logScores);
        for (int i = 0; i < scores.length; i++) {
            scores[i] = toOperatorReadyWeight(scores[i]);
        }
        return scores;
    }

    private double toOperatorReadyWeight(double finalWeight) {
        return Math.sqrt(finalWeight);
    }

    private double norm(double x) {
        return useOneNorm ? Math.abs(x) : x * x;
    }

    private static double mean(double[] values) {
        double total = 0.0;
        for (double value : values) {
            total += value;
        }
        return total / values.length;
    }

    public Node getNode(int i) {
        return treeInput.get().getNode(i);
    }

    public Tree getTree() {
        return (Tree) treeInput.get();
    }

    protected Base getBranchRateModel() {
        return branchRateModel;
    }

    protected int getDim() {
        return dim;
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

    public static final class ProposalScoreBreakdown {
        public final double attachmentHeight;
        public final double rawBaseContribution;
        public final double rawQuadraticContribution;
        public final double rawTotalContribution;
        public final double scaledBaseContribution;
        public final double scaledQuadraticContribution;
        public final double scaledTotalContribution;

        ProposalScoreBreakdown(
                double attachmentHeight,
                double rawBaseContribution,
                double rawQuadraticContribution,
                double scaledBaseContribution,
                double scaledQuadraticContribution) {
            this.attachmentHeight = attachmentHeight;
            this.rawBaseContribution = rawBaseContribution;
            this.rawQuadraticContribution = rawQuadraticContribution;
            this.rawTotalContribution = rawBaseContribution + rawQuadraticContribution;
            this.scaledBaseContribution = scaledBaseContribution;
            this.scaledQuadraticContribution = scaledQuadraticContribution;
            this.scaledTotalContribution = scaledBaseContribution + scaledQuadraticContribution;
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
