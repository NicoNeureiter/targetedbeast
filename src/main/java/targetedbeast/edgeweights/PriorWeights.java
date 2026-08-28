package targetedbeast.edgeweights;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Distribution;
import beast.base.inference.State;

@Description("Exact prior-aware edge weights for targeted operators. "
    + "These weights evaluate the candidate tree prior for each move and convert "
    + "that value into target proposal weights.")
public class PriorWeights extends Distribution implements EdgeWeights {

    private enum ProposalKind {
        INVALID,
        HEIGHT_ONLY,
        TOPOLOGY_CHANGE
    }

    final public Input<TreeInterface> treeInput = new Input<>("tree",
            "phylogenetic tree used for targeted proposals",
            Validate.REQUIRED);
    final public Input<Distribution> priorInput = new Input<>("prior",
            "tree prior used to score candidate local reattachments",
            Validate.REQUIRED);
    final public Input<Double> minWeightInput = new Input<>("minWeight",
            "minimum positive proposal weight returned by the prior scorer",
            1e-6);
    final public Input<Double> maxWeightInput = new Input<>("maxWeight",
            "maximum proposal weight returned by the prior scorer",
            1e6);
    final public Input<Double> temperatureInput = new Input<>("temperature",
            "temperature applied to exact local prior deltas before converting them to proposal weights",
            1.0);
    final public Input<Double> sourceEdgeOffsetInput = new Input<>("sourceEdgeOffset",
            "offset used in the simple source-edge heuristic 1 / (offset + branchLength)",
            0.1);

    private Tree tree;
    private Distribution prior;
    private double minWeight;
    private double maxWeight;
    private double temperature;
    private double sourceEdgeOffset;

    @Override
    public void initAndValidate() {
        TreeInterface treeInterface = treeInput.get();
        if (!(treeInterface instanceof Tree exactTree)) {
            throw new IllegalArgumentException("PriorWeights requires a beast.base.evolution.tree.Tree, but got "
                    + treeInterface.getClass().getName());
        }

        tree = exactTree;
        prior = priorInput.get();
        minWeight = minWeightInput.get();
        maxWeight = maxWeightInput.get();
        temperature = temperatureInput.get();
        sourceEdgeOffset = sourceEdgeOffsetInput.get();

        if (!(minWeight > 0.0)) {
            throw new IllegalArgumentException("minWeight must be > 0, but was " + minWeight);
        }
        if (!(maxWeight >= minWeight)) {
            throw new IllegalArgumentException("maxWeight must be >= minWeight, but was " + maxWeight);
        }
        if (!(temperature >= 0.0)) {
            throw new IllegalArgumentException("temperature must be >= 0, but was " + temperature);
        }
        if (!(sourceEdgeOffset > 0.0)) {
            throw new IllegalArgumentException("sourceEdgeOffset must be > 0, but was " + sourceEdgeOffset);
        }
    }

    @Override
    public void updateByOperator() {
    }

    @Override
    public void updateByOperatorWithoutNode(int ignore, List<Integer> nodes) {
    }

    @Override
    public void fakeUpdateByOperator() {
    }

    @Override
    public void prestore() {
    }

    @Override
    public void reset() {
    }

    @Override
    public double getEdgeWeights(int nodeNr) {
        return 1.0;
        // Node node = tree.getNode(nodeNr);
        // if (node == null || node.isRoot()) {
        //     return 0.0;
        // }
        // return 1.0 / (sourceEdgeOffset + node.getLength());
    }

    @Override
    public double[] getTargetWeights(int fromNodeNr, List<Node> toNodeNrs) {
        double[] targetHeights = new double[toNodeNrs.size()];
        for (int i = 0; i < toNodeNrs.size(); i++) {
            Node node = toNodeNrs.get(i);
            targetHeights[i] = node.getParent() == null ? node.getHeight() : node.getParent().getHeight();
        }
        return getTargetWeights(fromNodeNr, toNodeNrs, targetHeights);
    }

    @Override
    public double[] getTargetWeights(int fromNodeNr, List<Node> toNodeNrs, double toHeight) {
        double[] heights = new double[toNodeNrs.size()];
        for (int i = 0; i < toNodeNrs.size(); i++) {
            heights[i] = toHeight;
        }
        return getTargetWeights(fromNodeNr, toNodeNrs, heights);
    }

    @Override
    public double[] getTargetWeights(int fromNodeNr, List<Node> toNodeNrs, List<Double> toHeights) {
        if (toNodeNrs.size() != toHeights.size()) {
            throw new IllegalArgumentException("toNodeNrs and toHeights must have equal size");
        }
        double[] heights = new double[toHeights.size()];
        for (int i = 0; i < toHeights.size(); i++) {
            heights[i] = toHeights.get(i);
        }
        return getTargetWeights(fromNodeNr, toNodeNrs, heights);
    }

    private double[] getTargetWeights(int fromNodeNr, List<Node> toNodeNrs, double[] toHeights) {
        double[] weights = new double[toNodeNrs.size()];
        Node movingNode = tree.getNode(fromNodeNr);
        if (movingNode == null || movingNode.isRoot()) {
            return weights;
        }

        for (int i = 0; i < toNodeNrs.size(); i++) {
            weights[i] = targetWeightForCandidate(movingNode.getNr(), toNodeNrs.get(i).getNr(), toHeights[i]);
        }

        weights = clipMaxToSecondMaxRatio(weights, 3.0);
        normalise(weights);
        for (int i=0; i<weights.length; i++)
            weights[i] = 0.05 / weights.length + 0.95 * weights[i];
        
        return weights;
    }

    @Override
    public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
        double[] targetHeights = new double[toNodeNrs.size()];
        for (int i = 0; i < toNodeNrs.size(); i++) {
            Node node = tree.getNode(toNodeNrs.get(i));
            targetHeights[i] = node.getParent() == null ? node.getHeight() : node.getParent().getHeight();
        }
        return getTargetWeightsInteger(fromNodeNr, toNodeNrs, targetHeights);
    }

    @Override
    public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, double toHeight) {
        double[] heights = new double[toNodeNrs.size()];
        for (int i = 0; i < toNodeNrs.size(); i++) {
            heights[i] = toHeight;
        }
        return getTargetWeightsInteger(fromNodeNr, toNodeNrs, heights);
    }

    @Override
    public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, List<Double> toHeights) {
        if (toNodeNrs.size() != toHeights.size()) {
            throw new IllegalArgumentException("toNodeNrs and toHeights must have equal size");
        }
        double[] heights = new double[toHeights.size()];
        for (int i = 0; i < toHeights.size(); i++) {
            heights[i] = toHeights.get(i);
        }
        return getTargetWeightsInteger(fromNodeNr, toNodeNrs, heights);
    }

    private double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, double[] toHeights) {
        double[] weights = new double[toNodeNrs.size()];
        Node movingNode = tree.getNode(fromNodeNr);
        if (movingNode == null || movingNode.isRoot()) {
            return weights;
        }

        for (int i = 0; i < toNodeNrs.size(); i++) {
            Node candidateNode = tree.getNode(toNodeNrs.get(i));
            if (candidateNode == null) {
                weights[i] = minWeight;
                continue;
            }
            weights[i] = targetWeightForCandidate(movingNode.getNr(), candidateNode.getNr(), toHeights[i]);
        }

        weights = clipMaxToSecondMaxRatio(weights, 3.0);
        normalise(weights);
        for (int i=0; i<weights.length; i++)
            weights[i] = 0.05 / weights.length + 0.95 * weights[i];

        return weights;
    }

    
    private double targetWeightForCandidate(int movingNodeNr, int candidateNodeNr, double toHeight) {
        Node movingNode = tree.getNode(movingNodeNr);
        Node candidateNode = tree.getNode(candidateNodeNr);
        ProposalKind proposalKind = classifyCandidate(movingNode, candidateNode, toHeight);
        if (proposalKind == ProposalKind.INVALID) {
            return minWeight;
        }

        double candidateLogP = computeCandidateLogP(movingNode, candidateNode, toHeight, proposalKind);
        return boundedWeight(0.5 * temperature * candidateLogP);
    }

    private ProposalKind classifyCandidate(Node movingNode, Node candidateNode, double attachmentAge) {
        if (movingNode == null || candidateNode == null) {
            return ProposalKind.INVALID;
        }
        Node oldParent = movingNode.getParent();
        if (oldParent == null || oldParent.isRoot()) {
            return ProposalKind.INVALID;
        }
        if (!lineageCoversHeight(candidateNode, attachmentAge)) {
            return ProposalKind.INVALID;
        }
        Node sibling = getOtherChild(oldParent, movingNode);
        if (candidateNode == movingNode) {
            return ProposalKind.INVALID;
        }
        if (candidateNode == sibling || candidateNode == oldParent) {
            return ProposalKind.HEIGHT_ONLY;
        }
        if (candidateNode.isRoot()) {
            return ProposalKind.INVALID;
        }
        Node candidateParent = candidateNode.getParent();
        if (candidateParent == null || candidateParent == oldParent) {
            return ProposalKind.INVALID;
        }
        if (isDescendantOf(candidateNode, movingNode) || isDescendantOf(candidateNode, oldParent)) {
            return ProposalKind.INVALID;
        }
        return ProposalKind.TOPOLOGY_CHANGE;
    }

    private static boolean lineageCoversHeight(Node child, double height) {
        Node parent = child.getParent();
        return parent != null && child.getHeight() <= height && height <= parent.getHeight();
    }

	private static boolean isDescendantOf(Node node, Node ancestor) {
		Node current = node;
		while (current != null) {
			if (current == ancestor) {
				return true;
			}
			current = current.getParent();
		}
		return false;
	}

    private double computeCandidateLogP(Node movingNode, Node candidate, double attachmentAge,
            ProposalKind proposalKind) {
        if (proposalKind == ProposalKind.HEIGHT_ONLY) {
            return computeHeightOnlyCandidateLogP(movingNode, attachmentAge);
        }

        Node oldParent = movingNode.getParent();
        Node sibling = getOtherChild(oldParent, movingNode);
        Node grandParent = oldParent.getParent();
        Node candidateParent = candidate.getParent();

        if (grandParent == null || candidateParent == null) {
            return Double.NEGATIVE_INFINITY;
        }

        double previousHeight = oldParent.getHeight();

        double movedLogP;
        boolean heightChanged = false;
        boolean detachedFromGrandParent = false;
        boolean candidateAttachedToOldParent = false;
        boolean oldParentAttachedToCandidateParent = false;
        try {
            oldParent.setHeight(attachmentAge);
            heightChanged = true;
            replace(grandParent, oldParent, sibling);
            detachedFromGrandParent = true;
            replace(oldParent, sibling, candidate);
            candidateAttachedToOldParent = true;
            replace(candidateParent, candidate, oldParent);
            oldParentAttachedToCandidateParent = true;
            tree.setEverythingDirty(true);
            movedLogP = prior.calculateLogP();
        } catch (RuntimeException e) {
            movedLogP = Double.NEGATIVE_INFINITY;
        } finally {
            try {
                if (oldParentAttachedToCandidateParent) {
                    replace(candidateParent, oldParent, candidate);
                }
                if (candidateAttachedToOldParent) {
                    replace(oldParent, candidate, sibling);
                }
                if (detachedFromGrandParent) {
                    replace(grandParent, sibling, oldParent);
                }
                if (heightChanged) {
                    oldParent.setHeight(previousHeight);
                }
                tree.setEverythingDirty(true);
            } catch (RuntimeException e) {
                movedLogP = Double.NEGATIVE_INFINITY;
            }
        }

        double restoredLogP = prior.calculateLogP();
        if (!Double.isFinite(restoredLogP)) {
            throw new IllegalStateException("Prior produced non-finite restored logP for prior '"
                    + prior.getID() + "', movingNode='" + nodeId(movingNode)
                    + "', candidate='" + nodeId(candidate)
                    + "', attachmentAge=" + attachmentAge + ": " + restoredLogP);
        }
        if (!Double.isFinite(movedLogP)) {
            return Double.NEGATIVE_INFINITY;
        }
        return movedLogP;
    }

    private double computeHeightOnlyCandidateLogP(Node movingNode, double attachmentAge) {
        Node oldParent = movingNode.getParent();
        double previousHeight = oldParent.getHeight();

        double movedLogP;
        try {
            oldParent.setHeight(attachmentAge);
            tree.setEverythingDirty(true);
            movedLogP = prior.calculateLogP();
        } catch (RuntimeException e) {
            movedLogP = Double.NEGATIVE_INFINITY;
        } finally {
            oldParent.setHeight(previousHeight);
            tree.setEverythingDirty(true);
        }

        double restoredLogP = prior.calculateLogP();
        if (!Double.isFinite(restoredLogP)) {
            throw new IllegalStateException("Prior produced non-finite restored logP for prior '"
                    + prior.getID() + "', movingNode='" + nodeId(movingNode)
                    + "', attachmentAge=" + attachmentAge + ": " + restoredLogP);
        }
        if (!Double.isFinite(movedLogP)) {
            return Double.NEGATIVE_INFINITY;
        }
        return movedLogP;
    }

    private double boundedWeight(double logWeight) {
        double weight = Math.exp(logWeight);
        return Math.max(minWeight, Math.min(maxWeight, weight));
    }

    double[] clipMaxToSecondMaxRatio(double[] weights, double maxRatio) {
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

    /**
	 * Softmax: converts log-values to normalized probabilities.
	 */
	private static void normalise(double[] vals) {
		double s = 0.0;
		for (int i = 0; i < vals.length; i++) {
			s += vals[i];
		}
		for (int i = 0; i < vals.length; i++) {
			vals[i] /= s;
		}
	}

    private static Node getOtherChild(Node parent, Node child) {
        return parent.getLeft() == child ? parent.getRight() : parent.getLeft();
    }

    private static void replace(Node node, Node child, Node replacement) {
        node.removeChild(child);
        node.addChild(replacement);
    }

    private static String nodeId(Node node) {
        if (node == null) {
            return "<null>";
        }
        return node.getID() != null ? node.getID() : "node-" + node.getNr();
    }

    @Override
    protected boolean requiresRecalculation() {
        return true;
    }

    @Override
    public void store() {
        super.store();
    }

    @Override
    public void restore() {
        super.restore();
    }

    @Override
    public double minEdgeWeight() {
        return minWeight;
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
}