package targetedbeast.edgeweights;

import java.util.List;
import java.util.Random;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Distribution;
import beast.base.inference.State;

@Description("Heuristic prior-aware edge weights for targeted operators. "
        + "These weights do not evaluate the tree prior exactly. Instead, they bias proposals toward "
        + "shorter and younger local attachments so they can be composed with likelihood-informed weights.")
public class PriorHeuristicWeights extends Distribution implements EdgeWeights {

    private static final double LOG_WEIGHT_CLIP = 50.0;

    final public Input<TreeInterface> treeInput = new Input<>("tree",
            "phylogenetic tree used for targeted proposals",
            Validate.REQUIRED);
    final public Input<Double> minWeightInput = new Input<>("minWeight",
            "minimum positive proposal weight returned by the heuristic",
            1e-6);
    final public Input<Double> maxWeightInput = new Input<>("maxWeight",
            "maximum proposal weight returned by the heuristic",
            1e6);
    final public Input<Double> temperatureInput = new Input<>("temperature",
            "global scaling factor for all heuristic terms",
            1.0);
    final public Input<Double> sourceBranchScaleInput = new Input<>("sourceBranchScale",
            "strength with which longer source edges are favored",
            1.0);
    final public Input<Double> attachmentAgeScaleInput = new Input<>("attachmentAgeScale",
            "penalty on older proposed attachment heights",
            0.25);
    final public Input<Double> heightIncreaseScaleInput = new Input<>("heightIncreaseScale",
            "additional penalty on increases in the moving parent height",
            1.0);
    final public Input<Double> maxNewBranchScaleInput = new Input<>("maxNewBranchScale",
            "penalty on the largest candidate-dependent branch created by the move",
            1.0);

    private double minWeight;
    private double maxWeight;
    private double temperature;
    private double sourceBranchScale;
    private double attachmentAgeScale;
    private double heightIncreaseScale;
    private double maxNewBranchScale;

    @Override
    public void initAndValidate() {
        minWeight = minWeightInput.get();
        maxWeight = maxWeightInput.get();
        temperature = temperatureInput.get();
        sourceBranchScale = sourceBranchScaleInput.get();
        attachmentAgeScale = attachmentAgeScaleInput.get();
        heightIncreaseScale = heightIncreaseScaleInput.get();
        maxNewBranchScale = maxNewBranchScaleInput.get();

        if (!(minWeight > 0.0)) {
            throw new IllegalArgumentException("minWeight must be > 0, but was " + minWeight);
        }
        if (!(maxWeight >= minWeight)) {
            throw new IllegalArgumentException("maxWeight must be >= minWeight, but was " + maxWeight);
        }
        if (!(temperature >= 0.0)) {
            throw new IllegalArgumentException("temperature must be >= 0, but was " + temperature);
        }
        if (!(sourceBranchScale >= 0.0)) {
            throw new IllegalArgumentException("sourceBranchScale must be >= 0, but was " + sourceBranchScale);
        }
        if (!(attachmentAgeScale >= 0.0)) {
            throw new IllegalArgumentException("attachmentAgeScale must be >= 0, but was " + attachmentAgeScale);
        }
        if (!(heightIncreaseScale >= 0.0)) {
            throw new IllegalArgumentException("heightIncreaseScale must be >= 0, but was " + heightIncreaseScale);
        }
        if (!(maxNewBranchScale >= 0.0)) {
            throw new IllegalArgumentException("maxNewBranchScale must be >= 0, but was " + maxNewBranchScale);
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
        // return 1.0;

        Node node = treeInput.get().getNode(nodeNr);
        if (node == null || node.isRoot()) {
            return 0.0;
        }
        return 1. / (0.1 + node.getLength());


        // double lengthScale = getLengthScale();
        // double standardizedLength = (node.getLength() / lengthScale) - 1.0;
        // double logWeight = temperature * sourceBranchScale * standardizedLength;
        // return boundedWeight(logWeight);
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
        Node movingNode = treeInput.get().getNode(fromNodeNr);
        if (movingNode == null || movingNode.isRoot()) {
            return weights;
        }

        double lengthScale = getLengthScale();
        for (int i = 0; i < toNodeNrs.size(); i++) {
            weights[i] = targetWeightForCandidate(movingNode, toNodeNrs.get(i), toHeights[i], lengthScale);
        }
        return weights;
    }

    @Override
    public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
        double[] targetHeights = new double[toNodeNrs.size()];
        for (int i = 0; i < toNodeNrs.size(); i++) {
            Node node = treeInput.get().getNode(toNodeNrs.get(i));
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
        Node movingNode = treeInput.get().getNode(fromNodeNr);
        if (movingNode == null || movingNode.isRoot()) {
            return weights;
        }

        double lengthScale = getLengthScale();
        for (int i = 0; i < toNodeNrs.size(); i++) {
            Node candidateNode = treeInput.get().getNode(toNodeNrs.get(i));
            weights[i] = targetWeightForCandidate(movingNode, candidateNode, toHeights[i], lengthScale);
        }
        return weights;
    }

    private double targetWeightForCandidate(Node movingNode, Node candidateNode, double toHeight, double lengthScale) {
        if (candidateNode == null || candidateNode.isRoot() || movingNode.getParent() == null || candidateNode.getParent() == null) {
            return minWeight;
        }

        double oldParentHeight = movingNode.getParent().getHeight();
        double attachmentAge = Math.max(0.0, toHeight);
        double heightIncrease = Math.max(0.0, toHeight - oldParentHeight);
        double movingBranchLength = Math.max(0.0, toHeight - movingNode.getHeight());
        double candidateBranchLength = Math.max(0.0, toHeight - candidateNode.getHeight());
        double parentBranchLength = Math.max(0.0, candidateNode.getParent().getHeight() - toHeight);
        double maxNewLocalBranch = Math.max(movingBranchLength,
                Math.max(candidateBranchLength, parentBranchLength));

        double logWeight = -temperature * (
                attachmentAgeScale * (attachmentAge / lengthScale)
                        + heightIncreaseScale * (heightIncrease / lengthScale)
                        + maxNewBranchScale * (maxNewLocalBranch / lengthScale));
        return boundedWeight(logWeight);
    }

    private double boundedWeight(double logWeight) {
        double clipped = Math.max(-LOG_WEIGHT_CLIP, Math.min(LOG_WEIGHT_CLIP, logWeight));
        double weight = Math.exp(clipped);
        return Math.max(minWeight, Math.min(maxWeight, weight));
    }

    private double getLengthScale() {
        TreeInterface tree = treeInput.get();
        double totalLength = 0.0;
        int branchCount = 0;
        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
            Node node = tree.getNode(nodeNr);
            if (node == null || node.isRoot()) {
                continue;
            }
            totalLength += Math.max(0.0, node.getLength());
            branchCount++;
        }
        if (branchCount == 0) {
            return 1.0;
        }
        double meanLength = totalLength / branchCount;
        return meanLength > 0.0 ? meanLength : 1.0;
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