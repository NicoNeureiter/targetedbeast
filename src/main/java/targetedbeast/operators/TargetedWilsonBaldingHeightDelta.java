package targetedbeast.operators;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.operator.SubtreeSlide;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.util.InputUtil;
import beast.base.util.Randomizer;
import targetedbeast.edgeweights.EdgeWeights;

@Description("Implements a targeted Wilson-Balding move that first proposes a new parent height "
		+ "and then chooses a target lineage among lineages present at that height.")
public class TargetedWilsonBaldingHeightDelta extends TargetedWeightOperator {

	private static final double WEIGHT_EPS = 1E-16;

	public Input<EdgeWeights> edgeWeightsInput = new Input<>("edgeWeights",
			"input of weights to be used for targeted tree operations");

	public Input<Double> mutationLimitInput = new Input<>("mutationLimit",
			"Input of the number of mutations to be used as a limit", 15.0);

	public Input<Double> sizeInput = new Input<>("size",
			"additive height proposal size used when no SubtreeSlide is provided");

	public Input<SubtreeSlide> subtreeSlideInput = new Input<>("subtreeSlide",
			"optional SubtreeSlide operator whose tuned size is reused for the additive height proposal",
			Validate.XOR, sizeInput);

	public Input<Double> shrinkageInput = new Input<>("shrinkage",
			"optional factor in (0, 1] used to shrink the borrowed SubtreeSlide size", 1.0);

	public Input<Boolean> useEdgeLengthInput = new Input<>("useEdgeLength",
			"if true, the source edge is sampled proportional to branch length", false);

	public Input<Boolean> forceTopologyChangeInput = new Input<>("forceTopologyChange",
			"if true, proposals that only change the parent height are disallowed", false);

	double limit;
	EdgeWeights edgeWeights;
    HashMap<String, Integer> _rejects;
	private boolean warnedNoSource = false;

	@Override
	public void initAndValidate() {
        _rejects = new HashMap<>();

		limit = mutationLimitInput.get();
		edgeWeights = edgeWeightsInput.get();
		Double explicitSize = sizeInput.get();
		if (explicitSize != null && !(explicitSize > 0.0)) {
			throw new IllegalArgumentException("size must be > 0.0");
		}

		double shrinkage = shrinkageInput.get();
		if (!(shrinkage > 0.0 && shrinkage <= 1.0)) {
			throw new IllegalArgumentException("shrinkage must be in (0, 1]");
		}

		SubtreeSlide subtreeSlide = subtreeSlideInput.get();
		if (subtreeSlide != null && subtreeSlide.treeInput.get() != treeInput.get()) {
			throw new IllegalArgumentException("subtreeSlide must operate on the same tree as TargetedWilsonBaldingHeightDelta");
		}

		// Target weights have always been squared here, so that stays the no-transform default.
		initWeightTransforms(edgeWeights, true);
	}

	private static final class SourceSelection {
		final Node node;
		final double totalWeight;
		final double selectedWeight;

		SourceSelection(Node node, double totalWeight, double selectedWeight) {
			this.node = node;
			this.totalWeight = totalWeight;
			this.selectedWeight = selectedWeight;
		}
	}

	private static final class TargetSelection {
		final List<Integer> candidateNodeNrs;
		final double[] selectionMass;
		final double totalMass;

		TargetSelection(List<Integer> candidateNodeNrs, double[] selectionMass, double totalMass) {
			this.candidateNodeNrs = candidateNodeNrs;
			this.selectionMass = selectionMass;
			this.totalMass = totalMass;
		}
	}

    void registerReject(String s) {
        if (!_rejects.containsKey(s)) 
            _rejects.put(s, 0);

        // Increment counter
        _rejects.put(s, _rejects.get(s) + 1);
    }

	@Override
	public double proposal() {
        // if (Randomizer.nextDouble() < 0.001)
        //     System.out.println(getID() + ":  " + _rejects);

		// Freeze the transform coefficients, so the forward and reverse probabilities below
		// are computed with identical weights.
		startTransformProposal();

		boolean useEdgeLength = useEdgeLengthInput.get();
		Tree tree = (Tree) InputUtil.get(treeInput, this);

		double logHastingsRatio = 0.0;

		SourceSelection sourceSelection = selectSourceNode(tree, useEdgeLength);
		if (sourceSelection == null) {
            registerReject("no selectable source");
			return Double.NEGATIVE_INFINITY;
		}
		Node i = sourceSelection.node;
		logHastingsRatio -= Math.log(sourceSelection.selectedWeight / sourceSelection.totalWeight);

		// Source edges that are children of the root carry zero weight (see getSourceWeight),
		// so p is guaranteed not to be the root here.
		Node p = i.getParent();
		double oldHeight = p.getHeight();
		double newHeight = oldHeight + sampleHeightDelta();
		if (i.getHeight() > newHeight || tree.getRoot().getHeight() < newHeight) {
            registerReject("newHeight out of bounds");
			return Double.NEGATIVE_INFINITY;
		}

		TargetSelection forwardSelection = scoreTargetsAtHeight(tree, i, newHeight);
		if (forwardSelection == null) {
            registerReject("forwardSelection == null");
			return Double.NEGATIVE_INFINITY;
		}

		int forwardIndex = sampleIndexByMass(forwardSelection.selectionMass, forwardSelection.totalMass);
		if (forwardIndex < 0) {
            registerReject("forwardIndex < 0");
			return Double.NEGATIVE_INFINITY;
		}
		logHastingsRatio -= Math.log(forwardSelection.selectionMass[forwardIndex] / forwardSelection.totalMass);

		Node j = tree.getNode(forwardSelection.candidateNodeNrs.get(forwardIndex));
		Node jP = j.getParent();
		if (j.isRoot() || jP == null) {
            registerReject("j.isRoot() || jP == null");
			return Double.NEGATIVE_INFINITY;
		}

		final Node PiP = p.getParent();
		final Node CiP = getOtherChild(p, i);
		final boolean topologyChanged = j != CiP && j != p;
		if (forceTopologyChangeInput.get() && !topologyChanged) {
            registerReject("!topologyChanged");
			return Double.NEGATIVE_INFINITY;
		}

		p.setHeight(newHeight);
		if (topologyChanged) {
			replace(PiP, p, CiP);
			replace(p, CiP, j);
			replace(jP, j, p);
		}

		TargetSelection reverseSelection = scoreTargetsAtHeight(tree, i, oldHeight);
		if (reverseSelection == null) {
            registerReject("reverseSelection == null");
			return Double.NEGATIVE_INFINITY;
		}

		int reverseTargetNr = topologyChanged ? CiP.getNr() : getHeightOnlyReverseTargetNr(p, CiP, oldHeight);
		int reverseIndex = reverseSelection.candidateNodeNrs.indexOf(reverseTargetNr);
		if (reverseIndex < 0) {
            registerReject("reverseIndex < 0");
			return Double.NEGATIVE_INFINITY;
		}
		logHastingsRatio += Math.log(reverseSelection.selectionMass[reverseIndex] / reverseSelection.totalMass);

		Node mrca = getMRCA(PiP, p);
		mrca.makeDirty(Tree.IS_FILTHY);
		tree.setEverythingDirty(true);

		edgeWeights.prestore();
		edgeWeights.updateByOperator();

		// Reverse source-selection probability, evaluated on the moved tree T'. With a
		// transform the total is a nonlinear function of the whole vector, so it is rebuilt
		// rather than adjusted term by term.
		SourceWeights reverseSourceWeights = getSourceWeights(tree, useEdgeLength);
		double reverseSelectedWeight = reverseSourceWeights.weightFor(i.getNr());
		if (reverseSourceWeights.isDegenerate() || !(reverseSelectedWeight > 0.0)) {
			registerReject("!(reverseSelectedWeight > 0.0) || !(reverseTotalWeight > 0.0)");
			return Double.NEGATIVE_INFINITY;
		}
		logHastingsRatio += Math.log(reverseSelectedWeight / reverseSourceWeights.total);

        if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
            registerReject("logHR == -INF");
        }
		return logHastingsRatio;
	}

	private void warnNoSource() {
		if (!warnedNoSource) {
			warnedNoSource = true;
			System.err.println("WARNING: " + getID() + " has no movable source edge on this tree; "
					+ "every proposal will be rejected. Consider deactivating this operator.");
		}
	}

	/**
	 * Transformed source weights over the movable edges.
	 *
	 * <p>The root and its children are left out of the vector entirely rather than given
	 * weight zero: a source edge whose parent is the root cannot be moved by a Wilson-Balding
	 * step, and a transform adds its offset to every entry it receives, so a zero entry would
	 * acquire positive selection probability.
	 */
	private SourceWeights getSourceWeights(Tree tree, boolean useEdgeLength) {
		return sourceWeights(tree,
				node -> !(node.isRoot() || node.getParent().isRoot()),
				node -> useEdgeLength ? node.getLength() : edgeWeights.getEdgeWeights(node.getNr()));
	}

	private SourceSelection selectSourceNode(Tree tree, boolean useEdgeLength) {
		SourceWeights sourceWeights = getSourceWeights(tree, useEdgeLength);
		int m = sourceWeights.sampleIndex();
		if (m < 0) {
			// No movable source edge: every non-root edge is a child of the root
			// (e.g. a degenerate two-taxon tree). Reject gracefully instead of crashing.
			warnNoSource();
			return null;
		}
		return new SourceSelection(
				tree.getNode(sourceWeights.nodeNrs[m]),
				sourceWeights.total,
				sourceWeights.weights[m]);
	}

	private double sampleHeightDelta() {
		double size = getProposalSize();
		if (!(size > 0.0)) {
			throw new IllegalStateException("TargetedWilsonBaldingHeightDelta requires a positive proposal size");
		}

		SubtreeSlide subtreeSlide = subtreeSlideInput.get();
		boolean useGaussian = subtreeSlide == null || subtreeSlide.gaussianInput.get();
		if (useGaussian) {
			return Randomizer.nextGaussian() * size;
		}
		return (Randomizer.nextDouble() * size) - (size / 2.0);
	}

	private double getProposalSize() {
		SubtreeSlide subtreeSlide = subtreeSlideInput.get();
		if (subtreeSlide != null) {
			return subtreeSlide.getCoercableParameterValue() * shrinkageInput.get();
		}
		return sizeInput.get();
	}

	private int getHeightOnlyReverseTargetNr(Node parent, Node sibling, double reverseHeight) {
		return reverseHeight > parent.getHeight() ? parent.getNr() : sibling.getNr();
	}

	private TargetSelection scoreTargetsAtHeight(Tree tree, Node movingNode, double targetHeight) {
		List<Integer> candidateNodeNrs = getLineagesAtHeight(tree, targetHeight);
		if (candidateNodeNrs.isEmpty()) {
			return null;
		}

		Node parent = movingNode.getParent();
		Node sibling = getOtherChild(parent, movingNode);
		List<Integer> ancestors = collectCandidateAncestors(parent, candidateNodeNrs, targetHeight);

		edgeWeights.prestore();
		edgeWeights.updateByOperatorWithoutNode(movingNode.getNr(), ancestors);

		if (forceTopologyChangeInput.get()) {
			candidateNodeNrs.remove(Integer.valueOf(parent.getNr()));
			candidateNodeNrs.remove(Integer.valueOf(sibling.getNr()));
		}
		candidateNodeNrs.remove(Integer.valueOf(movingNode.getNr()));
		if (candidateNodeNrs.isEmpty()) {
			edgeWeights.reset();
			return null;
		}

		edgeWeights.setCurrentOperatorTag(getClass().getSimpleName());
		double[] rawWeights = edgeWeights.getTargetWeightsInteger(movingNode.getNr(), candidateNodeNrs, targetHeight);
		edgeWeights.reset();

		// Infeasible candidates were already removed from candidateNodeNrs above, which is
		// what the transform requires: it adds its offset to every entry it receives.
		double[] selectionMass = applyTargetTransform(rawWeights);
		double totalMass = 0.0;
		for (double mass : selectionMass) {
			totalMass += mass;
		}

		if (!(totalMass > WEIGHT_EPS)) {
			return null;
		}

		return new TargetSelection(candidateNodeNrs, selectionMass, totalMass);
	}

	private int sampleIndexByMass(double[] selectionMass, double totalMass) {
		double scaler = Randomizer.nextDouble() * totalMass;
		double cumulativeMass = 0.0;
		for (int idx = 0; idx < selectionMass.length; idx++) {
			cumulativeMass += selectionMass[idx];
			if (cumulativeMass > scaler) {
				return idx;
			}
		}
		return -1;
	}

	private boolean lineageCoversHeight(Node child, double height) {
		Node parent = child.getParent();
		return !child.isRoot() && child.getHeight() <= height && height <= parent.getHeight();
	}

	private List<Integer> getLineagesAtHeight(Tree tree, double height) {
		List<Integer> lineages = new ArrayList<>();
		for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
			if (lineageCoversHeight(tree.getNode(nodeNr), height)) {
				lineages.add(nodeNr);
			}
		}
		return lineages;
	}

	private List<Integer> collectCandidateAncestors(Node start, List<Integer> candidateNodeNrs, double maxHeight) {
		List<Integer> ancestors = new ArrayList<>();
		Node ancestor = start;
		while (ancestor != null && ancestor.getHeight() < maxHeight) {
			if (candidateNodeNrs.contains(ancestor.getNr())) {
				ancestors.add(ancestor.getNr());
			}
			ancestor = ancestor.getParent();
		}
		return ancestors;
	}

	private Node getMRCA(Node a, Node b) {
		Node iup = a;
		Node jup = b;
		while (iup != jup) {
			if (iup.getHeight() < jup.getHeight()) {
				assert !iup.isRoot();
				iup = iup.getParent();
			} else {
				assert !jup.isRoot();
				jup = jup.getParent();
			}
		}
		return jup;
	}

	@Override
	public void replace(final Node node, final Node child, final Node replacement) {
		node.removeChild(child);
		node.addChild(replacement);
		node.makeDirty(Tree.IS_FILTHY);
		replacement.makeDirty(Tree.IS_FILTHY);
	}
}