
package targetedbeast.operators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.util.InputUtil;
import beast.base.util.Randomizer;
import targetedbeast.edgeweights.EdgeWeights;
import targetedbeast.weighttransform.WeightTransform;

/**
 * WILSON, I. J. and D. J. BALDING, 1998 Genealogical inference from
 * microsatellite data. Genetics 150:499-51
 * http://www.genetics.org/cgi/ijlink?linkType=ABST&journalCode=genetics&resid=150/1/499
 */
@Description("Implements the unweighted Wilson-Balding branch swapping move. "
		+ "This move is similar to one proposed by WILSON and BALDING 1998  "
		+ "and involves removing a subtree and re-attaching it on a new parent branch. "
		+ "See <a href='http://www.genetics.org/cgi/content/full/161/3/1307/F1'>picture</a>.")
public class TargetedWilsonBalding extends TreeOperator {

    public Input<EdgeWeights> edgeWeightsInput = new Input<>("edgeWeights", "input of weights to be used for targetedn tree operations");

    public Input<Double> mutationLimitInput = new Input<>("mutationLimit", "Input of the number of mutations to be used as a limit", 15.0);

	public Input<Double> maxHeightRatioInput = new Input<>("maxHeightRatio", "optional upper bound on the multiplicative change in the parent height of the moved subtree", Double.POSITIVE_INFINITY);
	    
	public Input<Boolean> useAttachmentIntervalLengthInput = new Input<>(
			"useAttachmentIntervalLength",
			"if true, target-edge selection is weighted by the valid attachment-interval length",
			true);

    public Input<Boolean> useEdgeLengthInput = new Input<>("useEdgeLength", "if true, it uses the relative edge lengths for operations", false);

	public Input<WeightTransform> targetWeightTransformInput = new Input<>("targetWeightTransform",
			"optional transform applied to the target weights before they are normalized; "
					+ "when absent the target weights are squared, as they always have been");

	public Input<WeightTransform> edgeWeightTransformInput = new Input<>("edgeWeightTransform",
			"optional transform applied to the source-edge weights before they are normalized");

    double limit;

    EdgeWeights edgeWeights;

	WeightTransform targetWeightTransform;
	WeightTransform edgeWeightTransform;


    @Override
    public void initAndValidate() {
		limit = mutationLimitInput.get();
		edgeWeights = edgeWeightsInput.get();
		double maxHeightRatio = maxHeightRatioInput.get();
		if (!(maxHeightRatio >= 1.0)) {
			throw new IllegalArgumentException("maxHeightRatio must be >= 1.0");
		}

		targetWeightTransform = targetWeightTransformInput.get();
		edgeWeightTransform = edgeWeightTransformInput.get();

		if (targetWeightTransform != null
				&& targetWeightTransform.requiresLogSpace() != edgeWeights.isLogSpace()) {
			throw new IllegalArgumentException("targetWeightTransform expects "
					+ (targetWeightTransform.requiresLogSpace() ? "log" : "real")
					+ "-space weights but " + edgeWeights.getClass().getSimpleName()
					+ " returns " + (edgeWeights.isLogSpace() ? "log" : "real") + "-space weights");
		}
		// getEdgeWeights is real-space by contract in every implementation, and with
		// useEdgeLength=true the source weights are branch lengths rather than EdgeWeights
		// output, so a log-space transform is never correct here.
		if (edgeWeightTransform != null && edgeWeightTransform.requiresLogSpace()) {
			throw new IllegalArgumentException(
					"edgeWeightTransform must be a real-space transform; source edge weights are "
							+ "never in log space");
		}
    }

	private static final class AttachmentInterval {
		final double minAge;
		final double maxAge;

		AttachmentInterval(double minAge, double maxAge) {
			this.minAge = minAge;
			this.maxAge = maxAge;
		}

		double range() {
			return Math.max(0, maxAge - minAge);
		}

		boolean contains(double age) {
			return age >= minAge && age <= maxAge;
		}
	}

	private AttachmentInterval getAttachmentInterval(double movingNodeHeight, double movingParentHeight,
			double candidateHeight, double candidateParentHeight) {
		double minAge = Math.max(movingNodeHeight, candidateHeight);
		double maxAge = candidateParentHeight;

		double maxHeightRatio = maxHeightRatioInput.get();
		if (Double.isFinite(maxHeightRatio)) {
			minAge = Math.max(minAge, movingParentHeight / maxHeightRatio);
			maxAge = Math.min(maxAge, movingParentHeight * maxHeightRatio);
		}

		return new AttachmentInterval(minAge, maxAge);
	}

	private AttachmentInterval getAttachmentInterval(Node movingNode, Node candidate) {
		return getAttachmentInterval(
				movingNode.getHeight(),
				movingNode.getParent().getHeight(),
				candidate.getHeight(),
				candidate.getParent().getHeight());
	}

	private double sampleAttachmentAge(AttachmentInterval interval) {
		double range = interval.range();
		if (range <= 0.0) {
			return interval.minAge;
		}
		double epsilon = Math.max(1e-10, 1e-6 * range);
		double lower = interval.minAge + epsilon;
		double upper = interval.maxAge - epsilon;
		if (lower >= upper) {
			return 0.5 * (interval.minAge + interval.maxAge);
		}
		return lower + (Randomizer.nextDouble() * (upper - lower));
	}

	private boolean sameInterval(AttachmentInterval first, AttachmentInterval second) {
		return Math.abs(first.minAge - second.minAge) < 1e-12
				&& Math.abs(first.maxAge - second.maxAge) < 1e-12;
	}

	private double getAttachmentSelectionWeight(AttachmentInterval interval, boolean useAttachmentIntervalLength) {
		double range = interval.range();
		if (range <= 0.0) {
			return 0.0;
		}
		return useAttachmentIntervalLength ? range : 1.0;
	}

	private boolean hasAttachmentRange(AttachmentInterval interval) {
		return interval.range() > 0.0;
	}

	private void assertAgesWithinIntervals(List<Double> ages, List<AttachmentInterval> intervals, String context) {
		assert ages.size() == intervals.size() : "Age/interval size mismatch in " + context
				+ ": ages=" + ages.size() + ", intervals=" + intervals.size();
		for (int idx = 0; idx < ages.size(); idx++) {
			double age = ages.get(idx);
			AttachmentInterval interval = intervals.get(idx);
			assert interval.contains(age) : "Attachment age outside interval in " + context
					+ ": age=" + age
					+ ", minAge=" + interval.minAge
					+ ", maxAge=" + interval.maxAge
					+ ", index=" + idx;
		}
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

	private double getSourceWeight(Node node, boolean useEdgeLength) {
		if (node.isRoot()) {
			return 0.0;
		}
		return useEdgeLength ? node.getLength() : edgeWeights.getEdgeWeights(node.getNr());
	}

	private SourceSelection selectSourceNode(Tree tree, boolean useEdgeLength) {
		double totalWeight = 0.0;
		for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
			totalWeight += getSourceWeight(tree.getNode(nodeNr), useEdgeLength);
		}
		if (totalWeight < 1E-12) {
			throw new IllegalStateException("TargetedWilsonBalding has no selectable source edges");
		}

		double scaler = Randomizer.nextDouble() * totalWeight;
		double cumulativeWeight = 0.0;
		for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
			Node node = tree.getNode(nodeNr);
			double nodeWeight = getSourceWeight(node, useEdgeLength);
			cumulativeWeight += nodeWeight;
			if (cumulativeWeight > scaler) {
				return new SourceSelection(node, totalWeight, nodeWeight);
			}
		}

		throw new IllegalStateException("Failed to select source node despite positive total weight");
	}

	/**
	 * Reshape a target-weight vector.
	 *
	 * <p>Without a transform this is the squaring that has always been applied. With a
	 * transform, the transform's own exponent replaces it, so the two must never both run.
	 *
	 * <p>Every entry must be a genuinely selectable candidate: the transform's offset is
	 * added to all entries, so an infeasible candidate passed in as zero would acquire
	 * positive selection probability.
	 */
	private double[] applyTargetTransform(double[] rawWeights) {
		if (targetWeightTransform == null) {
			double[] out = new double[rawWeights.length];
			for (int k = 0; k < rawWeights.length; k++) {
				out[k] = rawWeights[k] * rawWeights[k];
			}
			return out;
		}
		return targetWeightTransform.apply(rawWeights);
	}

    @Override
    public double proposal() {
		// Freeze the transform coefficients for this proposal, so the forward and reverse
		// probabilities below are computed with identical weights.
		if (targetWeightTransform != null) {
			targetWeightTransform.startProposal();
		}
		if (edgeWeightTransform != null) {
			edgeWeightTransform.startProposal();
		}

        boolean useEdgeLength = useEdgeLengthInput.get();
		boolean useAttachmentIntervalLength = useAttachmentIntervalLengthInput.get();
        Tree tree = (Tree) InputUtil.get(treeInput, this);

        double logHastingsRatio = 0.0;

		SourceSelection sourceSelection = selectSourceNode(tree, useEdgeLength);

		Node i = sourceSelection.node;
		logHastingsRatio -= Math.log(sourceSelection.selectedWeight / sourceSelection.totalWeight);
		
		Node p = i.getParent();
		if (p.isRoot()) {
			return Double.NEGATIVE_INFINITY;
		}

		List<Node> candidateNodes = getCoExistingLineages(i, tree);
		if (candidateNodes.size() == 0) {
			return Double.NEGATIVE_INFINITY;
		}

		final Node CiP = getOtherChild(p, i);

		// get all nodes in target that are ancestors to p
		List<Integer> ancestors = new ArrayList<>();
		// make a copy of p
		Node ancestor = i.getParent();
		while (ancestor != null) {
			// check if p is in the target
			if (candidateNodes.contains(ancestor)) {
				ancestors.add(ancestor.getNr());
			}
			ancestor = ancestor.getParent();
		}

		// calculate the consensus sequences without the node i
		edgeWeights.prestore();
		edgeWeights.updateByOperatorWithoutNode(i.getNr(), ancestors);

		// remove i and p as potential targets
        candidateNodes.remove(i);
		candidateNodes.remove(p);

		List<Node> feasibleCandidateNodes = new ArrayList<>();
		List<Double> candidateAges = new ArrayList<>();
		List<AttachmentInterval> candidateIntervals = new ArrayList<>();
		for (Node c : candidateNodes) {
			AttachmentInterval interval = getAttachmentInterval(i, c);
			if (!hasAttachmentRange(interval)) {
				continue;
			}
			feasibleCandidateNodes.add(c);
			candidateAges.add(sampleAttachmentAge(interval));
			candidateIntervals.add(interval);
		}
		candidateNodes = feasibleCandidateNodes;
		if (candidateNodes.size() == 0) {
			edgeWeights.reset();
			return Double.NEGATIVE_INFINITY;
		}
		assertAgesWithinIntervals(candidateAges, candidateIntervals, "forward candidate scoring");

		double[] rawTargetWeights = edgeWeights.getTargetWeights(i.getNr(), candidateNodes, candidateAges);
		double[] distance = applyTargetTransform(rawTargetWeights);
		double[] forwardMass = new double[distance.length];
		for (int k = 0; k < distance.length; k++) {
			double intervalWeight = getAttachmentSelectionWeight(candidateIntervals.get(k), useAttachmentIntervalLength);
			forwardMass[k] = distance[k] * intervalWeight;
		}
		
        int siblingIdx = candidateNodes.indexOf(CiP);
		double[] selectionMass = forwardMass.clone();
		selectionMass[siblingIdx] = 0;
		

		double totalDistance = 0;
		for (int k = 0; k < candidateNodes.size(); k++) {
			totalDistance += selectionMass[k];				
		}

		if (totalDistance < 1E-12) {
			return Double.NEGATIVE_INFINITY;
		}

		double scaler2 = Randomizer.nextDouble() * totalDistance;
		double currDist = 0;
		int nodeNr = -1;
		for (int k = 0; k < candidateNodes.size(); k++) {
			currDist += selectionMass[k];
			if (currDist > scaler2) {				
				nodeNr = k;
				break;
			}
		}
        Node j = candidateNodes.get(nodeNr);
        AttachmentInterval forwardInterval = candidateIntervals.get(nodeNr);
		double newAge = sampleAttachmentAge(forwardInterval);
		assert forwardInterval != null;
		assert forwardInterval.contains(newAge) : "Sampled forward attachment age outside interval"
				+ ": newAge=" + newAge
				+ ", minAge=" + forwardInterval.minAge
				+ ", maxAge=" + forwardInterval.maxAge;
		logHastingsRatio -= Math.log(1. / forwardInterval.range());
		logHastingsRatio -= Math.log(selectionMass[nodeNr] / totalDistance);
      
		edgeWeights.reset();

		Node jP = j.getParent();

		// disallow moves that change the root.
		if (j.isRoot()) {
			return Double.NEGATIVE_INFINITY;
		}

		final int pnr = p.getNr();
		final int jPnr = jP.getNr();

		if (jPnr == pnr || j.getNr() == pnr || jPnr == i.getNr()) {
			return Double.NEGATIVE_INFINITY;
		}

		final Node PiP = p.getParent();

		double oldAge = p.getHeight();
		p.setHeight(newAge);

		// remove p from PiP, add the other child of p to PiP, frees p
		replace(PiP, p, CiP);
		// add j as child to p, removing CiP as child of p
		replace(p, CiP, j);
		// then parent node of j to p
		replace(jP, j, p);

		List<Node> reverseCandidateNodes = getCoExistingLineages(i, tree);
		reverseCandidateNodes.remove(i);
		reverseCandidateNodes.remove(p);
		if (reverseCandidateNodes.size() == 0) {
			return Double.NEGATIVE_INFINITY;
		}

		List<Integer> reverseAncestors = new ArrayList<>();
		Node reverseAncestor = i.getParent();
		while (reverseAncestor != null) {
			if (reverseCandidateNodes.contains(reverseAncestor)) {
				reverseAncestors.add(reverseAncestor.getNr());
			}
			reverseAncestor = reverseAncestor.getParent();
		}

		edgeWeights.prestore();
		edgeWeights.updateByOperatorWithoutNode(i.getNr(), reverseAncestors);

		Map<Integer, AttachmentInterval> forwardIntervalsByNode = new HashMap<>();
		Map<Integer, Double> forwardAgesByNode = new HashMap<>();
		for (int k = 0; k < candidateNodes.size(); k++) {
			forwardIntervalsByNode.put(candidateNodes.get(k).getNr(), candidateIntervals.get(k));
			forwardAgesByNode.put(candidateNodes.get(k).getNr(), candidateAges.get(k));
		}

		List<Node> feasibleReverseCandidateNodes = new ArrayList<>();
		List<AttachmentInterval> reverseCandidateIntervals = new ArrayList<>();
		List<Double> reverseCandidateAges = new ArrayList<>();
		for (Node candidate : reverseCandidateNodes) {
			AttachmentInterval reverseCandidateInterval = getAttachmentInterval(i, candidate);
			if (!hasAttachmentRange(reverseCandidateInterval)) {
				continue;
			}
			feasibleReverseCandidateNodes.add(candidate);
			reverseCandidateIntervals.add(reverseCandidateInterval);
			if (candidate == CiP) {
				reverseCandidateAges.add(oldAge);
				continue;
			}

			AttachmentInterval previousInterval = forwardIntervalsByNode.get(candidate.getNr());
			Double forwardAge = forwardAgesByNode.get(candidate.getNr());
			if (previousInterval != null && forwardAge != null && sameInterval(previousInterval, reverseCandidateInterval)) {
				reverseCandidateAges.add(forwardAge);
			} else {
				reverseCandidateAges.add(sampleAttachmentAge(reverseCandidateInterval));
			}
		}
		reverseCandidateNodes = feasibleReverseCandidateNodes;
		if (reverseCandidateNodes.size() == 0) {
			edgeWeights.reset();
			return Double.NEGATIVE_INFINITY;
		}
		assertAgesWithinIntervals(reverseCandidateAges, reverseCandidateIntervals, "reverse candidate scoring");

		int reverseSiblingIdx = reverseCandidateNodes.indexOf(CiP);
		if (reverseSiblingIdx < 0) {
			edgeWeights.reset();
			return Double.NEGATIVE_INFINITY;
		}
		int reverseNoOpIdx = reverseCandidateNodes.indexOf(j);

		double[] rawReverseWeights = edgeWeights.getTargetWeights(i.getNr(), reverseCandidateNodes, reverseCandidateAges);
		double[] reverseDistance = applyTargetTransform(rawReverseWeights);
		double[] reverseMass = new double[reverseDistance.length];
		for (int k = 0; k < reverseDistance.length; k++) {
			double intervalWeight = getAttachmentSelectionWeight(reverseCandidateIntervals.get(k), useAttachmentIntervalLength);
			reverseMass[k] = reverseDistance[k] * intervalWeight;
		}

		double[] reverseSelectionMass = reverseMass.clone();
		if (reverseNoOpIdx >= 0) {
			reverseSelectionMass[reverseNoOpIdx] = 0;
		}

		double reverseTotalDistance = 0;
		for (double mass : reverseSelectionMass) {
			reverseTotalDistance += mass;
		}
		double reverseSiblingDistance = reverseSelectionMass[reverseSiblingIdx];
		edgeWeights.reset();

		if (reverseTotalDistance < 1E-12 || reverseSiblingDistance < 1E-12) {
			return Double.NEGATIVE_INFINITY;
		}
		logHastingsRatio += Math.log(reverseSiblingDistance / reverseTotalDistance);

		AttachmentInterval reverseInterval = reverseCandidateIntervals.get(reverseSiblingIdx);

        assert reverseInterval != null;
        assert reverseInterval.contains(oldAge);
		logHastingsRatio += Math.log(1. / reverseInterval.range());

		// mark paths to common ancestor as changed
		Node mrca = getMRCA(PiP, p);
		mrca.makeDirty(3 - mrca.isDirty());
		
		// Reverse source-selection probability, evaluated on the moved tree T'.
		double iWeight;
		double reverseTotalWeight;
		if (useEdgeLength) {
			// A Wilson-Balding move only changes the moved subtree's parent edge length,
			// so the reverse total length is the forward total with that one edge updated.
			iWeight = getSourceWeight(i, useEdgeLength);
			reverseTotalWeight = sourceSelection.totalWeight - sourceSelection.selectedWeight + iWeight;
		} else {
			// Edge weights change for many edges after the move, so re-evaluate the full
			// source-weight sum on the moved tree. Sync the cached weights to T' first.
			edgeWeights.prestore();
			edgeWeights.updateByOperator();
			iWeight = getSourceWeight(i, useEdgeLength);
			reverseTotalWeight = 0.0;
			for (int nr = 0; nr < tree.getNodeCount(); nr++) {
				reverseTotalWeight += getSourceWeight(tree.getNode(nr), useEdgeLength);
			}
		}
		if (reverseTotalWeight < 1E-12) {
			return Double.NEGATIVE_INFINITY;
		}
		logHastingsRatio += Math.log(iWeight / reverseTotalWeight);

        return logHastingsRatio;
	}

	@Override
	public void optimize(final double logAlpha) {
		if (targetWeightTransform != null) {
			targetWeightTransform.optimize(logAlpha);
		}
		// Guard against the same object being wired to both inputs, which would otherwise
		// take two adaptation steps from one observation.
		if (edgeWeightTransform != null && edgeWeightTransform != targetWeightTransform) {
			edgeWeightTransform.optimize(logAlpha);
		}
	}
	
	/**
	 * Returns all nodes in the given tree whose parent's height exceeds the given node's height,
	 * excluding the root node.
	 *
	 * @param node the reference node whose height is used for comparison
	 * @param tree the tree containing the nodes to be filtered
	 */
	private List<Node> getCoExistingLineages(Node node, Tree tree) {
		return Arrays.stream(tree.getNodesAsArray())
			.filter(n -> !n.isRoot())
			.filter(n -> n.getParent().getHeight() > node.getHeight())
			.collect(Collectors.toList());
	}

	/**
	 * Finds the Most Recent Common Ancestor (MRCA) of two nodes in a tree.
	 *
	 * @param a the first node
	 * @param b the second node
	 * @return the node representing the MRCA of nodes a and b
	 */
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
		node.makeDirty(2);
		replacement.makeDirty(2);
	}

} // class WilsonBalding
