
package targetedbeast.operators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.util.InputUtil;
import beast.base.util.Randomizer;
import targetedbeast.edgeweights.EdgeWeights;

/**
 * WILSON, I. J. and D. J. BALDING, 1998 Genealogical inference from
 * microsatellite data. Genetics 150:499-51
 * http://www.genetics.org/cgi/ijlink?linkType=ABST&journalCode=genetics&resid=150/1/499
 */
@Description("Implements the unweighted Wilson-Balding branch swapping move. "
		+ "This move is similar to one proposed by WILSON and BALDING 1998  "
		+ "and involves removing a subtree and re-attaching it on a new parent branch. "
		+ "See <a href='http://www.genetics.org/cgi/content/full/161/3/1307/F1'>picture</a>.")
public class TargetedWilsonBalding extends TargetedWeightOperator {

    public Input<EdgeWeights> edgeWeightsInput = new Input<>("edgeWeights", "input of weights to be used for targetedn tree operations");

    public Input<Double> mutationLimitInput = new Input<>("mutationLimit", "Input of the number of mutations to be used as a limit", 15.0);

	public Input<Double> maxHeightRatioInput = new Input<>("maxHeightRatio", "optional upper bound on the multiplicative change in the parent height of the moved subtree", Double.POSITIVE_INFINITY);
	    
	public Input<Boolean> useAttachmentIntervalLengthInput = new Input<>(
			"useAttachmentIntervalLength",
			"if true, target-edge selection is weighted by the valid attachment-interval length",
			true);

    public Input<Boolean> useEdgeLengthInput = new Input<>("useEdgeLength", "if true, it uses the relative edge lengths for operations", false);

    double limit;

    EdgeWeights edgeWeights;

    @Override
    public void initAndValidate() {
		limit = mutationLimitInput.get();
		edgeWeights = edgeWeightsInput.get();
		double maxHeightRatio = maxHeightRatioInput.get();
		if (!(maxHeightRatio >= 1.0)) {
			throw new IllegalArgumentException("maxHeightRatio must be >= 1.0");
		}

		// Target weights have always been squared here, so that stays the no-transform default.
		initWeightTransforms(edgeWeights, true);
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

	/**
	 * Transformed source weights over the feasible (non-root) nodes.
	 *
	 * <p>The root is left out of the vector entirely rather than given weight zero: a
	 * transform adds its offset to every entry it receives, so a zero-weight root would
	 * become selectable and break the {@code p = i.getParent()} that follows.
	 */
	private SourceWeights getSourceWeights(Tree tree, boolean useEdgeLength) {
		return sourceWeights(tree,
				node -> !node.isRoot(),
				node -> useEdgeLength ? node.getLength() : edgeWeights.getEdgeWeights(node.getNr()));
	}

	private SourceSelection selectSourceNode(Tree tree, boolean useEdgeLength) {
		SourceWeights sourceWeights = getSourceWeights(tree, useEdgeLength);
		int m = sourceWeights.sampleIndex();
		if (m < 0) {
			throw new IllegalStateException("TargetedWilsonBalding has no selectable source edges");
		}
		return new SourceSelection(
				tree.getNode(sourceWeights.nodeNrs[m]),
				sourceWeights.total,
				sourceWeights.weights[m]);
	}

    @Override
    public double proposal() {
		// Freeze the transform coefficients for this proposal, so the forward and reverse
		// probabilities below are computed with identical weights.
		startTransformProposal();

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
		if (useEdgeLength && edgeWeightTransform == null) {
			// A Wilson-Balding move only changes the moved subtree's parent edge length, and
			// without a transform the total is a plain sum, so one term can be swapped out.
			iWeight = getSourceWeight(i, useEdgeLength);
			reverseTotalWeight = sourceSelection.totalWeight - sourceSelection.selectedWeight + iWeight;
		} else {
			// Either many edge weights changed, or the transform makes the total a nonlinear
			// function of the whole vector. Recompute it in full.
			if (!useEdgeLength) {
				// Sync the cached weights to T' before reading them.
				edgeWeights.prestore();
				edgeWeights.updateByOperator();
			}
			SourceWeights reverseWeights = getSourceWeights(tree, useEdgeLength);
			iWeight = reverseWeights.weightFor(i.getNr());
			reverseTotalWeight = reverseWeights.total;
		}
		if (reverseTotalWeight < 1E-12) {
			// BEAST does not restore calculation nodes on a direct reject: undo the cache update above.
			edgeWeights.reset();
			return Double.NEGATIVE_INFINITY;
		}
		logHastingsRatio += Math.log(iWeight / reverseTotalWeight);

        return logHastingsRatio;
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
