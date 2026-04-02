
package targetedbeast.operators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
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
public class TargetedWilsonBalding extends TreeOperator {

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
		return interval.minAge + (Randomizer.nextDouble() * interval.range());
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

    @Override
    public double proposal() {
        boolean useEdgeLength = useEdgeLengthInput.get();
		boolean useAttachmentIntervalLength = useAttachmentIntervalLengthInput.get();
        Tree tree = (Tree) InputUtil.get(treeInput, this);

        double logHastingsRatio = 0.0;

		SourceSelection sourceSelection = selectSourceNode(tree, useEdgeLength);

		Node i = sourceSelection.node;
		if (useEdgeLength) {
			logHastingsRatio -= Math.log(sourceSelection.selectedWeight / sourceSelection.totalWeight);
		}
		
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

		double[] distance = edgeWeights.getTargetWeights(i.getNr(), candidateNodes, candidateAges);
		double[] forwardMass = new double[distance.length];
        for (int k=0; k<distance.length; k++) {
			distance[k] = Math.pow(distance[k], 2);
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
        double newAge = candidateAges.get(nodeNr);
        AttachmentInterval forwardInterval = candidateIntervals.get(nodeNr);
		assert forwardInterval != null;
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
		double reverseTotalDistance = 0;
		double reverseSiblingDistance = 0;
		AttachmentInterval reverseInterval = null;
		for (int k = 0; k < candidateNodes.size(); k++) {
			Node candidate = candidateNodes.get(k);
			double candidateParentHeight = candidate == CiP ? PiP.getHeight() : candidate.getParent().getHeight();
			AttachmentInterval candidateReverseInterval = getAttachmentInterval(
					i.getHeight(),
					newAge,
					candidate.getHeight(),
					candidateParentHeight);
			double reverseMass = distance[k] * getAttachmentSelectionWeight(candidateReverseInterval, useAttachmentIntervalLength);
			if (k != nodeNr) {
				reverseTotalDistance += reverseMass;
			}
			if (k == siblingIdx) {
				reverseSiblingDistance = reverseMass;
				reverseInterval = candidateReverseInterval;
			}
		}

		if (reverseTotalDistance < 1E-12 || reverseSiblingDistance < 1E-12) {
			return Double.NEGATIVE_INFINITY;
		}
		logHastingsRatio += Math.log(reverseSiblingDistance / reverseTotalDistance);
		
		p.setHeight(newAge);

		// remove p from PiP, add the other child of p to PiP, frees p
		replace(PiP, p, CiP);
		// add j as child to p, removing CiP as child of p
		replace(p, CiP, j);
		// then parent node of j to p
		replace(jP, j, p);

        assert reverseInterval != null;
        assert reverseInterval.contains(oldAge);
		logHastingsRatio += Math.log(1. / reverseInterval.range());

		// mark paths to common ancestor as changed
		Node mrca = getMRCA(PiP, p);
		mrca.makeDirty(3 - mrca.isDirty());
		
		if (useEdgeLength) {
			double reverseTotalWeight = sourceSelection.totalWeight - sourceSelection.selectedWeight + i.getLength();
			logHastingsRatio += Math.log(i.getLength() / reverseTotalWeight);
		}
    	
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
