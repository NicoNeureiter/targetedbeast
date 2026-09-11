
package targetedbeast.operators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
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
public class TargetedWilsonBaldingFixedHeight extends TargetedWeightOperator {

	private static final double WEIGHT_EPS = 1E-12;

    public Input<EdgeWeights> edgeWeightsInput = new Input<>("edgeWeights", "input of weights to be used for targetedn tree operations");
    public Input<Double> mutationLimitInput = new Input<>("mutationLimit", "Input of the number of mutations to be used as a limit", 15.0);

	public Input<Boolean> useEdgeLengthInput = new Input<>("useEdgeLength", "if true, it uses the relative edge lengths for operations", false);

    
    double limit;
    
    EdgeWeights edgeWeights;

    HashMap<String, Integer> _rejects;
	
    @Override
    public void initAndValidate() {
		limit = mutationLimitInput.get();
		edgeWeights = edgeWeightsInput.get();
        _rejects = new HashMap<>();

		// Target weights have always been squared here, so that stays the no-transform default.
		initWeightTransforms(edgeWeights, true);
    }

    @Override
    public double proposal() {
        // if (Randomizer.nextDouble() < 0.001)
        //     System.out.println(getID() + ":  " + _rejects);

		// Freeze the transform coefficients, so the forward and reverse probabilities below
		// are computed with identical weights.
		startTransformProposal();

		if (useEdgeLengthInput.get()) {
			return lengthWeightedTreeProposal();
		} else {
            return edgeWeightsTreeProposal();
        }
    }

    boolean lineageCoversHeight(Node child, double height) {
        Node parent = child.getParent();
        return !child.isRoot() && child.getHeight() <= height && height <= parent.getHeight();
    }

    List<Integer> getLineagesAtHeight(Tree tree, double height) {
		List<Integer> lineages = new ArrayList<>();
		for (int k = 0; k < tree.getNodeCount(); k++)
			if (lineageCoversHeight(tree.getNode(k), height))
				lineages.add(k);
        return lineages;
    }

    void registerReject(String s) {
        if (!_rejects.containsKey(s)) 
            _rejects.put(s, 0);
        // Increment counter
        _rejects.put(s, _rejects.get(s) + 1);
    }

	private double edgeWeightsTreeProposal() {
        Tree tree = (Tree) InputUtil.get(treeInput, this);

        double logHastingsRatio = 0.0;

		SourceWeights sourceWeights = getSourceWeights(tree);
		int sourceIdx = sourceWeights.sampleIndex();
		if (sourceIdx < 0) {
			registerReject("sourceWeightBelowEPS");
			return Double.NEGATIVE_INFINITY;
		}

		int randomNode = sourceWeights.nodeNrs[sourceIdx];
		Node i = tree.getNode(randomNode);
		logHastingsRatio -= Math.log(sourceWeights.weights[sourceIdx] / sourceWeights.total);

		Node p = i.getParent();
		if (p.isRoot()) {
            registerReject("pIsRoot");
			return Double.NEGATIVE_INFINITY;
		}

        double toHeight = p.getHeight();

		List<Integer> coExistingNodes = getLineagesAtHeight(tree, toHeight);
		if (coExistingNodes.size() == 0) {
            registerReject("noCandidates");
			return Double.NEGATIVE_INFINITY;
		}

		final Node CiP = getOtherChild(p, i);

		// get all nodes in target that are ancestors to p
		List<Integer> ancestors = new ArrayList<>();
        ancestors.add(p.getNr());

		// calculate the consensus sequences without the node i
		edgeWeights.prestore();
		edgeWeights.updateByOperatorWithoutNode(i.getNr(), ancestors);

		// remove p as potential targets
		coExistingNodes.remove(coExistingNodes.indexOf(p.getNr()));
		try {
			coExistingNodes.remove(coExistingNodes.indexOf(i.getNr()));
		} catch (Exception e) {
            edgeWeights.reset();
			System.err.println("couldn't find node " + i.getNr() + " among coexisting nodes ");
			System.err.println(tree +";");
            registerReject("1?");
			return Double.NEGATIVE_INFINITY;
		}

		edgeWeights.setCurrentOperatorTag(getClass().getSimpleName());
		double[] rawTargetWeights = edgeWeights.getTargetWeightsInteger(i.getNr(), coExistingNodes, toHeight);
		// Transform the whole feasible vector first, then take the sibling's mass out of it:
		// the sibling is a genuine candidate for the reverse move, so it has to be present
		// while the offset is applied, and is only excluded from the forward draw afterwards.
		double[] distance = applyTargetTransform(rawTargetWeights);

		int siblingIdx = coExistingNodes.indexOf(CiP.getNr());
		assert siblingIdx >= 0 : "Sibling lineage should be present at the parent height";
		double siblingDistance = distance[siblingIdx];
		distance[siblingIdx] = 0;

		double totalDistance = 0;
		for (int k = 0; k < coExistingNodes.size(); k++) {
			totalDistance += distance[k];
        }

        if (totalDistance < WEIGHT_EPS) {
            // Risk of inaccurate Hastings ratio due to rounding errors
            edgeWeights.reset();
            registerReject("sumDistBelowEPS");
            return Double.NEGATIVE_INFINITY;
        }
		
		Node j = tree.getRoot();
		double scaler2 = Randomizer.nextDouble() * totalDistance;
		double currDist = 0;
		int nodeNr = -1;
		for (int k = 0; k < coExistingNodes.size(); k++) {
			currDist += distance[k];
			if (currDist > scaler2) {				
				nodeNr = k;
				j = treeInput.get().getNode(coExistingNodes.get(nodeNr));
				break;
			}
        }

		if ((totalDistance + siblingDistance - distance[nodeNr]) < WEIGHT_EPS) {
            // Risk of inaccurate Hastings ratio due to rounding errors
            edgeWeights.reset();
            registerReject("sumDistBackBelowEPS");
            return Double.NEGATIVE_INFINITY;
        }
		logHastingsRatio -= Math.log(distance[nodeNr] / totalDistance);
		logHastingsRatio += Math.log(siblingDistance / (totalDistance + siblingDistance - distance[nodeNr]));
      
		edgeWeights.reset();

		Node jP = j.getParent();

		// disallow moves that change the root.
		if (j.isRoot()) {
            registerReject("jIsRoot");
			return Double.NEGATIVE_INFINITY;
		}

		// final int pnr = p.getNr();
		// final int jPnr = jP.getNr();

        // if (j.getNr() == pnr) {
        //     registerReject("j.getNr()==pnr");
		// 	return Double.NEGATIVE_INFINITY;
        // }
        // if (jPnr == i.getNr()) {
        //     registerReject("jPnr==i.getNr()");
		// 	return Double.NEGATIVE_INFINITY;
		// }

		final Node PiP = p.getParent();

        if (j != CiP) {
            // remove p from PiP, add the other child of p to PiP, frees p
            replace(PiP, p, CiP);
            // add j as child to p, removing CiP as child of p
            replace(p, CiP, j);
            // then parent node of j to p
            replace(jP, j, p);
        }

		// mark paths to common ancestor as changed
		Node mrca = getMRCA(PiP, p);

		// mrca.makeDirty(3 - mrca.isDirty());
            // IS_CLEAN = 0, IS_DIRTY = 1, IS_FILTHY = 2;
            // IS_CLEAN -> 3 ??
            // IS_DIRTY -> 2 IS_FILTHY
            // IS_FILTHY -> 1 IS_DIRTY
		mrca.makeDirty(Tree.IS_FILTHY);
		tree.setEverythingDirty(true);

		edgeWeights.prestore();
		edgeWeights.updateByOperator();

		// Reverse source-selection probability, evaluated on the moved tree T'.
		SourceWeights reverseSourceWeights = getSourceWeights(tree);
		double reverseWeight = reverseSourceWeights.weightFor(randomNode);
		if (reverseSourceWeights.isDegenerate() || reverseWeight < WEIGHT_EPS) {
			registerReject("reverseEdgeWeightBelowEPS");
			return Double.NEGATIVE_INFINITY;
		}
		logHastingsRatio += Math.log(reverseWeight / reverseSourceWeights.total);

        return logHastingsRatio;
	}

	/**
	 * Transformed source weights over the feasible (non-root) nodes.
	 *
	 * <p>The root is left out of the vector entirely rather than given weight zero: a
	 * transform adds its offset to every entry it receives, so a zero-weight root would
	 * become selectable and break the {@code p = i.getParent()} that follows.
	 */
	private SourceWeights getSourceWeights(Tree tree) {
		return sourceWeights(tree,
				node -> !node.isRoot(),
				node -> edgeWeights.getEdgeWeights(node.getNr()));
	}

	/** Transformed source weights for the branch-length-weighted proposal. */
	private SourceWeights getLengthSourceWeights(Tree tree) {
		// A source edge whose parent is the root cannot be moved by this step, so it is left
		// out of the candidate set rather than drawn and then rejected.
		return sourceWeights(tree,
				node -> !(node.isRoot() || node.getParent().isRoot()),
				Node::getLength);
	}

	private double lengthWeightedTreeProposal() {
		Tree tree = (Tree) InputUtil.get(treeInput, this);

		double logHastingsRatio = 0.0;

		// choose a random node avoiding the root and its children
		SourceWeights sourceWeights = getLengthSourceWeights(tree);
		int sourceIdx = sourceWeights.sampleIndex();
		if (sourceIdx < 0) {
			registerReject("sourceWeightBelowEPS");
			return Double.NEGATIVE_INFINITY;
		}

		Node i = tree.getNode(sourceWeights.nodeNrs[sourceIdx]);
		logHastingsRatio -= Math.log(sourceWeights.weights[sourceIdx] / sourceWeights.total);

		Node p = i.getParent();
		if (p.isRoot()) {
			registerReject("pIsRoot");
			return Double.NEGATIVE_INFINITY;
		}

		// double minHeight = i.getHeight();
		double toHeight = p.getHeight();

		// List<Integer> coExistingNodes = getCoExistingLineages(p, tree).stream().map(n -> n.getNr()).collect(Collectors.toList());
		List<Integer> coExistingNodes = getLineagesAtHeight(tree, toHeight);
		if (coExistingNodes.size() == 0) {
			registerReject("noCandidates");
			return Double.NEGATIVE_INFINITY;
		}

		final Node CiP = getOtherChild(p, i);
        
		// calculate the consensus sequences without the node i
		edgeWeights.prestore();
		edgeWeights.updateByOperatorWithoutNode(i.getNr(), Arrays.asList(p.getNr()));

		// remove p as potential targets
		coExistingNodes.remove(coExistingNodes.indexOf(p.getNr()));
		try {
			coExistingNodes.remove(coExistingNodes.indexOf(i.getNr()));
		} catch (Exception e) {
			System.err.println("couldn't find node " + i.getNr() + " among coexisting nodes ");
			System.err.println(tree + ";");
            // edgeWeights.reset();
			// registerReject("1?");
			// return Double.NEGATIVE_INFINITY;
            throw new RuntimeException("Couldn't find current node among coexisting nodes.");
		}

		edgeWeights.setCurrentOperatorTag(getClass().getSimpleName());
		double[] rawTargetWeights = edgeWeights.getTargetWeightsInteger(i.getNr(), coExistingNodes, toHeight);
		// Transform the whole feasible vector first, then take the sibling's mass out of it;
		// see the comment on the same step in edgeWeightsTreeProposal.
		double[] distance = applyTargetTransform(rawTargetWeights);

		double siblingDistance = distance[coExistingNodes.indexOf(CiP.getNr())];
		distance[coExistingNodes.indexOf(CiP.getNr())] = 0;

		double totalDistance = 0;
		for (int k = 0; k < coExistingNodes.size(); k++) {
			totalDistance += distance[k];
		}

		if (totalDistance < 1E-15) {
			// Risk of inaccurate Hastings ratio due to rounding errors
            edgeWeights.reset();
			registerReject("sumDistBelowEPS");
			return Double.NEGATIVE_INFINITY;
		}

		Node j = tree.getRoot();
		double scaler2 = Randomizer.nextDouble() * totalDistance;
		double currDist = 0;
		int nodeNr = -1;
		for (int k = 0; k < coExistingNodes.size(); k++) {
			currDist += distance[k];
			if (currDist > scaler2) {
				nodeNr = k;
				j = treeInput.get().getNode(coExistingNodes.get(nodeNr));
				break;
			}
		}

		if ((totalDistance + siblingDistance - distance[nodeNr]) < 1E-15) {
			// Risk of inaccurate Hastings ratio due to rounding errors
            edgeWeights.reset();
			registerReject("sumDistBackBelowEPS");
			return Double.NEGATIVE_INFINITY;
		}
		logHastingsRatio -= Math.log(distance[nodeNr] / totalDistance);
		logHastingsRatio += Math.log(siblingDistance / (totalDistance + siblingDistance - distance[nodeNr]));

		edgeWeights.reset();

		Node jP = j.getParent();

		// disallow moves that change the root.
		if (j.isRoot()) {
			registerReject("jIsRoot");
			return Double.NEGATIVE_INFINITY;
		}

		final int pnr = p.getNr();
		final int jPnr = jP.getNr();

		if (j.getNr() == pnr) {
			registerReject("j.getNr()==pnr");
			return Double.NEGATIVE_INFINITY;
		}
		if (jPnr == i.getNr()) {
			registerReject("jPnr==i.getNr()");
			return Double.NEGATIVE_INFINITY;
		}

		final Node PiP = p.getParent();

		if (j != CiP) {
			// remove p from PiP, add the other child of p to PiP, frees p
			replace(PiP, p, CiP);
			// add j as child to p, removing CiP as child of p
			replace(p, CiP, j);
			// then parent node of j to p
			replace(jP, j, p);
		}

		// // mark paths to common ancestor as changed
		// Node mrca = getMRCA(PiP, p);
		// mrca.makeDirty(Tree.IS_FILTHY);

		// Reverse source-selection probability, evaluated on the moved tree T'. With a
		// transform the total is a nonlinear function of the whole vector, so it is rebuilt
		// rather than adjusted term by term.
		SourceWeights reverseSourceWeights = getLengthSourceWeights(tree);
		double reverseWeight = reverseSourceWeights.weightFor(i.getNr());
		if (reverseSourceWeights.isDegenerate() || reverseWeight < WEIGHT_EPS) {
			registerReject("reverseEdgeWeightBelowEPS");
			return Double.NEGATIVE_INFINITY;
		}
		logHastingsRatio += Math.log(reverseWeight / reverseSourceWeights.total);

        return logHastingsRatio;
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
