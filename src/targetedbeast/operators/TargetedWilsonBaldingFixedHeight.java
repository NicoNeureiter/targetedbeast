
package targetedbeast.operators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
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
public class TargetedWilsonBaldingFixedHeight extends TreeOperator {

	private static final double WEIGHT_EPS = 1E-16;

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
    }
    
    @Override
    public double proposal() {
        // if (Randomizer.nextDouble() < 0.0001)
        //     System.out.println(_rejects);

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

		double totalMutations = 0.0;
		for (int candidateNr = 0; candidateNr < tree.getNodeCount(); candidateNr++) {
			if (tree.getNode(candidateNr).isRoot()) { //  || tree.getNode(candidateNr).getParent().isRoot()) {
				continue;
			}
			totalMutations += edgeWeights.getEdgeWeights(candidateNr);
		}
		if (totalMutations < WEIGHT_EPS) {
			registerReject("sourceWeightBelowEPS");
			return Double.NEGATIVE_INFINITY;
		}

		double scaler = Randomizer.nextDouble() * totalMutations;
		int randomNode = -1;
		double currMuts = 0.0;
		for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
			if (tree.getNode(nodeNr).isRoot()) { // || tree.getNode(nodeNr).getParent().isRoot()) {
				continue;
			}
			currMuts += edgeWeights.getEdgeWeights(nodeNr);
			if (currMuts > scaler) {
				randomNode = nodeNr;
				break;
			}
		}
		if (randomNode < 0) {
			registerReject("noSourceChosen");
			return Double.NEGATIVE_INFINITY;
		}

		Node i = tree.getNode(randomNode);
		// logHastingsRatio -= Math.log(edgeWeights.getEdgeWeights(randomNode) / totalMutations);
		
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

		double[] distance = edgeWeights.getTargetWeightsInteger(i.getNr(), coExistingNodes, toHeight);
		
		int siblingIdx = coExistingNodes.indexOf(CiP.getNr());
		assert siblingIdx >= 0 : "Sibling lineage should be present at the parent height";
		double siblingDistance = distance[siblingIdx];
		siblingDistance = Math.pow(siblingDistance, 2);
		distance[siblingIdx] = 0;
		
		double totalDistance = 0;
		for (int k = 0; k < coExistingNodes.size(); k++) {
			distance[k] = Math.pow(distance[k], 2);
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

		// totalMutations = 0.0;
		// for (int reverseNodeNr = 0; reverseNodeNr < tree.getNodeCount(); reverseNodeNr++) {
		// 	if (tree.getNode(reverseNodeNr).isRoot() || tree.getNode(reverseNodeNr).getParent().isRoot()) {
		// 		continue;
		// 	}
		// 	totalMutations += edgeWeights.getEdgeWeights(reverseNodeNr);
		// }
		// // if (totalMutations < WEIGHT_EPS) {
		// // 	registerReject("reverseEdgeWeightBelowEPS");
		// // 	return Double.NEGATIVE_INFINITY;
		// // }

		// double reverseWeight = edgeWeights.getEdgeWeights(randomNode);

		// // if (reverseWeight < WEIGHT_EPS) {
		// // 	registerReject("reverseEdgeWeightBelowEPS");
		// // 	return Double.NEGATIVE_INFINITY;
		// // }

		// logHastingsRatio += Math.log(reverseWeight / totalMutations);
    	
        return logHastingsRatio;
	}

	private double lengthWeightedTreeProposal() {
		Tree tree = (Tree) InputUtil.get(treeInput, this);

		double logHastingsRatio = 0.0;

		// choose a random node avoiding root
		double totalLength = 0;
		List<Node> fromCandidates = Arrays.stream(tree.getNodesAsArray())
				.filter(n -> !(n.isRoot() || n.getParent().isRoot()))
				.toList();

		for (Node n : fromCandidates) {
			totalLength += n.getLength();
		}
		double scaler = Randomizer.nextDouble() * totalLength;
		Node i = null;
		double currLength = 0;
		for (Node n : fromCandidates) {
			currLength += n.getLength();
			if (currLength > scaler) {
				i = n;
				break;
			}
		}

		logHastingsRatio -= Math.log(i.getLength() / totalLength);
		totalLength -= i.getLength();

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

		double[] distance = edgeWeights.getTargetWeightsInteger(i.getNr(), coExistingNodes, toHeight);

		double siblingDistance = distance[coExistingNodes.indexOf(CiP.getNr())];
		siblingDistance = Math.pow(siblingDistance, 2);
		distance[coExistingNodes.indexOf(CiP.getNr())] = 0;

		double totalDistance = 0;
		for (int k = 0; k < coExistingNodes.size(); k++) {
			distance[k] = Math.pow(distance[k], 2);
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

		double reverseTotalLength = 0.0;
		for (Node candidate : Arrays.stream(tree.getNodesAsArray())
				.filter(n -> !(n.isRoot() || n.getParent().isRoot()))
				.toList()) {
			reverseTotalLength += candidate.getLength();
		}
		logHastingsRatio += Math.log(i.getLength() / reverseTotalLength);
    	
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
