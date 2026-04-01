
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
	
    public Input<Boolean> useRepeatedMutationsInput = new Input<>("useRepeatedMutations", "flag to indicate if repeated mutations should be used as weights for operations", false);
    
    public Input<Boolean> useEdgeLengthInput = new Input<>("useEdgeLength", "if true, it uses the relative edge lengths for operations", false);

    
    double limit;
    
    EdgeWeights edgeWeights;
    
    
    @Override
    public void initAndValidate() {
		limit = mutationLimitInput.get();
		edgeWeights = edgeWeightsInput.get();
    }

    
    @Override
    public double proposal() {
		if (useEdgeLengthInput.get()) {
			return LengthWeightedTreeProposal();
		} else {
			return EdgeWeightsTreeProposal();
		}
    }

	/**
	 * WARNING: Assumes strictly bifurcating beast.tree.
	 */
	/**
	 * override this for proposals,
	 *
	 * @return log of Hastings Ratio, or Double.NEGATIVE_INFINITY if proposal should
	 *         not be accepted *
	 */
	private double EdgeWeightsTreeProposal() {
        Tree tree = (Tree) InputUtil.get(treeInput, this);

        double logHastingsRatio = 0.0;

        // choose a random node avoiding root
        double totalMutations = 0;
    	for (int i = 0; i < tree.getNodeCount(); i++) {
			if (tree.getNode(i).isRoot())
				continue;			
			totalMutations += edgeWeights.getEdgeWeights(i);		
    	}
        if (totalMutations < 1E-12) {
            return Double.NEGATIVE_INFINITY;
        }

        double scaler = Randomizer.nextDouble() * totalMutations;
        int randomNode = -1;
        double currMuts = 0;
        for (int i = 0; i < tree.getNodeCount(); i++) {
			if (tree.getNode(i).isRoot())
				continue;	
        	currMuts +=  edgeWeights.getEdgeWeights(i);
			if (currMuts > scaler) {
				randomNode = i;
				break;
			}
        }
        assert randomNode > -1;
                
        // logHastingsRatio -= Math.log(edgeWeights.getEdgeWeights(randomNode) / totalMutations);

		Node i = tree.getNode(randomNode);
		
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

		// remove p as potential targets
		candidateNodes.remove(p);
		try {
			candidateNodes.remove(i);
		} catch (Exception e) {
			System.err.println("couldn't find node " + i.getNr() + " among coexisting nodes ");
			System.err.println(tree +";");
			return Double.NEGATIVE_INFINITY;
		}

        // candidateNodes.remove(candidateNodes.indexOf(CiP.getNr()));

        List<Double> candidateAges = new ArrayList<>();
        for (Node c : candidateNodes) {
            double newMinAge = Math.max(i.getHeight(), c.getHeight());
            double newRange = c.getParent().getHeight() - newMinAge;
            double newAge = newMinAge + (Randomizer.nextDouble() * newRange);
            candidateAges.add(newAge);
        }

		double[] distance = edgeWeights.getTargetWeights(i.getNr(), candidateNodes, candidateAges);
		
        int siblingIdx = candidateNodes.indexOf(CiP);
		double siblingDistance = distance[siblingIdx];
		siblingDistance = Math.pow(siblingDistance, 2);
		distance[siblingIdx] = 0;
		

		double totalDistance = 0;
		for (int k = 0; k < candidateNodes.size(); k++) {
			distance[k] = Math.pow(distance[k], 2);
			totalDistance += distance[k];				
		}

        if (totalDistance < 1E-12) {
            // Risk of inaccurate Hastings ratio due to rounding errors
            return Double.NEGATIVE_INFINITY;
        }
		
		Node j = tree.getRoot();
		double scaler2 = Randomizer.nextDouble() * totalDistance;
		double currDist = 0;
		int nodeNr = -1;
		for (int k = 0; k < candidateNodes.size(); k++) {
			currDist += distance[k];
			if (currDist > scaler2) {				
				nodeNr = k;
				break;
			}
		}
        j = candidateNodes.get(nodeNr);
        double newAge = candidateAges.get(nodeNr);

        if ((totalDistance + siblingDistance - distance[nodeNr]) < 1E-12) {
            // Risk of inaccurate Hastings ratio due to rounding errors
            return Double.NEGATIVE_INFINITY;
        }
		
		logHastingsRatio -= Math.log(distance[nodeNr] / totalDistance);
		logHastingsRatio += Math.log(siblingDistance / (totalDistance + siblingDistance - distance[nodeNr]));	      
      
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

		double newMinAge = Math.max(i.getHeight(), j.getHeight());
		double newRange = jP.getHeight() - newMinAge;
		double oldMinAge = Math.max(i.getHeight(), CiP.getHeight());
		double oldRange = PiP.getHeight() - oldMinAge;
		logHastingsRatio += Math.log(newRange / Math.abs(oldRange));
		
		p.setHeight(newAge);

		// remove p from PiP, add the other child of p to PiP, frees p
		replace(PiP, p, CiP);
		// add j as child to p, removing CiP as child of p
		replace(p, CiP, j);
		// then parent node of j to p
		replace(jP, j, p);

		// mark paths to common ancestor as changed
		Node mrca = getMRCA(PiP, p);
		mrca.makeDirty(3 - mrca.isDirty());
		
        edgeWeights.prestore();
        edgeWeights.updateByOperator();
        // recalculate hasting ratio
        // calculate tree length
        totalMutations = 0;
    	for (int k = 0; k < tree.getNodeCount(); k++) {
			if (tree.getNode(k).isRoot())
				continue;			
			totalMutations += edgeWeights.getEdgeWeights(k);	
    	}
    	
        // logHastingsRatio += Math.log(edgeWeights.getEdgeWeights(randomNode)/ totalMutations);
    	
        return logHastingsRatio;
	}

	
	private double LengthWeightedTreeProposal() {
        Tree tree = (Tree) InputUtil.get(treeInput, this);

        double logHastingsRatio = 0.0;

        // choose a random node avoiding root
        double totalLength = 0;
    	for (int i = 0; i < tree.getNodeCount(); i++) {
    		totalLength += tree.getNode(i).getLength();		
    	}
        double scaler = Randomizer.nextDouble() * totalLength;
        int randomNode = -1;
        double currLength = 0;
        for (int i = 0; i < tree.getNodeCount(); i++) {
        	currLength += tree.getNode(i).getLength();
			if (currLength > scaler) {
				randomNode = i;
				break;
			}
        }
                
        logHastingsRatio -= Math.log( tree.getNode(randomNode).getLength() / totalLength);
        totalLength -= tree.getNode(randomNode).getLength();
		
		Node i = tree.getNode(randomNode);
		
		Node p = i.getParent();
		if (p.isRoot()) {
			return Double.NEGATIVE_INFINITY;
		}

		double minHeight = i.getHeight();

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

		// remove p as potential targets
		candidateNodes.remove(p);
		try {
			candidateNodes.remove(i);
		} catch (Exception e) {
			System.err.println("couldn't find node " + i.getNr() + " among coexisting nodes ");
			System.err.println(tree +";");
			return Double.NEGATIVE_INFINITY;
		}
//		coExistingNodes.remove(coExistingNodes.indexOf(CiP.getNr()));
        List<Double> candidateAges = new ArrayList<>();
        for (Node c : candidateNodes) {
            double newMinAge = Math.max(i.getHeight(), c.getHeight());
            double newRange = c.getParent().getHeight() - newMinAge;
            double newAge = newMinAge + (Randomizer.nextDouble() * newRange);
            candidateAges.add(newAge);
        }

		double[] distance = edgeWeights.getTargetWeights(i.getNr(), candidateNodes, candidateAges);
		
        int siblingIdx = candidateNodes.indexOf(CiP);
		double siblingDistance = distance[siblingIdx];
		siblingDistance = Math.pow(siblingDistance, 2);
		distance[siblingIdx] = 0;
		
		
		double totalDistance = 0;
		for (int k = 0; k < candidateNodes.size(); k++) {
			distance[k] = Math.pow(distance[k], 2);
			totalDistance += distance[k];				
		}

        if (totalDistance < 1E-12) {
            // Risk of inaccurate Hastings ratio due to rounding errors
            return Double.NEGATIVE_INFINITY;
        }
		
		Node j = tree.getRoot();
		double scaler2 = Randomizer.nextDouble() * totalDistance;
		double currDist = 0;
		int nodeNr = -1;
		for (int k = 0; k < candidateNodes.size(); k++) {
			currDist += distance[k];
			if (currDist > scaler2) {				
				nodeNr = k;
				break;
			}
		}
        j = candidateNodes.get(nodeNr);
        double newAge = candidateAges.get(nodeNr);

        if ((totalDistance + siblingDistance - distance[nodeNr]) < 1E-12) {
            // Risk of inaccurate Hastings ratio due to rounding errors
            return Double.NEGATIVE_INFINITY;
        }
		
		logHastingsRatio -= Math.log(distance[nodeNr] / totalDistance);
		logHastingsRatio += Math.log(siblingDistance / (totalDistance + siblingDistance - distance[nodeNr]));	      
      
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

		double newMinAge = Math.max(i.getHeight(), j.getHeight());
		double newRange = jP.getHeight() - newMinAge;
		double oldMinAge = Math.max(i.getHeight(), CiP.getHeight());
		double oldRange = PiP.getHeight() - oldMinAge;
		logHastingsRatio += Math.log(newRange / Math.abs(oldRange));
		
		p.setHeight(newAge);

		// remove p from PiP, add the other child of p to PiP, frees p
		replace(PiP, p, CiP);
		// add j as child to p, removing CiP as child of p
		replace(p, CiP, j);
		// then parent node of j to p
		replace(jP, j, p);

		// mark paths to common ancestor as changed
		Node mrca = getMRCA(PiP, p);
		mrca.makeDirty(3 - mrca.isDirty());
		
		totalLength += i.getLength();
    	
    	logHastingsRatio += Math.log(i.getLength()/ totalLength);
    	
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
