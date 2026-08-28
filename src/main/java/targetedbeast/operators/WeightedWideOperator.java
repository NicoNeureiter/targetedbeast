
package targetedbeast.operators;

import java.util.ArrayList;
import java.util.List;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.util.InputUtil;
import beast.base.util.Randomizer;
import targetedbeast.edgeweights.EdgeWeights;

@Description("Picks a node, looks for all coexisting lineages and then performs a weighted move")
public class WeightedWideOperator extends TargetedWeightOperator {
	
    public Input<EdgeWeights> edgeWeightsInput = new Input<>("edgeWeights", "input of weights to be used for targetedn tree operations");

    public Input<Double> mutationLimitInput = new Input<>("mutationLimit", "Input of the number of mutations to be used as a limit", 15.0);

    public double limit;
    
    EdgeWeights edgeWeights;
    
    @Override
    public void initAndValidate() {
		limit = mutationLimitInput.get();
		edgeWeights = edgeWeightsInput.get();

		if (edgeWeights == null) {
			// The unweighted proposal reads no weights at all, so there is nothing to reshape.
			if (targetWeightTransformInput.get() != null || edgeWeightTransformInput.get() != null) {
				throw new IllegalArgumentException(getID() + ": weight transforms require "
						+ "edgeWeights; without it this operator proposes uniformly.");
			}
			return;
		}

		// Target weights have always been squared here, so that stays the no-transform default.
		initWeightTransforms(edgeWeights, true);
    }

    @Override
    public double proposal() {
		if (edgeWeights != null) {
			// Freeze the transform coefficients, so the forward and reverse probabilities
			// below are computed with identical weights.
			startTransformProposal();
			return weightedProposal();
		}else {
			return unweightedProposal();
		}
    }


    /**
     * WARNING: Assumes strictly bifurcating beast.tree.
     */
    /**
     * override this for proposals,
     *
     * @return log of Hastings Ratio, or Double.NEGATIVE_INFINITY if proposal should not be accepted *
     */
    private double weightedProposal() {
        Tree tree = (Tree) InputUtil.get(treeInput, this);

        double logHastingsRatio = 0.0;
//        System.out.println(tree + ";");
//        System.out.println(" ");
//        System.out.println(((ConsensusWeights) edgeWeights).getTree() + ";");

        // choose a random node avoiding root
        SourceWeights sourceWeights = getSourceWeights(tree);
        int sourceIdx = sourceWeights.sampleIndex();
        if (sourceIdx < 0) {
            return Double.NEGATIVE_INFINITY;
        }

        int randomNode = sourceWeights.nodeNrs[sourceIdx];
        logHastingsRatio -= Math.log(sourceWeights.weights[sourceIdx] / sourceWeights.total);

        Node i = tree.getNode(randomNode);
        Node p = i.getParent();
        // choose another random node to insert i above
        Node j = tree.getRoot();
        Node jP;
        final Node CiP = getOtherChild(p, i);

        
		if (p.isRoot()) {
			return Double.NEGATIVE_INFINITY;
		}		
        
        double currHeight = p.getHeight();
        
        List<Integer> coExistingNodes = new ArrayList<>();
		for (int k = 0; k < tree.getNodeCount(); k++) {
			if (tree.getNode(k).isRoot())
				continue;
			
			if (tree.getNode(k).getHeight() < currHeight && tree.getNode(k).getParent().getHeight() > currHeight)
				coExistingNodes.add(k);
		}
		if (coExistingNodes.size() == 0)
			return Double.NEGATIVE_INFINITY;
		
		// add the current node to the list
		coExistingNodes.add(CiP.getNr());

        List<Integer> ancestors = new ArrayList<>();
        Node ancestor = p;
        while (ancestor != null) {
            if (coExistingNodes.contains(ancestor.getNr())) {
                ancestors.add(ancestor.getNr());
            }
            ancestor = ancestor.getParent();
        }

        edgeWeights.prestore();
        edgeWeights.updateByOperatorWithoutNode(i.getNr(), ancestors);

		
        List<Double> toNodeHeights = coExistingNodes.stream()
                .map(nr -> tree.getNode(nr).getParent().getHeight())
                .toList();
        double[] rawTargetWeights = edgeWeights.getTargetWeightsInteger(i.getNr(), coExistingNodes, toNodeHeights);
        // Transform the whole feasible vector, CiP included: it is a genuine candidate for the
        // reverse move, so it has to be present while the offset is applied, and is only
        // excluded from the forward draw afterwards.
        double[] distance = applyTargetTransform(rawTargetWeights);
        int cipIdx = coExistingNodes.indexOf(CiP.getNr());
        double cipMass;
		double totalDistance = 0;
		for (int k = 0; k < coExistingNodes.size(); k++) {
			totalDistance += distance[k];
		}
        // Exclude CiP from forward selection: proposing to stay at the current
        // sibling would leave the tree unchanged but cause jPnr == pnr → -Inf.
        // Keep cipMass for the MH backward correction (probability of returning).
        cipMass = distance[cipIdx];
        totalDistance -= cipMass;
        if (totalDistance < 1e-12) {
            edgeWeights.reset();
            return Double.NEGATIVE_INFINITY;
        }

		double scaler2 = Randomizer.nextDouble() * totalDistance;
		double currDist = 0;
		int nodeNr = -1;
		for (int k = 0; k < coExistingNodes.size(); k++) {
            if (k == cipIdx) continue;
			currDist += distance[k];
			if (currDist > scaler2) {
				nodeNr = k;
				j = treeInput.get().getNode(coExistingNodes.get(nodeNr));
				break;
			}
		}
        if (nodeNr < 0) {
            edgeWeights.reset();
            return Double.NEGATIVE_INFINITY;
        }

		// calculate HR contribution
		logHastingsRatio -= Math.log(distance[nodeNr] / totalDistance);
        // Backward: probability of selecting CiP (original sibling) in the reverse move.
        // The scores are symmetric (target weights are computed on the tree with i removed,
        // which is identical in both directions), but the candidate set is not: the forward
        // set contains j and excludes CiP, the backward set contains CiP and excludes j.
        // So the backward normaliser swaps j's mass for CiP's mass.
        double backwardTotal = totalDistance - distance[nodeNr] + cipMass;
		logHastingsRatio += Math.log(cipMass / backwardTotal);

        edgeWeights.reset();
		
				
		jP = j.getParent();
    
        // disallow moves that change the root.
        if (j.isRoot()) {
            return Double.NEGATIVE_INFINITY;
        }

        assert jP != null;  // j != root tested above
        final int pnr = p.getNr();
        final int jPnr = jP.getNr();
        
        if ( jPnr == pnr || j.getNr() == pnr || jPnr == i.getNr())
            return Double.NEGATIVE_INFINITY;

        final Node PiP = p.getParent();

        // disconnect p
        // remove p from PiP, add the other child of p to PiP, frees p
        replace(PiP, p, CiP);
        // add j as child to p, removing CiP as child of p
        replace(p, CiP, j);
        // then parent node of j to p
        replace(jP, j, p);

        // mark paths to common ancestor as changed
        Node iup = PiP;
        Node jup = p;
        while (iup != jup) {
            if( iup.getHeight() < jup.getHeight() ) {
                assert !iup.isRoot();
                iup = iup.getParent();
            } else {
                assert !jup.isRoot();
                jup = jup.getParent();
            }
        }
        jup.makeDirty(3-jup.isDirty());
                
        edgeWeights.prestore();
        edgeWeights.updateByOperator();

        // Reverse source-selection probability, evaluated on the moved tree T'. With a
        // transform the total is a nonlinear function of the whole vector, so it is rebuilt
        // rather than adjusted term by term.
        SourceWeights reverseSourceWeights = getSourceWeights(tree);
        double reverseWeight = reverseSourceWeights.weightFor(randomNode);
        if (reverseSourceWeights.isDegenerate() || !(reverseWeight > 0.0)) {
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

    private double unweightedProposal() {
        Tree tree = (Tree) InputUtil.get(treeInput, this);

        double logHastingsRatio = 0.0;

        // choose a random node avoiding root
        double scaler = Randomizer.nextDouble() * tree.getNodeCount();
        int randomNode = -1;
        double currMuts = 0;
        for (int i = 0; i < tree.getNodeCount(); i++) {
        	currMuts ++;
			if (currMuts > scaler) {
				randomNode = i;
				break;
			}
        }
        
        Node i = tree.getNode(randomNode);
        
        if (i.isRoot())
            return Double.NEGATIVE_INFINITY;
        
        Node p = i.getParent();
        // choose another random node to insert i above
        Node j = tree.getRoot();
        Node jP;
        final Node CiP = getOtherChild(p, i);

        
		if (p.isRoot()) {
			return Double.NEGATIVE_INFINITY;
		}		
        
        double currHeight = p.getHeight();
        
        List<Integer> coExistingNodes = new ArrayList<>();
		for (int k = 0; k < tree.getNodeCount(); k++) {
			if (tree.getNode(k).isRoot())
				continue;
			
			if (tree.getNode(k).getHeight() < currHeight && tree.getNode(k).getParent().getHeight() > currHeight)
				coExistingNodes.add(k);
		}
		if (coExistingNodes.size() == 0)
			return Double.NEGATIVE_INFINITY;
		
		// add the current node to the list
		
		double scaler2 = Randomizer.nextDouble() * coExistingNodes.size();
		double currDist = 0;
		int nodeNr = -1;
		for (int k = 0; k < coExistingNodes.size(); k++) {
			currDist++;
			if (currDist > scaler2) {				
				nodeNr = k;
				j = treeInput.get().getNode(coExistingNodes.get(nodeNr));
				break;
			}
		}
				
		jP = j.getParent();
    
        // disallow moves that change the root.
        if (j.isRoot()) {
            return Double.NEGATIVE_INFINITY;
        }

        assert jP != null;  // j != root tested above
        final int pnr = p.getNr();
        final int jPnr = jP.getNr();
        
        if ( jPnr == pnr || j.getNr() == pnr || jPnr == i.getNr())
            return Double.NEGATIVE_INFINITY;

        final Node PiP = p.getParent();

        // disconnect p
        // remove p from PiP, add the other child of p to PiP, frees p
        replace(PiP, p, CiP);
        // add j as child to p, removing CiP as child of p
        replace(p, CiP, j);
        // then parent node of j to p
        replace(jP, j, p);

        // mark paths to common ancestor as changed
        Node iup = PiP;
        Node jup = p;
        while (iup != jup) {
            if( iup.getHeight() < jup.getHeight() ) {
                assert !iup.isRoot();
                iup = iup.getParent();
            } else {
                assert !jup.isRoot();
                jup = jup.getParent();
            }
        }
        jup.makeDirty(3-jup.isDirty());
                
    	return logHastingsRatio;
    }


} // class WilsonBalding
