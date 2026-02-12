package targetedbeast.edgeweights;

import java.util.*;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.branchratemodel.BranchRateModel;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.sitemodel.SiteModelInterface;
import beast.base.evolution.substitutionmodel.SubstitutionModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Distribution;
import beast.base.inference.State;
import targetedbeast.likelihood.SlowBeerLikelihoodCore;
import targetedbeast.likelihood.SlowTreeLikelihood;
import targetedbeast.util.LinearAlgebra;

@Description("Uses Felsenstein partial likelihoods from the tree likelihood calculation as edge weights. "
		+ "The partials at each node represent P(data below | state at node) and can be used to compute "
		+ "distances between nodes that reflect their similarity in terms of the underlying data.")
public class FelsensteinWeights extends Distribution implements EdgeWeights {
	
    final public Input<Alignment> dataInput = new Input<>("data", "sequence data for the beast.tree", Validate.REQUIRED);
    final public Input<Tree> treeInput = new Input<>("tree", "phylogenetic beast.tree with sequence data in the leafs", Validate.REQUIRED);
    final public Input<Double> maxWeightInput = new Input<>("maxWeight", "maximum weight for an edge", 10.0);
    final public Input<Double> minWeightInput = new Input<>("minWeight", "minimum weight for an edge", 0.1);
    final public Input<SlowTreeLikelihood> likelihoodInput = new Input<>("likelihood", "The tree likelihood used for calculating edge weights.", Validate.REQUIRED);

    protected SlowTreeLikelihood treelikelihood;
    protected SlowBeerLikelihoodCore likelihoodCore;
	
	// Leaf partials - stored separately since LikelihoodCore stores states, not partials, for leaves
	private double[][] leafPartials;

	// Partials computed for the "without node" case
	// Dimensions: [nodeCount][partialsSize] - only allocated for nodes that need it
	private double[][] partialsWithoutNode;

	// "Outside" partials: P(data NOT below node i | state at i)
	// Combined with regular partials gives P(all data | state at i)
	// Dimensions: [nodeCount][partialsSize]
	private double[][] outsidePartials;
	
	// Track which node was ignored when computing partialsWithoutNode
	private int currentIgnoredNode = -1;
	
	// For every leaf node, generate a sequence number
	// Leaf nodes with the same sequence have the same number
	private int[] sequenceID;
	private Alignment alignment;

	private boolean initialized = false;

	int stateCount;
	int patternCount;	
	int nodeCount;
	int leafNodeCount;
	int partialsSize;
	int matrixCount;
	int matrixSize; // stateCount * stateCount per rate category
	
	double maxWeight;
	double minWeight;

    // Arrays fields to avoid reallocating memory
    double[] _fromMatrix;
    double[] _toMatrix;
    double[] _probabilities;

	private static final double EPS = 1E-12;

	@Override
	public void initAndValidate() {
        treelikelihood = likelihoodInput.get();
		alignment = treelikelihood.dataInput.get();
		if (alignment == null) {
			throw new IllegalArgumentException("TreeLikelihood has no alignment (dataInput is null)");
		}
		if (dataInput.get() != alignment) {
			throw new IllegalArgumentException(
				"FelsensteinWeights requires its data input to be the same alignment object as used by the provided TreeLikelihood. " +
				"Use the same <data> in both. Got data='" + dataInput.get().getID() + "' but likelihood data='" + alignment.getID() + "'."
			);
		}

		if (alignment.getTaxonCount() != treeInput.get().getLeafNodeCount()) {
			throw new IllegalArgumentException(String.format(
					"Number of leaves in the tree (%d) does not match number of sequences (%d)",
					treeInput.get().getLeafNodeCount(), alignment.getTaxonCount()
            ));
		}

		stateCount = alignment.getMaxStateCount();
		patternCount = alignment.getPatternCount();
		nodeCount = treeInput.get().getNodeCount();
		leafNodeCount = treeInput.get().getLeafNodeCount();
		
		maxWeight = maxWeightInput.get();
		minWeight = minWeightInput.get();
		
		partialsWithoutNode = new double[nodeCount][];
		leafPartials = new double[leafNodeCount][];
		outsidePartials = new double[nodeCount][];
		
		initSequenceID();
	}
	
	/**
	 * Lazy initialization of likelihood core and leaf partials.
	 * Must be called lazily because TreeLikelihood may not be fully initialized during initAndValidate().
	 */
	public void ensureInitialized() {
		if (initialized) return;
		
        // Ensure likelihood core is SlowBeerLikelihoodCore
        if (treelikelihood.getLikelihoodCore() instanceof SlowBeerLikelihoodCore beerliCore) {
		    likelihoodCore = beerliCore;
        } else {
            throw new IllegalArgumentException(
                "FelsensteinWeights requires TreeLikelihood with BeerLikelihoodCore; got " +
                (treelikelihood.getLikelihoodCore() == null ? "null" : treelikelihood.getLikelihoodCore().getClass().getName())
            );
        }

		partialsSize = likelihoodCore.getPartialsSize();
		if (partialsSize <= 0) {
			// Ensure core has been initialized.
			treelikelihood.calculateLogP();
			partialsSize = likelihoodCore.getPartialsSize();
		}
        if (partialsSize <= 0)
            throw new IllegalStateException("Likelihood core has invalid partials size: " + partialsSize);

		// Require a SiteModel.Base to obtain category count; do not derive
		matrixCount = getSiteModel().getCategoryCount();
		if (matrixCount <= 0) {
			throw new IllegalStateException("SiteModel.getCategoryCount() returned " + matrixCount + ".");
		}

		// Derive patternCount from partialsSize to match what the core was actually initialized with
		// (the TreeLikelihood may use a filtered alignment with fewer patterns than dataInput)
		if (partialsSize <= 0 || partialsSize % (stateCount * matrixCount) != 0) {
			throw new IllegalStateException(
				"Likelihood core partialsSize (" + partialsSize + ") is not divisible by stateCount*matrixCount (" +
				stateCount + "*" + matrixCount + "=" + (stateCount * matrixCount) + "). Core may not be initialized."
			);
		}
		patternCount = partialsSize / (stateCount * matrixCount);
		matrixSize = (stateCount) * (stateCount);
		
		// Initialize leaf partials from sequence data
		TreeInterface tree = treeInput.get();
		for (int i = 0; i < leafNodeCount; i++) {
			leafPartials[i] = new double[partialsSize];
			initializeLeafPartials(tree.getNode(i), leafPartials[i]);
		}

        if (_probabilities == null)
            _probabilities = new double[matrixSize];
        if (_fromMatrix == null)
            _fromMatrix = new double[matrixCount * matrixSize];
        if (_toMatrix == null)
            _toMatrix = new double[matrixCount * matrixSize];
		
        assert partialsSize == patternCount * stateCount * matrixCount;

		initialized = true;
	}

	private void initSequenceID() {
		TreeInterface tree = treeInput.get();
		Alignment data = alignment;
		
		sequenceID = new int[tree.getNodeCount()];
		Map<String,Integer> sequenceMap = new HashMap<>();
		int k = 0;
		for (int i = 0; i < tree.getLeafNodeCount(); i++) {
			String taxon = tree.getNode(i).getID();
			String seq = data.getSequenceAsString(taxon); 
			if (!sequenceMap.containsKey(seq)) {
				sequenceMap.put(seq, k++);
			}
			sequenceID[i] = sequenceMap.get(seq);
		}
		for (int i = tree.getLeafNodeCount(); i < tree.getNodeCount(); i++) {
			sequenceID[i] = i;			
		}
	}
	
	/**
	 * Initialize partials for a leaf node based on sequence data.
	 */
	private void initializeLeafPartials(Node n, double[] partials) {
		Alignment data = alignment;
		int taxonIndex = data.getTaxonIndex(n.getID());
		if (taxonIndex < 0) {
			throw new IllegalArgumentException(
				"Leaf node '" + n.getID() + "' not found in alignment taxa. Available taxa: " + String.join(", ", data.getTaxaNames())
			);
		}
		
		for (int pattern = 0; pattern < patternCount; pattern++) {
			int state = data.getPattern(taxonIndex, pattern);
			
			for (int category = 0; category < matrixCount; category++) {
                int offset = getPartialOffset(pattern, category);
				
				if (state < stateCount) {
					// Unambiguous state
					for (int s = 0; s < stateCount; s++) {
						partials[offset + s] = (s == state) ? 1.0 : 0.0;
					}
				} else {
					// Ambiguous state - distribute probability equally
					for (int s = 0; s < stateCount; s++) {
						partials[offset + s] = 1.0;
					}
				}
			}
		}
	}

	/**
	 * Get partials for a node - from cache for leaves, from likelihood core for internal nodes.
	 */
	public double[] getPartials(int nodeNr) {
		ensureInitialized();
		if (nodeNr < leafNodeCount) {
			return leafPartials[nodeNr];
		} else {
			double[] partials = new double[partialsSize];
			likelihoodCore.getNodePartials(nodeNr, partials);
			return partials;
		}
	}
	
	/**
	 * Get the outsidePartials for a specific node (computed by updateByOperatorWithoutNode).
	 * Returns null if not computed for this node.
	 */
	public double[] getOutsidePartials(int nodeNr) {
		return outsidePartials[nodeNr];
	}
	
	/**
	 * Compute a divergence measure between two partial arrays.
	 * Uses a Hellinger-like distance which is symmetric and bounded.
	 * For each rate category, sums divergence across patterns, then averages across categories.
	 */
	private double computePartialsDivergence(double[] parentPartials, double[] childPartials) {
		if (parentPartials == null || childPartials == null) {
			return minWeight;
		}
		
		Alignment data = alignment;
		double totalDivergence = 0.0;
		
		// For each category, compute total divergence across all patterns
		for (int category = 0; category < matrixCount; category++) {
			double categoryDivergence = 0.0;
			
			for (int pattern = 0; pattern < patternCount; pattern++) {
				int offset = getPartialOffset(pattern, category);
				
				// Compute sums for normalization
				double parentSum = 0.0;
				double childSum = 0.0;
				for (int s = 0; s < stateCount; s++) {
					parentSum += parentPartials[offset + s];
					childSum += childPartials[offset + s];
				}
				
				if (parentSum <= 0 || childSum <= 0) {
					throw new RuntimeException("Partials sum to zero or negative - likelihood core may not be initialized");
				}
				
				// Compute Hellinger-like divergence for this pattern
				double patternDiv = 0.0;
				for (int s = 0; s < stateCount; s++) {
					double p = parentPartials[offset + s] / parentSum;
					double q = childPartials[offset + s] / childSum;
					double diff = Math.sqrt(p) - Math.sqrt(q);
					patternDiv += diff * diff;
				}
				categoryDivergence += data.getPatternWeight(pattern) * patternDiv;
			}
			
			totalDivergence += categoryDivergence;
		}
		
		return minWeight + totalDivergence / matrixCount;
	}
	
	@Override
    /**
     * Force an update to the weights in the middle of an operator, when the tree has changed from
     * the last likelihood update. Use sparingly, as the likelihood update is expensive.
     */
	public void updateByOperator() {
        partialsWithoutNode = new double[nodeCount][];  // Clear stale "without node" partials
	}

	@Override
	public void updateByOperatorWithoutNode(int ignore, List<Integer> nodes) {
		ensureInitialized();
        updateByOperator();

        // Recalculate the likelihood to ensure partials are up to date
        treeInput.get().setEverythingDirty(true);
        likelihoodInput.get().calculateLogP();
		
		// Ensure that the parent of the ignored node is always included in nodesToUpdate.
		// This is crucial because when ignore is a node i, its parent p becomes unary.
		// We must compute partialsWithoutNode[p] so that ancestors of p can properly
		// fetch p's partials through getPartialsForWithoutNode.
		Node parentOfIgnore = null;
		if (ignore >= 0 && ignore < nodeCount) {
			Node ignoreNode = treeInput.get().getNode(ignore);
			parentOfIgnore = ignoreNode.getParent();
		}
		
		List<Integer> nodesToUpdate = new ArrayList<>(nodes);
		if (parentOfIgnore != null && !parentOfIgnore.isRoot()) {
			if (!nodesToUpdate.contains(parentOfIgnore.getNr())) {
				nodesToUpdate.add(parentOfIgnore.getNr());
			}
		}
		
		// Ensure ALL ancestors from the ignored node to the root are in nodesToUpdate.
		// This is critical because partialsWithoutNode must be computed along the entire
		// ancestor path — otherwise getPartialsForWithoutNode falls back to normal partials
		// (which include the ignored node's contribution), producing incorrect results.
		if (ignore >= 0 && ignore < nodeCount) {
			Node ancestor = treeInput.get().getNode(ignore).getParent();
			while (ancestor != null) {
				if (!ancestor.isLeaf() && !nodesToUpdate.contains(ancestor.getNr())) {
					nodesToUpdate.add(ancestor.getNr());
				}
				ancestor = ancestor.getParent();
			}
		}
		
		// Track which node is being ignored and which nodes will have valid partialsWithoutNode
		currentIgnoredNode = ignore;
		
		// Compute modified partials that exclude the contribution of the ignored node
		Set<Integer> nodeSet = new HashSet<>(nodesToUpdate);
        
        Node root = treeInput.get().getRoot();
        Node effectiveRoot = becomesUnary(root, ignore) ? getIncludedChild(root, ignore) : root;
        
        // Initialize root's outside partials to uniform (1.0)
        // Frequencies are applied separately when computing likelihoods
        int rootNr = root.getNr();
        if (outsidePartials[rootNr] == null)
            outsidePartials[rootNr] = new double[partialsSize];
        Arrays.fill(outsidePartials[rootNr], 1.0);

        // Do the same with the effective root (if different from the original root)
        if (effectiveRoot != root){
            int effRootNr = effectiveRoot.getNr();
            if (outsidePartials[effRootNr] == null)
                outsidePartials[effRootNr] = new double[partialsSize];
            Arrays.fill(outsidePartials[effRootNr], 1.0);
        }

        updatePartialsWithoutNodeRecursive(effectiveRoot, ignore, nodeSet);

        // Recursively compute outside partials for all nodes below root
        // Must pass nodeSet so that we use partialsWithoutNode where appropriate
        updateOutsidePartialsRecursive(effectiveRoot, ignore, nodeSet);


        // TODO: stop outsidePartial calculation at the height of the ignore node
    }
	
	/**
	 * Recursively compute outside partials via pre-order (top-down) traversal.
	 * 
	 * For each child c of node n with sibling s:
	 *   outside[c] = transpose(T_nc) * (outside[n] ⊙ (T_ns * partials[s]))
	 * 
	 * where T_nc is the transition matrix from n to c, and ⊙ denotes element-wise multiplication.
	 * 
	 * @param node The current node being processed
	 * @param ignore Node number to ignore (for "without node" calculations)
	 * @param nodesToUpdate Set of nodes for which partialsWithoutNode have been computed
	 */
	private void updateOutsidePartialsRecursive(Node node, int ignore, Set<Integer> nodesToUpdate) {
		if (node.isLeaf())
			return;
		
		int nodeNr = node.getNr();
		double[] nodeOutside = outsidePartials[nodeNr];
		
		// Process each child
		Node leftChild = getEffectiveChild(node, 0, ignore);
		Node rightChild = getEffectiveChild(node, 1, ignore);

        // Compute the up-partials
        double[] rightUpPartial = applyTransitionUpWithoutNode(rightChild, node);
        updateChildOutsidePartials(node, leftChild, nodeOutside, rightUpPartial);
        updateOutsidePartialsRecursive(leftChild, ignore, nodesToUpdate);
		
        double[] leftUpPartial = applyTransitionUpWithoutNode(leftChild, node);
        updateChildOutsidePartials(node, rightChild, nodeOutside, leftUpPartial);
        updateOutsidePartialsRecursive(rightChild, ignore, nodesToUpdate);
	}

	
	/**
	 * Compute outside partials for a child node.
	 * 
	 * outside[child] = transpose(T_parent_to_child) * (outside[parent] ⊙ (T_parent_to_sibling * partials[sibling]))
	 * 
	 * @param parent The parent node
	 * @param child The child for which we're computing outside partials
	 * @param parentOutside The outside partials at the parent
	 * @param siblingContribution The sibling partial passed up to the parent
	 */
	private void updateChildOutsidePartials(Node parent, Node child, double[] parentOutside, double[] siblingContribution) {
		// Allocate outsidePartials for child if needed
        int childNr = child.getNr();
		if (outsidePartials[childNr] == null)
			outsidePartials[childNr] = new double[partialsSize];
		
		// Combine parent's outside with sibling's contribution (element-wise multiplication)
		double[] combinedMessage = LinearAlgebra.multiply(parentOutside, siblingContribution);
		
		// Apply transpose of transition matrix (going downward)
        double[] matrix = getTransitionMatrixWithoutNode(child, parent);
        applyTransposeTransitionMatrix(matrix, combinedMessage, outsidePartials[childNr]);
	}
	
	/**
	 * Get the full conditional likelihood at a node: P(all data | state at node)
	 * This is the element-wise product of partials (inside) and outside partials.
	 * 
	 * @param nodeNr The node number
	 * @param ignore Node to ignore (or -1 for none)
	 * @return Array of full conditional likelihoods indexed by state and pattern
	 */
	public double[] getFullConditionalLikelihoods(int nodeNr, int ignore) {
		ensureInitialized();
		
		double[] partials = getPartialsWithoutNode(nodeNr);
		double[] outside = outsidePartials[nodeNr];
		
        assert outside != null;
		
		// Element-wise product of inside and outside
		double[] fullConditional = LinearAlgebra.multiply(partials, outside);
		return fullConditional;
	}

	/**
	 * Recursively update partials for the "without node" case.
	 * Computes what partials would look like if the ignored node were removed.
	 * 
	 * Always recurses into children (post-order) so that deeper nodes in
	 * nodesToUpdate are reached, but only computes partialsWithoutNode for
	 * nodes that are in the nodesToUpdate set.
	 */
	private void updatePartialsWithoutNodeRecursive(Node n, int ignore, Set<Integer> nodesToUpdate) {
		final int nodeNr = n.getNr();

        // Leaf nodes have no partials to compute
        if (n.isLeaf())
            return;
		
		// Only compute partialsWithoutNode for nodes in nodesToUpdate
		if (!nodesToUpdate.contains(nodeNr))
			return;
		
		// Allocate array if needed
		if (partialsWithoutNode[nodeNr] == null)                                            // TODO: always keep allocated
			partialsWithoutNode[nodeNr] = new double[partialsSize];
		
		// Get children, excluding the ignored node
		Node child1 = getEffectiveChild(n, 0, ignore);
        Node child2 = getEffectiveChild(n, 1, ignore);

		assert (ignore==-1) || (child1.isLeaf() && getTree().getNode(ignore).isLeaf()) || child1.getNr() != ignore;
		assert (ignore==-1) || (child2.isLeaf() && getTree().getNode(ignore).isLeaf()) || child2.getNr() != ignore;
        
        // Update child partials recursively
        updatePartialsWithoutNodeRecursive(child1, ignore, nodesToUpdate);
        updatePartialsWithoutNodeRecursive(child2, ignore, nodesToUpdate);
        
        // Multiply the up-partials to get the parent partial
        double[] upPartials1 = applyTransitionUpWithoutNode(child1, n);
        double[] upPartials2 = applyTransitionUpWithoutNode(child2, n);
        partialsWithoutNode[nodeNr] = LinearAlgebra.multiply(upPartials1, upPartials2);
	}
	
	/**
	 * Apply transition matrix to partials (single-child pruning).
	 */
	public void applyTransitionMatrix(int childNodeNr, double[] childPartials, double[] outPartials) {
        ensureInitialized();
        double[] matrix = likelihoodCore.getNodeMatrices(childNodeNr);
		for (int category = 0; category < matrixCount; category++) {
			for (int pattern = 0; pattern < patternCount; pattern++) {
                int w = category * matrixSize; 
				int offset = getPartialOffset(pattern, category);
				for (int i = 0; i < stateCount; i++) {
					double sum = 0.0;
					for (int j = 0; j < stateCount; j++) {
						sum += matrix[w] * childPartials[offset + j];
                        w++;
					}
					outPartials[offset + i] = sum;
				}
			}
		}
	}

	public double[] applyTransitionMatrix(int childNodeNr, double[] childPartials) {
        double[] outPartials = new double[childPartials.length];
        applyTransitionMatrix(childNodeNr, childPartials, outPartials);
        return outPartials; 
    }

	/**
	 * Apply transition matrix to partials (single-child pruning).
	 */
	public void applyTransitionMatrix(double[] matrix, double[] childPartials, double[] outPartials) {
		for (int category = 0; category < matrixCount; category++) {
			for (int pattern = 0; pattern < patternCount; pattern++) {
                int w = category * matrixSize; 
				int offset = getPartialOffset(pattern, category);
				for (int i = 0; i < stateCount; i++) {
					double sum = 0.0;
					for (int j = 0; j < stateCount; j++) {
						sum += matrix[w] * childPartials[offset + j];
                        w++;
					}
					outPartials[offset + i] = sum;
				}
			}
		}
	}

	public double[] applyTransitionMatrix(double[] matrix, double[] childPartials) {
        double[] outPartials = new double[childPartials.length];
        applyTransitionMatrix(matrix, childPartials, outPartials);
        return outPartials; 
    }

	/**
	 * Apply the transpose of a provided transition matrix (per category) to partials.
	 * This computes: out[j] = sum_i T[i,j] * in[i]
	 * (whereas forward applies: out[i] = sum_j T[i,j] * in[j])
	 */
	public void applyTransposeTransitionMatrix(double[] matrix, double[] inPartials, double[] outPartials) {
		ensureInitialized();
		for (int category = 0; category < matrixCount; category++) {
			int matrixOffset = category * matrixSize;
			for (int pattern = 0; pattern < patternCount; pattern++) {
				int offset = getPartialOffset(pattern, category);
				for (int j = 0; j < stateCount; j++) {
					double sum = 0.0;
					for (int i = 0; i < stateCount; i++) {
                        double m = matrix[matrixOffset + i * stateCount + j];
						sum += m * inPartials[offset + i];
					}
					outPartials[offset + j] = sum;
				}
			}
		}
	}

	/**
	 * Apply the transpose of the transition matrix (for downward message passing).
	 */
	public void applyTransposeTransitionMatrix(int childNodeNr, double[] inPartials, double[] outPartials) {
		ensureInitialized();
		double[] matrix = likelihoodCore.getNodeMatrices(childNodeNr);
		applyTransposeTransitionMatrix(matrix, inPartials, outPartials);
	}

	public double[] applyTransposeTransitionMatrix(double[] matrix, double[] inPartials) {
        double[] outPartials = new double[inPartials.length];
        applyTransposeTransitionMatrix(matrix, inPartials, outPartials);
        return outPartials;
	}

	public double[] applyTransposeTransitionMatrix(int childNodeNr, double[] inPartials) {
        double[] outPartials = new double[inPartials.length];
        applyTransposeTransitionMatrix(childNodeNr, inPartials, outPartials);
        return outPartials;
	}


    int getPartialOffset(int pattern, int category) {
        return (category * patternCount + pattern) * stateCount;
    }
	
    /**
	 * Get partials for the "without node" case.
	 * If the node was updated as part of the "without" calculation, use those partials.
	 * Otherwise, read from the likelihood core.
	 */
	public double[] getPartialsWithoutNode(int nodeNr) {
		if (partialsWithoutNode[nodeNr] != null)
            return partialsWithoutNode[nodeNr];
		else
            return getPartials(nodeNr);
	}

	@Override
	public void fakeUpdateByOperator() {
		// Used for operators that change the tree without affecting edge weights
	}

	@Override
	protected boolean requiresRecalculation() {
		return treeInput.get().somethingIsDirty() || likelihoodInput.get().isDirtyCalculation();
	}

	@Override
	public void store() {
		super.store();
	}

	@Override
	public void prestore() {
		// No caching, nothing to prestore
	}

	@Override
	public void reset() {
		// No caching, nothing to reset
        treeInput.get().setEverythingDirty(true);
        likelihoodInput.get().calculateLogP();
	}

	@Override
	public void restore() {
		super.restore();
	}

	@Override
	public double getEdgeWeights(int nodeNr) {
        // Get parent node
        Node parent = treeInput.get().getNode(nodeNr).getParent();
        if (parent == null)
            return minWeight;  // Return minimum weight for the root node
		int parentNr = parent.getNr();

        // Get partials
        double[] parentPartials = getPartials(parentNr);
        double[] childPartials = getPartials(nodeNr);

        // Compute divergence
        double divergence = computePartialsDivergence(parentPartials, childPartials);

        // Clip and return
        return Math.max(minWeight, Math.min(maxWeight, divergence));
	}

	@Override	
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes) {
        return getTargetWeightsInteger(fromNodeNr,  toNodes.stream().map(n -> n.getNr()).toList());
    }
	
    /**
     * @param parent the parent
     * @param child  the child that you want the sister of
     * @return the other child of the given parent.
     */
    protected Node getOtherChild(final Node child) {
        Node parent = child.getParent();
        if (parent == null)
            return null;
        else if (parent.getLeft().getNr() == child.getNr()) {
            return parent.getRight();
        } else {
            return parent.getLeft();
        }
    }

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {

		// Use conventional partials for the fromNode (the same for all toNodes)
        Node fromNode = treeInput.get().getNode(fromNodeNr);
		double[] fromPartials = getPartials(fromNodeNr);

        // Compute the potential likelihood for all toNode candidates
		double[] weights = new double[toNodeNrs.size()];
		for (int k = 0; k < toNodeNrs.size(); k++) {
			int toNodeNr = toNodeNrs.get(k);
            Node toNode = treeInput.get().getNode(toNodeNr);
            
            // Use partialsWithoutNode (excludes fromNodeNr's contribution)
			double[] toPartials = getPartialsWithoutNode(toNodeNr);
			double newHeight = getToHeightWithoutNode(toNode, fromNode);

			// Compute the full proposed tree log-likelihood
			double logLikelihood = computeProposedTreeLogLikelihood(fromNode, toNode, fromPartials, toPartials, newHeight);
			// double logLikelihood = computePartialsAgreement(fromNode, toNode, fromPartials, toPartials, newHeight);
			weights[k] = EPS + logLikelihood / 2.0; // division by two to counteract squaring of probabilities in the operators
		}

        // Normalise probabilities
        double[] p = softmax(weights);
        return p;
	}


	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, double newHeight) {

		// Use conventional partials for the fromNode (the same for all toNodes)
        Node fromNode = treeInput.get().getNode(fromNodeNr);
        double[] fromPartials = getPartials(fromNodeNr);

        // Compute the potential likelihood for all toNode candidates
		double[] weights = new double[toNodeNrs.size()];
		for (int k = 0; k < toNodeNrs.size(); k++) {
			int toNodeNr = toNodeNrs.get(k);
            Node toNode = treeInput.get().getNode(toNodeNr);
            
            // Use partialsWithoutNode (excludes fromNodeNr's contribution)
			double[] toPartials = getPartialsWithoutNode(toNodeNr);

			// Compute the full proposed tree log-likelihood
			double logLikelihood = computeProposedTreeLogLikelihood(fromNode, toNode, fromPartials, toPartials, newHeight);
			// double logLikelihood = computePartialsAgreement(fromNode, toNode, fromPartials, toPartials, newHeight);
			weights[k] = EPS + logLikelihood / 2.0; // division by two to counteract squaring of probabilities in the operators
		}

        // Normalise probabilities
        double[] p = softmax(weights);
        return p;
	}

    public double computePartialsAgreement(Node fromNode, Node toNode, double[] fromPartials, double[] toPartials, double toHeight) {
        calculateTransitionMatrix(fromNode, toHeight, _fromMatrix);
        calculateTransitionMatrix(toNode, toHeight, _toMatrix);

        // Calculate partials at the potential new parent node
        double[] fromPartialsAtParent = applyTransitionMatrix(_fromMatrix, fromPartials);
        double[] toPartialsAtParent = applyTransitionMatrix(_toMatrix, toPartials);

        double[] targetPartialsPerCat = new double[toPartials.length];
        for (int i = 0; i < matrixCount * patternCount; i++) {
            int offset = i * stateCount;
            double toPartialSum = 0.001;
            for (int j = offset; j < offset + stateCount; j++)
                toPartialSum += toPartialsAtParent[j];
            for (int j = offset; j < offset + stateCount; j++)
                targetPartialsPerCat[j] = (toPartialsAtParent[j] / toPartialSum) * fromPartialsAtParent[j];
        }

        // Integrate across rate categories using the site model's category proportions
        double[] targetPartials = new double[patternCount * stateCount];
        double[] proportions = getSiteModel().getCategoryProportions(toNode);
        likelihoodCore.calculateIntegratePartials(targetPartialsPerCat, proportions, targetPartials);

        double[] patternLogLikelihoods = new double[patternCount];
        double[] freqs = getSubstitutionModel().getFrequencies();
        likelihoodCore.calculateLogLikelihoods(targetPartials, freqs, patternLogLikelihoods);

        // Sum over the log-likelihoods for different site patterns, applying pattern weights
        double weightedSum = 0.0;
        for (int p = 0; p < patternCount; p++) {
            weightedSum += alignment.getPatternWeight(p) * patternLogLikelihoods[p];
        }
        return weightedSum;
    }

	/**
	 * Determine the effective parent height for `toNode` when the current ignored node
	 * is removed. Unary nodes created by the ignored child become transparent.
	 */
	public double getToHeightWithoutNode(Node toNode, Node fromNode) {
        double minHeight = Math.max(toNode.getHeight(), fromNode.getHeight());
		Node effectiveParent = getEffectiveParent(toNode, currentIgnoredNode);
		if (effectiveParent == null) // toNode is root (when ignoring currentIgnoreNode)
			return minHeight + 0.1;
        else
		    return (minHeight + effectiveParent.getHeight()) / 2.0;
	}

    /**
     * Determine if a node becomes unary when the specified child is ignored.
     */
    public boolean becomesUnary(Node node, int ignore) {
        // Leaves never become unary 
        if (node.isLeaf())
            return false; 

        // Internal nodes become unary iff one child is ignored 
        return (node.getLeft().getNr() == ignore) || (node.getRight().getNr() == ignore);
    }

    /**
     * Get the child of `node` that is included (not ignored), assuming that exactly one child is ignored.
     */
    Node getIncludedChild(Node node, int ignore) {
        assert !node.isLeaf();

        if (node.getLeft().getNr() == ignore)
            return node.getRight();
        else if (node.getRight().getNr() == ignore)
            return node.getLeft();
        else
            throw new IllegalArgumentException("Node " + node.getID() + " does not have an ignored child " + ignore);
    }

    /**
     * Get the effective child of `node` at index `i`, accounting for the ignored node. If the child at index `i`
     * is the ignored node, return null. If the other child is the ignored node, return the included child
     * (unary node case). Otherwise, return the original child.
     */
    Node getEffectiveChild(Node node, int i, int ignore) {
        assert !node.isLeaf();
        Node child = node.getChild(i);
        if (child.getNr() == ignore)
            return null;
        else if (becomesUnary(child, ignore))
            return getIncludedChild(child, ignore);
        else
            return child;
    }

    /**
     * Get the effective parent of `node`, accounting for the ignored node. Unary nodes created by the ignored child become transparent,
     */
	public Node getEffectiveParent(Node node, int ignore) {
		if (ignore < 0) 
            return node.getParent();

		Node parent = node.getParent();
		while (parent != null && becomesUnary(parent, ignore)) {
			parent = parent.getParent();
		}
		return parent;
	}

    /**
     * Get the effective sibling of `node`, accounting for the ignored node. Unary nodes created by the ignored child become transparent,
     */
    Node getEffectiveSibling(final Node node, int ignore) {
        Node parent = getEffectiveParent(node, ignore);
        if (parent == null) 
            return null;
        else if (getEffectiveChild(parent, 0, ignore) == node)
            return getEffectiveChild(parent, 1, ignore);
        else if (getEffectiveChild(parent, 1, ignore) == node)
            return getEffectiveChild(parent, 0, ignore);
        else
            return null;
    }

	public double computeFullConditionalAgreement(Node fromNode, Node toNode, double[] fromFullConditional, double[] toFullConditional) {
		// Set `toHeight` at the midpoint of the edge above `toNode` (root node is a special case)
		double toHeight;
		if (toNode.isRoot())
			toHeight = toNode.getHeight() + 0.1;
		else
			toHeight = (toNode.getHeight() + getEffectiveParent(toNode, fromNode.getNr()).getHeight()) / 2.;

		return computeFullConditionalAgreement(fromNode, toNode, fromFullConditional, toFullConditional, toHeight);
	}

	public double computeFullConditionalAgreement(Node fromNode, Node toNode, double[] fromFullConditional, double[] toFullConditional, double toHeight) {
		calculateTransitionMatrix(fromNode, toHeight, _fromMatrix);
		calculateTransitionMatrix(toNode, toHeight, _toMatrix);

		// Calculate full conditional likelihoods at the potential new parent node
		double[] parentPartialsPerCat = new double[toFullConditional.length];
		likelihoodCore.calculatePartialsPartialsPruning(fromFullConditional, _fromMatrix,
														toFullConditional, _toMatrix,
														parentPartialsPerCat);

		// Integrate across rate categories using the site model's category proportions
		double[] parentPartials = new double[patternCount * stateCount];
		double[] proportions = getSiteModel().getCategoryProportions(toNode);
		likelihoodCore.calculateIntegratePartials(parentPartialsPerCat, proportions, parentPartials);

		double[] patternLogLikelihoods = new double[patternCount];
		double[] freqs = getSubstitutionModel().getFrequencies();
		likelihoodCore.calculateLogLikelihoods(parentPartials, freqs, patternLogLikelihoods);

		// Sum over the log-likelihoods for different site patterns, applying pattern weights
		double weightedSum = 0.0;
		for (int p = 0; p < patternCount; p++) {
			weightedSum += alignment.getPatternWeight(p) * patternLogLikelihoods[p];
		}
		return weightedSum;
	}

	/**
	 * Compute the log-likelihood of the proposed tree where fromNode's subtree is
	 * reattached on the edge above toNode at the given toHeight.
	 * 
	 * This properly accounts for ALL data in the tree by:
	 * 1. Computing partials at the hypothetical parent from fromNode and toNode
	 * 2. Propagating up to toNode's current parent (the grandparent)
	 * 3. Combining with the sibling of toNode at the grandparent level
	 * 4. Multiplying by outsidePartials[grandparent] for data above
	 * 
	 * Requires that updateByOperatorWithoutNode has been called first.
	 * 
	 * @param fromNode The node being moved
	 * @param toNode The target node (we attach on the edge above it)
	 * @param fromPartials Partials at fromNode (data in the moving subtree)
	 * @param toPartials Partials at toNode computed without fromNode
	 * @param toHeight Height of the hypothetical new parent
	 * @return Log-likelihood of the proposed tree
	 */
	public double computeProposedTreeLogLikelihood(Node fromNode, Node toNode, double[] fromPartials, double[] toPartials, double toHeight) {
        assert toHeight > 0;
        assert toHeight > toNode.getHeight();
        assert toHeight > fromNode.getHeight();
        assert !toNode.isRoot();

		// Step 1: Compute partials at the hypothetical parent from both children
		calculateTransitionMatrix(fromNode, toHeight, _fromMatrix);
		calculateTransitionMatrix(toNode, toHeight, _toMatrix);

		double[] hpPartialsPerCat = new double[partialsSize];
		likelihoodCore.calculatePartialsPartialsPruning(fromPartials, _fromMatrix,
														toPartials, _toMatrix,
														hpPartialsPerCat);

		// Step 2: Compute transition matrix from hypothetical parent up to grandparent.
		// Use toNode's branch rate for this new branch.
		// Use effective parent to skip any unary nodes created by ignoring fromNode
		Node grandparent = getEffectiveParent(toNode, currentIgnoredNode);
		double gpHeight = grandparent.getHeight();
		double[] hpToGpMatrix = new double[matrixCount * matrixSize];
		calculateTransitionMatrixForBranch(toHeight, gpHeight, toNode, hpToGpMatrix);

		// Step 3: Get the sibling of toNode and combine at grandparent level
		Node sibling = getEffectiveSibling(toNode, currentIgnoredNode);
		double[] gpPartialsPerCat = new double[partialsSize];
        applyTransitionMatrixToPartials(hpPartialsPerCat, hpToGpMatrix, gpPartialsPerCat);
		if (sibling != null) {
            double[] siblingUpPartials = applyTransitionUpWithoutNode(sibling, grandparent);    
            for (int i=0; i<partialsSize; i++)
                gpPartialsPerCat[i] *= siblingUpPartials[i];
		}

		// Step 4: Combine with outsidePartials at the grandparent
		double[] outsideGp = outsidePartials[grandparent.getNr()];
		double[] combinedPerCat = LinearAlgebra.multiply(gpPartialsPerCat, outsideGp);

        return computeLikelihoodFromPartials(toNode, combinedPerCat);
	}

    /**
     * Integrate across rate categories and compute log-likelihood
     * @param node The node at which the partials are computed (should already incorporate outside partials for data above)
     * @param combinedPartials Partials at the node, already multiplied by outside partials for data above
     * @return log-likelihood of the tree given the partials at the specified node (which should already incorporate outside partials for data above)
     */
    public double computeLikelihoodFromPartials(Node node, double[] combinedPartials) {
		double[] integratedPartials = new double[patternCount * stateCount];
		double[] proportions = getSiteModel().getCategoryProportions(node);
		likelihoodCore.calculateIntegratePartials(combinedPartials, proportions, integratedPartials);

		double[] patternLogLikelihoods = new double[patternCount];
		double[] freqs = getSubstitutionModel().getFrequencies();
		likelihoodCore.calculateLogLikelihoods(integratedPartials, freqs, patternLogLikelihoods);

		double weightedSum = 0.0;
		for (int p = 0; p < patternCount; p++)
			weightedSum += alignment.getPatternWeight(p) * patternLogLikelihoods[p];

		return weightedSum;
    }

    boolean allFinite(double[] arr) {
        for (double x : arr)
            if (!Double.isFinite(x))
                return false;
        return true;
    }

	/**
	 * Calculate transition matrix for a branch specified by heights, using the
	 * branch rate of the given reference node.
	 * 
	 * @param childHeight Height of the child end of the branch
	 * @param parentHeight Height of the parent end of the branch
	 * @param rateNode Node whose branch rate to use
	 * @param matrix Output matrix (size matrixCount * matrixSize)
	 */
	private void calculateTransitionMatrixForBranch(double childHeight, double parentHeight,
			Node rateNode, double[] matrix) {
		if (_probabilities == null) {
			_probabilities = new double[matrixSize];
		}
		double branchRate = getBranchRate(rateNode);
		for (int i = 0; i < matrixCount; i++) {
			final double jointBranchRate = getSiteModel().getRateForCategory(i, rateNode) * branchRate;
			getSubstitutionModel().getTransitionProbabilities(
				rateNode,
				parentHeight,
				childHeight,
				jointBranchRate,
				_probabilities
			);
			System.arraycopy(_probabilities, 0, matrix, i * matrixSize, matrixSize);
		}
	}

    // void applyTransitionUpWithoutNode(Node child, Node parent, int ignore, Set<Integer> nodesToUpdate, double[] outPartials) {
    //     double[] childPartial = getPartialsForWithoutNode(child.getNr(), ignore, nodesToUpdate);
    void applyTransitionUpWithoutNode(Node child, Node parent, double[] outPartials) {
        double[] childPartial = getPartialsWithoutNode(child.getNr());
        double[] matrix = getTransitionMatrixWithoutNode(child, parent);
        applyTransitionMatrix(matrix, childPartial, outPartials);
    }

    // double[] applyTransitionUpWithoutNode(Node child, Node parent, int ignore, Set<Integer> nodesToUpdate) {
    double[] applyTransitionUpWithoutNode(Node child, Node parent) {
        double[] outPartials = new double[partialsSize];
        // applyTransitionUpWithoutNode(child, parent, ignore, nodesToUpdate, outPartials);
        applyTransitionUpWithoutNode(child, parent, outPartials);
        return outPartials;
    }

    // void applyTransitionDownWithoutNode(Node child, Node parent, int ignore, Set<Integer> nodesToUpdate, double[] outPartials) {
    //     double[] parentPartial = getPartialsForWithoutNode(parent.getNr(), ignore, nodesToUpdate);
    //     double[] parentPartial = getPartialsForTargetWeight(parent.getNr(), nodesToUpdate);
    void applyTransitionDownWithoutNode(Node child, Node parent, double[] outPartials) {
        double[] parentPartial = getPartialsWithoutNode(parent.getNr());
        double[] matrix = getTransitionMatrixWithoutNode(child, parent);
        applyTransposeTransitionMatrix(matrix, parentPartial, outPartials);
    }

    // double[] applyTransitionDownWithoutNode(Node child, Node parent, int ignore, Set<Integer> nodesToUpdate) {
    double[] applyTransitionDownWithoutNode(Node child, Node parent, int ignore, Set<Integer> nodesToUpdate) {
        double[] outPartials = new double[partialsSize];
        // applyTransitionDownWithoutNode(child, parent, ignore, nodesToUpdate, outPartials);
        applyTransitionDownWithoutNode(child, parent, outPartials);
        return outPartials;
    }

    double[] getTransitionMatrixWithoutNode(Node child, Node parent) {
        if (child.getParent() == parent) {
            // `child`/`parent` are direct child/parent in original tree -> use the stored transition matrix
            return likelihoodCore.getNodeMatrices(child.getNr());
        } else {
            // `child`/`parent` are only connected via and ignored node -> calculate transition matrix in pruned tree
            // Important because the `toHeight` is different to the one in the original tree
            return calculateTransitionMatrix(child, parent.getHeight());
        }
    }


	/**
	 * Apply a transition matrix to partials (for propagating partials up through a branch).
	 * For each pattern and rate category: result[s] = sum_t T[s][t] * partials[t]
	 * 
	 * @param partials Input partials (per category)
	 * @param matrix Transition matrix (per category)
	 * @param result Output partials (per category)
	 */
	private void applyTransitionMatrixToPartials(double[] partials, double[] matrix, double[] result) {
		for (int cat = 0; cat < matrixCount; cat++) {
			int matrixOffset = cat * matrixSize;
			int partialsOffset = cat * patternCount * stateCount;
			for (int p = 0; p < patternCount; p++) {
				int patternOffset = partialsOffset + p * stateCount;
				for (int s = 0; s < stateCount; s++) {
					double sum = 0.0;
					for (int t = 0; t < stateCount; t++) {
						sum += matrix[matrixOffset + s * stateCount + t] * partials[patternOffset + t];
					}
					result[patternOffset + s] = sum;
				}
			}
		}
	}

    public void calculateTransitionMatrix(Node child, double toHeight, double[] matrix) {
        double branchRate = getBranchRate(child);
        for (int i = 0; i < matrixCount; i++) {
            final double jointBranchRate = getSiteModel().getRateForCategory(i, child) * branchRate;            
            getSubstitutionModel().getTransitionProbabilities(
                child,
                toHeight,
                child.getHeight(),
                jointBranchRate,
                _probabilities
            );
            System.arraycopy(_probabilities, 0, matrix, i * matrixSize, matrixSize);
        }
    }

    public double[] calculateTransitionMatrix(Node child, double toHeight) {
        double[] matrix = new double[matrixSize * matrixCount];
        calculateTransitionMatrix(child, toHeight, matrix);
        return matrix;
    }

    public SubstitutionModel getSubstitutionModel() {
        return likelihoodInput.get().getSubstitutionModel();
    }

    public SiteModelInterface.Base getSiteModel() {
        if (treelikelihood.siteModelInput.get() instanceof SiteModel.Base siteModel)
			return siteModel;
		else
			throw new IllegalStateException(
				"FelsensteinWeights requires SiteModel.Base from the TreeLikelihood; got " + treelikelihood.siteModelInput.get().getClass().getName()
			);
    }

    public BranchRateModel getBranchRateModel() {
        return likelihoodInput.get().branchRateModelInput.get();
    }

    public double getBranchRate(Node node) {
        BranchRateModel brm = getBranchRateModel();
        if (brm == null) {
            return 1.0;
        } else {
            return brm.getRateForBranch(node);
        }
    }

    public Tree getTree() {
        return treeInput.get();
    }

	// =================================================================
	// ======================== Utility Methods ========================
	// =================================================================

	/**
	 * Numerically stable log-sum-exp.
	 */
	private static double logSumExp(double[] logVals) {
		double maxTerm = Double.NEGATIVE_INFINITY;
		for (double logx : logVals) {
			maxTerm = Math.max(maxTerm, logx);
		}
		if (maxTerm == Double.NEGATIVE_INFINITY) {
			return Double.NEGATIVE_INFINITY;
		}
		double sum = 0.0;
		for (double logx : logVals) {
			sum += Math.exp(logx - maxTerm);
		}
		return maxTerm + Math.log(sum);
	}

	/**
	 * Softmax: converts log-values to normalized probabilities.
	 */
	private static double[] softmax(double[] logVals) {
		double logSum = logSumExp(logVals);
		double[] result = new double[logVals.length];
		for (int i = 0; i < logVals.length; i++) {
			result[i] = Math.exp(logVals[i] - logSum);
		}
		return result;
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
		// Not used for edge weights
	}

}
