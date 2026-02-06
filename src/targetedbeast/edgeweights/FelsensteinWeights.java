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

	private static final double EPS = 1E-8;

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
			String leaves = "?";
			if (treeInput.get() instanceof Tree) {
				leaves = String.join(", ", ((Tree) treeInput.get()).getTaxaNames());
			}
			throw new IllegalArgumentException(String.format(
					"The number of leaves in the tree (%d) does not match the number of sequences (%d). "
							+ "The tree has leaves [%s], while the data refers to taxa [%s].",
					treeInput.get().getLeafNodeCount(), alignment.getTaxonCount(), leaves,
					String.join(", ", alignment.getTaxaNames())));
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
	private void ensureInitialized() {
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
		SiteModel.Base siteModel = (treelikelihood.siteModelInput.get() instanceof SiteModel.Base)
			? (SiteModel.Base) treelikelihood.siteModelInput.get() : null;
		if (siteModel == null) {
			throw new IllegalStateException(
				"FelsensteinWeights requires SiteModel.Base from the TreeLikelihood; got " +
				(treelikelihood.siteModelInput.get() == null ? "null" : treelikelihood.siteModelInput.get().getClass().getName())
			);
		}

		matrixCount = siteModel.getCategoryCount();
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
						partials[offset + s] = 1.0 / stateCount;
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
	
	/**
	 * Compute a similarity measure between the partials of two nodes.
	 * Uses the dot product of normalized partial distributions.
	 * For each rate category, multiplies across patterns (sums log), then averages across categories.
	 */
	private double computePartialsSimilarity(double[] partials1, double[] partials2) {
		// return 1.;
        
        Alignment data = alignment;
        double[] categoryLogSims = new double[matrixCount];
		
		// For each category, compute total log-similarity across all patterns
		for (int category = 0; category < matrixCount; category++) {
			categoryLogSims[category] = 0.0;
			
			for (int pattern = 0; pattern < patternCount; pattern++) {
				int offset = getPartialOffset(pattern, category);
				
				// Compute sums for normalization
				double sum1 = 0.0;
				double sum2 = 0.0;
				for (int s = 0; s < stateCount; s++) {
					sum1 += partials1[offset + s];
					sum2 += partials2[offset + s];
				}
				
				if (sum1 <= EPS || sum2 <= EPS) {
                    continue;
				}
				
				// Compute dot product for this pattern
				double patternSim = 0.0;
				for (int s = 0; s < stateCount; s++) {
					double p1 = partials1[offset + s] / sum1;
					double p2 = partials2[offset + s] / sum2;
					patternSim += p1 * p2 + EPS;
				}
				// Add small epsilon for numerical stability
				categoryLogSims[category] += data.getPatternWeight(pattern) * Math.log(patternSim);
			}
		}
		
        // Sum up the probabilities of different rate categories and transform the result back to log space
        // log(sum(sim_c))) = log(sum(exp(logsim_c)))
        return logSumExp(categoryLogSims);
	}

	@Override
    /**
     * Force an update to the weights in the middle of an operator, when the tree has changed from
     * the last likelihood update. Use sparingly, as the likelihood update is expensive.
     */
	public void updateByOperator() {
		// likelihoodInput.get().calculateLogP();  // Force full partial recalculation
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
		
		// Track which node is being ignored and which nodes will have valid partialsWithoutNode
		currentIgnoredNode = ignore;
		
		// Compute modified partials that exclude the contribution of the ignored node
		Set<Integer> nodeSet = new HashSet<>(nodesToUpdate);
        
        Node root = treeInput.get().getRoot();
		updatePartialsWithoutNodeRecursive(root, ignore, nodeSet);

        // Compute outside partials via downward pass (pre-order traversal)
        // At root: outside = uniform (no data above root; frequencies applied separately)
        int rootNr = root.getNr();
        if (outsidePartials[rootNr] == null)
            outsidePartials[rootNr] = new double[partialsSize];
        
        // Initialize root's outside partials to uniform (1.0)
        // Frequencies are applied separately when computing likelihoods
        Arrays.fill(outsidePartials[rootNr], 1.0);
        
        // Recursively compute outside partials for all nodes below root
        updateOutsidePartials(root, ignore);
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
	 */
	private void updateOutsidePartials(Node node, int ignore) {
		if (node.isLeaf())
			return;
		
		int nodeNr = node.getNr();
		double[] nodeOutside = outsidePartials[nodeNr];
		
		// Process each child
		Node leftChild = node.getLeft();
		Node rightChild = node.getRight();
		
		if (leftChild != null && leftChild.getNr() != ignore) {
			computeChildOutsidePartials(node, leftChild, rightChild, nodeOutside, ignore);
			updateOutsidePartials(leftChild, ignore);
		}
		
		if (rightChild != null && rightChild.getNr() != ignore) {
			computeChildOutsidePartials(node, rightChild, leftChild, nodeOutside, ignore);
			updateOutsidePartials(rightChild, ignore);
		}
	}
	
	/**
	 * Compute outside partials for a child node.
	 * 
	 * outside[child] = transpose(T_parent_to_child) * (outside[parent] ⊙ (T_parent_to_sibling * partials[sibling]))
	 * 
	 * @param parent The parent node
	 * @param child The child for which we're computing outside partials
	 * @param sibling The sibling of the child
	 * @param parentOutside The outside partials at the parent
	 * @param ignore Node number to ignore
	 */
	private void computeChildOutsidePartials(Node parent, Node child, Node sibling, 
			                                    double[] parentOutside, int ignore) {
		int childNr = child.getNr();
		
		// Allocate if needed
		if (outsidePartials[childNr] == null)
			outsidePartials[childNr] = new double[partialsSize];
		
		double[] childOutside = outsidePartials[childNr];
		
		// Get sibling's contribution (partials transformed through parent->sibling transition)
		double[] siblingContribution;
		if (sibling != null && sibling.getNr() != ignore) {
			double[] siblingPartials = getPartialsForWithoutNode(sibling.getNr(), ignore, 
					Collections.emptySet());
			siblingContribution = applyTransitionMatrix(sibling.getNr(), siblingPartials);
		} else {
			// No sibling contribution (sibling is ignored or doesn't exist)
			siblingContribution = new double[partialsSize];
			Arrays.fill(siblingContribution, 1.0);
		}
		
		// Combine parent's outside with sibling's contribution (element-wise multiplication)
		double[] combinedMessage = new double[partialsSize];
		for (int i = 0; i < partialsSize; i++)
			combinedMessage[i] = parentOutside[i] * siblingContribution[i];
		
		// Apply transpose of transition matrix (going downward)
		applyTransposeTransitionMatrix(child.getNr(), combinedMessage, childOutside);
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
		
		double[] partials = getPartialsForTargetWeight(nodeNr, ignore);
		double[] outside = outsidePartials[nodeNr];
		
		if (outside == null) {
			// Outside partials not yet computed, return just the partials
			return partials;
		}
		
		// Element-wise product of inside and outside
		double[] fullConditional = new double[partialsSize];
		for (int i = 0; i < partialsSize; i++)
			fullConditional[i] = partials[i] * outside[i];
		
		return fullConditional;
	}


    /// ------------------------------------------------------------------------------------------------

	/**
	 * Recursively update partials for the "without node" case.
	 * Computes what partials would look like if the ignored node were removed.
	 */
	private void updatePartialsWithoutNodeRecursive(Node n, int ignore, Set<Integer> nodesToUpdate) {
		final int nodeNr = n.getNr();

        // Return if `n` is a leaf node or doesn't need updating 
        if (n.isLeaf() || !nodesToUpdate.contains(nodeNr))
            return;
		
		// Recurse to children first (post-order traversal)
		for (Node child : n.getChildren())
			updatePartialsWithoutNodeRecursive(child, ignore, nodesToUpdate);
		
		// Allocate array if needed
		if (partialsWithoutNode[nodeNr] == null)
			partialsWithoutNode[nodeNr] = new double[partialsSize];
		
		// Get children, excluding the ignored node
		Node child1 = n.getChild(0);
		Node child2 = n.getChild(1);
		boolean ignore1 = (child1.getNr() == ignore);
		boolean ignore2 = (child2.getNr() == ignore);
		
		assert !(ignore1 && ignore2) : "We can't ignore both children";
        
        if (ignore1 || ignore2) {
			// One child ignored - unary node case
			Node validChild = ignore1 ? child2 : child1;
			double[] childPartials = getPartialsForWithoutNode(validChild.getNr(), ignore, nodesToUpdate);
			applyTransitionMatrix(validChild.getNr(), childPartials, partialsWithoutNode[nodeNr]);

            // TODO: double check whether the transtion matrix is correctly applied

		} else {
			// Both children valid - standard Felsenstein pruning
			double[] partials1 = getPartialsForWithoutNode(child1.getNr(), ignore, nodesToUpdate);
			double[] partials2 = getPartialsForWithoutNode(child2.getNr(), ignore, nodesToUpdate);
			likelihoodCore.calculatePartialsPartialsPruning(
                partials1, likelihoodCore.getNodeMatrices(child1),
				partials2, likelihoodCore.getNodeMatrices(child2),
                partialsWithoutNode[nodeNr]
            );
		}
	}

	// private void applyTransitionMatrix(int childNodeNr, double[] childPartials, double[] outPartials) {
	// 	double[] matrix = new double[matrixSize];
	// 	for (int category = 0; category < matrixCount; category++) {
	// 		for (int pattern = 0; pattern < patternCount; pattern++) {
	// 			int offset = getPartialOffset(pattern, category);
	// 			for (int i = 0; i < stateCount; i++) {
	// 				double sum = 0.0;
	// 				for (int j = 0; j < stateCount; j++) {
	// 					sum += matrix[i * stateCount + j] * childPartials[offset + j];
	// 				}
	// 				outPartials[offset + i] = sum;
	// 			}
	// 		}
	// 	}
	// }
	
	/**
	 * Apply transition matrix to partials (single-child pruning).
	 */
	private void applyTransitionMatrix(int childNodeNr, double[] childPartials, double[] outPartials) {
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

	/**
	 * Apply the transpose of the transition matrix (for downward message passing).
	 * This computes: out[j] = sum_i T[i,j] * in[i]
	 * (whereas forward applies: out[i] = sum_j T[i,j] * in[j])
	 */
	private void applyTransposeTransitionMatrix(int childNodeNr, double[] inPartials, double[] outPartials) {
		double[] matrix = likelihoodCore.getNodeMatrices(childNodeNr);
		
		for (int category = 0; category < matrixCount; category++) {
			int matrixOffset = category * matrixSize;
			
			for (int pattern = 0; pattern < patternCount; pattern++) {
				int offset = getPartialOffset(pattern, category);
				
				// Apply transpose: out[j] = sum_i T[i,j] * in[i]
				for (int j = 0; j < stateCount; j++) {
					double sum = 0.0;
					for (int i = 0; i < stateCount; i++) {
						// T[i,j] is at matrix[i * stateCount + j]
						sum += matrix[matrixOffset + i * stateCount + j] * inPartials[offset + i];
					}
					outPartials[offset + j] = sum;
				}
			}
		}
	}
	


	private double[] applyTransitionMatrix(int childNodeNr, double[] childPartials) {
        double[] outPartials = new double[childPartials.length];
        applyTransitionMatrix(childNodeNr, childPartials, outPartials);
        return outPartials; 
    }

    int getPartialOffset(int pattern, int category) {
        // return (pattern * matrixCount + category) * stateCount;
        return (category * patternCount + pattern) * stateCount;
    }
	
	/**
	 * Get partials for the "without node" case.
	 * If the node was updated as part of the "without" calculation, use those partials.
	 * Otherwise, read from the likelihood core.
	 */
	public double[] getPartialsForWithoutNode(int nodeNr, int ignore, Set<Integer> nodesToUpdate) {
		if (nodeNr == ignore) {
			return null;
		}
		if (nodesToUpdate.contains(nodeNr) && nodeNr >= leafNodeCount) {
			// Use the computed "without" partials
			return partialsWithoutNode[nodeNr];
		} else {
			// Use normal partials from core
			return getPartials(nodeNr);
		}
	}
	
	/**
	 * Get partials of node `nodeNr` ignoring the effect of the subtree below `nodeNr`.
	 */
	public double[] getPartialsForTargetWeight(int nodeNr, int ignoreNodeNr) {
		// Check if we have valid partialsWithoutNode for this node,
		// computed with ignoreNodeNr as the ignored node
		if (currentIgnoredNode == ignoreNodeNr && 
			nodeNr >= leafNodeCount &&
			partialsWithoutNode[nodeNr] != null) {
			return partialsWithoutNode[nodeNr];
		}

		// Otherwise use the regular partials
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

    double hypotheticalEdgeLength = 0.1;
    double[] fromMatrix;
    double[] toMatrix;
    double[] _probabilities;

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
        Node fromNode = treeInput.get().getNode(fromNodeNr);

        treelikelihood.calculateLogP();

        // treelikelihood.calculateLogP();
		double[] weights = new double[toNodeNrs.size()];
        // double[] fromMatrix = likelihoodCore.getNodeMatrices(fromNodeNr);

        if (_probabilities == null)
            _probabilities = new double[matrixSize];
        if (fromMatrix == null)
            fromMatrix = new double[matrixCount * matrixSize];
        if (toMatrix == null)
            toMatrix = new double[matrixCount * matrixSize];

		// Use computePartialsAgreement with conventional partials
		double[] fromPartials = getPartials(fromNodeNr);

		for (int k = 0; k < toNodeNrs.size(); k++) {
			int toNodeNr = toNodeNrs.get(k);
            Node toNode = treeInput.get().getNode(toNodeNr);
            
            // Use partialsWithoutNode (excludes fromNodeNr's contribution)
			double[] toPartials = getPartialsForTargetWeight(toNodeNr, fromNodeNr);

			double toHeight = getToHeightWithoutNode(toNode);
			weights[k] = computePartialsAgreement(fromNode, toNode, fromPartials, toPartials, toHeight);
		}
        double[] p = softmax(weights);

        // Add epsilon to avoid 0 weights
        for (int i=0; i<p.length; i++){
            p[i] += EPS;
        }

        return p;
	}


	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, double newHeight) {
        Node fromNode = treeInput.get().getNode(fromNodeNr);
		double[] weights = new double[toNodeNrs.size()];

        // Make sure arrays are initialized
        if (_probabilities == null)
            _probabilities = new double[matrixSize];
        if (fromMatrix == null)
            fromMatrix = new double[matrixCount * matrixSize];
        if (toMatrix == null)
            toMatrix = new double[matrixCount * matrixSize];

		// Use computePartialsAgreement with conventional partials
		double[] fromPartials = getPartials(fromNodeNr);

		for (int k = 0; k < toNodeNrs.size(); k++) {
			int toNodeNr = toNodeNrs.get(k);
            Node toNode = treeInput.get().getNode(toNodeNr);
            
            // Use partialsWithoutNode (excludes fromNodeNr's contribution)
			double[] toPartials = getPartialsForTargetWeight(toNodeNr, fromNodeNr);
			weights[k] = computePartialsAgreement(fromNode, toNode, fromPartials, toPartials, newHeight);

			// // Compute the full proposed tree log-likelihood
			// double logLikelihood = computeProposedTreeLogLikelihood(fromNode, toNode, fromPartials, toPartials, newHeight);
			// weights[k] = logLikelihood / 2.0; // division by two to counteract squaring of probabilities in the operators
		}

        // Normalise probabilities
        double[] p = softmax(weights);

        // Add epsilon to avoid 0 weights
        for (int i=0; i<p.length; i++)
            p[i] += EPS;

        // p = clipAndNormalize(p, 0.8);

        return p;
	}

    public double computePartialsAgreement(Node fromNode, Node toNode, double[] fromPartials, double[] toPartials) {
        // Set `toHeight` at the midpoint of the edge above `toNode` (root node is a special case)
        double toHeight; 
        if (toNode.isRoot())
            toHeight = toNode.getHeight() + hypotheticalEdgeLength;
        else
            toHeight = (toNode.getHeight() + toNode.getParent().getHeight()) / 2.;

        return computePartialsAgreement(fromNode, toNode, fromPartials, toPartials, toHeight);
    }

    public double computePartialsAgreement(Node fromNode, Node toNode, double[] fromPartials, double[] toPartials, double toHeight) {
        calculateTransitionMatrix(fromNode, toHeight, fromMatrix);
        calculateTransitionMatrix(toNode, toHeight, toMatrix);

        // Calculate partials at the potential new parent node
        double[] parentPartialsPerCat = new double[toPartials.length];
        likelihoodCore.calculatePartialsPartialsPruning(fromPartials, fromMatrix,
                                                        toPartials, toMatrix,
                                                        parentPartialsPerCat);

        // Integrate across rate categories using the site model's category proportions
        double[] parentPartials = new double[patternCount * stateCount];
        double[] proportions = getSiteModel().getCategoryProportions(toNode);
        likelihoodCore.calculateIntegratePartials(parentPartialsPerCat, proportions, parentPartials);

        double[] patternLogLikelihoods = new double[patternCount];
        double[] freqs = treelikelihood.getSubstitutionModel().getFrequencies();
        likelihoodCore.calculateLogLikelihoods(parentPartials, freqs, patternLogLikelihoods);

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
	private double getToHeightWithoutNode(Node toNode) {
		if (toNode.isRoot())
			return toNode.getHeight() + hypotheticalEdgeLength;
		Node effectiveParent = getEffectiveParentForWithoutNode(toNode, currentIgnoredNode);
		if (effectiveParent == null)
			return toNode.getHeight() + hypotheticalEdgeLength;
		return (toNode.getHeight() + effectiveParent.getHeight()) / 2.0;
	}

	private Node getEffectiveParentForWithoutNode(Node node, int ignore) {
		if (ignore < 0)
			return node.getParent();
		Node parent = node.getParent();
		while (parent != null &&
				(parent.getLeft().getNr() == ignore || parent.getRight().getNr() == ignore)) {
			parent = parent.getParent();
		}
		return parent;
	}

	public double computeFullConditionalAgreement(Node fromNode, Node toNode, double[] fromFullConditional, double[] toFullConditional) {
		// Set `toHeight` at the midpoint of the edge above `toNode` (root node is a special case)
		double toHeight;
		if (toNode.isRoot())
			toHeight = toNode.getHeight() + hypotheticalEdgeLength;
		else
			toHeight = (toNode.getHeight() + toNode.getParent().getHeight()) / 2.;

		return computeFullConditionalAgreement(fromNode, toNode, fromFullConditional, toFullConditional, toHeight);
	}

	public double computeFullConditionalAgreement(Node fromNode, Node toNode, double[] fromFullConditional, double[] toFullConditional, double toHeight) {
		calculateTransitionMatrix(fromNode, toHeight, fromMatrix);
		calculateTransitionMatrix(toNode, toHeight, toMatrix);

		// Calculate full conditional likelihoods at the potential new parent node
		double[] parentPartialsPerCat = new double[toFullConditional.length];
		likelihoodCore.calculatePartialsPartialsPruning(fromFullConditional, fromMatrix,
														toFullConditional, toMatrix,
														parentPartialsPerCat);

		// Integrate across rate categories using the site model's category proportions
		double[] parentPartials = new double[patternCount * stateCount];
		double[] proportions = getSiteModel().getCategoryProportions(toNode);
		likelihoodCore.calculateIntegratePartials(parentPartialsPerCat, proportions, parentPartials);

		double[] patternLogLikelihoods = new double[patternCount];
		double[] freqs = treelikelihood.getSubstitutionModel().getFrequencies();
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
	public double computeProposedTreeLogLikelihood(Node fromNode, Node toNode,
			double[] fromPartials, double[] toPartials, double toHeight) {

		// Step 1: Compute partials at the hypothetical parent from both children
		calculateTransitionMatrix(fromNode, toHeight, fromMatrix);
		calculateTransitionMatrix(toNode, toHeight, toMatrix);

		double[] hpPartialsPerCat = new double[partialsSize];
		likelihoodCore.calculatePartialsPartialsPruning(fromPartials, fromMatrix,
														toPartials, toMatrix,
														hpPartialsPerCat);

		if (toNode.isRoot()) {
			// toNode is the root: the hypothetical parent becomes the new root.
			// No data above, just integrate and apply frequencies.
			double[] hpPartials = new double[patternCount * stateCount];
			double[] proportions = getSiteModel().getCategoryProportions(toNode);
			likelihoodCore.calculateIntegratePartials(hpPartialsPerCat, proportions, hpPartials);

			double[] patternLogLikelihoods = new double[patternCount];
			double[] freqs = treelikelihood.getSubstitutionModel().getFrequencies();
			likelihoodCore.calculateLogLikelihoods(hpPartials, freqs, patternLogLikelihoods);

			double weightedSum = 0.0;
			for (int p = 0; p < patternCount; p++)
				weightedSum += alignment.getPatternWeight(p) * patternLogLikelihoods[p];
			return weightedSum;
		}

		// Step 2: Compute transition matrix from hypothetical parent up to grandparent.
		// Use toNode's branch rate for this new branch.
		Node grandparent = toNode.getParent();
		double gpHeight = grandparent.getHeight();
		double[] hpToGpMatrix = new double[matrixCount * matrixSize];
		calculateTransitionMatrixForBranch(toHeight, gpHeight, toNode, hpToGpMatrix);

		// Step 3: Get the sibling of toNode and combine at grandparent level
		Node sibling = getOtherChild(toNode);
		double[] gpPartialsPerCat = new double[partialsSize];

		if (sibling != null && sibling.getNr() != currentIgnoredNode) {
			// Normal case: sibling exists and is not the ignored node
			double[] siblingPartials = getPartialsForTargetWeight(sibling.getNr(), currentIgnoredNode);
			double[] siblingMatrix = new double[matrixCount * matrixSize];
			calculateTransitionMatrix(sibling, gpHeight, siblingMatrix);

			likelihoodCore.calculatePartialsPartialsPruning(hpPartialsPerCat, hpToGpMatrix,
															siblingPartials, siblingMatrix,
															gpPartialsPerCat);
		} else {
			// Sibling is the ignored node (fromNode) or doesn't exist: grandparent is
			// effectively unary. Just propagate hpPartialsPerCat up through the transition matrix.
			applyTransitionMatrixToPartials(hpPartialsPerCat, hpToGpMatrix, gpPartialsPerCat);
		}

		// Step 4: Combine with outsidePartials at the grandparent
		double[] outsideGp = outsidePartials[grandparent.getNr()];
		double[] combinedPerCat = new double[partialsSize];
        
        assert (outsideGp != null);

		// if (outsideGp != null) {
        for (int i = 0; i < partialsSize; i++)
            combinedPerCat[i] = gpPartialsPerCat[i] * outsideGp[i];
		// } else {
		// 	// Grandparent is root or outside not computed — just use gpPartialsPerCat
		// 	System.arraycopy(gpPartialsPerCat, 0, combinedPerCat, 0, partialsSize);
		// }

		// Step 5: Integrate across rate categories and compute log-likelihood
		double[] integratedPartials = new double[patternCount * stateCount];
		double[] proportions = getSiteModel().getCategoryProportions(toNode);
		likelihoodCore.calculateIntegratePartials(combinedPerCat, proportions, integratedPartials);

		double[] patternLogLikelihoods = new double[patternCount];
		double[] freqs = treelikelihood.getSubstitutionModel().getFrequencies();
		likelihoodCore.calculateLogLikelihoods(integratedPartials, freqs, patternLogLikelihoods);

		double weightedSum = 0.0;
		for (int p = 0; p < patternCount; p++)
			weightedSum += alignment.getPatternWeight(p) * patternLogLikelihoods[p];
		return weightedSum;
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
		double branchRate = getBranchRate(rateNode);
		for (int i = 0; i < matrixCount; i++) {
			final double jointBranchRate = getSiteModel().getRateForCategory(i, rateNode) * branchRate;
			treelikelihood.getSubstitutionModel().getTransitionProbabilities(
				rateNode,
				parentHeight,
				childHeight,
				jointBranchRate,
				_probabilities
			);
			System.arraycopy(_probabilities, 0, matrix, i * matrixSize, matrixSize);
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
            treelikelihood.getSubstitutionModel().getTransitionProbabilities(
                child,
                toHeight,
                child.getHeight(),
                jointBranchRate,
                _probabilities
            );
            System.arraycopy(_probabilities, 0, matrix, i * matrixSize, matrixSize);
        }
    }

    public SubstitutionModel getSubstitutionModel() {
        return likelihoodInput.get().getSubstitutionModel();
    }

    public SiteModelInterface.Base getSiteModel() {
        return (SiteModelInterface.Base) likelihoodInput.get().siteModelInput.get();
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
