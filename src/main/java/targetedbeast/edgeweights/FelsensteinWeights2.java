package targetedbeast.edgeweights;

import java.io.PrintStream;
import java.util.*;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.likelihood.BeerLikelihoodCore;
import beast.base.evolution.likelihood.LikelihoodCore;
import beast.base.spec.evolution.likelihood.TreeLikelihood;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Distribution;
import beast.base.inference.State;

@Description("Uses Felsenstein partial likelihoods from the tree likelihood calculation as edge weights. "
		+ "The partials at each node represent P(data below | state at node) and can be used to compute "
		+ "distances between nodes that reflect their similarity in terms of the underlying data. "
		+ "This version caches edge weights for efficiency.")
public class FelsensteinWeights2 extends Distribution implements EdgeWeights {
	
    final public Input<Alignment> dataInput = new Input<>("data", "sequence data for the beast.tree", Validate.REQUIRED);
    
    final public Input<TreeInterface> treeInput = new Input<>("tree", "phylogenetic beast.tree with sequence data in the leafs", Validate.REQUIRED);
    
    final public Input<Double> maxWeightInput = new Input<>("maxWeight", "maximum weight for an edge", 10.0);
    
    final public Input<Double> minWeightInput = new Input<>("minWeight", "minimum weight for an edge", 0.1);

    final public Input<TreeLikelihood> likelihoodInput = new Input<>("likelihood", "The tree likelihood used for calculating edge weights.", Validate.REQUIRED);

	protected int hasDirt;

	// Partials for the "without node" case only - used by updateByOperatorWithoutNode
	// Uses double-buffering for efficient store/restore
	// Dimensions: [2][nodeCount][partialsSize]
	private double[][][] partialsWithoutNode;
	
	// Leaf partials - stored separately since LikelihoodCore stores states, not partials, for leaves
	// Dimensions: [leafNodeCount][partialsSize]
	private double[][] leafPartials;
	
	// Temporary buffer for reading partials from likelihood core
	private double[] tempPartials;

	// Active index for partialsWithoutNode (double-buffering)
	private int[] partialsWithoutNodeIndex;
	private int[] storedPartialsWithoutNodeIndex;
	
	// Track which node was ignored when computing partialsWithoutNode
	private int currentIgnoredNode = -1;
	
	// Edge mutations cache with double-buffering
	public double[][] edgeMutations;
	private int[] activeMutationsIndex;
	private int[] storedActiveMutationsIndex;

    protected TreeLikelihood treelikelihood;
    protected BeerLikelihoodCore likelihoodCore;
    private Alignment alignment;
	
	// For every leaf node, generate a sequence number
	// Leaf nodes with the same sequence have the same number
	private int[] sequenceID;

	private boolean operatorUpdated = false;
	private boolean weightsInitialized = false;

	int stateCount;
	int patternCount;	
	int nodeCount;
	int leafNodeCount;
	int partialsSize;
	int matrixCount; // number of rate categories
	int matrixSize;  // stateCount * stateCount per rate category
	
	double maxWeight;
	double minWeight;
	
	private static final double EPS = 1E-10;
    

	@Override
	public void initAndValidate() {
        treelikelihood = likelihoodInput.get();
        alignment = treelikelihood.dataInput.get();
        if (alignment == null) {
            throw new IllegalArgumentException("TreeLikelihood has no alignment (dataInput is null)");
        }
        if (dataInput.get() != alignment) {
            throw new IllegalArgumentException(
                "FelsensteinWeights2 requires its data input to be the same alignment object as used by the provided TreeLikelihood. " +
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
		
		// Initialize arrays
		edgeMutations = new double[2][nodeCount];
		activeMutationsIndex = new int[nodeCount];
		storedActiveMutationsIndex = new int[nodeCount];
		
		// PartialsWithoutNode - only allocated when needed
		partialsWithoutNode = new double[2][nodeCount][];
		partialsWithoutNodeIndex = new int[nodeCount];
		storedPartialsWithoutNodeIndex = new int[nodeCount];
		
		// Leaf partials - allocated lazily
		leafPartials = new double[leafNodeCount][];
		
		initSequenceID();
		
		// Note: We cannot access the likelihood core partials during initAndValidate
		// because the likelihood has not been computed yet.
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
	 * Ensures the likelihood core is initialized and returns it.
	 */
	private BeerLikelihoodCore getLikelihoodCore() {
		if (likelihoodCore == null) {
            // Ensure likelihood core is BeerLikelihood
            if (treelikelihood.getLikelihoodCore() instanceof BeerLikelihoodCore beerliCore) {
                likelihoodCore = beerliCore;
            } else {
                throw new IllegalArgumentException(
                    "FelsensteinWeights requires TreeLikelihood with BeerLikelihoodCore; got " +
                    (treelikelihood.getLikelihoodCore() == null ? "null" : treelikelihood.getLikelihoodCore().getClass().getName())
                );
            }

            // BEAST3's core BeerLikelihoodCore keeps partialsSize protected and exposes no
            // getPartialsSize() (targetedbeast's own SlowBeerLikelihoodCore adds one). Recompute
            // it exactly as BeerLikelihoodCore.initialize does — patternCount * nrOfStates *
            // matrixCount — from the same public values TreeLikelihood.initCore feeds it.
            treelikelihood.calculateLogP();
			
			// Get matrixCount from SiteModel rather than computing from partialsSize
			SiteModel.Base siteModel = (treelikelihood.siteModelInput.get() instanceof SiteModel.Base)
				? (SiteModel.Base) treelikelihood.siteModelInput.get() : null;
			if (siteModel == null) {
				throw new IllegalStateException(
					"FelsensteinWeights2 requires SiteModel.Base from the TreeLikelihood; got " +
					(treelikelihood.siteModelInput.get() == null ? "null" : treelikelihood.siteModelInput.get().getClass().getName())
				);
			}
			matrixCount = siteModel.getCategoryCount();
			if (matrixCount <= 0) {
				throw new IllegalStateException("SiteModel.getCategoryCount() returned " + matrixCount + ".");
			}
			
			// patternCount is what TreeLikelihood.initCore passes to the core: alignment.getPatternCount()
			patternCount = treelikelihood.dataInput.get().getPatternCount();
			if (patternCount <= 0) {
				throw new IllegalStateException("Alignment reports patternCount " + patternCount + ".");
			}
			partialsSize = patternCount * stateCount * matrixCount;
			matrixSize = stateCount * stateCount;
			
			// Allocate temp buffer for reading partials
			tempPartials = new double[partialsSize];
		}
		return likelihoodCore;
	}

	/**
	 * Get the size of the partials array for a node.
	 */
    protected int getPartialSize() {
        getLikelihoodCore();
        return partialsSize;
    }
    
    /**
     * Ensure weights are initialized by computing edge weights for all nodes.
     * Called on first access after the likelihood has been computed.
     */
    private void ensureWeightsInitialized() {
    	if (!weightsInitialized) {
    		// Initialize leaf partials
    		initializeAllLeafPartials();
    		// Compute initial edge weights
    		computeAllEdgeWeights();
    		weightsInitialized = true;
    	}
    }
    
    /**
     * Initialize partials for all leaf nodes from sequence data.
     */
    private void initializeAllLeafPartials() {
    	getLikelihoodCore(); // ensure partialsSize is set
    	TreeInterface tree = treeInput.get();
    	for (int i = 0; i < leafNodeCount; i++) {
    		if (leafPartials[i] == null) {
    			leafPartials[i] = new double[partialsSize];
    		}
    		initializeLeafPartials(tree.getNode(i), leafPartials[i]);
    	}
    }
    
    /**
     * Compute edge weights for all edges by reading partials from likelihood core.
     */
    private void computeAllEdgeWeights() {
    	TreeInterface tree = treeInput.get();
    	computeEdgeWeightsRecursive(tree.getRoot());
    }
    
    /**
     * Recursively compute edge weights for a subtree.
     */
    private void computeEdgeWeightsRecursive(Node n) {
    	if (n.isLeaf()) {
    		return;
    	}
    	
    	// Recurse to children first
    	computeEdgeWeightsRecursive(n.getLeft());
    	computeEdgeWeightsRecursive(n.getRight());
    	
    	final int nodeNr = n.getNr();
    	final int leftNr = n.getLeft().getNr();
    	final int rightNr = n.getRight().getNr();
    	
    	// Switch to new buffer for edge mutations
    	activeMutationsIndex[leftNr] = 1 - activeMutationsIndex[leftNr];
    	activeMutationsIndex[rightNr] = 1 - activeMutationsIndex[rightNr];
    	
    	// Compute edge weights as divergence between parent and children
    	double leftDivergence = computePartialsDivergenceFromCore(nodeNr, leftNr);
    	double rightDivergence = computePartialsDivergenceFromCore(nodeNr, rightNr);

    	edgeMutations[activeMutationsIndex[leftNr]][leftNr] = Math.max(minWeight, Math.min(maxWeight, leftDivergence));
    	edgeMutations[activeMutationsIndex[rightNr]][rightNr] = Math.max(minWeight, Math.min(maxWeight, rightDivergence));
    }

	@Override
	public void updateByOperator() {
		ensureWeightsInitialized();
		operatorUpdated = true;
		// Recompute edge weights from likelihood core
		computeAllEdgeWeights();
	}

	@Override
	public void updateByOperatorWithoutNode(int ignore, List<Integer> nodes) {
		ensureWeightsInitialized();
		operatorUpdated = true;
		
		// For the "without node" case, we need to compute modified partials
		// that exclude the contribution of the ignored node
		updatePartialsWithoutNode(treeInput.get().getRoot(), ignore, nodes);
	}

	@Override
	public void fakeUpdateByOperator() {
		operatorUpdated = true;
		// Used for operators that change the tree, but without affecting edge weights
	}
	
	/**
	 * Get partials for a node, reading from likelihood core for internal nodes
	 * or from our leaf partials cache for leaves.
	 */
	private double[] getPartialsFromCore(int nodeNr) {
		if (nodeNr < leafNodeCount) {
			// Leaf node - use cached leaf partials
			return leafPartials[nodeNr];
		} else {
			// Internal node - read from likelihood core
			getLikelihoodCore().getNodePartials(nodeNr, tempPartials);
			return tempPartials;
		}
	}
	
	/**
	 * Compute divergence between two nodes by reading partials from likelihood core.
	 */
	private double computePartialsDivergenceFromCore(int parentNr, int childNr) {
		// Need separate buffers since both might be internal nodes
		double[] parentPartials;
		double[] childPartials;
		
        assert parentNr >= leafNodeCount : "Parent node " + parentNr + " cannot be a leaf node";
        parentPartials = new double[partialsSize];
        getLikelihoodCore().getNodePartials(parentNr, parentPartials);
		
		if (childNr < leafNodeCount) {
			childPartials = leafPartials[childNr];
		} else {
			childPartials = new double[partialsSize];
			getLikelihoodCore().getNodePartials(childNr, childPartials);
		}
		
		return computePartialsDivergence(parentPartials, childPartials);
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
		
		getLikelihoodCore(); // ensure matrixCount is set
		
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
					continue; // Skip patterns with zero partials
				}
				
				// Compute Hellinger-like divergence for this pattern
				double patternDiv = 0.0;
				for (int s = 0; s < stateCount; s++) {
					double p = parentPartials[offset + s] / parentSum;
					double q = childPartials[offset + s] / childSum;
					double diff = Math.sqrt(p) - Math.sqrt(q);
					patternDiv += diff * diff;
				}
				categoryDivergence += alignment.getPatternWeight(pattern) * patternDiv;
			}
			
			totalDivergence += categoryDivergence;
		}
		
		return minWeight + totalDivergence / matrixCount;
	}
	
	/**
	 * Get the offset into the partials array for a given pattern and category.
	 * Layout is: [category][pattern][state]
	 */
	private int getPartialOffset(int pattern, int category) {
		return (category * patternCount + pattern) * stateCount;
	}
	
	/**
	 * Initialize partials for a leaf node based on the sequence data.
	 * For each pattern and rate category, we set probability 1.0 for the observed state and 0 for others.
	 * Ambiguous states get equal probability for all compatible states.
	 */
	private void initializeLeafPartials(Node n, double[] partials) {
		Alignment data = alignment;
		int taxonIndex = data.getTaxonIndex(n.getID());
		if (taxonIndex < 0) {
			throw new IllegalArgumentException(
				"Leaf node '" + n.getID() + "' not found in alignment taxa. Available taxa: " + String.join(", ", data.getTaxaNames())
			);
		}
		
		// Ensure matrixCount is initialized
		getLikelihoodCore();
		
		for (int pattern = 0; pattern < patternCount; pattern++) {
			int state = data.getPattern(taxonIndex, pattern);
			
			// Fill in partials for all rate categories
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
	 * Update partials for nodes excluding a specific node's contribution.
	 * Used when computing target weights for operators that need to evaluate
	 * what the partials would look like if a node were removed.
	 * 
	 * @param root The root of the tree
	 * @param ignore The node number to ignore in partial calculations
	 * @param changedNodes List of nodes that have changed and need updating
	 */
	private void updatePartialsWithoutNode(Node root, int ignore, List<Integer> changedNodes) {
		// Track which node is being ignored
		currentIgnoredNode = ignore;
		
		// Mark which nodes need updating based on the path from ignored node to root
		Set<Integer> nodesToUpdate = new HashSet<>(changedNodes);
		
		// Update partials recursively
		updatePartialsWithoutNodeRecursive(root, ignore, nodesToUpdate);
	}
	
	/**
	 * Recursively update partials for the "without node" case.
	 */
	private void updatePartialsWithoutNodeRecursive(Node n, int ignore, Set<Integer> nodesToUpdate) {
		if (n.isLeaf()) {
			return;
		}
		
		// Recurse to children first (post-order traversal)
		updatePartialsWithoutNodeRecursive(n.getLeft(), ignore, nodesToUpdate);
		updatePartialsWithoutNodeRecursive(n.getRight(), ignore, nodesToUpdate);
		
		final int nodeNr = n.getNr();
		if (!nodesToUpdate.contains(nodeNr)) {
			return;
		}
		
		// Switch to new buffer
		partialsWithoutNodeIndex[nodeNr] = 1 - partialsWithoutNodeIndex[nodeNr];
		
		final int leftNr = n.getLeft().getNr();
		final int rightNr = n.getRight().getNr();
		
		// Allocate array if needed
		int activeInd = partialsWithoutNodeIndex[nodeNr];
		if (partialsWithoutNode[activeInd][nodeNr] == null) {
			partialsWithoutNode[activeInd][nodeNr] = new double[partialsSize];
		}
		
		double[] currentPartials = partialsWithoutNode[activeInd][nodeNr];
		
		if (leftNr != ignore && rightNr != ignore) {
			// Both children are valid - combine their partials
			double[] leftPartials = getPartialsForWithoutNode(leftNr, ignore, nodesToUpdate);
			double[] rightPartials = getPartialsForWithoutNode(rightNr, ignore, nodesToUpdate);
			if (leftPartials != null && rightPartials != null) {
				for (int i = 0; i < currentPartials.length; i++) {
					currentPartials[i] = leftPartials[i] * rightPartials[i];
				}
			}
		} else if (leftNr == ignore) {
			// Only use right child
			double[] rightPartials = getPartialsForWithoutNode(rightNr, ignore, nodesToUpdate);
			if (rightPartials != null) {
				System.arraycopy(rightPartials, 0, currentPartials, 0, currentPartials.length);
			}
		} else { // rightNr == ignore
			// Only use left child
			double[] leftPartials = getPartialsForWithoutNode(leftNr, ignore, nodesToUpdate);
			if (leftPartials != null) {
				System.arraycopy(leftPartials, 0, currentPartials, 0, currentPartials.length);
			}
		}
	}
	
	/**
	 * Get partials for the "without node" case.
	 * If the node was updated as part of the "without" calculation, use those partials.
	 * Otherwise, read from the likelihood core.
	 */
	private double[] getPartialsForWithoutNode(int nodeNr, int ignore, Set<Integer> nodesToUpdate) {
		if (nodeNr == ignore) {
			return null;
		}
		if (nodesToUpdate.contains(nodeNr) && nodeNr >= leafNodeCount) {
			// Use the computed "without" partials
			return partialsWithoutNode[partialsWithoutNodeIndex[nodeNr]][nodeNr];
		} else {
			// Use normal partials from core
			return getPartialsFromCore(nodeNr);
		}
	}

	/**
	 * Check state for changed variables and update temp results if necessary.
	 */
	@Override
	protected boolean requiresRecalculation() {
		hasDirt = Tree.IS_CLEAN;

		if (dataInput.get().isDirtyCalculation()) {
			hasDirt = Tree.IS_FILTHY;
			return true;
		}
		
		if (!operatorUpdated) {
			ensureWeightsInitialized();
			// Recompute edge weights from likelihood core
			computeAllEdgeWeights();
		}
		
		operatorUpdated = false;
		return treeInput.get().somethingIsDirty();
	}

	@Override
	public void store() {
		super.store();
		if (!operatorUpdated) {
			System.arraycopy(partialsWithoutNodeIndex, 0, storedPartialsWithoutNodeIndex, 0, partialsWithoutNodeIndex.length);
			System.arraycopy(activeMutationsIndex, 0, storedActiveMutationsIndex, 0, activeMutationsIndex.length);
		}
	}

	@Override
	public void prestore() {
		System.arraycopy(partialsWithoutNodeIndex, 0, storedPartialsWithoutNodeIndex, 0, partialsWithoutNodeIndex.length);
		System.arraycopy(activeMutationsIndex, 0, storedActiveMutationsIndex, 0, activeMutationsIndex.length);
	}

	@Override
	public void reset() {
		// undoes any previous calculation
		System.arraycopy(storedPartialsWithoutNodeIndex, 0, partialsWithoutNodeIndex, 0, partialsWithoutNodeIndex.length);
		System.arraycopy(storedActiveMutationsIndex, 0, activeMutationsIndex, 0, activeMutationsIndex.length);
		operatorUpdated = false;
	}

	@Override
	public void restore() {
		super.restore();
		System.arraycopy(storedPartialsWithoutNodeIndex, 0, partialsWithoutNodeIndex, 0, partialsWithoutNodeIndex.length);
		System.arraycopy(storedActiveMutationsIndex, 0, activeMutationsIndex, 0, activeMutationsIndex.length);
	}

	/**
	 * Checksum over the currently-active cached state (edge mutations and the cached
	 * without-node partials). Lets beast's reject validation (-Dbeast.debug=true) detect a
	 * cache that was not correctly restored after a rejected proposal. Only the active
	 * buffers are hashed -- the inactive/stored halves carry no guarantee.
	 */
	@Override
	protected int getChecksum() {
		int h = 1;
		int n = treeInput.get().getNodeCount();
		for (int nr = 0; nr < n; nr++) {
			h = 31 * h + Double.hashCode(getEdgeMutations(nr));
			h = 31 * h + Arrays.hashCode(partialsWithoutNode[partialsWithoutNodeIndex[nr]][nr]);
		}
		return h;
	}

	public double getEdgeMutations(int i) {
		return edgeMutations[activeMutationsIndex[i]][i];
	}

	/**
	 * Get the partials for a node, reading from likelihood core or leaf cache.
	 */
	public double[] getNodePartials(int nr) {
		if (nr < leafNodeCount) {
			return leafPartials[nr];
		} else {
			// Allocate new array for result since tempPartials gets overwritten
			double[] result = new double[partialsSize];
			getLikelihoodCore().getNodePartials(nr, result);
			return result;
		}
	}

	@Override
	public double getEdgeWeights(int nodeNr) {
		return minWeight + getEdgeMutations(nodeNr);
	}
	
	/**
	 * Get partials of node `nodeNr` ignoring the effect of the subtree below `fromNodeNr`.
	 * Uses partialsWithoutNode if available and computed with the correct ignored node.
	 */
	private double[] getPartialsForTargetWeight(int nodeNr, int fromNodeNr) {
		// Check if we have valid partialsWithoutNode for this node,
		// computed with fromNodeNr as the ignored node
		if (currentIgnoredNode == fromNodeNr && 
			nodeNr >= leafNodeCount &&
			partialsWithoutNode[partialsWithoutNodeIndex[nodeNr]][nodeNr] != null) {
			return partialsWithoutNode[partialsWithoutNodeIndex[nodeNr]][nodeNr];
		}

		// Otherwise use the regular partials
		return getNodePartials(nodeNr);
	}

	@Override	
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes) {
		double[] weights = new double[toNodes.size()];
		double[] fromPartials = getNodePartials(fromNodeNr);
		
		for (int k = 0; k < toNodes.size(); k++) {
			int nodeNo = toNodes.get(k).getNr();
			// Use partialsWithoutNode (excludes fromNodeNr's contribution)
			double[] toPartials = getPartialsForTargetWeight(nodeNo, fromNodeNr);
			weights[k] = computePartialsSimilarity(fromPartials, toPartials);
		}

		// Normalise probabilities
		double[] p = softmax(weights);
		
		// Add epsilon to avoid 0 weights
		for (int i = 0; i < p.length; i++)
			p[i] += EPS;

		return p;
	}
	
	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
		double[] weights = new double[toNodeNrs.size()];
		double[] fromPartials = getNodePartials(fromNodeNr);
		
		for (int k = 0; k < toNodeNrs.size(); k++) {
			int nodeNo = toNodeNrs.get(k);
			// Use partialsWithoutNode (excludes fromNodeNr's contribution)
			double[] toPartials = getPartialsForTargetWeight(nodeNo, fromNodeNr);
			weights[k] = computePartialsSimilarity(fromPartials, toPartials);
		}

		// Normalise probabilities
		double[] p = softmax(weights);
		
		// Add epsilon to avoid 0 weights
		for (int i = 0; i < p.length; i++)
			p[i] += EPS;
		return p;
	}
	
	/**
	 * Compute a similarity measure between the partials of two nodes.
	 * Uses the dot product of normalized partial distributions.
	 * For each rate category, multiplies across patterns (sums log), then averages across categories.
	 */
	private double computePartialsSimilarity(double[] partials1, double[] partials2) {
		// Ensure matrixCount is initialized
		getLikelihoodCore();
		
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
				
				if (sum1 <= 0 || sum2 <= 0) {
					continue;
				}
				
				// Compute dot product for this pattern
				double patternSim = 0.0;
				for (int s = 0; s < stateCount; s++) {
					double p1 = partials1[offset + s] / sum1;
					double p2 = partials2[offset + s] / sum2;
					patternSim += p1 * p2;
				}
				// Add small epsilon for numerical stability
				categoryLogSims[category] += alignment.getPatternWeight(pattern) * Math.log(patternSim + 1E-7);
			}
		}
		
		return logSumExp(categoryLogSims);
	}
	
	// ========== Utility Methods ==========

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
	
	
	@Override
	public void init(PrintStream out) {
		out.println("#NEXUS\n");
		out.println("Begin trees;");
	}

	@Override
	public void log(long sample, PrintStream out) {
		Tree tree = (Tree) treeInput.get();
		out.print("tree STATE_" + sample + " = ");
		final String newick = toNewick(tree.getRoot());
		out.print(newick);
		out.print(";");
	}
	
	public String getTree() {
		Tree tree = (Tree) treeInput.get();
		return toNewick(tree.getRoot());
	}

	/**
	 * Generate a Newick string with edge weights annotated.
	 */
	public String toNewick(Node n) {
		final StringBuilder buf = new StringBuilder();
		if (!n.isLeaf()) {
			buf.append("(");
			boolean isFirst = true;
			for (Node child : n.getChildren()) {
				if (isFirst)
					isFirst = false;
				else
					buf.append(",");
				buf.append(toNewick(child));
			}
			buf.append(")");

			if (n.getID() != null)
				buf.append(n.getID());
		} else {
			if (n.getID() != null)
				buf.append(n.getID());
		}

		final int nodeNr = n.getNr();
		double weight = edgeMutations[activeMutationsIndex[nodeNr]][nodeNr];
		buf.append("[&weight=").append(String.format("%.4f", weight)).append("]");
		buf.append(":").append(n.getLength());

		return buf.toString();
	}

	/**
	 * @see beast.base.core.Loggable
	 */
	@Override
	public void close(PrintStream out) {
		out.print("End;");
	}


	@Override
	public double minEdgeWeight() {
		return minWeight;
	}

} // class FelsensteinWeights
