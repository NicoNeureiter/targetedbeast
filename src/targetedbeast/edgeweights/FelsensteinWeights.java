package targetedbeast.edgeweights;

import java.util.*;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.likelihood.LikelihoodCore;
import beast.base.evolution.likelihood.TreeLikelihood;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Distribution;
import beast.base.inference.State;

@Description("Uses Felsenstein partial likelihoods from the tree likelihood calculation as edge weights. "
		+ "The partials at each node represent P(data below | state at node) and can be used to compute "
		+ "distances between nodes that reflect their similarity in terms of the underlying data.")
public class FelsensteinWeights extends Distribution implements EdgeWeights {
	
    final public Input<Alignment> dataInput = new Input<>("data", "sequence data for the beast.tree", Validate.REQUIRED);
    final public Input<TreeInterface> treeInput = new Input<>("tree", "phylogenetic beast.tree with sequence data in the leafs", Validate.REQUIRED);
    final public Input<Double> maxWeightInput = new Input<>("maxWeight", "maximum weight for an edge", 10.0);
    final public Input<Double> minWeightInput = new Input<>("minWeight", "minimum weight for an edge", 0.1);
    final public Input<TreeLikelihood> likelihoodInput = new Input<>("likelihood", "The tree likelihood used for calculating edge weights.", Validate.REQUIRED);

    protected TreeLikelihood treelikelihood;
    protected LikelihoodCore likelihoodCore;
	
	// Leaf partials - stored separately since LikelihoodCore stores states, not partials, for leaves
	private double[][] leafPartials;

	// Partials computed for the "without node" case
	// Dimensions: [nodeCount][partialsSize] - only allocated for nodes that need it
	private double[][] partialsWithoutNode;
	
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
		
		initSequenceID();
	}
	
	/**
	 * Lazy initialization of likelihood core and leaf partials.
	 * Must be called lazily because TreeLikelihood may not be fully initialized during initAndValidate().
	 */
	private void ensureInitialized() {
		if (initialized) return;
		
		likelihoodCore = treelikelihood.getLikelihoodCore();
		partialsSize = likelihoodCore.getPartialsSize();
		if (partialsSize <= 0) {
			// Ensure core has been initialized.
			treelikelihood.calculateLogP();
			partialsSize = likelihoodCore.getPartialsSize();
		}

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
		matrixSize = stateCount * stateCount;
		
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
	private double[] getPartials(int nodeNr) {
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
				
				if (sum1 <= 0 || sum2 <= 0) {
                    continue;
					// throw new IllegalStateException(
					// 	"Partials sum to zero or negative (sum1=" + sum1 + ", sum2=" + sum2 + ") " +
					// 	"at pattern=" + pattern + ", category=" + category + "."
					// );
				}
				
				// Compute dot product for this pattern
				double patternSim = 0.0;
				for (int s = 0; s < stateCount; s++) {
					double p1 = partials1[offset + s] / sum1;
					double p2 = partials2[offset + s] / sum2;
					patternSim += p1 * p2;
				}
				// Add small epsilon for numerical stability
				categoryLogSims[category] += data.getPatternWeight(pattern) * Math.log(patternSim + 1E-7);
			}
		}
		
        return logSumExp(categoryLogSims);
	}

	@Override
    /**
     * Force an update to the weights in the middle of an operator, when the tree has changed from
     * the last likelihood update. Use sparingly, as the likelihood update is expensive.
     */
	public void updateByOperator() {
		likelihoodInput.get().calculateLogP();  // Force full partial recalculation
        partialsWithoutNode = new double[nodeCount][];  // Clear stale "without node" partials
	}

	@Override
	public void updateByOperatorWithoutNode(int ignore, List<Integer> nodes) {
		ensureInitialized();
		// Track which node is being ignored and which nodes will have valid partialsWithoutNode
		currentIgnoredNode = ignore;
		
		// Compute modified partials that exclude the contribution of the ignored node
		Set<Integer> nodesToUpdate = new HashSet<>(nodes);
		updatePartialsWithoutNodeRecursive(treeInput.get().getRoot(), ignore, nodesToUpdate);
	}
	
	/**
	 * Recursively update partials for the "without node" case.
	 * Computes what partials would look like if the ignored node were removed.
	 * 
	 * Uses the Felsenstein pruning algorithm:
	 * P_parent(s) = prod_{c in children} sum_{s'} P(s'|s,t_c) * P_c(s')
	 * 
	 * where P(s'|s,t_c) is the transition probability from state s to s' over branch length t_c.
	 */
	private void updatePartialsWithoutNodeRecursive(Node n, int ignore, Set<Integer> nodesToUpdate) {
		if (n.isLeaf()) {
			return;
		}
		
		// Recurse to children first (post-order traversal)
		for (Node child : n.getChildren()) {
			updatePartialsWithoutNodeRecursive(child, ignore, nodesToUpdate);
		}
		
		final int nodeNr = n.getNr();
		if (!nodesToUpdate.contains(nodeNr)) {
			return;
		}
		
		// Allocate array if needed
		if (partialsWithoutNode[nodeNr] == null) {
			partialsWithoutNode[nodeNr] = new double[partialsSize];
		}
		
		double[] currentPartials = partialsWithoutNode[nodeNr];
		
		// Get children, excluding the ignored node
		List<Node> validChildren = new ArrayList<>();
		for (Node child : n.getChildren()) {
			if (child.getNr() != ignore) {
				validChildren.add(child);
			}
		}
		
		if (validChildren.isEmpty()) {
			// All children are ignored - set uniform partials
			Arrays.fill(currentPartials, 1.0);
		} else {
			// Initialize with 1s for multiplication
			Arrays.fill(currentPartials, 1.0);
			
			// Temporary buffer for single rate category matrix
			double[] matrix = new double[matrixSize];
			
			// For each valid child, apply transition matrix and multiply into result
			for (Node child : validChildren) {
				double[] childPartials = getPartialsForWithoutNode(child.getNr(), ignore, nodesToUpdate);
				if (childPartials != null) {
					// Apply transition matrix for each rate category
					// Matrix layout: matrix[i * stateCount + j] = P(from j -> to i)
					for (int category = 0; category < matrixCount; category++) {
						// Get the transition matrix for this child's branch and rate category
						likelihoodCore.getNodeMatrix(child.getNr(), category, matrix);
						
						for (int pattern = 0; pattern < patternCount; pattern++) {
							int partialsOffset = getPartialOffset(pattern, category);
							
							for (int i = 0; i < stateCount; i++) {
								double sum = 0.0;
								for (int j = 0; j < stateCount; j++) {
									// matrix[i * stateCount + j] = P(from state j to state i)
									sum += matrix[i * stateCount + j] * childPartials[partialsOffset + j];
								}
								currentPartials[partialsOffset + i] *= sum;
							}

							// // Rescale per (pattern, category) to avoid numerical underflow.
							// double rescale = 0.0;
							// for (int i = 0; i < stateCount; i++) {
							// 	rescale += currentPartials[partialsOffset + i];
							// }
							// if (!(rescale > 0.0) || Double.isNaN(rescale) || Double.isInfinite(rescale)) {
							// 	Arrays.fill(currentPartials, partialsOffset, partialsOffset + stateCount, 1.0);
							// } else {
							// 	for (int i = 0; i < stateCount; i++) {
							// 		currentPartials[partialsOffset + i] /= rescale;
							// 	}
							// }
						}
					}
				}
			}
		}
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
	private double[] getPartialsForWithoutNode(int nodeNr, int ignore, Set<Integer> nodesToUpdate) {
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
	private double[] getPartialsForTargetWeight(int nodeNr, int fromNodeNr) {
		// Check if we have valid partialsWithoutNode for this node,
		// computed with fromNodeNr as the ignored node
		if (currentIgnoredNode == fromNodeNr && 
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
		double[] weights = new double[toNodes.size()];
		double[] fromPartials = getPartials(fromNodeNr);
		
		for (int k = 0; k < toNodes.size(); k++) {
			int nodeNo = toNodes.get(k).getNr();
            // Use partialsWithoutNode (excludes fromNodeNr's contribution)
			double[] toPartials = getPartialsForTargetWeight(nodeNo, fromNodeNr);
			weights[k] = computePartialsSimilarity(fromPartials, toPartials);
		}

        // Normalise probabilities
        double[] p = softmax(weights);
        
        // Add epsilon to avoid 0 weights
        for (int i=0; i<p.length; i++)
            p[i] += EPS;

        return p;
    }

	private static final double EPS = 1E-10;
	
	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
		double[] weights = new double[toNodeNrs.size()];
		double[] fromPartials = getPartials(fromNodeNr);
		
		for (int k = 0; k < toNodeNrs.size(); k++) {
			int nodeNo = toNodeNrs.get(k);
            // Use partialsWithoutNode (excludes fromNodeNr's contribution)
            double[] toPartials = getPartialsForTargetWeight(nodeNo, fromNodeNr);
            weights[k] = computePartialsSimilarity(fromPartials, toPartials);
		}

        // Normalise probabilities
        double[] p = softmax(weights);
        
        // Add epsilon to avoid 0 weights
        for (int i=0; i<p.length; i++)
            p[i] += EPS;
        return p;
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
