package targetedbeast.edgeweights;

import java.util.*;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.branchratemodel.BranchRateModel;
import beast.base.evolution.likelihood.LikelihoodCore;
import beast.base.evolution.likelihood.TreeLikelihood;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.sitemodel.SiteModelInterface;
import beast.base.evolution.substitutionmodel.SubstitutionModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Distribution;
import beast.base.inference.State;
import targetedbeast.util.LinearAlgebra;

@Description("Uses Felsenstein partial likelihoods from the tree likelihood calculation as edge weights. "
		+ "The partials at each node represent P(data below | state at node) and can be used to compute "
		+ "distances between nodes that reflect their similarity in terms of the underlying data.")
public class PartialWeights extends Distribution implements EdgeWeights {
	
    final public Input<Alignment> dataInput = new Input<>("data", "sequence data for the beast.tree", Validate.REQUIRED);
    final public Input<Tree> treeInput = new Input<>("tree", "phylogenetic beast.tree with sequence data in the leafs", Validate.REQUIRED);
    final public Input<Double> maxWeightInput = new Input<>("maxWeight", "maximum weight for an edge", Double.POSITIVE_INFINITY);
    final public Input<Double> minWeightInput = new Input<>("minWeight", "minimum weight for an edge", 1E-10);
	final public Input<Boolean> logSpaceInput = new Input<>("logSpace",
			"if true, the target weights are returned as raw log-weights instead of being passed "
					+ "through a softmax, so that a log-space WeightTransform can do the conversion "
					+ "to real space with an adaptable temperature", false);
	final public Input<TreeLikelihood> likelihoodInput = new Input<>("likelihood", "The tree likelihood used for calculating edge weights.", Validate.REQUIRED);

	protected TreeLikelihood treelikelihood;
	protected LikelihoodCore likelihoodCore;
	
	// Leaf partials - stored separately since LikelihoodCore stores states, not partials, for leaves
	private double[][] leafPartials;

	// Partials computed for the "without node" case
	// Dimensions: [nodeCount][partialsSize] - only allocated for nodes that need it
	private double[][] partialsWithoutNode;
	private boolean[] partialsWithoutNodeValid;
	
	// Track which node was ignored when computing partialsWithoutNode
	private int currentIgnoredNode = -1;

	// The pruned tree partialsWithoutNode were computed on, by core label; cachedFrom is null when
	// there are none
	private Node cachedFrom;
	private final Map<Node, Integer> coreLabels = new IdentityHashMap<>();
	private Node[] prunedParents;
	private double[] prunedHeights;
	
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
    double[] _targetPartials;
    double[] _patternLogLikelihoods;
    
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
		partialsWithoutNodeValid = new boolean[nodeCount];
		prunedParents = new Node[nodeCount];
		prunedHeights = new double[nodeCount];
		leafPartials = new double[leafNodeCount][];
		
		initSequenceID();
	}
	
	/**
	 * Lazy initialization of likelihood core and leaf partials.
	 * Must be called lazily because TreeLikelihood may not be fully initialized during initAndValidate().
	 */
	public void ensureInitialized() {
		if (initialized) return;

		treelikelihood.calculateLogP();

		likelihoodCore = treelikelihood.getLikelihoodCore();
		if (likelihoodCore == null) {
			throw new IllegalArgumentException(
				"PartialWeights requires a TreeLikelihood-compatible object exposing getLikelihoodCore(); got " +
				treelikelihood.getClass().getName()
			);
		}

		// Require a SiteModel.Base to obtain category count; do not derive
		matrixCount = getSiteModel().getCategoryCount();
		if (matrixCount <= 0) {
			throw new IllegalStateException("SiteModel.getCategoryCount() returned " + matrixCount + ".");
		}

		matrixSize = (stateCount) * (stateCount);
		partialsSize = patternCount * stateCount * matrixCount;
		if (partialsSize <= 0) {
			throw new IllegalStateException(
				"Invalid partial dimensions: patternCount=" + patternCount +
				", stateCount=" + stateCount + ", matrixCount=" + matrixCount
			);
		}
		
		// Initialize leaf partials from sequence data
		TreeInterface tree = treeInput.get();
		for (int i = 0; i < leafNodeCount; i++) {
			leafPartials[i] = new double[partialsSize];
			initializeLeafPartials(tree.getNode(i), leafPartials[i]);
		}

        for (int i = 0; i < nodeCount; i++) {
            if (partialsWithoutNode[i] == null) {
                partialsWithoutNode[i] = new double[partialsSize];
                partialsWithoutNodeValid[i] = false;
            }
        }

        if (_probabilities == null)
            _probabilities = new double[matrixSize];
        if (_fromMatrix == null)
            _fromMatrix = new double[matrixCount * matrixSize];
        if (_toMatrix == null)
            _toMatrix = new double[matrixCount * matrixSize];
        if (_targetPartials == null)
            _targetPartials = new double[patternCount * stateCount];
        if (_patternLogLikelihoods == null)
            _patternLogLikelihoods = new double[patternCount];
		
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
			likelihoodCore.getNodePartials(coreLabel(nodeNr), partials);
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
	
	@Override
    /**
     * Force an update to the weights in the middle of an operator, when the tree has changed from
     * the last likelihood update. Use sparingly, as the likelihood update is expensive.
     */
	public void updateByOperator() {
		// Source weights are branch lengths, so there is nothing to update.
	}

	/**
	 * Computes the partials without {@code ignore} for all its ancestors; {@code nodes} is unused.
	 * They depend only on the pruned tree, which a move of {@code ignore} leaves unchanged, so the
	 * reverse pass reuses the forward pass's values: the likelihood core still holds the old tree.
	 */
	@Override
	public void updateByOperatorWithoutNode(int ignore, List<Integer> nodes) {
		ensureInitialized();
		currentIgnoredNode = ignore;
		Node from = treeInput.get().getNode(ignore);
		if (from == cachedFrom && prunedTreeUnchanged(from)) {
			return;
		}
		forgetPartialsWithoutNode();
		Arrays.fill(partialsWithoutNodeValid, false);

		Set<Integer> nodeSet = new HashSet<>();
		for (Node a = from.getParent(); a != null; a = a.getParent()) {
			nodeSet.add(a.getNr());
		}
        Node root = treeInput.get().getRoot();
        Node effectiveRoot = becomesUnary(root, ignore) ? getIncludedChild(root, ignore) : root;
        updatePartialsWithoutNodeRecursive(effectiveRoot, ignore, nodeSet);
		recordPrunedTree(from);
    }

	/** Forgets the cached partials without a node; call whenever the likelihood core may have changed. */
	void forgetPartialsWithoutNode() {
		cachedFrom = null;
		coreLabels.clear();
	}

	/** Number under which the core and the cache hold {@code nodeNr}'s partials; differs after a root change. */
	private int coreLabel(int nodeNr) {
		if (cachedFrom == null) {
			return nodeNr;
		}
		return coreLabels.get(treeInput.get().getNode(nodeNr));
	}

	private void recordPrunedTree(Node from) {
		Node parent = from.getParent();
		for (Node n : treeInput.get().getNodesAsArray()) {
			coreLabels.put(n, n.getNr());
			prunedParents[n.getNr()] = prunedParent(n, parent);
			prunedHeights[n.getNr()] = n.getHeight();
		}
		cachedFrom = from;
	}

	private boolean prunedTreeUnchanged(Node from) {
		Node parent = from.getParent();
		for (Node n : treeInput.get().getNodesAsArray()) {
			if (n == parent || n == from) {
				continue;
			}
			int label = coreLabels.get(n);
			if (prunedParents[label] != prunedParent(n, parent) || prunedHeights[label] != n.getHeight()) {
				return false;
			}
		}
		return true;
	}

	/** n's parent once {@code pruned} is removed, null for the root. */
	private static Node prunedParent(Node n, Node pruned) {
		Node p = n.getParent();
		return p == pruned ? p.getParent() : p;
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

		// Get children, excluding the ignored node
		Node child1 = getEffectiveChild(n, 0, ignore);
        Node child2 = getEffectiveChild(n, 1, ignore);

		assert (ignore==-1) || (child1.isLeaf() && getTree().getNode(ignore).isLeaf()) || child1.getNr() != ignore;
		assert (ignore==-1) || (child2.isLeaf() && getTree().getNode(ignore).isLeaf()) || child2.getNr() != ignore;
        
        // Update child partials recursively
        updatePartialsWithoutNodeRecursive(child1, ignore, nodesToUpdate);
        updatePartialsWithoutNodeRecursive(child2, ignore, nodesToUpdate);

		// Only materialize partials at requested nodes
		if (!nodesToUpdate.contains(nodeNr)) {
			return;
		}
        
        // Multiply the up-partials to get the parent partial
        double[] upPartials1 = applyTransitionUpWithoutNode(child1, n);
        double[] upPartials2 = applyTransitionUpWithoutNode(child2, n);
        LinearAlgebra.multiply(upPartials1, upPartials2, partialsWithoutNode[nodeNr]);
        partialsWithoutNodeValid[nodeNr] = true;
	}
	
	/**
	 * Apply transition matrix to partials (single-child pruning).
	 */
	public void applyTransitionMatrix(int childNodeNr, double[] childPartials, double[] outPartials) {
        ensureInitialized();
        double[] matrix = new double[matrixSize];
		for (int category = 0; category < matrixCount; category++) {
			likelihoodCore.getNodeMatrix(childNodeNr, category, matrix);
			for (int pattern = 0; pattern < patternCount; pattern++) {
				int w = 0;
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

	private double[] getNodeMatrices(int nodeNr) {
		double[] matrices = new double[matrixCount * matrixSize];
		for (int category = 0; category < matrixCount; category++) {
			likelihoodCore.getNodeMatrix(nodeNr, category, _probabilities);
			System.arraycopy(_probabilities, 0, matrices, category * matrixSize, matrixSize);
		}
		return matrices;
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

    int getPartialOffset(int pattern, int category) {
        return (category * patternCount + pattern) * stateCount;
    }
	
    /**
	 * Get partials for the "without node" case.
	 * If the node was updated as part of the "without" calculation, use those partials.
	 * Otherwise, read from the likelihood core.
	 */
	public double[] getPartialsWithoutNode(int nodeNr) {
		int label = coreLabel(nodeNr);
		if (partialsWithoutNodeValid[label])
            return partialsWithoutNode[label];
		else
            return getPartials(nodeNr);
	}

	@Override
	public void fakeUpdateByOperator() {
		// Used for operators that change the tree without affecting edge weights
	}

	@Override
	protected boolean requiresRecalculation() {
		return getTree().somethingIsDirty() || likelihoodInput.get().isDirtyCalculation();
	}

	@Override
	public void store() {
		// Called before the likelihood recomputes the core's partials.
		forgetPartialsWithoutNode();
		super.store();
	}

	@Override
	public void prestore() {
		// No caching, nothing to prestore
	}

	@Override
	public void reset() {
		// No caching, nothing to reset
        // getTree().setEverythingDirty(true);
        // likelihoodInput.get().calculateLogP();
	}

	@Override
	public void restore() {
		forgetPartialsWithoutNode();
		super.restore();
	}

	@Override
	public double getEdgeWeights(int nodeNr) {
        return 0.1 + getTree().getNode(nodeNr).getLength();
        // // Get parent node
        // Node parent = getTree().getNode(nodeNr).getParent();
        // if (parent == null)
        //     return minWeight;  // Return minimum weight for the root node
		// int parentNr = parent.getNr();

        // // Get partials
        // double[] parentPartials = getPartials(parentNr);
        // double[] childPartials = getPartials(nodeNr);

        // // Compute divergence
        // double divergence = computePartialsDivergence(parentPartials, childPartials);
        // // double divergence = getAveragePartialDiff(parentPartials, childPartials);

        // // Clip and return
        // return Math.max(minWeight, Math.min(maxWeight, divergence));
	}

	@Override	
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes) {
        return getTargetWeightsInteger(fromNodeNr,  toNodes.stream().map(n -> n.getNr()).toList());
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
            Node toNode = resolveScoringNode(treeInput.get().getNode(toNodeNr), fromNode);

            // Cannot move to edges that end below fromNode
            double parentHeight = toNode.isRoot() ? Double.POSITIVE_INFINITY : toNode.getParent().getHeight();
            if (parentHeight <= fromNode.getHeight()) {
                weights[k] = Double.NEGATIVE_INFINITY;
                continue;
            }

            // Use partialsWithoutNode (excludes fromNodeNr's contribution)
			double[] toPartials = getPartialsWithoutNode(toNode.getNr());
			double newHeight = getToHeightWithoutNode(toNode, fromNode);

			// // Compute the full proposed tree log-likelihood
			double toWeight = computePartialsAgreement(fromNode, toNode, fromPartials, toPartials, newHeight);
			weights[k] = EPS + toWeight / 2.0; // division by two to counteract squaring of probabilities in the operators
            // double partialDistance = getAveragePartialDiff(fromPartials, toPartials);
            // weights[k] = 1. / partialDistance;
		}

        weights = clipMaxToSecondMaxDiff(weights, 2.);
        // In log-space mode a WeightTransform performs the exponentiation instead, with an
        // adaptable temperature; normalize(softmax(x)^b) == softmax(b*x), so the two modes
        // differ only in who chooses b.
        return logSpaceInput.get() ? weights : softmax(weights);

        // weights = clipMaxToSecondMaxRatio(weights, 10.);
        // return weights;
	}

	/**
	 * Scores fromNode's own parent p as p's other child: p's partials include fromNode, and
	 * attaching above p is the same as attaching above that child.
	 */
	private Node resolveScoringNode(Node toNode, Node fromNode) {
		Node parent = fromNode.getParent();
		if (toNode != parent) {
			return toNode;
		}
		return parent.getLeft() == fromNode ? parent.getRight() : parent.getLeft();
	}

    double[] clipMaxToSecondMaxRatio(double[] weights, double maxRatio) {
		if (weights == null || weights.length < 2) {
			return weights;
		}

		int firstIdx = -1;
		int secondIdx = -1;
		double first = Double.NEGATIVE_INFINITY;
		double second = Double.NEGATIVE_INFINITY;

		for (int i = 0; i < weights.length; i++) {
			double w = weights[i];
			if (w > first) {
				second = first;
				secondIdx = firstIdx;
				first = w;
				firstIdx = i;
			} else if (w > second) {
				second = w;
				secondIdx = i;
			}
		}

		if (firstIdx < 0 || secondIdx < 0 || first <= maxRatio * second) {
			return weights;
		}

		double[] clipped = Arrays.copyOf(weights, weights.length);
		clipped[firstIdx] = maxRatio * second;
		return clipped;
    }

    double[] clipMaxToSecondMaxDiff(final double[] weights, final double maxDiff) {
		if (weights == null || weights.length < 2) {
			return weights;
		}

		int firstIdx = -1;
		int secondIdx = -1;
		double first = Double.NEGATIVE_INFINITY;
		double second = Double.NEGATIVE_INFINITY;

		for (int i = 0; i < weights.length; i++) {
			double w = weights[i];
			if (w > first) {
				second = first;
				secondIdx = firstIdx;
				first = w;
				firstIdx = i;
			} else if (w > second) {
				second = w;
				secondIdx = i;
			}
		}

		if (firstIdx < 0 || secondIdx < 0 || first <= second + maxDiff) {
			return weights;
		}

		double[] clipped = Arrays.copyOf(weights, weights.length);
		clipped[firstIdx] = second + maxDiff;
		return clipped;
    }

	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, double newHeight) {

		// Use conventional partials for the fromNode (the same for all toNodes)
        Node fromNode = treeInput.get().getNode(fromNodeNr);
        double[] fromPartials = getPartials(fromNodeNr);

        // Compute the potential likelihood for all toNode candidates
		double[] weights = new double[toNodeNrs.size()];
		for (int k = 0; k < toNodeNrs.size(); k++) {
			int toNodeNr = toNodeNrs.get(k);
            Node toNode = resolveScoringNode(treeInput.get().getNode(toNodeNr), fromNode);

            // Use partialsWithoutNode (excludes fromNodeNr's contribution)
			double[] toPartials = getPartialsWithoutNode(toNode.getNr());

			// Compute the full proposed tree log-likelihood
			double toWeight = computePartialsAgreement(fromNode, toNode, fromPartials, toPartials, newHeight);
			weights[k] = EPS + toWeight / 2.0; // division by two to counteract squaring of probabilities in the operators

            // double partialDistance = getAveragePartialDiff(fromPartials, toPartials);
            // weights[k] = 1. / partialDistance;
		}

        weights = clipMaxToSecondMaxDiff(weights, 2.);
        // In log-space mode a WeightTransform performs the exponentiation instead, with an
        // adaptable temperature; normalize(softmax(x)^b) == softmax(b*x), so the two modes
        // differ only in who chooses b.
        return logSpaceInput.get() ? weights : softmax(weights);

        // weights = clipMaxToSecondMaxRatio(weights, 10.);
        // return weights;
	}

    public double getAveragePartialDiff(double[] fromPartial, double[] toPartial) {
        double totalDiff = EPS;
        int offset = 0;
        double wSum = 0.0;
		// for (int i = 0; i < matrixCount * patternCount; i++) {
        for (int category = 0; category < matrixCount; category++) {
            for (int pattern = 0; pattern < patternCount; pattern++) {
                double w = alignment.getPatternWeight(pattern);

                // int offset = i * stateCount;
                double toPartialSum = 0;
                double fromPartialSum = 0;
                for (int j = offset; j < offset + stateCount; j++) {
                    fromPartialSum += fromPartial[j];
                    toPartialSum += toPartial[j];
                }
                if (toPartialSum <= 0.0 || fromPartialSum <= 0.0) {
                    continue;
                } else {
                    for (int j = offset; j < offset + stateCount; j++) {
                        double diff = (toPartial[j] / toPartialSum) - (fromPartial[j] / fromPartialSum);
                        totalDiff += w * Math.abs(diff);
                        wSum += w;
                    }
                }

                offset += stateCount; 
            }
        }
        // return totalDiff / (matrixCount * patternCount);
        return totalDiff / wSum;
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
            // double toPartialSum = 1E-6;
            // double fromPartialSum = 1E-6;
            double toPartialSum = 0;
            double fromPartialSum = 0;
            for (int j = offset; j < offset + stateCount; j++) {
                toPartialSum += toPartialsAtParent[j];
                fromPartialSum += fromPartialsAtParent[j];
            }
            if (toPartialSum <= EPS || fromPartialSum <= EPS)
                for (int j = offset; j < offset + stateCount; j++)
                    targetPartialsPerCat[j] = 1.0 / stateCount / stateCount;
            else
                for (int j = offset; j < offset + stateCount; j++)
                    targetPartialsPerCat[j] = (toPartialsAtParent[j] / toPartialSum) * (fromPartialsAtParent[j] / fromPartialSum);
        }

        return computeLikelihoodFromPartials(toNode, targetPartialsPerCat);
    }


    /**
     * Integrate across rate categories and compute log-likelihood
     * @param node The node at which the partials are computed (should already incorporate outside partials for data above)
     * @param combinedPartials Partials at the node, already multiplied by outside partials for data above
     * @return log-likelihood of the tree given the partials at the specified node (which should already incorporate outside partials for data above)
     */
    public double computeLikelihoodFromPartials(Node node, double[] combinedPartials) {
		double[] proportions = getSiteModel().getCategoryProportions(node);
		integratePartials(combinedPartials, proportions, _targetPartials);

		double[] freqs = getSubstitutionModel().getFrequencies();
		likelihoodCore.calculateLogLikelihoods(_targetPartials, freqs, _patternLogLikelihoods);

		double weightedSum = 0.0;
		for (int p = 0; p < patternCount; p++)
			weightedSum += alignment.getPatternWeight(p) * _patternLogLikelihoods[p];

		return weightedSum;
    }

	private void integratePartials(double[] inPartials, double[] proportions, double[] outPartials) {
		int u = 0;
		int v = 0;
		for (int k = 0; k < patternCount; k++) {
			for (int i = 0; i < stateCount; i++) {
				outPartials[u] = inPartials[v] * proportions[0];
				u++;
				v++;
			}
		}

		for (int l = 1; l < matrixCount; l++) {
			u = 0;
			for (int k = 0; k < patternCount; k++) {
				for (int i = 0; i < stateCount; i++) {
					outPartials[u] += inPartials[v] * proportions[l];
					u++;
					v++;
				}
			}
		}
	}

	/**
	 * Determine the effective parent height for `toNode` when the current ignored node
	 * is removed. Unary nodes created by the ignored child become transparent.
	 */
	public double getToHeightWithoutNode(Node toNode, Node fromNode) {
        double minHeight = Math.max(toNode.getHeight(), fromNode.getHeight());
		Node effectiveParent = getEffectiveParent(toNode, currentIgnoredNode);

        if (effectiveParent == null) // toNode is effectively root (when ignoring currentIgnoredNode)
			return minHeight + 0.1;

        assert effectiveParent.getHeight() > fromNode.getHeight();
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

    void applyTransitionUpWithoutNode(Node child, Node parent, double[] outPartials) {
        double[] childPartial = getPartialsWithoutNode(child.getNr());
        double[] matrix = getTransitionMatrixWithoutNode(child, parent);
        applyTransitionMatrix(matrix, childPartial, outPartials);
    }

    double[] applyTransitionUpWithoutNode(Node child, Node parent) {
        double[] outPartials = new double[partialsSize];
        applyTransitionUpWithoutNode(child, parent, outPartials);
        return outPartials;
    }

    double[] getTransitionMatrixWithoutNode(Node child, Node effectiveParent) {
        if (child.getParent() == effectiveParent) {
            // `child`/`parent` are direct child/parent in original tree -> use the stored transition matrix
			return getNodeMatrices(child.getNr());
        } else {
            // `child`/`parent` are only connected via and ignored node -> calculate transition matrix in pruned tree
            // Important because the `toHeight` is different to the one in the original tree
            return calculateTransitionMatrix(child, effectiveParent.getHeight());
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
		return getSiteModel().substModelInput.get();
    }

    public SiteModelInterface.Base getSiteModel() {
        if (treelikelihood.siteModelInput.get() instanceof SiteModel.Base siteModel)
			return siteModel;
		else
			throw new IllegalStateException(
				"PartialWeights requires SiteModel.Base from the TreeLikelihood; got " + treelikelihood.siteModelInput.get().getClass().getName()
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
	public boolean isLogSpace() {
		return logSpaceInput.get();
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
