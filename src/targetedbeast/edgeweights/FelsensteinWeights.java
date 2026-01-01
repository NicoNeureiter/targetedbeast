package targetedbeast.edgeweights;

import java.io.PrintStream;
import java.util.*;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.likelihood.LikelihoodCore;
import beast.base.evolution.likelihood.TreeLikelihood;
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
    
    final public Input<Double> minWeightInput = new Input<>("minWeight", "minimum weight for an edge", 0.0001);

    final public Input<TreeLikelihood> likelihoodInput = new Input<>("likelihood", "The tree likelihood used for calculating edge weights.", Validate.REQUIRED);

	protected int hasDirt;

	// Stores the normalized partials for each node (2 copies for store/restore via active index)
	// Dimensions: [2][nodeCount][partialsSize]
	private double[][][] nodePartials;

	private boolean[] changed;
	private boolean[] changedChildren;

	private int[] activeIndex;
	private int[] storedActiveIndex;

	private int[] activeMutationsIndex;
	private int[] storedActiveMutationsIndex;

    protected TreeLikelihood treelikelihood;
    protected LikelihoodCore likelihoodCore;
	
	// for every leaf node, generate a sequence number
	// leaf nodes with the same sequence have the same number
	private int[] sequenceID;

	public double[][] edgeMutations;

	private boolean operatorUpdated = false;
	private boolean weightsInitialized = false;

	int stateCount;
	int patternCount;	
	int nodeCount;
	int partialsSize;
	int matrixCount; // number of rate categories
	
	double maxWeight;
	double minWeight;

	@Override
	public void initAndValidate() {
        treelikelihood = likelihoodInput.get();

		if (dataInput.get().getTaxonCount() != treeInput.get().getLeafNodeCount()) {
			String leaves = "?";
			if (treeInput.get() instanceof Tree) {
				leaves = String.join(", ", ((Tree) treeInput.get()).getTaxaNames());
			}
			throw new IllegalArgumentException(String.format(
					"The number of leaves in the tree (%d) does not match the number of sequences (%d). "
							+ "The tree has leaves [%s], while the data refers to taxa [%s].",
					treeInput.get().getLeafNodeCount(), dataInput.get().getTaxonCount(), leaves,
					String.join(", ", dataInput.get().getTaxaNames())));
		}

		stateCount = dataInput.get().getMaxStateCount();
		patternCount = dataInput.get().getPatternCount();
		nodeCount = treeInput.get().getNodeCount();
		
		maxWeight = maxWeightInput.get();
		minWeight = minWeightInput.get();
		
		// Initialize arrays
		edgeMutations = new double[2][nodeCount];
		nodePartials = new double[2][nodeCount][];

		activeIndex = new int[nodeCount];
		storedActiveIndex = new int[nodeCount];

		activeMutationsIndex = new int[nodeCount];
		storedActiveMutationsIndex = new int[nodeCount];

		changed = new boolean[nodeCount];
		changedChildren = new boolean[nodeCount];
		
		initSequenceID();
		
		// Note: We cannot access the likelihood core partials during initAndValidate
		// because the likelihood has not been computed yet. The partials will be 
		// retrieved during the first call to updateByOperator() or calculateLogP().
	}

	
	private void initSequenceID() {
		TreeInterface tree = treeInput.get();
		Alignment data = dataInput.get();
		
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
	private LikelihoodCore getLikelihoodCore() {
		if (likelihoodCore == null) {
			likelihoodCore = treelikelihood.getLikelihoodCore();
			partialsSize = likelihoodCore.getPartialsSize();
			// Compute matrixCount (number of rate categories) from partialsSize
			// partialsSize = patternCount * stateCount * matrixCount
			matrixCount = partialsSize / (patternCount * stateCount);
		}
		return likelihoodCore;
	}

	/**
	 * Get the size of the partials array for a node.
	 */
    protected int getPartialSize() {
        return getLikelihoodCore().getPartialsSize();
    }
    
    /**
     * Ensure weights are initialized by computing partials for all nodes.
     * Called on first access after the likelihood has been computed.
     */
    private void ensureWeightsInitialized() {
    	if (!weightsInitialized) {
    		// First time - compute weights for all nodes
    		Arrays.fill(changed, true);
    		Arrays.fill(changedChildren, true);
    		updateNodePartials(treeInput.get().getRoot());
    		weightsInitialized = true;
    	}
    }

	@Override
	public void updateByOperator() {
		ensureWeightsInitialized();
		operatorUpdated = true;
		Arrays.fill(changed, false);
		Arrays.fill(changedChildren, false);
		getFilthyNodes(treeInput.get().getRoot());
		updateNodePartials(treeInput.get().getRoot());
	}

	@Override
	public void updateByOperatorWithoutNode(int ignore, List<Integer> nodes) {
		ensureWeightsInitialized();
		Arrays.fill(changed, false);
		Arrays.fill(changedChildren, false);
		for (Integer nodeNo : nodes) {
			changed[nodeNo] = true;
		}
		updatePartialsWithoutNode(treeInput.get().getRoot(), ignore);
	}

	@Override
	public void fakeUpdateByOperator() {
		operatorUpdated = true;
		// used for operators that change the tree, but without affecting the order of
		// the patterns
	}

	private boolean getFilthyNodes(Node node) {
		// compute the number of patterns for each node.
		if (node.isLeaf()) {
			return false;
		} else {
			boolean left = getFilthyNodes(node.getLeft());
			boolean right = getFilthyNodes(node.getRight());
			if (left || right || node.isDirty() > 1) {
				changed[node.getNr()] = true;
				if (node.isDirty() == 3) { 
					return false;
				}
			}
		}
		return changed[node.getNr()];
	}

	/**
	 * Update node partials from the likelihood core and compute edge weights.
	 * The partials P(data below | state at node) are retrieved from the tree likelihood.
	 * Edge weights are computed as divergence measures between parent and child partials.
	 */
	private void updateNodePartials(Node n) {
		final int nodeNr = n.getNr();
		
		if (n.isLeaf()) {
			// For leaf nodes, we don't have partials in the usual sense 
			// (they have states, not partials), but we can still compute
			// a normalized distribution for each pattern based on the data
			if (changed[nodeNr] || nodePartials[activeIndex[nodeNr]][nodeNr] == null) {
				// Note: We do NOT call treelikelihood.calculateLogP() here as that would
				// corrupt the cached likelihood state during operator proposals.
				// The partials should already be computed from previous MCMC steps.
				
				activeIndex[nodeNr] = 1 - activeIndex[nodeNr];
				int activeInd = activeIndex[nodeNr];
				
				if (nodePartials[activeInd][nodeNr] == null) {
					nodePartials[activeInd][nodeNr] = new double[getPartialSize()];
				}
				
				// For leaves, create partials from the state information
				// This is a simplified representation - actual partials for leaves
				// are handled specially in the likelihood calculation
				initializeLeafPartials(n, nodePartials[activeInd][nodeNr]);
			}
			return;
		}
		
		if (changed[nodeNr]) {
			// Recurse to children first
			updateNodePartials(n.getLeft());
			updateNodePartials(n.getRight());

			final int leftNr = n.getLeft().getNr();
			final int rightNr = n.getRight().getNr();

			// Note: We do NOT call treelikelihood.calculateLogP() here as that would
			// corrupt the cached likelihood state during operator proposals.
			// The partials should already be computed from previous MCMC steps.

			// Switch to new active index for this node
			activeIndex[nodeNr] = 1 - activeIndex[nodeNr];
			activeMutationsIndex[leftNr] = 1 - activeMutationsIndex[leftNr];
			activeMutationsIndex[rightNr] = 1 - activeMutationsIndex[rightNr];

			int activeInd = activeIndex[nodeNr];
			
			// Allocate array if needed
			if (nodePartials[activeInd][nodeNr] == null) {
				nodePartials[activeInd][nodeNr] = new double[getPartialSize()];
			}
			
			// Get partials from tree likelihood for this internal node
			getLikelihoodCore().getNodePartials(nodeNr, nodePartials[activeInd][nodeNr]);
			
			// Compute edge weights as divergence between parent and children
			double leftDivergence = computePartialsDivergence(nodeNr, leftNr);
			double rightDivergence = computePartialsDivergence(nodeNr, rightNr);
			
			edgeMutations[activeMutationsIndex[leftNr]][leftNr] = Math.max(minWeight, Math.min(maxWeight, leftDivergence));
			edgeMutations[activeMutationsIndex[rightNr]][rightNr] = Math.max(minWeight, Math.min(maxWeight, rightDivergence));
		} else {
			updateNodePartials(n.getLeft());
			updateNodePartials(n.getRight());
		}
	}
	
	/**
	 * Initialize partials for a leaf node based on the sequence data.
	 * For each pattern and rate category, we set probability 1.0 for the observed state and 0 for others.
	 * Ambiguous states get equal probability for all compatible states.
	 * 
	 * The partials array is organized as [pattern][category][state], flattened to:
	 * partials[pattern * stateCount * matrixCount + category * stateCount + state]
	 */
	private void initializeLeafPartials(Node n, double[] partials) {
		Alignment data = dataInput.get();
		int nodeNr = n.getNr();
		
		// Ensure matrixCount is initialized
		getLikelihoodCore();
		
		for (int pattern = 0; pattern < patternCount; pattern++) {
			int state = data.getPattern(nodeNr, pattern);
			
			// Fill in partials for all rate categories
			for (int category = 0; category < matrixCount; category++) {
				int offset = (pattern * matrixCount + category) * stateCount;
				
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
	 * Compute a divergence measure between the partials of two nodes.
	 * Uses a Hellinger-like distance which is symmetric and bounded.
	 * Returns a value that is higher when the partials are more different.
	 * 
	 * The partials array is organized as [pattern][category][state], flattened.
	 * We integrate over rate categories by summing partials across categories.
	 */
	private double computePartialsDivergence(int parentNr, int childNr) {
		double[] parentPartials = nodePartials[activeIndex[parentNr]][parentNr];
		double[] childPartials = nodePartials[activeIndex[childNr]][childNr];
		
		if (parentPartials == null || childPartials == null) {
			return minWeight;
		}
		
		// Ensure matrixCount is initialized
		getLikelihoodCore();
		
		double totalDivergence = 0.0;
		
		// Compute divergence for each pattern (using a simplified Hellinger-like measure)
		// that works well with partial likelihoods
		for (int pattern = 0; pattern < patternCount; pattern++) {
			// Sum partials across all rate categories for this pattern
			double parentSum = 0.0;
			double childSum = 0.0;
			double[] parentPatternPartials = new double[stateCount];
			double[] childPatternPartials = new double[stateCount];
			
			for (int category = 0; category < matrixCount; category++) {
				int offset = (pattern * matrixCount + category) * stateCount;
				for (int s = 0; s < stateCount; s++) {
					parentPatternPartials[s] += parentPartials[offset + s];
					childPatternPartials[s] += childPartials[offset + s];
					parentSum += parentPartials[offset + s];
					childSum += childPartials[offset + s];
				}
			}
			
			if (parentSum > 0 && childSum > 0) {
				// Compute Hellinger-like distance: sum of (sqrt(p) - sqrt(q))^2
				double patternDiv = 0.0;
				for (int s = 0; s < stateCount; s++) {
					double p = parentPatternPartials[s] / parentSum;
					double q = childPatternPartials[s] / childSum;
					double diff = Math.sqrt(p) - Math.sqrt(q);
					patternDiv += diff * diff;
				}
				totalDivergence += patternDiv;
			}
		}
		
		// Scale the divergence to a reasonable range
		return totalDivergence;
	}
	
	/**
	 * Update partials for a node, ignoring a specific node in the computation.
	 * Used when computing target weights for operators.
	 */
	private void updatePartialsWithoutNode(Node n, int ignore) {
		if (n.isLeaf()) {
			return;
		}
		
		final int nodeNr = n.getNr();
		if (changed[nodeNr]) {
			activeIndex[nodeNr] = 1 - activeIndex[nodeNr];

			final int leftNr = n.getLeft().getNr();
			final int rightNr = n.getRight().getNr();

			int activeInd = activeIndex[nodeNr];
			int activeIndLeft = activeIndex[leftNr];
			int activeIndRight = activeIndex[rightNr];
			
			if (nodePartials[activeInd][nodeNr] == null) {
				nodePartials[activeInd][nodeNr] = new double[getPartialSize()];
			}
			
			double[] currentPartials = nodePartials[activeInd][nodeNr];
			
			if (leftNr != ignore && rightNr != ignore) {
				// Both children are valid - combine their partials
				double[] leftPartials = nodePartials[activeIndLeft][leftNr];
				double[] rightPartials = nodePartials[activeIndRight][rightNr];
				if (leftPartials != null && rightPartials != null) {
					for (int i = 0; i < currentPartials.length; i++) {
						currentPartials[i] = leftPartials[i] * rightPartials[i];
					}
				}
			} else if (leftNr == ignore) {
				// Only use right child
				double[] rightPartials = nodePartials[activeIndRight][rightNr];
				if (rightPartials != null) {
					System.arraycopy(rightPartials, 0, currentPartials, 0, currentPartials.length);
				}
			} else if (rightNr == ignore) {
				// Only use left child
				double[] leftPartials = nodePartials[activeIndLeft][leftNr];
				if (leftPartials != null) {
					System.arraycopy(leftPartials, 0, currentPartials, 0, currentPartials.length);
				}
			}
		}
		updatePartialsWithoutNode(n.getLeft(), ignore);
		updatePartialsWithoutNode(n.getRight(), ignore);
	}

	/**
	 * check state for changed variables and update temp results if necessary *
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
			Arrays.fill(changed, false);
			Arrays.fill(changedChildren, false);
			getFilthyNodes(treeInput.get().getRoot());
			updateNodePartials(treeInput.get().getRoot());
		}
		
		operatorUpdated = false;
		return treeInput.get().somethingIsDirty();
	}

	@Override
	public void store() {
		super.store();
		if (!operatorUpdated) { // avoid storing again if the operator has already done it
			if (operatorUpdated) {
				operatorUpdated = false;
				return;
			}
			System.arraycopy(activeIndex, 0, storedActiveIndex, 0, activeIndex.length);
			System.arraycopy(activeMutationsIndex, 0, storedActiveMutationsIndex, 0, activeMutationsIndex.length);
		}
	}

	@Override
	public void prestore() {
		System.arraycopy(activeIndex, 0, storedActiveIndex, 0, activeIndex.length);
		System.arraycopy(activeMutationsIndex, 0, storedActiveMutationsIndex, 0, activeMutationsIndex.length);
	}

	@Override
	public void reset() {
		// undoes any previous calculation
		System.arraycopy(storedActiveIndex, 0, activeIndex, 0, activeIndex.length);
		System.arraycopy(storedActiveMutationsIndex, 0, activeMutationsIndex, 0, activeMutationsIndex.length);
		operatorUpdated = false;
	}

	@Override
	public void restore() {
		super.restore();
		System.arraycopy(storedActiveIndex, 0, activeIndex, 0, activeIndex.length);
		System.arraycopy(storedActiveMutationsIndex, 0, activeMutationsIndex, 0, activeMutationsIndex.length);
	}

	public double getEdgeMutations(int i) {
		return edgeMutations[activeMutationsIndex[i]][i];
	}
	
	public boolean getChanged(int i) {
		return changed[i];
	}

	/**
	 * Get the stored partials for a node.
	 */
	public double[] getNodePartials(int nr) {
		return nodePartials[activeIndex[nr]][nr];
	}

	@Override
	public double getEdgeWeights(int nodeNr) {
		return minWeight + getEdgeMutations(nodeNr);
	}

	@Override	
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodeNrs) {
		double[] weights = new double[toNodeNrs.size()];
		double[] fromPartials = getNodePartials(fromNodeNr);
		
		for (int k = 0; k < toNodeNrs.size(); k++) {
			int nodeNo = toNodeNrs.get(k).getNr();
			if (sequenceID[nodeNo] == sequenceID[fromNodeNr]) {
				// Same sequence - maximum weight
				weights[k] = maxWeight;				
			} else {
				double[] toPartials = getNodePartials(nodeNo);
				// Calculate similarity between partials (inverse of divergence)
				double similarity = computePartialsSimilarity(fromPartials, toPartials);
				weights[k] = similarity;
			}
		}
		return weights;
	}
	
	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
		double[] weights = new double[toNodeNrs.size()];
		double[] fromPartials = getNodePartials(fromNodeNr);
		
		for (int k = 0; k < toNodeNrs.size(); k++) {
			int nodeNo = toNodeNrs.get(k);
			if (sequenceID[nodeNo] == sequenceID[fromNodeNr]) {
				// Same sequence - maximum weight
				weights[k] = maxWeight;
			} else {
				double[] toPartials = getNodePartials(nodeNo);
				// Calculate similarity between partials (inverse of divergence)
				double similarity = computePartialsSimilarity(fromPartials, toPartials);
				weights[k] = similarity;
			}
		}		
		return weights;
	}
	
	/**
	 * Compute a similarity measure between the partials of two nodes.
	 * Uses the dot product of normalized partial distributions, which gives
	 * higher values when the partials are more similar.
	 * 
	 * The partials array is organized as [pattern][category][state], flattened.
	 * We integrate over rate categories by summing partials across categories.
	 */
	private double computePartialsSimilarity(double[] partials1, double[] partials2) {
		// if (partials1 == null || partials2 == null) {
		// 	return minWeight;
		// }
		
		// Ensure matrixCount is initialized
		getLikelihoodCore();
		
		double totalSimilarity = 0.0;
		
		// Compute similarity for each pattern using Bhattacharyya coefficient
		for (int pattern = 0; pattern < patternCount; pattern++) {
			// Sum partials across all rate categories for this pattern
			double sum1 = 0.0;
			double sum2 = 0.0;
			double[] patternPartials1 = new double[stateCount];
			double[] patternPartials2 = new double[stateCount];
			
			for (int category = 0; category < matrixCount; category++) {
				int offset = (pattern * matrixCount + category) * stateCount;
				for (int s = 0; s < stateCount; s++) {
					patternPartials1[s] += partials1[offset + s];
					patternPartials2[s] += partials2[offset + s];
					sum1 += partials1[offset + s];
					sum2 += partials2[offset + s];
				}
			}
			
			if (sum1 > 0 && sum2 > 0) {
				// Compute Bhattacharyya coefficient
				double patternSim = 0.0;
				for (int s = 0; s < stateCount; s++) {
					double p1 = patternPartials1[s] / sum1;
					double p2 = patternPartials2[s] / sum2;
					patternSim += Math.sqrt(p1 * p2);
				}
				totalSimilarity += patternSim;
			}
		}
		
		// Normalize to [0, 1] range and apply min weight
		double normalizedSimilarity = totalSimilarity / patternCount;
		return Math.max(minWeight, normalizedSimilarity);
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
