package targetedbeast.edgeweights;

import java.io.PrintStream;
import java.util.*;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.core.Log;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Distribution;
import beast.base.inference.State;
import targetedbeast.util.Alignment2PCA;

@Description("Keeps track of the distances in PCA space. "
		+ "PCAWeights is a distribution to ensure that it is updated correctly")
public class PCAWeightsBrownian extends Distribution implements EdgeWeights {
	private static final double BROWNIAN_SCALE = 1.0;
	private static final double VARIANCE_FLOOR = 1e-12;
	
    final public Input<Alignment> dataInput = new Input<>("data", "sequence data for the beast.tree", Validate.REQUIRED);
    
    final public Input<TreeInterface> treeInput = new Input<>("tree", "phylogenetic beast.tree with sequence data in the leafs", Validate.REQUIRED);
    
    final public Input<Double> maxWeightInput = new Input<>("maxWeight", "maximum weight for an edge", 10.0);
    
    final public Input<Double> minWeightInput = new Input<>("minWeight", "maximum weight for an edge", 0.0);

	final public Input<Integer> dimensionInput = new Input<>("dimension", "dimension of PCA points", 2);

    final public Input<String> valueInput = new Input<>("value", "comma separated list of taxon=x1 x2 x3 pairs, "
    		+ "where <dimension> number of dimensions are specified, e.g. taxon1=0.2 0.4, taxon2=0.4 0.3, taxon3=0.9 0.1. "
    		+ "If value is specified, data is ignored");

	final public Input<Boolean> distanceBasedInput = new Input<>("distanceBased", "flag to indicate PCA is done based on distance matrix (if true) or normalised alignment (if false)", true);
	final public Input<Boolean> compressedInput = new Input<>("compressed", "flag to indicate matrix should have remove duplicate entries(if true) or leave then in, which is slower (if false)", true);
	final public Input<Boolean> useOneNormInput = new Input<>("useOneNorm", "flag to indicate distance uses one norm (if true) or two norm (if false)", true);
	final public Input<Double> offsetInput = new Input<>("offset", "offset in weight", 0.01);


	protected int hasDirt;

	// 2 x nr of taxa x dimensions
	private double[][][] points;
	private double[][][] variances;

	private boolean[] changed;
	private boolean[] changedChildren;

	private int[] activeIndex;
	private int[] storedActiveIndex;

	private int[] activeMutationsIndex;
	private int[] storedActiveMutationsIndex;
	
	public double[][] edgeMutations;
	
	private boolean operatorUpdated = false;
	private double offset;
	private boolean useOneNorm;

	int stateCount;
	int patternCount;	
	int maxStateCount;
	
	double maxWeight;
	double minWeight;
	
	double totalMuts[];

	private int dim;

	@Override
	public void initAndValidate() {
		dim = dimensionInput.get();
		offset = offsetInput.get();
		useOneNorm = useOneNormInput.get();
		
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
		maxStateCount = dataInput.get().getMaxStateCount();
				
		edgeMutations = new double[2][treeInput.get().getNodeCount()];
		// should probably be changes to a non double
		points = new double[2][treeInput.get().getNodeCount()][dimensionInput.get()];
		variances = new double[2][treeInput.get().getNodeCount()][dimensionInput.get()];
		if (valueInput.get() != null) {
			parseValue();
		} else {
			calcValue();
		}

		activeIndex = new int[treeInput.get().getNodeCount()];
		storedActiveIndex = new int[treeInput.get().getNodeCount()];

		activeMutationsIndex = new int[treeInput.get().getNodeCount()];
		storedActiveMutationsIndex = new int[treeInput.get().getNodeCount()];
		

		changed = new boolean[treeInput.get().getNodeCount()];
		changedChildren = new boolean[treeInput.get().getNodeCount()];
			
		maxWeight = maxWeightInput.get();
		minWeight = minWeightInput.get();
		
		totalMuts = new double[patternCount];

		updateWeights();
	}

	
	private void calcValue() {
		Map<String, double[]> map = Alignment2PCA.getPoints(dataInput.get(), 
				dimensionInput.get(), 
				distanceBasedInput.get(),
				compressedInput.get());

		int taxonCount = treeInput.get().getLeafNodeCount();
		int dim = dimensionInput.get();
		// List<String> taxa = treeInput.get().getTaxonset().asStringList();
        String[] taxa = ((Tree) treeInput.get()).getTaxaNames();
		for (int i = 0; i < taxonCount; i++) {
			String taxon = taxa[i];
			double [] p = map.get(taxon);
			for (int j = 0; j < dim; j++) {
				points[0][i][j] = p[j];
			}
		}
	}
	

	private void parseValue() {
		String [] strs0 = valueInput.get().split(",");
        List<String> labels = treeInput.get().getTaxonset().asStringList();
        int dimension = dimensionInput.get();
		if (strs0.length != labels.size()) {
			Log.warning("Number of points specified (" + strs0.length +" should equal number of taxa (" + labels.size()+")");
		}
		
		for  (int i = 0; i < strs0.length; i++) {
			String trait = strs0[i];
            trait = trait.replaceAll("\\s+", " ");
            String[] strs = trait.split("=");
            if (strs.length != 2) {
                throw new IllegalArgumentException("could not parse trait: " + trait);
            }
            String taxonID = normalize(strs[0]);
            int taxonNr = labels.indexOf(taxonID);
            
            String [] point = strs[1].split("\\s+");
            if (point.length != dimension) {
                throw new IllegalArgumentException("could not parse trait: " + trait + " since dimension is not " + dimension);
            }
            for (int j = 0; j < dimension; j++) {
            	points[0][taxonNr][j] = Double.parseDouble(point[j]);
            }
		}
	}

	/**
     * remove start and end spaces
     */
    protected String normalize(String str) {
        if (str.charAt(0) == ' ') {
            str = str.substring(1);
        }
        if (str.endsWith(" ")) {
            str = str.substring(0, str.length() - 1);
        }
        return str;
    }

	private void updateWeights() {
		Arrays.fill(changed, true);
		Arrays.fill(changedChildren, true);
		getNodeConsensusSequences(treeInput.get().getRoot());
	}

	@Override
	public void updateByOperator() {
		operatorUpdated = true;
		Arrays.fill(changed, false);
		Arrays.fill(changedChildren, false);
		getFilthyNodes(treeInput.get().getRoot());
		getNodeConsensusSequences(treeInput.get().getRoot());
	}

    Node getOtherChild(final Node parent, final Node child) {
        if (parent.getLeft().getNr() == child.getNr()) {
            return parent.getRight();
        } else {
            return parent.getLeft();
        }
    }

    Node getSibling(final Node child) {
        return getOtherChild(child.getParent(), child);
    }


	@Override
	public void updateByOperatorWithoutNode(int ignore, List<Integer> nodes) {
        updateByOperator();

		Arrays.fill(changed, false);
		Arrays.fill(changedChildren, false);
		for (Integer nodeNo : nodes) {
			changed[nodeNo] = true;
		}

		Node ancestor = treeInput.get().getNode(ignore).getParent();
		while (ancestor != null) {
			changed[ancestor.getNr()] = true;
			ancestor = ancestor.getParent();
		}
        getConsensusWithoutNode(treeInput.get().getRoot(), ignore);
	}

    public double mean(double[][] matrix) {
        double sum = 0;
        double n = 0;
        for (double[] row : matrix) {
            for (double val : row) {
                sum += val;
                n++;
            }
        }
        return sum / n;
    }

    public double mean(double[][][] arr) {
        double sum = 0;
        double n = 0;
        for (double[][] matrix : arr) {
            for (double[] row : matrix) {
                for (double val : row) {
                    sum += val;
                    n++;
                }
            }
        }
        return sum / n;
    }

    public double[][] activePoints(double[][][] points, int[] activeIndex) {
        double[][] a = new double[points[0].length][];

        for (int i = 0; i < points[0].length; i++) {
            a[i] = points[activeIndex[i]][i];
        }

        return a;
    }
    
    public static double[][] deepCopy(double[][] matrix) {
        final double[][] result = new double[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            result[i] = Arrays.copyOf(matrix[i], matrix[i].length);
        }
        return result;
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
				} else {
				}
			}
		}

		return changed[node.getNr()];
	}

	private void getNodeConsensusSequences(Node n) {
		if (n.isLeaf()) {
			// the active index for leaves is always 0, could be changed to make the arrays
			// shorter
			return;
		} else {
			final int nodeNr = n.getNr();
            // compare the patterns of the two lineages
            getNodeConsensusSequences(n.getLeft());
            getNodeConsensusSequences(n.getRight());

			if (changed[nodeNr]) {
				
				final int leftNr = n.getLeft().getNr();
				final int rightNr = n.getRight().getNr();

				// set this node index to the active index
				activeIndex[nodeNr] = 1 - activeIndex[nodeNr];
				activeMutationsIndex[leftNr] = 1 - activeMutationsIndex[leftNr];
				activeMutationsIndex[rightNr] = 1 - activeMutationsIndex[rightNr];

				int activeInd = activeIndex[nodeNr];
				int activeIndLeft = activeIndex[leftNr];
				int activeIndRight = activeIndex[rightNr];
				double sumLeft = 0;
				double sumRight = 0;
				
				final double[] leftconsensus = points[activeIndLeft][leftNr];
				final double[] rightconsensus = points[activeIndRight][rightNr];
				final double[] leftVariance = variances[activeIndLeft][leftNr];
				final double[] rightVariance = variances[activeIndRight][rightNr];
				final double [] currentconsensus = points[activeInd][nodeNr];
				final double[] currentVariance = variances[activeInd][nodeNr];

				updateParentPosterior(n.getLeft(), n.getRight(), leftconsensus, rightconsensus,
						leftVariance, rightVariance, currentconsensus, currentVariance);

				for (int i = 0; i < dim; i++) {
					sumLeft += norm(leftconsensus[i] - currentconsensus[i]);
					sumRight += norm(rightconsensus[i] - currentconsensus[i]);
				}
				edgeMutations[activeMutationsIndex[leftNr]][leftNr] = Math.min(maxWeight, Math.sqrt(sumLeft));
				edgeMutations[activeMutationsIndex[rightNr]][rightNr] = Math.min(maxWeight, Math.sqrt(sumRight));				
			}
		}
	}

	private void getConsensusWithoutNode(Node n, int ignore) {
		if (n.isLeaf()) {
			return;
		} else {
            
			getConsensusWithoutNode(n.getLeft(), ignore);
			getConsensusWithoutNode(n.getRight(), ignore);

			final int nodeNr = n.getNr();
			if (changed[nodeNr]) {

				// set this node index to the active index
				activeIndex[nodeNr] = 1 - activeIndex[nodeNr];

				final int leftNr = n.getLeft().getNr();
				final int rightNr = n.getRight().getNr();

				int activeInd = activeIndex[nodeNr];
				int activeIndLeft = activeIndex[leftNr];
				int activeIndRight = activeIndex[rightNr];

				final double[] leftconsensus = points[activeIndLeft][leftNr]; 
				final double[] rightconsensus = points[activeIndRight][rightNr];
				final double[] leftVariance = variances[activeIndLeft][leftNr];
				final double[] rightVariance = variances[activeIndRight][rightNr];
				final double [] currentconsensus = points[activeInd][nodeNr];
				final double[] currentVariance = variances[activeInd][nodeNr];
				
				if (leftNr != ignore && rightNr != ignore) {
					updateParentPosterior(n.getLeft(), n.getRight(), leftconsensus, rightconsensus,
							leftVariance, rightVariance, currentconsensus, currentVariance);
				} else if (leftNr == ignore) {
					copyChildPosteriorToParent(n.getRight(), rightconsensus, rightVariance, currentconsensus, currentVariance);
				} else if (rightNr == ignore) {
					copyChildPosteriorToParent(n.getLeft(), leftconsensus, leftVariance, currentconsensus, currentVariance);
				}
			}

			return;
		}
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
			Arrays.fill(changed, false);
			Arrays.fill(changedChildren, false);
			getFilthyNodes(treeInput.get().getRoot());
			getNodeConsensusSequences(treeInput.get().getRoot());
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

//	public void unstore() {
//	}

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

//	public byte[] getConsensus(int nr) {
//		return consensus[activeIndex[nr]][nr];
//	}


	public double[] getPoints(int nr) {
		return points[activeIndex[nr]][nr];
	}

	double[] getVariances(int nr) {
		return variances[activeIndex[nr]][nr];
	}
	
	@Override
	public double getEdgeWeights(int nodeNr) {
		return getEdgeMutations(nodeNr);
	}

	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes) {
        return getTargetWeightsInteger(fromNodeNr,  toNodes.stream().map(n -> n.getNr()).toList());
		// double[] distances = new double[toNodes.size()];
		// double[] currConsensus = getPoints(fromNodeNr);
		
		// for (int k = 0; k < toNodes.size(); k++) {
        //     Node toNode = toNodes.get(k);
		// 	int nodeNo = toNode.getNr();
        //     double sum = offset;
		// 	double[] consensus = getPoints(nodeNo);
            
        //     // Node parent = toNode.getParent();
        //     // if (parent != null) {
        //     //     double[] parentConsensus = getPoints(parent.getNr());
        //     //     consensus = interpolatePoints(consensus, parentConsensus);
        //     // }
                

		// 	// calculate the distance between the two consensus
		// 	for (int l = 0; l < dim; l++) {
		// 		sum += norm(currConsensus[l] - consensus[l]);
		// 	}
		// 	distances[k] = 1 / invNorm(sum);
		// }
		// return distances;
	}

	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
		// double[] logScores = new double[toNodeNrs.size()];
		// for (int k = 0; k < toNodeNrs.size(); k++) {
		// 	int nodeNr = toNodeNrs.get(k);
		// 	double toHeight = getDefaultAttachmentHeight(getNode(nodeNr), fromNodeNr);
		// 	logScores[k] = getAttachmentLogScore(fromNodeNr, nodeNr, fromNodeNr, toHeight);
		// }
		// return exponentiateLogScores(logScores);
        double[] distances = new double[toNodeNrs.size()];
		for (int k = 0; k < toNodeNrs.size(); k++) {
			int nodeNr = toNodeNrs.get(k);
			double toHeight = getDefaultAttachmentHeight(getNode(nodeNr), fromNodeNr);
            distances[k] = 1. / getMeanDist(fromNodeNr, nodeNr, fromNodeNr, toHeight);
        }
        return distances;
	}

    @Override
    public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes, double toHeight) {
        return getTargetWeightsInteger(fromNodeNr, toNodes.stream().map(Node::getNr).toList(), toHeight);
    }

    @Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, double toHeight) {
		// double[] logScores = new double[toNodeNrs.size()];
		double[] distances = new double[toNodeNrs.size()];
		for (int k = 0; k < toNodeNrs.size(); k++) {
			// logScores[k] = getAttachmentLogScore(fromNodeNr, toNodeNrs.get(k), fromNodeNr, toHeight);
            distances[k] = 1. / getMeanDist(fromNodeNr, toNodeNrs.get(k), fromNodeNr, toHeight);
		}
		// return exponentiateLogScores(logScores);
		return distances;
	}

    @Override
    public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes, List<Double> toHeights) {
        return getTargetWeightsInteger(fromNodeNr, toNodes.stream().map(Node::getNr).toList(), toHeights);
    }

    @Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, List<Double> toHeights) {
		// double[] logScores = new double[toNodeNrs.size()];
		double[] distances = new double[toNodeNrs.size()];
		for (int k = 0; k < toNodeNrs.size(); k++) {
			// logScores[k] = getAttachmentLogScore(fromNodeNr, toNodeNrs.get(k), fromNodeNr, toHeights.get(k));
            distances[k] = 1. / getMeanDist(fromNodeNr, toNodeNrs.get(k), fromNodeNr, toHeights.get(k));
		}
		// return exponentiateLogScores(logScores);
		return distances;
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

	boolean becomesUnary(Node node, int ignore) {
		if (ignore < 0 || node == null || node.isLeaf()) {
			return false;
		}
		return node.getLeft().getNr() == ignore || node.getRight().getNr() == ignore;
	}

	private void updateParentPosterior(Node leftChild, Node rightChild,
			double[] leftMean, double[] rightMean,
			double[] leftVar, double[] rightVar,
			double[] parentMean, double[] parentVar) {
		for (int i = 0; i < dim; i++) {
			double propagatedLeftVar = propagatedVariance(leftVar[i], leftChild.getLength());
			double propagatedRightVar = propagatedVariance(rightVar[i], rightChild.getLength());
			double leftPrecision = 1.0 / propagatedLeftVar;
			double rightPrecision = 1.0 / propagatedRightVar;
			double precisionSum = leftPrecision + rightPrecision;
			parentVar[i] = Math.max(1.0 / precisionSum, VARIANCE_FLOOR);
			parentMean[i] = parentVar[i] * (leftMean[i] * leftPrecision + rightMean[i] * rightPrecision);
		}
	}

	private void copyChildPosteriorToParent(Node child, double[] childMean, double[] childVar,
			double[] parentMean, double[] parentVar) {
		for (int i = 0; i < dim; i++) {
			parentMean[i] = childMean[i];
			parentVar[i] = propagatedVariance(childVar[i], child.getLength());
		}
	}

	private double propagatedVariance(double variance, double branchLength) {
		return Math.max(variance + BROWNIAN_SCALE * Math.max(branchLength, 0.0), VARIANCE_FLOOR);
	}

	private double getDefaultAttachmentHeight(Node node, int ignore) {
		Node parent = getEffectiveParent(node, ignore);
		double minHeight = Math.max(node.getHeight(), getNode(ignore).getHeight());
		if (parent == null) {
			return minHeight + 0.1;
		}
		return 0.5 * (minHeight + parent.getHeight());
	}

	private double getAttachmentLogScore(int fromNodeNr, int toNodeNr, int ignore, double toHeight) {
		GaussianState sourceState = getSourceStateAtHeight(fromNodeNr, toHeight);
		GaussianState targetState = getAttachmentState(getNode(toNodeNr), ignore, toHeight);
		double logScore = 0.0;
		for (int i = 0; i < dim; i++) {
			double combinedVariance = Math.max(sourceState.variance[i] + targetState.variance[i], VARIANCE_FLOOR);
			double diff = sourceState.mean[i] - targetState.mean[i];
			logScore += -0.5 * (Math.log(combinedVariance) + (diff * diff) / combinedVariance);
		}
		return logScore;
	}

	private double getMeanDist(int fromNodeNr, int toNodeNr, int ignore, double toHeight) {
		GaussianState sourceState = getSourceStateAtHeight(fromNodeNr, toHeight);
		GaussianState targetState = getAttachmentState(getNode(toNodeNr), ignore, toHeight);
        // Node toNodeParent = getEffectiveParent(getNode(toNodeNr), ignore);
        // GaussianState targetParentState = getAttachmentState(toNodeParent, ignore, toHeight);

		double dist = 1E-12;
		for (int i = 0; i < dim; i++) {
            // double targetInterpMean = 0.5 * targetState.mean[i] + 0.5 * targetParentState.mean[i];
			// dist += sqr(sourceState.mean[i] - targetInterpMean);
			dist += sqr(sourceState.mean[i] - targetState.mean[i]);
		}
		return Math.sqrt(dist / dim);
	}

    public double sqr(double x) {
        return x * x;
    }

	private GaussianState getSourceStateAtHeight(int fromNodeNr, double toHeight) {
		double[] mean = Arrays.copyOf(getPoints(fromNodeNr), dim);
		double[] variance = Arrays.copyOf(getVariances(fromNodeNr), dim);
		double branchLength = Math.max(0.0, toHeight - getNode(fromNodeNr).getHeight());
		for (int i = 0; i < dim; i++) {
			variance[i] = propagatedVariance(variance[i], branchLength);
		}
		return new GaussianState(mean, variance);
	}

	private GaussianState getAttachmentState(Node node, int ignore, double toHeight) {
		double[] childMean = getPoints(node.getNr());
		double[] childVariance = getVariances(node.getNr());
		double childBranchLength = Math.max(0.0, toHeight - node.getHeight());

		Node parent = getEffectiveParent(node, ignore);
		if (parent == null) {
			double[] mean = Arrays.copyOf(childMean, dim);
			double[] variance = Arrays.copyOf(childVariance, dim);
			for (int i = 0; i < dim; i++) {
				variance[i] = propagatedVariance(variance[i], childBranchLength);
			}
			return new GaussianState(mean, variance);
		}

		double[] attachmentMean = new double[dim];
		double[] attachmentVariance = new double[dim];
		double[] parentMean = getPoints(parent.getNr());
		double[] parentVariance = getVariances(parent.getNr());
		double parentBranchLength = Math.max(0.0, parent.getHeight() - toHeight);

		for (int i = 0; i < dim; i++) {
			double childVarAtAttachment = propagatedVariance(childVariance[i], childBranchLength);
			double parentVarAtAttachment = propagatedVariance(parentVariance[i], parentBranchLength);
			double childPrecision = 1.0 / childVarAtAttachment;
			double parentPrecision = 1.0 / parentVarAtAttachment;
			double precisionSum = childPrecision + parentPrecision;
			attachmentVariance[i] = Math.max(1.0 / precisionSum, VARIANCE_FLOOR);
			attachmentMean[i] = attachmentVariance[i] *
					(childMean[i] * childPrecision + parentMean[i] * parentPrecision);
		}

		return new GaussianState(attachmentMean, attachmentVariance);
	}

	private double[] exponentiateLogScores(double[] logScores) {
		double[] scores = new double[logScores.length];
		double maxLogScore = Double.NEGATIVE_INFINITY;
		for (double logScore : logScores) {
			if (logScore > maxLogScore) {
				maxLogScore = logScore;
			}
		}
		if (!Double.isFinite(maxLogScore)) {
			Arrays.fill(scores, 1.0);
			return scores;
		}
		for (int i = 0; i < logScores.length; i++) {
			scores[i] = Math.max(Math.exp(logScores[i] - maxLogScore), VARIANCE_FLOOR) + offset;
		}
		return scores;
	}

	private static final class GaussianState {
		private final double[] mean;
		private final double[] variance;

		private GaussianState(double[] mean, double[] variance) {
			this.mean = mean;
			this.variance = variance;
		}
	}

    double[] interpolatePoints(double[] p1, double[] p2) {
        return interpolatePoints(p1, p2, 0.5);
    }
	
    double[] interpolatePoints(double[] p1, double[] p2, double w1) {
        double[] p = new double[p1.length];
        for (int i = 0; i < p1.length; i++) {
            p[i] = w1 * p1[i] + (1 - w1) * p2[i];
        }
        return p;
    }


    private double norm(double x) {
        if (useOneNorm)
            return Math.abs(x);
        else 
            return  x * x;
    }
	@Override
	public List<String> getArguments() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public List<String> getConditions() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public void sample(State state, Random random) {
		// TODO Auto-generated method stub
		
	}
	
	
	@Override
	public void init(PrintStream out) {
//    	out.print("mutations\t");
//        Node node = treeInput.get().getRoot();
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
		
		// calculate the total number of mutations
//		double totalMutations = 0;
//		for (int i = 0; i < tree.getNodeCount(); i++) {
//			if (tree.getNode(i).isRoot())
//				continue;
//			totalMutations += edgeMutations[activeMutationsIndex[i]][i];
//		}
	}

    public Node getNode(int i) {
        return treeInput.get().getNode(i);
    }
	
	public Tree getTree() {
		Tree tree = (Tree) treeInput.get();
		return tree;
	}

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
		if (!n.isRoot()) {
			buf.append(
					"[&sum=" +  edgeMutations[activeMutationsIndex[nodeNr]][nodeNr]);
			buf.append("]");
		} else {
			buf.append("[&sum=" + edgeMutations[activeMutationsIndex[nodeNr]][nodeNr] + "]");
		}
		buf.append(":").append(n.getLength());

		return buf.toString();
	}

	/**
	 * @see beast.base.core.Loggable *
	 */
	@Override
	public void close(PrintStream out) {
		out.print("End;");
	}


	@Override
	public double minEdgeWeight() {
		return minWeight;
	}

} 
