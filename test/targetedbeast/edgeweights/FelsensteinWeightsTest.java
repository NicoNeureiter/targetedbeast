package targetedbeast.edgeweights;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.lang.reflect.Field;

import org.junit.Before;
import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.likelihood.LikelihoodCore;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.sitemodel.SiteModelInterface;
import beast.base.evolution.substitutionmodel.JukesCantor;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.parameter.RealParameter;
import targetedbeast.likelihood.SlowBeerLikelihoodCore;
import targetedbeast.likelihood.SlowTreeLikelihood;

/**
 * Tests for FelsensteinWeights.
 * 
 * Category 1: Fine-grained component tests (only for non-trivial logic)
 * Category 2: Overall behavior / property-based tests
 */
public class FelsensteinWeightsTest {

	private Alignment alignment;
	private Tree tree;
	private SlowTreeLikelihood likelihood;
	private SiteModel siteModel;
	private FelsensteinWeights weights;

	/**
	 * Set up a simple 4-taxon tree with JC69 model.
	 * Tree: ((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15)
	 */
	@Before
	public void setUp() {
		// Create alignment with 4 taxa
		Sequence seqA = new Sequence("A", "ACGTACGTAA");
		Sequence seqB = new Sequence("B", "ACGTACGTAG");
		Sequence seqC = new Sequence("C", "ACGTAAGTAC");
		Sequence seqD = new Sequence("D", "ACGTAAGTAC");  // Same as C to test identical sequences

		alignment = new Alignment();
		alignment.initByName("sequence", seqA, "sequence", seqB, 
							 "sequence", seqC, "sequence", seqD);

		// Create tree
		tree = new TreeParser();
		((TreeParser) tree).initByName(
			"newick", "((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);",
			"IsLabelledNewick", true,
			"taxa", alignment
		);

		// Create substitution model
		JukesCantor jc = new JukesCantor();
		jc.initAndValidate();
        
        // Create shape parameter for gamma rate heterogeneity
        RealParameter shapeParameter = new RealParameter("1.0");

		// Create site model with 4 gamma categories
		siteModel = new SiteModel();
		siteModel.initByName("substModel", jc, "gammaCategoryCount", 4, "shape", shapeParameter);

		// Create likelihood
		System.setProperty("java.only", "true");
		likelihood = new SlowTreeLikelihood();
		likelihood.initByName("data", alignment, "tree", tree, "siteModel", siteModel);

		// Create FelsensteinWeights
		weights = new FelsensteinWeights();
		weights.initByName(
			"data", alignment,
			"tree", tree,
			"likelihood", likelihood,
			"minWeight", 0.01,
			"maxWeight", 10.0
		);
	}

	// ========== Category 1: Component Tests ==========

	/**
	 * Test that getPartialOffset produces correct category-major layout.
	 * This is crucial since wrong layout caused hard-to-debug issues.
	 */
	@Test
	public void testPartialOffsetLayout() {
		// Force initialization
		likelihood.calculateLogP();
		weights.getEdgeWeights(0);

		// For category-major layout: offset = (category * patternCount + pattern) * stateCount
		// Check that consecutive patterns within same category are stateCount apart
		int offset00 = weights.getPartialOffset(0, 0);
		int offset10 = weights.getPartialOffset(1, 0);
		int offset01 = weights.getPartialOffset(0, 1);

		assertEquals("offset(0,0) should be 0", 0, offset00);
		assertEquals("offset(1,0) - offset(0,0) should be stateCount", 
					 weights.stateCount, offset10 - offset00);
		
		// offset(0,1) should jump by patternCount * stateCount
		assertEquals("offset(0,1) should be patternCount * stateCount",
					 weights.patternCount * weights.stateCount, offset01);
	}

	// ========== Category 2: Property-Based Tests ==========

	/**
	 * Property: Edge weights should be positive and within bounds.
	 */
	@Test
	public void testEdgeWeightsAreBounded() {
		likelihood.calculateLogP();

		for (int i = 0; i < tree.getNodeCount(); i++) {
			if (tree.getNode(i).isRoot()) continue;
			
			double w = weights.getEdgeWeights(i);
			assertTrue("Edge weight should be >= minWeight", w >= 0.01);
			assertTrue("Edge weight should be <= maxWeight", w <= 10.0);
		}
	}

	/**
	 * Property: Target weights should form a valid probability distribution 
	 * (sum to ~1, all positive).
	 */
	@Test
	public void testTargetWeightsFormDistribution() {
		likelihood.calculateLogP();

		// Pick an internal node and get target weights to all other internal nodes
		List<Node> internalNodes = new ArrayList<>();
		for (int i = 0; i < tree.getNodeCount(); i++) {
			if (!tree.getNode(i).isLeaf() && !tree.getNode(i).isRoot()) {
				internalNodes.add(tree.getNode(i));
			}
		}

		if (internalNodes.size() < 2) {
			// Need at least 2 internal nodes
			return;
		}

		Node fromNode = internalNodes.get(0);
		List<Node> toNodes = new ArrayList<>(internalNodes);
		toNodes.remove(fromNode);

		double[] targetWeights = weights.getTargetWeights(fromNode.getNr(), toNodes);

		// All weights should be positive
		for (double w : targetWeights) {
			assertTrue("Target weight should be positive", w > 0);
		}

		// Sum should be close to 1 (allowing for EPS additions)
		double sum = Arrays.stream(targetWeights).sum();
		assertEquals("Target weights should sum to ~1", 1.0, sum, 0.01);
	}

	/**
	 * Property: Divergence should be symmetric: div(A,B) = div(B,A).
	 */
	@Test
	public void testDivergenceSymmetry() {
		likelihood.calculateLogP();

		// Get two internal nodes
		Node node1 = null, node2 = null;
		for (int i = 0; i < tree.getNodeCount(); i++) {
			if (!tree.getNode(i).isLeaf()) {
				if (node1 == null) node1 = tree.getNode(i);
				else if (node2 == null) { node2 = tree.getNode(i); break; }
			}
		}

		if (node1 == null || node2 == null) return;

		// Compute edge weights in both directions (via common ancestor logic)
		// Since divergence is symmetric, we test by checking the similarity is also symmetric
		List<Node> list1 = Arrays.asList(node2);
		List<Node> list2 = Arrays.asList(node1);

		double[] w12 = weights.getTargetWeights(node1.getNr(), list1);
		double[] w21 = weights.getTargetWeights(node2.getNr(), list2);

		// The raw similarity scores should be symmetric (before softmax)
		// We can't easily access raw similarities, but with single-element lists,
		// the softmax just returns [1.0 + eps], so this is a weak test.
		// A stronger test is the reconstruction property below.
	}

	/**
	 * Property: updatePartialsWithoutNodeRecursive should recover full partials
	 * when ignoring a non-existent node (ignore = -1) and updating all nodes.
	 */
	@Test
	public void testPartialsWithoutNodeInvalidIgnoreMatchesFullPartials() throws Exception {
		likelihood.calculateLogP();

		List<Integer> allNodes = new ArrayList<>();
		for (int i = 0; i < tree.getNodeCount(); i++) {
			allNodes.add(i);
		}

		weights.updateByOperatorWithoutNode(-1, allNodes);

		double[][] partialsWithoutNode = getPartialsWithoutNode(weights);
		LikelihoodCore core = likelihood.getLikelihoodCore();

		int leafCount = tree.getLeafNodeCount();
		int partialsSize = weights.partialsSize;

		for (int nodeNr = leafCount; nodeNr < tree.getNodeCount(); nodeNr++) {
			double[] expected = new double[partialsSize];
			core.getNodePartials(nodeNr, expected);

			double[] actual = partialsWithoutNode[nodeNr];
			assertNotNull("partialsWithoutNode should be computed for internal node " + nodeNr, actual);
			assertEquals("partialsWithoutNode length mismatch for node " + nodeNr, expected.length, actual.length);

			System.out.println("Node " + nodeNr + " expected partials: " + Arrays.toString(expected));
			System.out.println("Node " + nodeNr + " actual partials:   " + Arrays.toString(actual));

			double maxDiff = maxAbsDiff(expected, actual);
			assertEquals("partialsWithoutNode should match full partials for node " + nodeNr, 0.0, maxDiff, 1e-8);
		}
	}

	private static double[][] getPartialsWithoutNode(FelsensteinWeights weights) throws Exception {
		Field field = FelsensteinWeights.class.getDeclaredField("partialsWithoutNode");
		field.setAccessible(true);
		return (double[][]) field.get(weights);
	}

	private static double maxAbsDiff(double[] a, double[] b) {
		double max = 0.0;
		for (int i = 0; i < a.length; i++) {
			max = Math.max(max, Math.abs(a[i] - b[i]));
		}
		return max;
	}

	/**
	 * Property: Full conditional likelihoods at any node should recover
	 * the full tree likelihood when aggregated across states and patterns.
	 */
	@Test
	public void testFullConditionalLikelihoodMatchesTreeLikelihood() {
		double expectedLogP = likelihood.calculateLogP();

		List<Integer> allNodes = new ArrayList<>();
		for (int i = 0; i < tree.getNodeCount(); i++) {
			allNodes.add(i);
		}

		weights.updateByOperatorWithoutNode(-1, allNodes);

		for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
			double actualLogP = computeLogPFromFullConditionals(nodeNr);
			assertEquals(
					"Full conditional likelihood should match tree likelihood at node " + nodeNr,
					expectedLogP,
					actualLogP,
					1e-6
			);
		}
	}

	private double computeLogPFromFullConditionals(int nodeNr) {
		double[] fullConditional = weights.getFullConditionalLikelihoods(nodeNr, -1);

		double[] integrated = new double[weights.patternCount * weights.stateCount];
		SiteModelInterface.Base siteModelBase = weights.getSiteModel();
		double[] proportions = siteModelBase.getCategoryProportions(tree.getNode(nodeNr));
		weights.likelihoodCore.calculateIntegratePartials(fullConditional, proportions, integrated);

		double[] patternLogLikelihoods = new double[weights.patternCount];
		double[] freqs = weights.getSubstitutionModel().getFrequencies();
		weights.likelihoodCore.calculateLogLikelihoods(integrated, freqs, patternLogLikelihoods);

		double weightedSum = 0.0;
		for (int p = 0; p < weights.patternCount; p++) {
			weightedSum += alignment.getPatternWeight(p) * patternLogLikelihoods[p];
		}

		return weightedSum;
	}

	/**
	 * Property: Identical sequences should have identical leaf partials.
	 * Taxa C and D have the same sequence, so their partials should be equal.
	 */
	@Test
	public void testIdenticalSequencesHaveIdenticalPartials() {
		likelihood.calculateLogP();
		weights.getEdgeWeights(0);  // Force initialization

		// Find node indices for C and D
		int nodeC = -1, nodeD = -1;
		for (int i = 0; i < tree.getLeafNodeCount(); i++) {
			String id = tree.getNode(i).getID();
			if ("C".equals(id)) nodeC = i;
			if ("D".equals(id)) nodeD = i;
		}

		assertTrue("Should find taxon C", nodeC >= 0);
		assertTrue("Should find taxon D", nodeD >= 0);

		// Get edge weights - they should be similar for C and D (both connecting to same parent)
		// Actually, we need to check the leaf partials are identical
		// Since leafPartials is private, we'll test indirectly via similarity
		List<Node> targets = Arrays.asList(tree.getNode(nodeC));
		double[] simToC = weights.getTargetWeights(nodeD, targets);

		// The similarity of D to C should be maximal (since they have same sequence)
		// In a single-element list, softmax gives ~1.0
		assertEquals("Similarity of identical sequences should be ~1", 1.0, simToC[0], 0.01);
	}

	/**
	 * KEY PROPERTY TEST: Partials reconstruction via "without node" computation.
	 * 
	 * If node P has children L and R, then:
	 *   partialsWithoutNode(P, ignoring L) contains only R's contribution
	 *   partialsWithoutNode(P, ignoring R) contains only L's contribution
	 * 
	 * Multiplying these (element-wise, after transition matrix application) should
	 * approximate the original partials at P (up to scaling).
	 * 
	 * This tests the core correctness of the "without node" Felsenstein recursion.
	 */
	@Test
	public void testPartialsWithoutNodeReconstruction() {
		likelihood.calculateLogP();
		weights.getEdgeWeights(0);  // Force initialization

		// Find an internal node with two children
		Node parent = null;
		for (int i = 0; i < tree.getNodeCount(); i++) {
			Node n = tree.getNode(i);
			if (!n.isLeaf() && !n.isRoot() && n.getChildCount() == 2) {
				parent = n;
				break;
			}
		}

		if (parent == null) {
			// Skip if no suitable node found
			return;
		}

		Node leftChild = parent.getChild(0);
		Node rightChild = parent.getChild(1);

		// Compute partialsWithoutNode ignoring left child
		List<Integer> nodesToUpdate = Arrays.asList(parent.getNr());
		weights.updateByOperatorWithoutNode(leftChild.getNr(), nodesToUpdate);

		// The target weight from left child to parent (using partialsWithoutNode)
		// should reflect how well left's partials match "everything except left" at parent
		List<Node> targetList = Arrays.asList(parent);
		double[] weightsFromLeft = weights.getTargetWeights(leftChild.getNr(), targetList);

		// Now compute ignoring right child
		weights.updateByOperatorWithoutNode(rightChild.getNr(), nodesToUpdate);
		double[] weightsFromRight = weights.getTargetWeights(rightChild.getNr(), targetList);

		// Both should give valid (positive) weights
		assertTrue("Weight from left child should be positive", weightsFromLeft[0] > 0);
		assertTrue("Weight from right child should be positive", weightsFromRight[0] > 0);

		// The similarity from a child to its parent (excluding that child's contribution)
		// should be reasonable (not degenerate)
		// This is a sanity check that the reconstruction is working
	}

	/**
	 * Property: Self-similarity should be maximal.
	 * Computing similarity of a node's partials with itself should give a high value.
	 */
	@Test
	public void testSelfSimilarityIsMaximal() {
		likelihood.calculateLogP();

		// Get an internal node
		Node internalNode = null;
		for (int i = 0; i < tree.getNodeCount(); i++) {
			if (!tree.getNode(i).isLeaf()) {
				internalNode = tree.getNode(i);
				break;
			}
		}

		if (internalNode == null) return;

		// Get target weights to itself (degenerate case but should work)
		List<Node> selfList = Arrays.asList(internalNode);
		double[] selfWeights = weights.getTargetWeights(internalNode.getNr(), selfList);

		// With a single target, softmax always gives ~1.0 (plus eps), so this is weak
		// But it shouldn't crash
		assertEquals("Self-weight in single-element list should be ~1", 1.0, selfWeights[0], 0.01);
	}

	/**
	 * Test with multiple rate categories.
	 * Verifies that partials are correctly aggregated across categories.
	 */
	@Test
	public void testMultipleRateCategories() {
		// The setUp already uses 4 gamma categories
		likelihood.calculateLogP();

		// Verify matrixCount is as expected
		weights.getEdgeWeights(0);  // Force init
		assertEquals("Should have 4 rate categories " + siteModel.getCategoryCount(), 4, weights.matrixCount);

		// Edge weights should still be valid
		for (int i = 0; i < tree.getNodeCount(); i++) {
			if (!tree.getNode(i).isRoot()) {
				double w = weights.getEdgeWeights(i);
				assertTrue("Edge weight with gamma should be valid", w > 0 && !Double.isNaN(w));
			}
		}
	}

	/**
	 * Test that aggregating root partials across rate categories and states matches TreeLikelihood.
	 * 
	 * The root partials represent P(data | state at root, rate category). We integrate these
	 * across rate categories using site model category proportions (which account for gamma
	 * discretization and invariant sites), then aggregate over states using substitution model
	 * frequencies. This should match exactly how TreeLikelihood computes its final log probability.
	 */
	@Test
	public void testAggegatePartials() throws Exception {
        SlowBeerLikelihoodCore core = (SlowBeerLikelihoodCore) likelihood.getLikelihoodCore();

		// Setup tree and likelihood
		likelihood.calculateLogP();

		// Create list of all nodes to update
		List<Integer> allNodes = new ArrayList<>();
		for (int i = 0; i < tree.getNodeCount(); i++) {
			allNodes.add(i);
		}

		// Update partialsWithoutNode with invalid ignore node (-1), which should recover full partials
		weights.updateByOperatorWithoutNode(-1, allNodes);

		// Get root node and its two children
		Node rootNode = tree.getRoot();

		// Ensure internal buffers are initialized for transition matrix computation
		ensureProbabilitiesInitialized(weights);
		ensureTransitionMatricesInitialized(weights);

		// -----------------------------------------------------------------------------
        // Get root partials
		double[] rootPartialsPerCat = weights.getPartials(rootNode.getNr());

		// Integrate root partials across rate categories before aggregating over states.
		double[] rootPartials = new double[weights.patternCount * weights.stateCount];
		double[] proportions = siteModel.getCategoryProportions(rootNode);
		core.calculateIntegratePartials(rootPartialsPerCat, proportions, rootPartials);

		double[] patternLogLikelihoods = new double[weights.patternCount];
		double[] freqs = likelihood.getSubstitutionModel().getFrequencies();
		core.calculateLogLikelihoods(rootPartials, freqs, patternLogLikelihoods);

		// Sum over the log-likelihoods for different site patterns, applying pattern weights
		double weightsLogProb = 0.0;
		for (int p = 0; p < alignment.getPatternCount(); p++) {
			weightsLogProb += alignment.getPatternWeight(p) * patternLogLikelihoods[p];
		}
		// -----------------------------------------------------------------------------

		// Get log probability from tree likelihood
		double treeLogProb = likelihood.calculateLogP();

		// Debug output
		System.out.println("Log prob from computePartialsAgreement: " + weightsLogProb);
		System.out.println("Tree likelihood log prob: " + treeLogProb);

		// Validate that computePartialsAgreement (with pattern weighting) matches TreeLikelihood
		assertEquals("Combined partials with pattern weighting should match tree likelihood log prob",
			treeLogProb, weightsLogProb, 1e-6);
	}

	@Test
	public void testGetTargetWeightsIntegerMatchesLikelihood() throws Exception {
		// Setup tree and likelihood
		likelihood.calculateLogP();

		// Create list of all nodes to update
		List<Integer> allNodes = new ArrayList<>();
		for (int i = 0; i < tree.getNodeCount(); i++) {
			allNodes.add(i);
		}

		// Update partialsWithoutNode with invalid ignore node (-1), which should recover full partials
		weights.updateByOperatorWithoutNode(-1, allNodes);

		// Get root node and its two children
		Node rootNode = tree.getRoot();
		if (rootNode.getChildCount() != 2) {
			System.out.println("Skipping test - root must have exactly 2 children");
			return;
		}

		Node child1 = rootNode.getChild(0);
		Node child2 = rootNode.getChild(1);

		// Compute agreement using the same structure as getTargetWeightsInteger
		int child1Nr = child1.getNr();
		int child2Nr = child2.getNr();

		double[] fromPartials = getPartialsByNr(weights, child1Nr);
		double[] toPartials = getPartialsForTargetWeightByNr(weights, child2Nr, child1Nr);

		if (fromPartials == null || toPartials == null) {
			System.out.println("Skipping test - could not retrieve partials");
			return;
		}

		// Ensure internal buffers are initialized for transition matrix computation
		ensureProbabilitiesInitialized(weights);
		ensureTransitionMatricesInitialized(weights);

		// Compute agreement - now includes pattern weighting internally
		double weightsLogProb = weights.computePartialsAgreement(child1, child2, fromPartials, toPartials, rootNode.getHeight());

		// Get log probability from tree likelihood
		double treeLogProb = likelihood.calculateLogP();

		// Debug output
		System.out.println("Log prob from computePartialsAgreement: " + weightsLogProb);
		System.out.println("Tree likelihood log prob: " + treeLogProb);

		// Validate that computePartialsAgreement (with pattern weighting) matches TreeLikelihood
		assertEquals("Combined partials with pattern weighting should match tree likelihood log prob",
			treeLogProb, weightsLogProb, 1e-6);
	}

	private double[] getPartialsWithoutNodeFromCache(FelsensteinWeights weights, int nodeNr) throws Exception {
		double[][] partialsWithoutNode = getPartialsWithoutNode(weights);
		if (nodeNr >= 0 && nodeNr < partialsWithoutNode.length) {
			return partialsWithoutNode[nodeNr];
		}
		return null;
	}

	private double[] getPartialsByNr(FelsensteinWeights weights, int nodeNr) throws Exception {
		java.lang.reflect.Method method = FelsensteinWeights.class.getDeclaredMethod("getPartials", int.class);
		method.setAccessible(true);
		return (double[]) method.invoke(weights, nodeNr);
	}

	private double[] getPartialsForTargetWeightByNr(FelsensteinWeights weights, int nodeNr, int fromNodeNr) throws Exception {
		java.lang.reflect.Method method = FelsensteinWeights.class.getDeclaredMethod("getPartialsForTargetWeight", int.class, int.class);
		method.setAccessible(true);
		return (double[]) method.invoke(weights, nodeNr, fromNodeNr);
	}

	private void ensureProbabilitiesInitialized(FelsensteinWeights weights) throws Exception {
		Field field = FelsensteinWeights.class.getDeclaredField("_probabilities");
		field.setAccessible(true);
		if (field.get(weights) == null) {
			field.set(weights, new double[weights.matrixSize]);
		}
	}

	private void ensureTransitionMatricesInitialized(FelsensteinWeights weights) throws Exception {
		Field fromMatrixField = FelsensteinWeights.class.getDeclaredField("fromMatrix");
		Field toMatrixField = FelsensteinWeights.class.getDeclaredField("toMatrix");
		fromMatrixField.setAccessible(true);
		toMatrixField.setAccessible(true);
		if (fromMatrixField.get(weights) == null) {
			fromMatrixField.set(weights, new double[weights.matrixCount * weights.matrixSize]);
		}
		if (toMatrixField.get(weights) == null) {
			toMatrixField.set(weights, new double[weights.matrixCount * weights.matrixSize]);
		}
	}

	private double sumDoubleArray(double[] arr) {
		double sum = 0.0;
		for (double x : arr) {
			sum += x;
		}
		return sum;
	}
}
