package targetedbeast.edgeweights;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import java.lang.reflect.Field;

import org.junit.Before;
import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.likelihood.LikelihoodCore;
import beast.base.evolution.sitemodel.SiteModel;
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
		Sequence seqA = new Sequence("A", "ACGTACGT-A");
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

        weights.updateByOperator();
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
			assertTrue("Edge weight should be finite", Double.isFinite(w));
			assertTrue("Edge weight should be >= minWeight", w >= 0.01);
			assertTrue("Edge weight should be <= maxWeight", w <= 10.0);
		}
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

		List<Integer> allNodes = IntStream.range(0, tree.getNodeCount()).boxed().toList();

		weights.updateByOperatorWithoutNode(-1, allNodes);

		for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
			double actualLogP = computeLogPFromFullConditionals(nodeNr);
            System.out.println(expectedLogP + "   " + actualLogP);
			assertEquals(
					"Full conditional likelihood should match tree likelihood at node " + nodeNr,
					expectedLogP,
					actualLogP,
					1e-10
			);
		}
	}

	private double computeLogPFromFullConditionals(int nodeNr) {
		double[] fullConditional = weights.getFullConditionalLikelihoods(nodeNr, -1);
        return computeLogPFromFullConditionals(nodeNr, fullConditional);
    }

    private double computeLogPFromFullConditionals(int nodeNr, double[] fullConditional) {
		double[] integrated = new double[weights.patternCount * weights.stateCount];
		double[] proportions = weights.getSiteModel().getCategoryProportions(tree.getNode(nodeNr));
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

        String partialStringC = Arrays.toString(weights.getPartials(nodeC));
        String partialStringD = Arrays.toString(weights.getPartials(nodeD));
        
		// Check whether leaf partials are equal
		assertEquals("Identical sequences should result in identical leaf partials", partialStringC, partialStringD);
	}

    private static List<Integer> collectAncestors(Node nodeI) {
        List<Integer> ancestors = new ArrayList<>();
        Node ancestor = nodeI.getParent();
        while (ancestor != null) {
            ancestors.add(ancestor.getNr());
            ancestor = ancestor.getParent();
        }
        return ancestors;
    }

    // TODO| For the test to work we need a larger tree, because the internalNode 
    // TODO| is not allowed to be the root or child of the root.
    // 
	// /**
	//  * Property: Self-similarity should be maximal.
	//  * Computing similarity of a node's partials with itself should give a high value.
	//  */
	// @Test
	// public void testSelfSimilarityIsMaximal() {
	// 	// Get an internal node
	// 	Node internalNode = null;
	// 	for (Node n : tree.getNodesAsArray()) {
	// 		if (!(n.isLeaf() || n.isRoot())) {
	// 			internalNode = n;
	// 			break;
	// 		}
	// 	}
	// 	assert (internalNode != null);

    //     List<Integer> ancestors = collectAncestors(internalNode);
    //     System.out.println(tree.toString());
    //     System.out.println(internalNode.getNr());
    //     weights.updateByOperatorWithoutNode(internalNode.getNr(), ancestors);

	// 	// Get target weights to itself (degenerate case but should work)
	// 	List<Node> selfList = Arrays.asList(internalNode);
	// 	double[] selfWeights = weights.getTargetWeights(internalNode.getNr(), selfList);

	// 	// With a single target, softmax always gives ~1.0 (plus eps), so this is weak
	// 	// But it shouldn't crash
	// 	assertEquals("Self-weight in single-element list should be ~1", 1.0, selfWeights[0], 0.01);
	// }

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
	public void testAggregatePartials() throws Exception {
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
    
	private void ensureProbabilitiesInitialized(FelsensteinWeights weights) throws Exception {
		Field field = FelsensteinWeights.class.getDeclaredField("_probabilities");
		field.setAccessible(true);
		if (field.get(weights) == null) {
			field.set(weights, new double[weights.matrixSize]);
		}
	}

	private void ensureTransitionMatricesInitialized(FelsensteinWeights weights) throws Exception {
		Field fromMatrixField = FelsensteinWeights.class.getDeclaredField("_fromMatrix");
		Field toMatrixField = FelsensteinWeights.class.getDeclaredField("_toMatrix");
		fromMatrixField.setAccessible(true);
		toMatrixField.setAccessible(true);
		if (fromMatrixField.get(weights) == null) {
			fromMatrixField.set(weights, new double[weights.matrixCount * weights.matrixSize]);
		}
		if (toMatrixField.get(weights) == null) {
			toMatrixField.set(weights, new double[weights.matrixCount * weights.matrixSize]);
		}
	}

}
