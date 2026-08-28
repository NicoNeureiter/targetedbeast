package targetedbeast.edgeweights;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;

public class CombinedEdgeWeightsTest {

	private static final String TEST_NEWICK = "((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);";

	private Tree tree;
	private CombinedEdgeWeights combinedWeights;

	@Before
	public void setUp() {
		tree = parseTree(TEST_NEWICK);

		ConstantWeights constantWeights = new ConstantWeights();
		constantWeights.initByName("tree", tree);

		PriorHeuristicWeights priorWeights = new PriorHeuristicWeights();
		priorWeights.initByName(
				"tree", tree,
				"temperature", 1.0,
				"sourceBranchScale", 1.0,
				"attachmentAgeScale", 0.25,
				"heightIncreaseScale", 1.0,
				"maxNewBranchScale", 1.0,
				"minWeight", 1e-6,
				"maxWeight", 1e6);

		combinedWeights = new CombinedEdgeWeights();
		combinedWeights.initByName("edgeWeights", constantWeights, "edgeWeights", priorWeights);
	}

	@Test
	public void explicitTargetHeightsPropagateThroughCombinedWeights() {
		int movingNode = findNodeById(tree, "A").getNr();
		int candidateNode = findNodeById(tree, "C").getNr();

		double[] weightedByHeight = combinedWeights.getTargetWeightsInteger(
				movingNode,
				List.of(candidateNode, candidateNode),
				List.of(0.05, 0.14));

		assertTrue(weightedByHeight[0] > weightedByHeight[1]);

		double[] defaultHeights = combinedWeights.getTargetWeightsInteger(
				movingNode,
				List.of(candidateNode, candidateNode));
		assertEquals(defaultHeights[0], defaultHeights[1], 1e-12);
	}

	@Test
	public void nodeAndIntegerHeightAwareOverloadsAgree() {
		Node movingNode = findNodeById(tree, "A");
		Node candidateNode = findNodeById(tree, "C");
		List<Double> targetHeights = List.of(0.05, 0.14);

		double[] byInteger = combinedWeights.getTargetWeightsInteger(
				movingNode.getNr(),
				List.of(candidateNode.getNr(), candidateNode.getNr()),
				targetHeights);
		double[] byNode = combinedWeights.getTargetWeights(
				movingNode.getNr(),
				List.of(candidateNode, candidateNode),
				targetHeights);

		assertEquals(byInteger[0], byNode[0], 1e-12);
		assertEquals(byInteger[1], byNode[1], 1e-12);
	}

	private static Tree parseTree(String newick) {
		Tree parsedTree = new TreeParser();
		((TreeParser) parsedTree).initByName(
				"newick", newick,
				"IsLabelledNewick", true,
				"taxa", null);
		return parsedTree;
	}

	private static Node findNodeById(Tree searchTree, String id) {
		for (Node node : searchTree.getNodesAsArray()) {
			if (id.equals(node.getID())) {
				return node;
			}
		}
		throw new IllegalArgumentException("Node not found: " + id);
	}
}