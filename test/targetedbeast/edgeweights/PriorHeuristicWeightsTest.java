package targetedbeast.edgeweights;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.speciation.YuleModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;

public class PriorHeuristicWeightsTest {

    private static final String TEST_NEWICK = "((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);";

    private Alignment alignment;
    private Tree tree;
    private PriorHeuristicWeights weights;

    @Before
    public void setUp() {
        Sequence seqA = new Sequence("A", "ACGTACGTAA");
        Sequence seqB = new Sequence("B", "ACGTACGTAT");
        Sequence seqC = new Sequence("C", "ACGTACGTGA");
        Sequence seqD = new Sequence("D", "ACGTACGTGT");

        alignment = new Alignment();
        alignment.initByName("sequence", seqA,
                "sequence", seqB,
                "sequence", seqC,
                "sequence", seqD);

        tree = parseTree(TEST_NEWICK);

        weights = new PriorHeuristicWeights();
        weights.initByName(
                "tree", tree,
                "temperature", 1.0,
                "sourceBranchScale", 1.0,
                "attachmentAgeScale", 0.25,
                "heightIncreaseScale", 1.0,
                "maxNewBranchScale", 1.0,
                "minWeight", 1e-6,
                "maxWeight", 1e6);
    }

    @Test
    public void longerSourceBranchGetsHigherSelectionWeight() {
        int a = findNodeNrById(tree, "A");
        int c = findNodeNrById(tree, "C");

        assertTrue(tree.getNode(c).getLength() > tree.getNode(a).getLength());
        assertTrue(weights.getEdgeWeights(c) > weights.getEdgeWeights(a));
    }

    @Test
    public void targetWeightDecreasesForOlderAttachmentOnSameCandidateEdge() {
        int a = findNodeNrById(tree, "A");
        int c = findNodeNrById(tree, "C");

        double[] targetWeights = weights.getTargetWeightsInteger(a, List.of(c, c), List.of(0.05, 0.14));

        assertTrue(targetWeights[0] > targetWeights[1]);
        assertTrue(targetWeights[0] >= weights.minEdgeWeight());
        assertTrue(targetWeights[1] >= weights.minEdgeWeight());
    }

    @Test
    public void defaultTargetHeightMatchesExplicitCandidateParentHeight() {
        int a = findNodeNrById(tree, "A");
        int c = findNodeNrById(tree, "C");
        double explicitHeight = tree.getNode(c).getParent().getHeight();

        double implicit = weights.getTargetWeightsInteger(a, List.of(c))[0];
        double explicit = weights.getTargetWeightsInteger(a, List.of(c), explicitHeight)[0];

        assertEquals(explicit, implicit, 1e-12);
    }

    @Test
    public void heuristicMatchesYuleDirectionForEarlierVersusLaterAttachment() {
        int a = findNodeNrById(tree, "A");
        int c = findNodeNrById(tree, "C");

        double earlierHeight = 0.05;
        double laterHeight = 0.14;

        double earlierHeuristic = weights.getTargetWeightsInteger(a, List.of(c), earlierHeight)[0];
        double laterHeuristic = weights.getTargetWeightsInteger(a, List.of(c), laterHeight)[0];

        double earlierPriorLogP = computeYuleLogPAfterMove(earlierHeight);
        double laterPriorLogP = computeYuleLogPAfterMove(laterHeight);

        assertTrue("Expected earlier attachment to have better heuristic score", earlierHeuristic > laterHeuristic);
        assertTrue("Expected earlier attachment to have higher Yule prior", earlierPriorLogP > laterPriorLogP);
    }

    private double computeYuleLogPAfterMove(double newParentHeight) {
        Tree movedTree = parseTree(TEST_NEWICK);

        Node movingNode = findNodeById(movedTree, "A");
        Node oldParent = movingNode.getParent();
        Node sibling = oldParent.getLeft() == movingNode ? oldParent.getRight() : oldParent.getLeft();
        Node grandParent = oldParent.getParent();
        Node candidate = findNodeById(movedTree, "C");
        Node candidateParent = candidate.getParent();

        oldParent.setHeight(newParentHeight);
        replace(grandParent, oldParent, sibling);
        replace(oldParent, sibling, candidate);
        replace(candidateParent, candidate, oldParent);

        movedTree.setEverythingDirty(true);

        YuleModel treePrior = new YuleModel();
        treePrior.initByName("tree", movedTree, "birthDiffRate", "2.0");
        return treePrior.calculateLogP();
    }

    private static Tree parseTree(String newick) {
        Tree parsedTree = new TreeParser();
        ((TreeParser) parsedTree).initByName(
                "newick", newick,
                "IsLabelledNewick", true,
                "taxa", null);
        return parsedTree;
    }

    private static void replace(Node node, Node child, Node replacement) {
        node.removeChild(child);
        node.addChild(replacement);
    }

    private static int findNodeNrById(Tree searchTree, String id) {
        return findNodeById(searchTree, id).getNr();
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