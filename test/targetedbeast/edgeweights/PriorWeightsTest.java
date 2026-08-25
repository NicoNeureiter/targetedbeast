package targetedbeast.edgeweights;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import beast.base.evolution.speciation.YuleModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;

public class PriorWeightsTest {

    private static final String TEST_NEWICK = "((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);";

    private Tree tree;
    private YuleModel prior;
    private PriorWeights weights;

    @Before
    public void setUp() {
        tree = parseTree(TEST_NEWICK);

        prior = new YuleModel();
        prior.initByName("tree", tree, "birthDiffRate", "2.0");

        weights = new PriorWeights();
        weights.initByName(
                "tree", tree,
                "prior", prior,
                "temperature", 1.0,
                "minWeight", 1e-12,
                "maxWeight", 1e12);
    }

    @Test
    public void sourceWeightsAreNeutralForNonRootEdges() {
        int a = findNodeNrById(tree, "A");
        int c = findNodeNrById(tree, "C");

        assertEquals(1.0, weights.getEdgeWeights(a), 0.0);
        assertEquals(1.0, weights.getEdgeWeights(c), 0.0);
    }

    @Test
    public void targetWeightsTrackExactPriorRatioAndRestoreBaseline() {
        int a = findNodeNrById(tree, "A");
        int c = findNodeNrById(tree, "C");

        double earlierHeight = 0.05;
        double laterHeight = 0.14;
        double baseline = prior.calculateLogP();

        double[] targetWeights = weights.getTargetWeightsInteger(a, List.of(c, c), List.of(earlierHeight, laterHeight));

        double earlierPriorLogP = computeYuleLogPAfterMove(earlierHeight);
        double laterPriorLogP = computeYuleLogPAfterMove(laterHeight);
        double expectedRatio = Math.exp(0.5 * ((earlierPriorLogP - baseline) - (laterPriorLogP - baseline)));

        assertTrue(targetWeights[0] > targetWeights[1]);
        assertEquals(expectedRatio, targetWeights[0] / targetWeights[1], 1e-12);
        assertEquals(baseline, prior.calculateLogP(), 1e-12);
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
    public void noOpSiblingCandidateHasUnitWeight() {
        int a = findNodeNrById(tree, "A");
        int b = findNodeNrById(tree, "B");
        double baseline = prior.calculateLogP();

        double[] siblingWeight = weights.getTargetWeightsInteger(a, List.of(b));

        assertEquals(1.0, siblingWeight[0], 1e-12);
        assertEquals(baseline, prior.calculateLogP(), 1e-12);
    }

    @Test
    public void heightOnlySiblingCandidatesUseExactPriorRatio() {
        int a = findNodeNrById(tree, "A");
        int b = findNodeNrById(tree, "B");

        double lowerHeight = 0.03;
        double upperHeight = 0.08;
        double baseline = prior.calculateLogP();

        double[] targetWeights = weights.getTargetWeightsInteger(a, List.of(b, b), List.of(lowerHeight, upperHeight));

        double lowerPriorLogP = computeYuleLogPAfterHeightOnlyMove(lowerHeight);
        double upperPriorLogP = computeYuleLogPAfterHeightOnlyMove(upperHeight);
        double expectedRatio = Math.exp(0.5 * ((lowerPriorLogP - baseline) - (upperPriorLogP - baseline)));

        assertEquals(expectedRatio, targetWeights[0] / targetWeights[1], 1e-12);
        assertEquals(baseline, prior.calculateLogP(), 1e-12);
    }

    @Test
    public void heightOnlyParentCandidatesUseExactPriorRatio() {
        int a = findNodeNrById(tree, "A");
        int ab = findNodeById(tree, "A").getParent().getNr();

        double lowerHeight = 0.12;
        double upperHeight = 0.18;
        double baseline = prior.calculateLogP();

        double[] targetWeights = weights.getTargetWeightsInteger(a, List.of(ab, ab), List.of(lowerHeight, upperHeight));

        double lowerPriorLogP = computeYuleLogPAfterHeightOnlyMove(lowerHeight);
        double upperPriorLogP = computeYuleLogPAfterHeightOnlyMove(upperHeight);
        double expectedRatio = Math.exp(0.5 * ((lowerPriorLogP - baseline) - (upperPriorLogP - baseline)));

        assertEquals(expectedRatio, targetWeights[0] / targetWeights[1], 1e-12);
        assertEquals(baseline, prior.calculateLogP(), 1e-12);
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

    private double computeYuleLogPAfterHeightOnlyMove(double newParentHeight) {
        Tree movedTree = parseTree(TEST_NEWICK);

        Node movingNode = findNodeById(movedTree, "A");
        Node oldParent = movingNode.getParent();
        oldParent.setHeight(newParentHeight);

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