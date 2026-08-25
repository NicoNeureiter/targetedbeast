package targetedbeast.edgeweights;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;

public class PCAWeightsBrownianTest {

    private Alignment alignment;
    private Tree tree;
    private PCAWeightsBrownian weights;

    @Before
    public void setUp() {
        alignment = new Alignment();
        alignment.initByName(
                "sequence", new Sequence("A", "ACGTACGTAA"),
                "sequence", new Sequence("B", "ACGTACGTAT"),
                "sequence", new Sequence("C", "ACGTACGTGA"),
                "dataType", "nucleotide");

        tree = new TreeParser();
        ((TreeParser) tree).initByName(
                "newick", "((A:0.1,B:0.2):0.3,C:0.4);",
                "IsLabelledNewick", true,
                "taxa", alignment);

        weights = new PCAWeightsBrownian();
        weights.initByName(
                "tree", tree,
                "data", alignment,
                "dimension", 1,
                "value", "A=0.0,B=2.0,C=4.0",
                "offset", 0.01);
    }

    @Test
    public void internalPosteriorMatchesBrownianPrecisionWeighting() {
        Node ab = findInternalNodeWithChildren("A", "B");
        Node root = tree.getRoot();

        double aMean = weights.getPoints(findNodeNrById("A"))[0];
        double bMean = weights.getPoints(findNodeNrById("B"))[0];
        double cMean = weights.getPoints(findNodeNrById("C"))[0];

        double abLeftVar = ab.getLeft().getLength();
        double abRightVar = ab.getRight().getLength();
        double expectedAbVariance = 1.0 / (1.0 / abLeftVar + 1.0 / abRightVar);
        double expectedAbMean = expectedAbVariance * (aMean / abLeftVar + bMean / abRightVar);

        assertEquals(expectedAbMean, weights.getPoints(ab.getNr())[0], 1e-12);
        assertEquals(expectedAbVariance, weights.getVariances(ab.getNr())[0], 1e-12);

        double rootLeftVar = expectedAbVariance + ab.getLength();
        double rootRightVar = root.getRight().getLength();
        double expectedRootVariance = 1.0 / (1.0 / rootLeftVar + 1.0 / rootRightVar);
        double expectedRootMean = expectedRootVariance * (expectedAbMean / rootLeftVar + cMean / rootRightVar);

        assertEquals(expectedRootMean, weights.getPoints(root.getNr())[0], 1e-12);
        assertEquals(expectedRootVariance, weights.getVariances(root.getNr())[0], 1e-12);
    }

    @Test
    public void targetWeightsPreferCloserCandidate() {
        int fromNodeNr = findNodeNrById("A");
        int bNodeNr = findNodeNrById("B");
        int cNodeNr = findNodeNrById("C");

        double[] targetWeights = weights.getTargetWeightsInteger(fromNodeNr, List.of(bNodeNr, cNodeNr));

        assertEquals(2, targetWeights.length);
        assertTrue(Double.isFinite(targetWeights[0]));
        assertTrue(Double.isFinite(targetWeights[1]));
        assertTrue(targetWeights[0] > 0.0);
        assertTrue(targetWeights[1] > 0.0);
        assertTrue("Closer candidate should have larger Brownian weight", targetWeights[0] > targetWeights[1]);
    }

    @Test
    public void fixedHeightChangesRelativeTargetScores() {
        int fromNodeNr = findNodeNrById("A");
        int bNodeNr = findNodeNrById("B");
        int cNodeNr = findNodeNrById("C");

        double[] lowerWeights = weights.getTargetWeightsInteger(fromNodeNr, List.of(bNodeNr, cNodeNr), 0.05);
        double[] upperWeights = weights.getTargetWeightsInteger(fromNodeNr, List.of(bNodeNr, cNodeNr), 0.18);

        assertEquals(2, lowerWeights.length);
        assertEquals(2, upperWeights.length);
        assertNotEquals(lowerWeights[0] / lowerWeights[1], upperWeights[0] / upperWeights[1], 1e-12);
    }

    @Test
    public void ignoredSiblingUsesEffectiveParent() {
        Node a = tree.getNode(findNodeNrById("A"));
        Node b = tree.getNode(findNodeNrById("B"));

        assertSame(tree.getRoot(), weights.getEffectiveParent(b, a.getNr()));

        weights.prestore();
        weights.updateByOperatorWithoutNode(a.getNr(), collectAncestors(a));
        double[] withoutNodeWeights = weights.getTargetWeightsInteger(a.getNr(), List.of(b.getNr(), findNodeNrById("C")));
        weights.reset();

        assertEquals(2, withoutNodeWeights.length);
        assertTrue(withoutNodeWeights[0] > 0.0);
        assertTrue(withoutNodeWeights[1] > 0.0);
        assertArrayEquals(withoutNodeWeights, withoutNodeWeights, 0.0);
    }

    private Node findInternalNodeWithChildren(String leftId, String rightId) {
        for (Node node : tree.getNodesAsArray()) {
            if (node.isLeaf()) {
                continue;
            }
            if (node.getLeft().isLeaf() && node.getRight().isLeaf()) {
                String left = node.getLeft().getID();
                String right = node.getRight().getID();
                if ((leftId.equals(left) && rightId.equals(right)) || (leftId.equals(right) && rightId.equals(left))) {
                    return node;
                }
            }
        }
        throw new IllegalArgumentException("Could not find internal node for children " + leftId + " and " + rightId);
    }

    private int findNodeNrById(String id) {
        for (Node node : tree.getNodesAsArray()) {
            if (id.equals(node.getID())) {
                return node.getNr();
            }
        }
        throw new IllegalArgumentException("Could not find node with id " + id);
    }

    private List<Integer> collectAncestors(Node node) {
        List<Integer> ancestors = new java.util.ArrayList<>();
        Node current = node.getParent();
        while (current != null) {
            ancestors.add(current.getNr());
            current = current.getParent();
        }
        return ancestors;
    }
}
