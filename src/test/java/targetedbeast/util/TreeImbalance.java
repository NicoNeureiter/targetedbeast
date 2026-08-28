package targetedbeast.util;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;

/**
 * Tree-topology imbalance metrics.
 *
 * <p>Pure functions of tree shape, with no dependency on any testing
 * infrastructure. Used as quantisation functions in detailed-balance tests, but
 * generally applicable as summaries of tree balance.
 */
public final class TreeImbalance {

    private TreeImbalance() {
    }

    /**
     * Compute tree-level imbalance as a weighted average of per-node imbalances.
     *
     * @param tree The tree to compute imbalance for
     * @return weighted average of internal-node imbalances, or 0 if undefined
     */
    public static double treeImbalance(Tree tree) {
        double weightedSum = 0.0;
        double totalWeight = 0.0;

        // Traverse all internal nodes and compute weighted imbalance
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isLeaf()) {
                ImbalanceResult result = nodeImbalance(node);
                if (!Double.isNaN(result.imbalance)) {
                    weightedSum += result.imbalance * result.weight;
                    totalWeight += result.weight;
                }
            }
        }

        // Return weighted average if there are any internal nodes
        return totalWeight > 0 ? weightedSum / totalWeight : 0.0;
    }

    /**
     * Absolute difference in leaf count between the root's two children.
     */
    public static int rootImbalance(Node root) {
        return Math.abs(root.getLeft().getLeafNodeCount() - root.getRight().getLeafNodeCount());
    }

    /**
     * Compute the imbalance metric for a single node.
     *
     * @param node The node to compute imbalance for
     * @return ImbalanceResult containing imbalance and weight
     */
    public static ImbalanceResult nodeImbalance(Node node) {
        int size = node.getLeafNodeCount();
        double I;
        double w = 1.0;

        if (node.getChildCount() < 2 || size < 4) {
            I = Double.NaN;
        } else {
            Node c1 = node.getChild(0);
            Node c2 = node.getChild(1);

            int bigger = Math.max(c1.getLeafNodeCount(), c2.getLeafNodeCount());
            double m = Math.ceil(size / 2.0);

            I = (bigger - m) / (size - m - 1);
        }

        if (size % 2 == 1) {  // odd
            w = 1.0;
        } else {              // even
            w = 1.0 - 1.0 / size;
            if (I == 0) {
                w *= 2;
            } else {
                assert Double.isNaN(I) || I > 0 : "I should be NaN or positive, but was: " + I;
            }
        }
        if (Double.isNaN(I)) {
            I = 0.0;
            w = 0.0;
        }
        return new ImbalanceResult(I, w);
    }

    /**
     * Holds the imbalance and weight values produced for a single node.
     */
    public static final class ImbalanceResult {
        public final double imbalance;
        public final double weight;

        public ImbalanceResult(double imbalance, double weight) {
            this.imbalance = imbalance;
            this.weight = weight;
        }
    }
}
