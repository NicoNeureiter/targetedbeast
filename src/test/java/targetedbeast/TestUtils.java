package targetedbeast;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;

/**
 * The two members these tests used from BEAST core's test.beast.BEASTTestCase.
 * That class lives in beast-base's test tree, which is not published as an artifact,
 * so it is reproduced here rather than depended on.
 */
public class TestUtils {

    public static final double PRECISION = 1e-6;

    /** The standard six-primate tree, matching the alignments the tests build inline. */
    public static Tree getTree(Alignment data) throws Exception {
        TreeParser tree = new TreeParser();
        tree.initByName("taxa", data,
                "newick", "((((human:0.024003,(chimp:0.010772,bonobo:0.010772):0.013231):0.012035,"
                        + "gorilla:0.036038):0.033087000000000005,orangutan:0.069125):0.030456999999999998,"
                        + "siamang:0.099582);",
                "IsLabelledNewick", true);
        tree.setID("Tree.t:tree");
        return tree;
    }
}
