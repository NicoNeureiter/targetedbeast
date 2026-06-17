package targetedbeast;

import org.junit.BeforeClass;
import org.junit.Test;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.operator.WilsonBalding;
import beast.base.evolution.speciation.YuleModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeDistribution;
import beast.base.evolution.tree.TreeUtils;
import beast.base.inference.DirectSimulator;
import beast.base.inference.Distribution;
import java.util.*;
import java.util.function.Function;

import static targetedbeast.util.TreeImbalance.rootImbalance;


/**
 * Detailed-balance test for the targeted tree operators.
 *
 * <p>Specialises {@link DetailedBalanceTest} for {@link Tree} states sampled from
 * a {@link YuleModel} prior, with tree-level quantisation functions
 * (root child height, tree length, imbalance). Each {@code @Test} method builds a
 * {@link TreeTrial} with one operator/edge-weight combination, defined inline.
 */
public class WilsonBaldingDBTest extends DetailedBalanceTest<Tree> {

    protected static final int NUM_SAMPLES = 20_000;
    protected static final int NUM_TAXA = 5;
    protected static final double BIRTH_DIFF_RATE = 2.0;
    protected static Alignment alignment;

    @BeforeClass
    public static void setUpClass() {
        alignment = createDummyAlignment();
    }

    @Override
    protected int getNumSamples() {
        return NUM_SAMPLES;
    }

    @Override
    protected List<StateMapper<Tree>> getStateMappers() {
        return List.of(
            new StateMapper<Tree>("RootFirstChildHeight", tree -> {
                List<Node> rootChildren = tree.getRoot().getChildren();
                double firstChildHeight = Math.max(rootChildren.get(0).getHeight(), rootChildren.get(1).getHeight());
                return String.format("%.0f", 2 * firstChildHeight);
            }),
            new StateMapper<Tree>("TreeLength", tree -> {
                double treeLength = TreeUtils.getTreeLength(tree, tree.getRoot());
                return String.format("%.0f", 2 * treeLength);
            }),
            new StateMapper<Tree>("TreeImbalance", tree -> String.valueOf(rootImbalance(tree.getRoot())))
        );
    }

    @Test
    public void testWilsonBalding() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            WilsonBalding operator = new WilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0);
            return operator;
        }));
    }

    /**
     * A {@link Trial} over trees: a single Yule-distributed tree is repeatedly
     * re-drawn by a {@link DirectSimulator} and perturbed by one operator, both
     * of which are constructed once for the whole test. The operator is supplied
     * as a function of the tree it operates on.
     */
    class TreeTrial implements Trial<Tree> {
        final Tree tree = new Tree();
        final TreeDistribution prior;
        final DirectSimulator simulator = new DirectSimulator();
        final TreeOperator operator;

        TreeTrial(Function<Tree, TreeOperator> operatorFactory) {
            tree.initByName("taxonset", getTaxonSet(NUM_TAXA));
            prior = getPrior(tree);
            simulator.initByName("distribution", prior, "nSamples", 1);
            operator = operatorFactory.apply(tree);
        }

        @Override
        public Tree nextSample() throws Exception {
            simulator.run();
            return tree;
        }

        @Override
        public Distribution prior() {
            return prior;
        }

        @Override
        public double proposal() {
            return operator.proposal();
        }
    }

    protected TaxonSet getTaxonSet(int numTaxa) {
        TaxonSet taxonSet = new TaxonSet();
        for (int i = 0; i < numTaxa; i++) {
            taxonSet.initByName("taxon", new Taxon(String.valueOf(i)));
        }
        return taxonSet;
    }

    protected TreeDistribution getPrior(Tree tree) {
        YuleModel treePrior = new YuleModel();
        treePrior.initByName("tree", tree, "birthDiffRate", String.valueOf(BIRTH_DIFF_RATE));
        return treePrior;
    }

    static public Alignment createDummyAlignment() {
        Alignment data = new Alignment();
        data.initByName(
            "sequence", new Sequence("0", "TGATAAAGAGTTACTAGAGTAAATAATAGGAGCTCCCCCTAGACTATG"),
            "sequence", new Sequence("1", "CGATACAGAATTACTAGAGTAAATAATAGGAGTATCCCCCTGACTATA"),
            "sequence", new Sequence("2", "TGATAAAGAAATACTAGAGTAAATAATAGGAGTTTCCCCTTGACTAAG"),
            "sequence", new Sequence("3", "AGATATAGAGTTACTAGAGTAAATAATAGAGGTACCCGCTTGACAATG"),
            "sequence", new Sequence("4", "TGACA-AGAGTTACTAGAGTAAAAAATAGAGGTCTCCCCTTCAGTATG")
        );
        return data;
    }

}
