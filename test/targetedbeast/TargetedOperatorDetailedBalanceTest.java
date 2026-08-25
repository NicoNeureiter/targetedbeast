package targetedbeast;

import targetedbeast.alignment.ConsensusAlignment;
import targetedbeast.edgeweights.CombinedEdgeWeights;
import targetedbeast.edgeweights.ConstantWeights;
import targetedbeast.edgeweights.EdgeWeights;
import targetedbeast.edgeweights.LearnedWeights;
import targetedbeast.edgeweights.PCAWeights;
import targetedbeast.edgeweights.PCAWeights2;
import targetedbeast.edgeweights.PCAWeightsBrownian;
import targetedbeast.edgeweights.ParsimonyWeights2;
import targetedbeast.edgeweights.PartialWeights;
import beast.base.evolution.likelihood.TreeLikelihood;
import targetedbeast.operators.IntervalScaleOperator;
import targetedbeast.operators.RangeSlide;
import targetedbeast.operators.TargetedWilsonBalding;
import targetedbeast.operators.TargetedWilsonBaldingFixedHeight;
import targetedbeast.operators.WeightedWideOperator;
import org.junit.BeforeClass;
import org.junit.Test;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.branchratemodel.StrictClockModel;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.speciation.YuleModel;
import beast.base.evolution.substitutionmodel.JukesCantor;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeDistribution;
import beast.base.evolution.tree.TreeUtils;
import beast.base.inference.DirectSimulator;
import beast.base.inference.Distribution;
import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;
import static targetedbeast.util.TreeImbalance.rootImbalance;


/**
 * Detailed-balance test for the targeted tree operators.
 *
 * <p>Specialises {@link DetailedBalanceTest} for {@link Tree} states sampled from
 * a {@link YuleModel} prior, with tree-level quantisation functions
 * (root child height, tree length, imbalance). Each {@code @Test} method builds a
 * {@link TreeTrial} with one operator/edge-weight combination, defined inline.
 */
public class TargetedOperatorDetailedBalanceTest extends DetailedBalanceTest<Tree> {

    /** Sample count; override with -Ddb.numSamples=... to raise the test's statistical power. */
    protected static final int NUM_SAMPLES = Integer.getInteger("db.numSamples", 100_000);
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
        testDetailedBalance(
            new TreeTrial(tree -> createOperator(
                TargetedWilsonBalding::new, tree, 
                "edgeWeights", getConstantWeights(tree))
            )
        );
        // testDetailedBalance(new TreeTrial(tree -> {
        //     TreeOperator op = new TargetedWilsonBalding();
        //     op.initByName("tree", tree, "weight", 1.0, "edgeWeights", getConstantWeights(tree));
        //     return op;
        // }));
    }

    TreeOperator createOperator(final Supplier<? extends TreeOperator> factory, final Tree tree, final Object... initArgs){
        TreeOperator operator = factory.get();
        operator.initByName("tree", tree, "weight", 1.0);
        operator.initByName(initArgs);
        return operator;
    }

    @Test
    public void testTargWilsonBalding() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getPCAWeights(tree), "useEdgeLength", false);
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingFixedNodeWeights() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.edgeweights.FixedNodeWeights edgeWeights = new targetedbeast.edgeweights.FixedNodeWeights();
            edgeWeights.initByName("tree", tree);
            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", edgeWeights, "useEdgeLength", false);
            return operator;
        }));
    }

    /**
     * Isolates the source-selection term: source weights vary strongly with tree structure,
     * target weights are uniform, and nothing is cached. Detailed balance here requires the
     * operator to include the source-selection probability in its Hastings ratio.
     */
    @Test
    public void testTargWilsonBaldingStructuralSourceWeights() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.edgeweights.StructuralSourceWeights edgeWeights =
                    new targetedbeast.edgeweights.StructuralSourceWeights();
            edgeWeights.initByName("tree", tree);
            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", edgeWeights, "useEdgeLength", false);
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingStructuralWeights() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.edgeweights.StructuralWeights edgeWeights = new targetedbeast.edgeweights.StructuralWeights();
            edgeWeights.initByName("tree", tree);
            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", edgeWeights, "useEdgeLength", false);
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingPartial() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getPartialWeights(tree));
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingFixedHeightEdgeLength() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TreeOperator operator = new TargetedWilsonBaldingFixedHeight();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getParsimonyWeights(tree), "useEdgeLength", true);
            return operator;
        }));
    }



    @Test
    public void testIntervalScaleOperator() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TreeOperator op = new IntervalScaleOperator();
            op.initByName("tree", tree, "weight", "1");
            return op;
        }));
    }

    // @Test
    // public void testTargWilsonBaldingLearnedWeights() throws Exception {
    //     testDetailedBalance(new TreeTrial(tree -> {
    //         TreeLikelihood treeLikelihood = new TreeLikelihood();
    //         treeLikelihood.initByName(
    //                 "tree", tree,
    //                 "siteModel", getSiteModel(),
    //                 "branchRateModel", getStrictClockModel(),
    //                 "data", alignment,
    //                 "implementation", "BeerLikelihoodCore4");

    //         LearnedWeights edgeWeights = new LearnedWeights();
    //         edgeWeights.initByName(
    //                 "tree", tree,
    //                 "data", alignment,
    //                 "likelihood", treeLikelihood,
    //                 "dimension", 3,
    //                 "nTrainingTrees", 5,
    //                 "maxIterations", 200);

    //         TargetedWilsonBalding operator = new TargetedWilsonBalding();
    //         operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", edgeWeights);
    //         return operator;
    //     }));
    // }

    @Test
    public void testTargWilsonBaldingBrownianWeights() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TargetedWilsonBalding operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getPCAWeightsBrownian(tree));
            return operator;
        }));
    }



    @Test
    public void testWeightedWide() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            WeightedWideOperator operator = new WeightedWideOperator();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getPCAWeights(tree));
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
        final List<targetedbeast.edgeweights.EdgeWeights> edgeWeights = new ArrayList<>();

        TreeTrial(Function<Tree, TreeOperator> operatorFactory) {
            tree.initByName("taxonset", getTaxonSet(NUM_TAXA));
            prior = getPrior(tree);
            simulator.initByName("distribution", prior, "nSamples", 1);
            operator = operatorFactory.apply(tree);
            // Collect any edge-weight caches the operator depends on so the harness can keep
            // them in sync with the tree (see nextSample).
            for (beast.base.core.Input<?> input : operator.getInputs().values()) {
                if (input.get() instanceof targetedbeast.edgeweights.EdgeWeights ew) {
                    edgeWeights.add(ew);
                }
            }
        }

        @Override
        public Tree nextSample() throws Exception {
            simulator.run();
            // The DirectSimulator redraws the tree outside the MCMC State machinery, so the
            // operators' incremental edge-weight caches are NOT resynced the way they would be
            // in a real chain (via requiresRecalculation after each accept/restore). Without
            // this, the operator's within-proposal incremental updates build on a stale base
            // (the previous sample's tree), producing tree-inconsistent, path-dependent weights
            // that break detailed balance. Force a fresh, CLEAN base here to mimic production:
            // setEverythingDirty(true) triggers a full recompute; setEverythingDirty(false) then
            // clears the flags so the operator's own move drives the incremental updates.
            for (targetedbeast.edgeweights.EdgeWeights ew : edgeWeights) {
                tree.setEverythingDirty(true);
                ew.updateByOperator();
            }
            tree.setEverythingDirty(false);
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
        treePrior.initByName(
                "tree", tree,
                "birthDiffRate", "" + BIRTH_DIFF_RATE
        );
        return treePrior;
    }

    static public Alignment createDummyAlignment() {
        Alignment data = new Alignment();
        data.initByName(
            "sequence", new Sequence("0", "TGATAAAGAGTTACTAGAGTAAATAATAGGAGCTCCCCCTAGACTATG"),
            "sequence", new Sequence("1", "CGATACAGAATTACTAGAGTAAATAATAGGAGTATCCCCCTGACTATA"),
            "sequence", new Sequence("2", "TGATAAAGAAATACTAGAGTAAATAATAGGAGTTTCCCCTTGACTAAG"),
            "sequence", new Sequence("3", "AGATATAGAGTTACTAGAGTAAATAATAGAGGTACCCGCTTGACAATG"),
            "sequence", new Sequence("4", "TGACA-AGAGTTACTAGAGTAAAAAATAGAGGTCTCCCCTTCAGTATG"),
            "dataType", "nucleotide"
        );
        return data;
    }

    /*
     * Builders for the substitution/clock models and edge weights used by the operators above.
     */

    SiteModel getSiteModel() {
        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", new JukesCantor());
        return siteModel;
    }

    StrictClockModel getStrictClockModel() {
        StrictClockModel clock = new StrictClockModel();
        clock.initByName("clock.rate", "1.0");
        return clock;
    }

    ConstantWeights getConstantWeights(Tree tree) {
        ConstantWeights edgeWeights = new ConstantWeights();
        edgeWeights.initByName("tree", tree);
        return edgeWeights;
    }

    ParsimonyWeights2 getParsimonyWeights(Tree tree) {
        ConsensusAlignment consAlignment = new ConsensusAlignment();
        consAlignment.initByName("data", alignment);
        ParsimonyWeights2 edgeWeights = new ParsimonyWeights2();
        edgeWeights.initByName("tree", tree, "data", consAlignment);
        return edgeWeights;
    }

    EdgeWeights getPCAWeights(Tree tree) {
        PCAWeights2 edgeWeights = new PCAWeights2();
        edgeWeights.initByName("tree", tree, "data", alignment, "dimension", 3);
        return edgeWeights;
    }

    PCAWeightsBrownian getPCAWeightsBrownian(Tree tree) {
        PCAWeightsBrownian edgeWeights = new PCAWeightsBrownian();
        edgeWeights.initByName("tree", tree, "data", alignment, "dimension", 3);
        return edgeWeights;
    }





    PartialWeights getPartialWeights(Tree tree) {
        TreeLikelihood treeLikelihood = new TreeLikelihood();
        treeLikelihood.initByName("tree", tree, "siteModel", getSiteModel(), "data", alignment, "implementation", "BeerLikelihoodCore4");
        PartialWeights edgeWeights = new PartialWeights();
        edgeWeights.initByName("tree", tree, "likelihood", treeLikelihood, "data", alignment);
        return edgeWeights;
    }

}
