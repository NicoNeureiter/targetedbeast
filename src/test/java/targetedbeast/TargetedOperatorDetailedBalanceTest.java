package targetedbeast;

import targetedbeast.alignment.ConsensusAlignment;
import targetedbeast.edgeweights.CombinedEdgeWeights;
import targetedbeast.edgeweights.ConstantWeights;
import targetedbeast.edgeweights.EdgeWeights;
import targetedbeast.edgeweights.LearnedWeights;
import targetedbeast.edgeweights.PCAWeights;
import targetedbeast.edgeweights.PCAWeights2;
import targetedbeast.edgeweights.PCAWeightsBrownian;
import targetedbeast.edgeweights.PCAWeightsBrownianFullLikelihood;
import targetedbeast.edgeweights.ParsimonyWeights2;
import targetedbeast.edgeweights.PartialWeights;
import beast.base.spec.evolution.likelihood.TreeLikelihood;
import targetedbeast.operators.IntervalScaleOperator;
import targetedbeast.operators.RangeSlide;
import targetedbeast.operators.TargetedWilsonBalding;
import targetedbeast.operators.TargetedWilsonBaldingFixedHeight;
import targetedbeast.operators.TargetedWilsonBaldingHeightDelta;
import targetedbeast.operators.WeightedWideOperator;
import org.junit.BeforeClass;
import org.junit.Test;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.evolution.operator.TreeOperator;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.speciation.YuleModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
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
    public void testTargWilsonBaldingFixedHeight() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TreeOperator operator = new TargetedWilsonBaldingFixedHeight();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getParsimonyWeights(tree));
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingMaxHeightRatio() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getParsimonyWeights(tree),
                    "useEdgeLength", false,
                    "maxHeightRatio", 1.0);
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingHeightDelta() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TreeOperator operator = new TargetedWilsonBaldingHeightDelta();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getParsimonyWeights(tree), "size", 0.25);
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingHeightDeltaEdgeLength() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TreeOperator operator = new TargetedWilsonBaldingHeightDelta();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getParsimonyWeights(tree), "useEdgeLength", true, "size", 0.25);
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingBrownianFullLikelihoodWeights() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TargetedWilsonBalding operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getPCAWeightsBrownianFullLikelihood(tree));
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


    /**
     * Detailed balance with a target-weight transform whose coefficients are deliberately
     * far from the defaults. The transform must be applied identically in the forward and
     * reverse blocks; if it is not, the forward and backward acceptance flows diverge.
     *
     * <p>optimise="false" is required: the harness accumulates flow across samples, which is
     * only meaningful for a fixed proposal kernel.
     */
    @Test
    public void testTargWilsonBaldingTargetTransform() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 3.7, "offset", 0.3, "optimise", false);

            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getPCAWeights(tree),
                    "useEdgeLength", false,
                    "targetWeightTransform", transform);
            return operator;
        }));
    }

    /**
     * A grid tuner makes the operator a mixture over several fixed-coefficient kernels, one per
     * grid point. Each is a valid MH move whose Hastings ratio comes from its own coefficients,
     * frozen for the whole proposal, so the mixture satisfies detailed balance too - provided
     * the coefficients are never averaged across grid points. This is what checks that.
     *
     * <p>explorationProb="1.0" keeps the mixing weights uniform forever, which is what makes the
     * mixture a fixed kernel and so meaningful to accumulate flow across.
     */
    @Test
    public void testTargWilsonBaldingGridTransform() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.weighttransform.GridTuner tuner =
                    new targetedbeast.weighttransform.GridTuner();
            tuner.initByName("exponents", "1 3.7 12", "explorationProb", 1.0);

            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 3.7, "offset", 0.3, "tuner", tuner);

            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getPCAWeights(tree),
                    "useEdgeLength", false,
                    "targetWeightTransform", transform);
            return operator;
        }));
    }

    /**
     * The same, with an offset large enough that the proposal is close to uniform over the
     * candidate set. Catches normalization errors that a small offset would mask.
     */
    @Test
    public void testTargWilsonBaldingTargetTransformHeavyOffset() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 1.0, "offset", 8.0, "optimise", false);

            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getPCAWeights(tree),
                    "useEdgeLength", false,
                    "targetWeightTransform", transform);
            return operator;
        }));
    }

    /**
     * Detailed balance for the source-selection path under a transform.
     *
     * <p>StructuralSourceWeights gives uniform target weights (so the target term cancels)
     * and strongly depth-dependent, uncached source weights, so a mistake in the reverse
     * source term cannot be blamed on stale caches.
     */
    @Test
    public void testTargWilsonBaldingSourceTransform() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.edgeweights.StructuralSourceWeights edgeWeights =
                    new targetedbeast.edgeweights.StructuralSourceWeights();
            edgeWeights.initByName("tree", tree);

            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 0.6, "offset", 0.4, "optimise", false);

            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", edgeWeights,
                    "useEdgeLength", false,
                    "edgeWeightTransform", transform);
            return operator;
        }));
    }

    /**
     * The source transform with useEdgeLength=true. This path used to update the reverse
     * total incrementally, which is only valid when the total is a plain sum; a transform
     * makes it nonlinear, so the whole vector must be recomputed.
     */
    @Test
    public void testTargWilsonBaldingSourceTransformEdgeLength() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 2.5, "offset", 0.2, "optimise", false);

            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getPCAWeights(tree),
                    "useEdgeLength", true,
                    "edgeWeightTransform", transform);
            return operator;
        }));
    }

    /** The transform must never make the root selectable as a source edge. */
    @Test
    public void testTargWilsonBaldingSourceTransformNeverSelectsRoot() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            // A huge offset makes the proposal nearly uniform over source edges, which is
            // when a wrongly-included root would be picked most often.
            transform.initByName("exponent", 1.0, "offset", 50.0, "optimise", false);

            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getPCAWeights(tree),
                    "useEdgeLength", false,
                    "edgeWeightTransform", transform);
            return operator;
        }));
    }

    // ------------------------------------------------------------------------------------
    // The same two transform paths on the other operators that accept them. Every transform
    // here sets optimise=false: an adapting coefficient changes the kernel between proposals,
    // which is legitimate in a chain but makes a fixed-kernel balance test meaningless.
    // ------------------------------------------------------------------------------------

    @Test
    public void testTargWilsonBaldingFixedHeightTargetTransform() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 3.0, "offset", 0.4, "optimise", false);

            TreeOperator operator = new TargetedWilsonBaldingFixedHeight();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getParsimonyWeights(tree),
                    "targetWeightTransform", transform);
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingFixedHeightSourceTransform() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.edgeweights.StructuralSourceWeights edgeWeights =
                    new targetedbeast.edgeweights.StructuralSourceWeights();
            edgeWeights.initByName("tree", tree);

            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 0.6, "offset", 0.4, "optimise", false);

            TreeOperator operator = new TargetedWilsonBaldingFixedHeight();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", edgeWeights,
                    "edgeWeightTransform", transform);
            return operator;
        }));
    }

    /** The branch-length source path, whose reverse total is rebuilt rather than adjusted. */
    @Test
    public void testTargWilsonBaldingFixedHeightSourceTransformEdgeLength() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 2.5, "offset", 0.2, "optimise", false);

            TreeOperator operator = new TargetedWilsonBaldingFixedHeight();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getParsimonyWeights(tree),
                    "useEdgeLength", true,
                    "edgeWeightTransform", transform);
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingHeightDeltaTargetTransform() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 3.0, "offset", 0.4, "optimise", false);

            TreeOperator operator = new TargetedWilsonBaldingHeightDelta();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getParsimonyWeights(tree),
                    "size", 0.25,
                    "targetWeightTransform", transform);
            return operator;
        }));
    }

    /**
     * The source path with useEdgeLength=false, which is the mode whose source-selection
     * term used to be omitted from the Hastings ratio entirely.
     */
    @Test
    public void testTargWilsonBaldingHeightDeltaSourceTransform() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.edgeweights.StructuralSourceWeights edgeWeights =
                    new targetedbeast.edgeweights.StructuralSourceWeights();
            edgeWeights.initByName("tree", tree);

            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 0.6, "offset", 0.4, "optimise", false);

            TreeOperator operator = new TargetedWilsonBaldingHeightDelta();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", edgeWeights,
                    "size", 0.25,
                    "edgeWeightTransform", transform);
            return operator;
        }));
    }

    @Test
    public void testWeightedWideTargetTransform() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 3.0, "offset", 0.4, "optimise", false);

            WeightedWideOperator operator = new WeightedWideOperator();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getPCAWeights(tree),
                    "targetWeightTransform", transform);
            return operator;
        }));
    }

    @Test
    public void testWeightedWideSourceTransform() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.edgeweights.StructuralSourceWeights edgeWeights =
                    new targetedbeast.edgeweights.StructuralSourceWeights();
            edgeWeights.initByName("tree", tree);

            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 0.6, "offset", 0.4, "optimise", false);

            WeightedWideOperator operator = new WeightedWideOperator();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", edgeWeights,
                    "edgeWeightTransform", transform);
            return operator;
        }));
    }

    /** Baseline for the operator below: RangeSlide had no balance coverage at all. */
    @Test
    public void testRangeSlide() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            RangeSlide operator = new RangeSlide();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getPCAWeights(tree),
                    "useWeightedStep", true,
                    "size", 0.25,
                    "optimise", false);
            return operator;
        }));
    }

    /**
     * RangeSlide uses its target weights as they come rather than squaring them, so a
     * transform here replaces an identity rather than an exponent of 2.
     */
    @Test
    public void testRangeSlideTargetTransform() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 3.0, "offset", 0.4, "optimise", false);

            RangeSlide operator = new RangeSlide();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getPCAWeights(tree),
                    "useWeightedStep", true,
                    "size", 0.25,
                    "optimise", false,
                    "targetWeightTransform", transform);
            return operator;
        }));
    }

    @Test
    public void testRangeSlideSourceTransform() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.edgeweights.StructuralSourceWeights edgeWeights =
                    new targetedbeast.edgeweights.StructuralSourceWeights();
            edgeWeights.initByName("tree", tree);

            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 0.6, "offset", 0.4, "optimise", false);

            RangeSlide operator = new RangeSlide();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", edgeWeights,
                    "useWeightedStep", true,
                    "size", 0.25,
                    "optimise", false,
                    "edgeWeightTransform", transform);
            return operator;
        }));
    }

    PCAWeightsBrownianFullLikelihood getPCAWeightsBrownianFullLikelihood(Tree tree) {
        PCAWeightsBrownianFullLikelihood edgeWeights = new PCAWeightsBrownianFullLikelihood();
        edgeWeights.initByName("tree", tree, "data", alignment, "dimension", 3);
        return edgeWeights;
    }
}
