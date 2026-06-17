package targetedbeast;

import targetedbeast.alignment.ConsensusAlignment;
import targetedbeast.edgeweights.CombinedEdgeWeights;
import targetedbeast.edgeweights.ConstantWeights;
import targetedbeast.edgeweights.FelsensteinWeights;
import targetedbeast.edgeweights.LearnedWeights;
import targetedbeast.edgeweights.PCAWeights;
import targetedbeast.edgeweights.PCAWeights2;
import targetedbeast.edgeweights.PCAWeightsBrownian;
import targetedbeast.edgeweights.PCAWeightsBrownianFullLikelihood;
import targetedbeast.edgeweights.ParsimonyWeights2;
import targetedbeast.edgeweights.PartialWeights;
import targetedbeast.edgeweights.PriorHeuristicWeights;
import beast.base.evolution.likelihood.TreeLikelihood;
import targetedbeast.operators.TargetedWilsonBalding;
import targetedbeast.operators.TargetedWilsonBaldingHeightDelta;
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

    protected static final int NUM_SAMPLES = 40_000;
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
            TargetedWilsonBalding operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getConstantWeights(tree));
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBalding() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TargetedWilsonBalding operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getPCAWeights(tree), "useEdgeLength", true);
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingMaxHeightRatio() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TargetedWilsonBalding operator = new TargetedWilsonBalding();
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
    public void testTargWilsonBaldingFelsenstein() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TargetedWilsonBalding operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getLikelihoodWeights(tree));
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingPartial() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TargetedWilsonBalding operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getPartialWeights(tree));
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingFixedHeight() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TargetedWilsonBaldingFixedHeight operator = new TargetedWilsonBaldingFixedHeight();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getParsimonyWeights(tree));
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingFixedHeightEdgeLength() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TargetedWilsonBaldingFixedHeight operator = new TargetedWilsonBaldingFixedHeight();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getParsimonyWeights(tree), "useEdgeLength", true);
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingHeightDelta() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TargetedWilsonBaldingHeightDelta operator = new TargetedWilsonBaldingHeightDelta();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getParsimonyWeights(tree), "size", 0.25);
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingHeightDeltaEdgeLength() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TargetedWilsonBaldingHeightDelta operator = new TargetedWilsonBaldingHeightDelta();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getParsimonyWeights(tree), "useEdgeLength", true, "size", 0.25);
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingLearnedWeights() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TreeLikelihood treeLikelihood = new TreeLikelihood();
            treeLikelihood.initByName(
                    "tree", tree,
                    "siteModel", getSiteModel(),
                    "branchRateModel", getStrictClockModel(),
                    "data", alignment,
                    "implementation", "BeerLikelihoodCore4");

            LearnedWeights edgeWeights = new LearnedWeights();
            edgeWeights.initByName(
                    "tree", tree,
                    "data", alignment,
                    "likelihood", treeLikelihood,
                    "dimension", 3,
                    "nTrainingTrees", 5,
                    "maxIterations", 200);

            TargetedWilsonBalding operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", edgeWeights);
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingBrownianWeights() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TargetedWilsonBalding operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getPCAWeightsBrownian(tree));
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
    public void testTargWilsonBaldingPriorHeuristicWeights() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TargetedWilsonBalding operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getPriorHeuristicWeights(tree));
            return operator;
        }));
    }

    @Test
    public void testTargWilsonBaldingCombinedPriorFelsensteinWeights() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            TargetedWilsonBalding operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getCombinedPriorFelsensteinWeights(tree));
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

    PCAWeights2 getPCAWeights(Tree tree) {
        PCAWeights2 edgeWeights = new PCAWeights2();
        edgeWeights.initByName("tree", tree, "data", alignment, "dimension", 3);
        return edgeWeights;
    }

    PCAWeightsBrownian getPCAWeightsBrownian(Tree tree) {
        PCAWeightsBrownian edgeWeights = new PCAWeightsBrownian();
        edgeWeights.initByName("tree", tree, "data", alignment, "dimension", 3);
        return edgeWeights;
    }

    PCAWeightsBrownianFullLikelihood getPCAWeightsBrownianFullLikelihood(Tree tree) {
        PCAWeightsBrownianFullLikelihood edgeWeights = new PCAWeightsBrownianFullLikelihood();
        edgeWeights.initByName("tree", tree, "data", alignment, "dimension", 3);
        return edgeWeights;
    }

    FelsensteinWeights getLikelihoodWeights(Tree tree) {
        TreeLikelihood treeLikelihood = new TreeLikelihood();
        treeLikelihood.initByName("tree", tree, "siteModel", getSiteModel(), "data", alignment, "implementation", "BeerLikelihoodCore4");
        FelsensteinWeights edgeWeights = new FelsensteinWeights();
        edgeWeights.initByName("tree", tree, "likelihood", treeLikelihood, "data", alignment);
        return edgeWeights;
    }

    PriorHeuristicWeights getPriorHeuristicWeights(Tree tree) {
        PriorHeuristicWeights edgeWeights = new PriorHeuristicWeights();
        edgeWeights.initByName(
                "tree", tree,
                "temperature", 1.0,
                "sourceBranchScale", 1.0,
                "attachmentAgeScale", 0.25,
                "heightIncreaseScale", 1.0,
                "maxNewBranchScale", 1.0,
                "minWeight", 1e-6,
                "maxWeight", 1e6);
        return edgeWeights;
    }

    CombinedEdgeWeights getCombinedPriorFelsensteinWeights(Tree tree) {
        CombinedEdgeWeights edgeWeights = new CombinedEdgeWeights();
        edgeWeights.initByName(
                "edgeWeights", getPriorHeuristicWeights(tree),
                "edgeWeights", getLikelihoodWeights(tree));
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
