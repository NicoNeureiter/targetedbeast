package targetedbeast;

import targetedbeast.alignment.ConsensusAlignment;
import targetedbeast.edgeweights.ConstantWeights;
import targetedbeast.edgeweights.FelsensteinWeights;
import targetedbeast.edgeweights.PCAWeights;
import targetedbeast.edgeweights.ParsimonyWeights2;
import targetedbeast.likelihood.SlowTreeLikelihood;
import targetedbeast.operators.TargetedWilsonBalding;
import targetedbeast.operators.TargetedWilsonBaldingFixedHeight;
import targetedbeast.util.Counter;
import targetedbeast.util.DefaultHashMap;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
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


public class DetailedBalanceTest {

    protected static final int NUM_SAMPLES = 200_000;
    protected static final int NUM_TAXA = 5;
    protected static final double BIRTH_DIFF_RATE = 2.0;
    protected static final double TOLERANCE = 0.001;
    protected static Alignment alignment;

    public interface TreeGroupMapper {
        public String op(Tree tree);
    }

    public static Map<String, TreeGroupMapper> treeGroupers;

    @BeforeClass
    public static void setUpClass() {
        alignment = createDummyAlignment();

        treeGroupers = new HashMap<>();

        // treeGroupers.put("RootHeight1Decimal", tree -> String.format("%.1f", (1 * tree.getRoot().getHeight())));
        treeGroupers.put("RootFirstChildHeight", tree -> {
            List<Node> rootChildren = tree.getRoot().getChildren();
            double firstChildHeight = Math.max(rootChildren.get(0).getHeight(),
                                               rootChildren.get(1).getHeight());
            return String.format("%.0f", 2 * firstChildHeight);
        });

        treeGroupers.put("TreeLength", tree -> {
            double treeLength = TreeUtils.getTreeLength(tree, tree.getRoot());
            return String.format("%.0f", 2 * treeLength);
        });

        treeGroupers.put("TreeImbalance", tree -> String.valueOf(rootImbalance(tree.getRoot())));
    }

    @Test
    public void testWilsonBalding() throws Exception {
        TargetedWilsonBaldingConstFactory opFactory = new TargetedWilsonBaldingConstFactory();
        testDetailedBalance(opFactory);
    }

    @Test
    public void testTargWilsonBalding() throws Exception {
        TargetedWilsonBaldingFactory opFactory = new TargetedWilsonBaldingFactory();
        testDetailedBalance(opFactory);
    }

    @Test
    public void testTargWilsonBaldingFelsenstein() throws Exception {
        TargetedWilsonBaldingFelsensteinFactory opFactory = new TargetedWilsonBaldingFelsensteinFactory();
        testDetailedBalance(opFactory);
    }

    @Test
    public void testTargWilsonBaldingFixedHeight() throws Exception {
        TargetedWilsonBaldingFactory opFactory = new TargetedWilsonBaldingFactory();
        testDetailedBalance(opFactory);
    }

    public void testDetailedBalance(OperatorFactory operatorFactory) throws Exception {
        // Create a map from mapper name to data structures
        Map<String, Counter<String>> groupCounters = new HashMap<>();
        Map<String, Counter<String>> proposalCounters = new HashMap<>();
        Map<String, DefaultHashMap<String, Double>> flows = new HashMap<>();

        // Initialize data structures for each mapper
        for (String mapperName : treeGroupers.keySet()) {
            groupCounters.put(mapperName, new Counter<>());
            proposalCounters.put(mapperName, new Counter<>());
            flows.put(mapperName, new DefaultHashMap<>(0.0));
        }

        Tree tree = new Tree();
        tree.initByName("taxonset", getTaxonSet(NUM_TAXA));

        Distribution prior = getPrior(tree);
        DirectSimulator simulator = new DirectSimulator();
        simulator.initByName("distribution", prior, "nSamples", 1);

        // Create operator once outside the loop to avoid memory issues
        TreeOperator operator = operatorFactory.getOperator(tree);

        for (int i = 0; i < NUM_SAMPLES; i++) {

            simulator.run();

            // Compute all beforeKeys for all mappers
            Map<String, String> beforeKeys = new HashMap<>();
            for (String mapperName : treeGroupers.keySet()) {
                TreeGroupMapper mapper = treeGroupers.get(mapperName);
                String beforeKey = mapper.op(tree);
                beforeKeys.put(mapperName, beforeKey);
                groupCounters.get(mapperName).increment(beforeKey);
            }

            double logPBefore = prior.calculateLogP();

            double logHR = operator.proposal();

            double logPAfter = prior.calculateLogP();

            double pAccept = Math.min(1, Math.exp(logPAfter - logPBefore + logHR));

            // Compute all afterKeys for all mappers and update flow counters
            for (String mapperName : treeGroupers.keySet()) {
                TreeGroupMapper mapper = treeGroupers.get(mapperName);
                String afterKey = mapper.op(tree);
                String beforeKey = beforeKeys.get(mapperName);

                String transitionKey = beforeKey + "-" + afterKey;
                // String keyNoTransition = beforeKey + "-" + beforeKey;

                // Count number of proposals
                proposalCounters.get(mapperName).increment(transitionKey);

                // Update the expected flow
                flows.get(mapperName).put(transitionKey, flows.get(mapperName).get(transitionKey) + pAccept);
                // flows.get(mapperName).put(keyNoTransition, flows.get(mapperName).get(keyNoTransition) + (1 - pAccept));

            }
        }

        // Verify detailed balance for each mapper
        for (String mapperName : treeGroupers.keySet()) {
            System.out.println("\n=== Detailed balance test for mapper: " + mapperName + " ===");
            Counter<String> groupCounter = groupCounters.get(mapperName);
            Counter<String> proposalCounter = proposalCounters.get(mapperName);
            DefaultHashMap<String, Double> flow = flows.get(mapperName);

            System.out.println(groupCounter);
            System.err.println(proposalCounter);

            for (String group : groupCounter.keySet()) {
                for (String toGroup : groupCounter.keySet()) {
                    // Only check each pair once (by excluding on order of the pair)
                    if (group.compareTo(toGroup) < 0 ) 
                        continue;
                    
                    // Skip self-loops (trivially symmetric)
                    if (group.equals(toGroup)) 
                        continue;

                    String keyForward = group + "-" + toGroup;
                    String keyBackward = toGroup + "-" + group;

                    double transitionsForward = flow.get(keyForward);
                    double transitionsBackward = flow.get(keyBackward);

                    // Skip low count/high variance groups
                    if (Math.min(groupCounter.getCount(group), groupCounter.getCount(toGroup)) < 20) 
                        continue;

                    // Skip low count/high variance groups
                    if (Math.min(proposalCounter.getCount(keyForward), proposalCounter.getCount(keyBackward)) < 20) 
                        continue;

                    double forwVariance = estimateFlowVariance(groupCounter.getCount(group), proposalCounter.getCount(keyForward), transitionsForward);
                    double backVariance = estimateFlowVariance(groupCounter.getCount(toGroup), proposalCounter.getCount(keyBackward), transitionsBackward);
                    double tolerance = 3 * Math.sqrt(forwVariance + backVariance);

                    /*
                    Operator steps should follow detailed balance between classes
                    Detailed balance: p_i * q_ij = p_j * q_ji
                    */

                    if (Math.max(transitionsForward, transitionsBackward) > 0)
                        System.out.println(
                            String.format("%-8s:  %8.4f <-> %8.4f    (tol=%8.4f     diff=%8.4f)", keyForward, transitionsForward, transitionsBackward, tolerance, Math.abs(transitionsForward - transitionsBackward))
                        );
                    Assert.assertEquals(
                        "Detailed balance [" + mapperName + "] " + group + "-" + toGroup,
                        transitionsForward, transitionsBackward, tolerance);
                }
            }
        }
    }

    double estimateFlowVariance(int stateCounts, int proposalCounts, double totalFlow) {
        double proposalFrequency = (double) proposalCounts / stateCounts;
        double meanAcceptRate = (totalFlow / proposalCounts);
        System.err.println(meanAcceptRate);
        return NUM_SAMPLES * meanAcceptRate * proposalFrequency * (1 - proposalFrequency);
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
        Sequence seq0 = new Sequence("0", "TGATAAAGAGTTACTAGAGTAAATAATAGGAGCTCCCCCTAGACTATG");
        Sequence seq1 = new Sequence("1", "CGATACAGAATTACTAGAGTAAATAATAGGAGTATCCCCCTGACTATA");
        Sequence seq2 = new Sequence("2", "TGATAAAGAAATACTAGAGTAAATAATAGGAGTTTCCCCTTGACTAAG");
        Sequence seq3 = new Sequence("3", "AGATATAGAGTTACTAGAGTAAATAATAGAGGTACCCGCTTGACAATG");
        Sequence seq4 = new Sequence("4", "TGACA-AGAGTTACTAGAGTAAAAAATAGAGGTCTCCCCTTCAGTATG");
        // Sequence seq5 = new Sequence("5", "CGACGAAGAGTTACTAGAGTAAATAACAGGGGTTTCCCCTTAACCATAGGAGTCGAACCCATCCTTGAGAATCCCTGCCACCCGTCGCACCCTGTTCTAAGTAAGGGGTTATACCCTTCCCATACTAAGAAATTTAGGTTAAACACAGACCAAGAGCC");

        Alignment data = new Alignment();
        data.initByName(
            "sequence", seq0,
            "sequence", seq1,
            "sequence", seq2,
            "sequence", seq3,
            "sequence", seq4,
            // "sequence", seq5,
            "dataType", "nucleotide"
        );
        return data;
    }

    /*
     * Factories to pass to the test function to create various operators 
     */

    abstract class OperatorFactory {

        abstract public TreeOperator getOperator(Tree tree);

        SiteModel getSiteModel() {
            SiteModel siteModel = new SiteModel();
            siteModel.initByName("substModel", new JukesCantor());
            return siteModel;
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

        PCAWeights getPCAWeights(Tree tree) {
            PCAWeights edgeWeights = new PCAWeights();
            edgeWeights.initByName("tree", tree, "data", alignment, "dimension", 3);
            return edgeWeights;
        }
        
        FelsensteinWeights getLikelihoodWeights(Tree tree) {
            SlowTreeLikelihood treeLikelihood = new SlowTreeLikelihood();
            treeLikelihood.initByName("tree", tree, "siteModel", getSiteModel(), "data", alignment, "implementation", "SlowBeerLikelihoodCore4");
            FelsensteinWeights edgeWeights = new FelsensteinWeights();
            edgeWeights.initByName("tree", tree, "likelihood", treeLikelihood, "data", alignment);
            return edgeWeights;
        }
        
    }

    class TargetedWilsonBaldingConstFactory extends OperatorFactory {

        @Override
        public TargetedWilsonBalding getOperator(Tree tree) {
            // alignment = new SimulatedAlignment();
            // alignment.initByName("tree", tree, "siteModel", getSiteModel(), "sequenceLength", 5, "dataType", "nucleotide");
            TargetedWilsonBalding operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getConstantWeights(tree));
            return operator;
        }
    }

    class TargetedWilsonBaldingFactory extends OperatorFactory {

        @Override
        public TargetedWilsonBalding getOperator(Tree tree) {
            TargetedWilsonBalding operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getParsimonyWeights(tree));
            return operator;
        }
    }

    class TargetedWilsonBaldingFelsensteinFactory extends OperatorFactory {

        @Override
        public TargetedWilsonBalding getOperator(Tree tree) {
            TargetedWilsonBalding operator = new TargetedWilsonBalding();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getLikelihoodWeights(tree));
            return operator;
        }
    }


    class TargetedWilsonBaldingFixedHeightFactory extends OperatorFactory {

        @Override
        public TargetedWilsonBaldingFixedHeight getOperator(Tree tree) {
            TargetedWilsonBaldingFixedHeight operator = new TargetedWilsonBaldingFixedHeight();
            operator.initByName("tree", tree, "weight", 1.0, "edgeWeights", getPCAWeights(tree));
            return operator;
        }
    }


    /**
     * Compute tree-level imbalance as weighted sum of node imbalances
     * @param tree The tree to compute imbalance for
     * @return Weighted sum of node imbalances
     */
    static public double treeImbalance(Tree tree) {
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

    static public int rootImbalance(Node root) {
        return Math.abs(root.getLeft().getLeafNodeCount() - root.getRight().getLeafNodeCount());
    }

    /**
     * Helper class to hold imbalance and weight values
     */
    static class ImbalanceResult {
        public final double imbalance;
        public final double weight;
        
        public ImbalanceResult(double imbalance, double weight) {
            this.imbalance = imbalance;
            this.weight = weight;
        }
    }
    
    /**
     * Compute node imbalance metric
     * @param node The node to compute imbalance for
     * @return ImbalanceResult containing imbalance and weight
     */
    static public ImbalanceResult nodeImbalance(Node node) {
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

}
