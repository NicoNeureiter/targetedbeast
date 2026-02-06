package targetedbeast;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.junit.BeforeClass;
import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.likelihood.TreeLikelihood;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.substitutionmodel.JukesCantor;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.DirectSimulator;
import beast.base.evolution.speciation.YuleModel;
import beast.base.util.Randomizer;
import targetedbeast.edgeweights.FelsensteinWeights;
import targetedbeast.edgeweights.ParsimonyWeights2;
import targetedbeast.likelihood.SlowTreeLikelihood;

/**
 * Test to verify that target weights computed by FelsensteinWeights are invariant
 * when the ignored subtree is moved via Wilson-Balding.
 * 
 * The key insight: when ignoring node i, its parent p becomes unary. A unary node
 * in Felsenstein pruning is "transparent" - transition(A→p) × transition(p→CiP) = transition(A→CiP).
 * So target weights from i to candidate nodes should be the same before/after moving i.
 */
public class FelsensteinWeightsWithoutNodeTest {

    private static Alignment alignment;

    @BeforeClass
    public static void setUpClass() {
        alignment = DetailedBalanceTest.createDummyAlignment();
    }

    /**
     * Test that target weights are invariant when computed before vs after
     * a Wilson-Balding style move of the ignored node.
     */
    @Test
    public void targetWeightsInvariantAfterMove() throws Exception {
        // Randomizer.setSeed(42);
        
        Tree tree = new Tree();
        tree.initByName("taxonset", getTaxonSet(alignment.getTaxonCount()));

        // Simulate a tree from a Yule prior
        YuleModel yulePrior = new YuleModel();
        yulePrior.initByName("tree", tree, "birthDiffRate", "2.0");
        DirectSimulator simulator = new DirectSimulator();
        simulator.initByName("distribution", yulePrior, "nSamples", 1);
        simulator.run();

        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", new JukesCantor());

        SlowTreeLikelihood treeLikelihood = new SlowTreeLikelihood();
        treeLikelihood.initByName(
                "tree", tree,
                "siteModel", siteModel,
                "data", alignment,
                "implementation", "SlowBeerLikelihoodCore4");

        FelsensteinWeights edgeWeights = new FelsensteinWeights();
        edgeWeights.initByName("tree", tree, "likelihood", treeLikelihood, "data", alignment);
        // ParsimonyWeights2 edgeWeights = new ParsimonyWeights2();
        // edgeWeights.initByName("tree", tree, "data", alignment);

        // Initialize likelihoods
        treeLikelihood.calculateLogP();

        // Pick a node i that can be moved (not root, parent not root)
        Node nodeI = pickMovableNode(tree);
        Node p = nodeI.getParent();
        Node CiP = getOtherChild(p, nodeI);
        Node PiP = p.getParent();

        // Get candidate nodes (edges that exist at nodeI's height, excluding i and p)
        List<Node> candidates = getCoExistingLineages(nodeI, tree);
        candidates.remove(p);
        candidates.remove(nodeI);
        
        // Find a target j different from CiP
        Node j = null;
        for (Node candidate : candidates) {
            if (candidate != CiP && !candidate.isRoot()) {
                j = candidate;
                break;
            }
        }
        if (j == null) {
            System.out.println("Skipping test - no suitable target node found");
            return;
        }
        Node jP = j.getParent();

        // Compute ancestors of i for partialsWithoutNode
        List<Integer> ancestorsBefore = collectAncestors(nodeI, candidates);

        // Compute target weights BEFORE the move
        edgeWeights.prestore();
        edgeWeights.updateByOperatorWithoutNode(nodeI.getNr(), ancestorsBefore);
        double[] weightsBefore = edgeWeights.getTargetWeights(nodeI.getNr(), candidates);
        edgeWeights.reset();

        // Store the weights to specific nodes for comparison
        int idxCiP = candidates.indexOf(CiP);
        int idxJ = candidates.indexOf(j);
        double weightToCiPBefore = weightsBefore[idxCiP];
        double weightToJBefore = weightsBefore[idxJ];

        // Capture node numbers and branch rates before the move
        int nodeINrBefore = nodeI.getNr();
        int pNrBefore = p.getNr();
        int ciPNrBefore = CiP.getNr();
        int jNrBefore = j.getNr();

        double nodeIRateBefore = edgeWeights.getBranchRate(nodeI);
        double pRateBefore = edgeWeights.getBranchRate(p);
        double ciPRateBefore = edgeWeights.getBranchRate(CiP);
        double jRateBefore = edgeWeights.getBranchRate(j);

        // Perform a Wilson-Balding move: detach i+p, reattach on edge above j
        double newHeight = (j.getHeight() + jP.getHeight()) / 2.0;
        p.setHeight(newHeight);
        
        // Detach: PiP adopts CiP directly
        PiP.removeChild(p);
        PiP.addChild(CiP);
        
        // Reattach: p goes between j and jP
        jP.removeChild(j);
        jP.addChild(p);
        p.removeChild(CiP);
        p.addChild(j);

        // Recompute likelihoods after the move
        treeLikelihood.calculateLogP();

        // Verify node indices are stable across the move
        assertEquals("nodeI index should be invariant", nodeINrBefore, nodeI.getNr());
        assertEquals("p index should be invariant", pNrBefore, p.getNr());
        assertEquals("CiP index should be invariant", ciPNrBefore, CiP.getNr());
        assertEquals("j index should be invariant", jNrBefore, j.getNr());

        // Verify branch rates are stable across the move
        assertEquals("nodeI branch rate should be invariant", nodeIRateBefore, edgeWeights.getBranchRate(nodeI), 1e-12);
        assertEquals("p branch rate should be invariant", pRateBefore, edgeWeights.getBranchRate(p), 1e-12);
        assertEquals("CiP branch rate should be invariant", ciPRateBefore, edgeWeights.getBranchRate(CiP), 1e-12);
        assertEquals("j branch rate should be invariant", jRateBefore, edgeWeights.getBranchRate(j), 1e-12);

        // Compute ancestors of i for the new tree
        List<Node> candidatesAfter = getCoExistingLineages(nodeI, tree);
        candidatesAfter.remove(p);
        candidatesAfter.remove(nodeI);
        List<Integer> ancestorsAfter = collectAncestors(nodeI, candidatesAfter);

        // Compute target weights AFTER the move
        edgeWeights.prestore();
        edgeWeights.updateByOperatorWithoutNode(nodeI.getNr(), ancestorsAfter);
        double[] weightsAfter = edgeWeights.getTargetWeights(nodeI.getNr(), candidatesAfter);
        edgeWeights.reset();

        // Find CiP and j in the new candidate list
        int idxCiPAfter = candidatesAfter.indexOf(CiP);
        int idxJAfter = candidatesAfter.indexOf(j);

        // The weights from i to CiP and j should be the same before/after
        // because:
        // 1. i's subtree is unchanged
        // 2. The "tree without i" (with p as unary) has the same effective partials
        System.out.println(idxCiPAfter + " | " + weightToCiPBefore);
        if (idxCiPAfter >= 0) {
            double weightToCiPAfter = weightsAfter[idxCiPAfter];
            System.out.println(weightToCiPAfter);
            assertEquals("Weight to CiP should be invariant", 
                    weightToCiPBefore, weightToCiPAfter, 1e-9);
        }
        System.out.println(idxJAfter + " | " + weightToJBefore);
        if (idxJAfter >= 0) {
            double weightToJAfter = weightsAfter[idxJAfter];
            System.out.println(weightToJAfter);
            assertEquals("Weight to j should be invariant", 
                    weightToJBefore, weightToJAfter, 1e-9);
        }
    }

    private static Node pickMovableNode(Tree tree) {
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isRoot() && !node.getParent().isRoot()) {
                return node;
            }
        }
        throw new IllegalStateException("No suitable non-root node found for move test");
    }

    private static Node getOtherChild(Node parent, Node child) {
        for (Node c : parent.getChildren()) {
            if (c != child) return c;
        }
        throw new IllegalArgumentException("Child not found in parent");
    }

    private static List<Node> getCoExistingLineages(Node node, Tree tree) {
        if (node.isRoot())
            return new ArrayList<>();
        else
            return Arrays.stream(tree.getNodesAsArray())
                .filter(n -> !n.isRoot())
                .filter(n -> n.getParent().getHeight() >= node.getParent().getHeight())
                .filter(n -> n.getHeight() < node.getParent().getHeight())
                .collect(Collectors.toList());
    }

    private static List<Integer> collectAncestors(Node nodeI, List<Node> candidates) {
        List<Integer> ancestors = new ArrayList<>();
        Node ancestor = nodeI.getParent();
        while (ancestor != null) {
            if (candidates.contains(ancestor)) {
                ancestors.add(ancestor.getNr());
            }
            ancestor = ancestor.getParent();
        }
        return ancestors;
    }

    private TaxonSet getTaxonSet(int numTaxa) {
        TaxonSet taxonSet = new TaxonSet();
        for (int i = 0; i < numTaxa; i++) {
            taxonSet.initByName("taxon", new Taxon(String.valueOf(i)));
        }
        return taxonSet;
    }
}
