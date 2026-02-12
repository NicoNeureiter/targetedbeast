package targetedbeast.edgeweights;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.BeforeClass;
import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.substitutionmodel.JukesCantor;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.DirectSimulator;
import beast.base.evolution.speciation.YuleModel;
import targetedbeast.DetailedBalanceTest;
import targetedbeast.likelihood.SlowTreeLikelihood;
import targetedbeast.util.LinearAlgebra;

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
        // Ancestors should be recomputed each time since the node's position changes
        List<Integer> ancestorsBefore = collectAncestors(nodeI);

        // Compute target weights BEFORE the move
        edgeWeights.prestore();
        edgeWeights.updateByOperatorWithoutNode(nodeI.getNr(), ancestorsBefore);
        double[] weightsBefore = edgeWeights.getTargetWeights(nodeI.getNr(), candidates);
        
        // Store partials for all nodes (for comparison after the move)
        double[][] partialsWithoutNodeBefore = new double[tree.getNodeCount()][];
        double[][] outsidePartialsBefore = new double[tree.getNodeCount()][];
        for (int i = 0; i < tree.getNodeCount(); i++) {
            double[] pwn = edgeWeights.getPartialsWithoutNode(i);
            double[] op = edgeWeights.getOutsidePartials(i);
            if (pwn != null) partialsWithoutNodeBefore[i] = Arrays.copyOf(pwn, pwn.length);
            if (op != null) outsidePartialsBefore[i] = Arrays.copyOf(op, op.length);
        }
        
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
        double newHeight = edgeWeights.getToHeightWithoutNode(j, nodeI);
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

        // // Recompute likelihoods after restoring the tree
        // treeLikelihood.calculateLogP();

        // Now compute target weights AFTER the "move and reverse" operation
        // This tests that the weights are invariant even if we internally perform moves
        List<Node> candidatesAfter = getCoExistingLineages(nodeI, tree);
        candidatesAfter.remove(p);
        candidatesAfter.remove(nodeI);
        List<Integer> ancestorsAfter = collectAncestors(nodeI);

        assertEquals(candidates, candidatesAfter);

        // Compute target weights AFTER the move
        edgeWeights.prestore();
        edgeWeights.updateByOperatorWithoutNode(nodeI.getNr(), ancestorsAfter);
        double[] weightsAfter = edgeWeights.getTargetWeights(nodeI.getNr(), candidatesAfter);
        
        // CRITICAL TEST: Verify that partials computed with same ignored node are identical
        // before and after the move. This isolates whether the issue is in partials computation
        // or in the weights calculation.
        List<Integer> nodesWithChangedPartials = new ArrayList<>();
        List<Integer> nodesWithChangedOutsidePartials = new ArrayList<>();
        for (int i = 0; i < tree.getNodeCount(); i++) {
            if (i == p.getNr()) continue;
            // double[] pwnAfter = edgeWeights.getPartialsForWithoutNode(i, nodeI.getNr(), new HashSet<>(ancestorsAfter));
            double[] pwnAfter = edgeWeights.getPartialsWithoutNode(i);
            double[] opAfter = edgeWeights.getOutsidePartials(i);
            
            double[] pwnBefore = partialsWithoutNodeBefore[i];
            // Check partialsWithoutNode
            if (pwnBefore != null || pwnAfter != null) {
                if (pwnBefore == null) {
                    throw new AssertionError("Node " + i + " has partialsWithoutNode after move but not before!");
                }
                if (pwnAfter == null) {
                    throw new AssertionError("Node " + i + " has partialsWithoutNode before move but not after!");
                }
                if (!allClose(pwnBefore, pwnAfter, 1E-6)) {
                    nodesWithChangedPartials.add(i);
                }
            }
            
            // Check outsidePartials
            if (outsidePartialsBefore[i] != null || opAfter != null) {
                if (outsidePartialsBefore[i] == null) {
                    throw new AssertionError("Node " + i + " has outsidePartials after move but not before!");
                }
                if (opAfter == null) {
                    throw new AssertionError("Node " + i + " has outsidePartials before move but not after!");
                }
                if (!allClose(outsidePartialsBefore[i], opAfter, 1E-6)) {
                    nodesWithChangedOutsidePartials.add(i);
                }
                for (int k = 0; k < outsidePartialsBefore[i].length; k++) {
                    assertEquals("outsidePartials[" + i + "][" + k + "] should be invariant when ignoring same node",
                            outsidePartialsBefore[i][k], opAfter[k], 1e-6);
                }
            }
        }
        assertTrue("partialsWithoutNode not invariant: " + nodesWithChangedPartials.toString(), nodesWithChangedPartials.isEmpty());
        assertTrue("outsidePartials not invariant: " + nodesWithChangedOutsidePartials.toString(), nodesWithChangedOutsidePartials.isEmpty());
        System.out.println("Partials consistency check PASSED!");
        
        edgeWeights.reset();

        // Find CiP and j in the candidate list (should be the same as before)
        int idxCiPAfter = candidatesAfter.indexOf(CiP);
        int idxJAfter = candidatesAfter.indexOf(j);

        // The weights from i to CiP and j should be the same before/after
        // because:
        // 1. i's subtree is unchanged
        // 2. The "tree without i" (with p as unary) has the same effective partials
        if (idxCiPAfter >= 0) {
            double weightToCiPAfter = weightsAfter[idxCiPAfter];
            assertEquals("Weight to CiP should be invariant", 
                    weightToCiPBefore, weightToCiPAfter, 1e-9);
        }
        if (idxJAfter >= 0) {
            double weightToJAfter = weightsAfter[idxJAfter];
            assertEquals("Weight to j should be invariant", 
                    weightToJBefore, weightToJAfter, 1e-9);
        }
    }

    boolean allClose(double[] a1, double[] a2, double tol) {
        for (int i=0; i<a1.length; i++) {
            if (Math.abs(a1[i] - a2[i]) > tol)
                return false;
        }
        return true;
    }

    /**
     * Test that inner products of partialsWithoutNode and outsidePartials
     * are consistent across nodes when ignoring a node (i.e. reflect the
     * pruned tree likelihood up to a constant).
     */
    @Test
    public void prunedTreeLikelihoodConsistentAcrossNodes() throws Exception {
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

        // Initialize likelihoods
        treeLikelihood.calculateLogP();

        // Pick a node i that can be moved (not root, parent not root)
        Node nodeI = pickMovableNode(tree);

        // Use candidate set to define ancestors to update (as in main test)
        List<Node> candidates = getCoExistingLineages(nodeI, tree);
        candidates.remove(nodeI.getParent());
        candidates.remove(nodeI);
        List<Integer> ancestors = collectAncestors(nodeI);

        // Compute partialsWithoutNode and outsidePartials for ignored node
        edgeWeights.prestore();
        edgeWeights.updateByOperatorWithoutNode(nodeI.getNr(), ancestors);

        Double referencePrunedLikelihood = null;
        int referenceNodeNr = -1;
        for (int i = 0; i < tree.getNodeCount(); i++) {
            if (i == nodeI.getNr() || i == nodeI.getParent().getNr())
                continue;

            // double[] partials = edgeWeights.getPartialsForWithoutNode(i, nodeI.getNr(), new HashSet<>(ancestors));
            double[] partials = edgeWeights.getPartialsWithoutNode(i);
            double[] outside = edgeWeights.getOutsidePartials(i);

            if (partials == null || outside == null) {
                continue;
            }
            double[] combinedPartials = LinearAlgebra.multiply(partials, outside);
            double prunedLikelihoodAtNode = edgeWeights.computeLikelihoodFromPartials(tree.getNode(i), combinedPartials);

            if (referencePrunedLikelihood == null) {
                referencePrunedLikelihood = prunedLikelihoodAtNode;
                referenceNodeNr = i;
            } else {
                assertEquals(
                        "partialsWithoutNode ⊙ outsidePartials inner product should be consistent across nodes when ignoring same node (ref node "
                                + referenceNodeNr + ")",
                        referencePrunedLikelihood,
                        prunedLikelihoodAtNode,
                        1e-9);
            }
        }

        if (referencePrunedLikelihood == null) {
            throw new AssertionError("No nodes had both partialsWithoutNode and outsidePartials computed");
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
                .filter(n -> n.getParent().getHeight() > node.getHeight())
                // .filter(n -> n.getHeight() < node.getParent().getHeight())
                .collect(Collectors.toList());
    }

    private static List<Integer> collectAncestors(Node nodeI) {
        List<Integer> ancestors = new ArrayList<>();
        Node ancestor = nodeI.getParent();
        while (ancestor != null) {
            ancestors.add(ancestor.getNr());
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
