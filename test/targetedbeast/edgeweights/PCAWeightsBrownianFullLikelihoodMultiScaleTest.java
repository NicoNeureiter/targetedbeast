package targetedbeast.edgeweights;

import static org.junit.Assert.assertFalse;

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;

public class PCAWeightsBrownianFullLikelihoodMultiScaleTest {

    private Tree tree;
    private PCAWeightsBrownianFullLikelihood weights;

    @Before
    public void setUp() {
        Alignment alignment = new Alignment();
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

        weights = new PCAWeightsBrownianFullLikelihood();
        weights.initByName(
                "tree", tree,
                "data", alignment,
                "dimension", 2,
                "value", "A=0.0 0.0,B=2.0 0.5,C=4.0 3.0",
                "offset", 0.01);
    }

    @Test
    public void perDimensionProposalScalesWeightAxisContributionsIndependently() {
        int aNr = findNodeNrById("A");
        int bNr = findNodeNrById("B");
        int cNr = findNodeNrById("C");
        double toHeight = 0.2;

        weights.setProposalLogLikelihoodScales(new double[] {2.0, 0.0});
        double[] firstAxisOnly = normalizeAfterOperatorSquare(
                weights.getTargetWeightsInteger(aNr, List.of(bNr, cNr), toHeight));

        weights.setProposalLogLikelihoodScales(new double[] {0.0, 2.0});
        double[] secondAxisOnly = normalizeAfterOperatorSquare(
                weights.getTargetWeightsInteger(aNr, List.of(bNr, cNr), toHeight));

        assertFalse(java.util.Arrays.equals(firstAxisOnly, secondAxisOnly));
    }

    private int findNodeNrById(String id) {
        for (beast.base.evolution.tree.Node node : tree.getNodesAsArray()) {
            if (id.equals(node.getID())) {
                return node.getNr();
            }
        }
        throw new IllegalArgumentException("Node not found: " + id);
    }

    private static double[] normalizeAfterOperatorSquare(double[] operatorWeights) {
        double total = 0.0;
        double[] normalized = new double[operatorWeights.length];
        for (int i = 0; i < operatorWeights.length; i++) {
            normalized[i] = operatorWeights[i] * operatorWeights[i];
            total += normalized[i];
        }
        for (int i = 0; i < normalized.length; i++) {
            normalized[i] /= total;
        }
        return normalized;
    }
}