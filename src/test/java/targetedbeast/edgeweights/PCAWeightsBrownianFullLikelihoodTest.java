package targetedbeast.edgeweights;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.Before;
import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;

public class PCAWeightsBrownianFullLikelihoodTest {

    private Alignment alignment;
    private Tree tree;
    private PCAWeightsBrownianFullLikelihood weights;

    @Before
    public void setUp() {
        alignment = new Alignment();
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
                "dimension", 1,
                "value", "A=0.0,B=2.0,C=4.0",
                "offset", 0.01);
    }

    @Test
    public void outsideMessageForInternalChildMatchesSiblingContribution() {
        weights.updateByOperatorWithoutNode(-1, Collections.emptyList());

        Node ab = findInternalNodeWithChildren("A", "B");
        Node sibling = tree.getRoot().getChild(0) == ab ? tree.getRoot().getChild(1) : tree.getRoot().getChild(0);

        double expectedMean = weights.getPoints(sibling.getNr())[0];
        double expectedVariance = weights.getVariances(sibling.getNr())[0] + sibling.getLength() + ab.getLength();

        assertEquals(expectedMean, weights.getOutsideMeans(ab.getNr())[0], 1e-12);
        assertEquals(expectedVariance, weights.getOutsideVariances(ab.getNr())[0], 1e-12);
    }

    @Test
    public void fullConditionalCombinesInsideAndOutsideGaussians() {
        weights.updateByOperatorWithoutNode(-1, Collections.emptyList());

        Node ab = findInternalNodeWithChildren("A", "B");
        PCAWeightsBrownianFullLikelihood.GaussianMessage full = weights.getFullConditional(ab.getNr());

        double insideMean = weights.getPoints(ab.getNr())[0];
        double insideVariance = weights.getVariances(ab.getNr())[0];
        double outsideMean = weights.getOutsideMeans(ab.getNr())[0];
        double outsideVariance = weights.getOutsideVariances(ab.getNr())[0];

        double expectedVariance = 1.0 / (1.0 / insideVariance + 1.0 / outsideVariance);
        double expectedMean = expectedVariance * (insideMean / insideVariance + outsideMean / outsideVariance);

        assertEquals(expectedMean, full.mean[0], 1e-12);
        assertEquals(expectedVariance, full.variance[0], 1e-12);
    }

    @Test
    public void treeLikelihoodMatchesFullConditionalAtAnyNode() {
        weights.updateByOperatorWithoutNode(-1, Collections.emptyList());

        double treeLogLikelihood = weights.getApproximateTreeLogLikelihood();
        Node ab = findInternalNodeWithChildren("A", "B");

        assertEquals(treeLogLikelihood, weights.getApproximateTreeLogLikelihoodAtNode(tree.getRoot().getNr()), 1e-10);
        assertEquals(treeLogLikelihood, weights.getApproximateTreeLogLikelihoodAtNode(ab.getNr()), 1e-10);
        assertEquals(treeLogLikelihood, weights.getApproximateTreeLogLikelihoodAtNode(findNodeNrById("A")), 1e-10);
    }

        @Test
        public void finiteRootPriorAddsExpectedRootPenaltyToTreeLogLikelihood() {
        PCAWeightsBrownianFullLikelihood flatRootWeights = new PCAWeightsBrownianFullLikelihood();
        flatRootWeights.initByName(
            "tree", tree,
            "data", alignment,
            "dimension", 1,
            "value", "A=0.0,B=2.0,C=4.0",
            "rootPriorVariance", Double.POSITIVE_INFINITY,
            "offset", 0.01);

        PCAWeightsBrownianFullLikelihood finiteRootWeights = new PCAWeightsBrownianFullLikelihood();
        finiteRootWeights.initByName(
            "tree", tree,
            "data", alignment,
            "dimension", 1,
            "value", "A=0.0,B=2.0,C=4.0",
            "rootPriorVariance", 0.5,
            "offset", 0.01);

        int rootNr = tree.getRoot().getNr();
        double flatLogLikelihood = flatRootWeights.getApproximateTreeLogLikelihood();
        double rootMean = flatRootWeights.getPoints(rootNr)[0];
        double rootVariance = flatRootWeights.getVariances(rootNr)[0];
        double expected = flatLogLikelihood - 0.5 * (Math.log(rootVariance + 0.5)
            + (rootMean * rootMean) / (rootVariance + 0.5));

        assertEquals(expected, finiteRootWeights.getApproximateTreeLogLikelihood(), 1e-12);
        }

    @Test
    public void rootPriorVariancesInputOverridesScalarRootPriorVariancePerAxis() {
        PCAWeightsBrownianFullLikelihood traitRootWeights = new PCAWeightsBrownianFullLikelihood();
        traitRootWeights.initByName(
            "tree", tree,
            "data", alignment,
            "dimension", 2,
            "value", "A=0.0 1.0,B=2.0 3.0,C=4.0 5.0",
            "rootPriorVariance", Double.POSITIVE_INFINITY,
            "rootPriorVariances", "0.5 2.0",
            "offset", 0.01);

        assertArrayEquals(new double[] {0.5, 2.0}, traitRootWeights.getRootPriorVariances(), 1e-12);
    }

    @Test
    public void insideWithoutNodeAtRootCombinesRemainingSubtrees() {
        int aNr = findNodeNrById("A");
        Node ab = findInternalNodeWithChildren("A", "B");
        Node b = ab.getLeft().getNr() == aNr ? ab.getRight() : ab.getLeft();
        Node c = tree.getRoot().getChild(0) == ab ? tree.getRoot().getChild(1) : tree.getRoot().getChild(0);

        weights.updateByOperatorWithoutNode(aNr, List.of(tree.getRoot().getNr()));

        double bAtRootVariance = weights.getVariances(b.getNr())[0] + b.getLength() + ab.getLength();
        double cAtRootVariance = weights.getVariances(c.getNr())[0] + c.getLength();
        double expectedVariance = 1.0 / (1.0 / bAtRootVariance + 1.0 / cAtRootVariance);
        double expectedMean = expectedVariance *
                (weights.getPoints(b.getNr())[0] / bAtRootVariance + weights.getPoints(c.getNr())[0] / cAtRootVariance);

        assertEquals(expectedMean, weights.getInsideWithoutMeans(tree.getRoot().getNr())[0], 1e-12);
        assertEquals(expectedVariance, weights.getInsideWithoutVariances(tree.getRoot().getNr())[0], 1e-12);
    }

    @Test
    public void targetWeightsPreferCloserCandidate() {
        int aNr = findNodeNrById("A");
        int bNr = findNodeNrById("B");
        int cNr = findNodeNrById("C");

        double[] targetWeights = weights.getTargetWeightsInteger(aNr, List.of(bNr, cNr));

        assertEquals(2, targetWeights.length);
        assertTrue(targetWeights[0] > 0.0);
        assertTrue(targetWeights[1] > 0.0);
        assertTrue(targetWeights[0] > targetWeights[1]);
    }

    @Test
    public void inverseMeanDistanceProposalUsesInverseAttachmentMeanDistance() {
        PCAWeightsBrownianFullLikelihood meanDistanceWeights = new PCAWeightsBrownianFullLikelihood();
        meanDistanceWeights.initByName(
                "tree", tree,
                "data", alignment,
                "dimension", 1,
                "value", "A=0.0,B=2.0,C=4.0",
                "offset", 0.01,
                "useInverseMeanDistanceProposal", true);

        int aNr = findNodeNrById("A");
        int bNr = findNodeNrById("B");
        int cNr = findNodeNrById("C");
        double toHeight = 0.2;

        double[] targetWeights = meanDistanceWeights.getTargetWeightsInteger(aNr, List.of(bNr, cNr), toHeight);

        assertEquals(2, targetWeights.length);
        assertEquals(Math.sqrt(0.5), targetWeights[0], 1e-9);
        assertEquals(0.5, targetWeights[1], 1e-9);
        assertTrue(targetWeights[0] > targetWeights[1]);

        double[] normalized = normalizeAfterOperatorSquare(targetWeights);
        assertEquals(2.0 / 3.0, normalized[0], 1e-9);
        assertEquals(1.0 / 3.0, normalized[1], 1e-9);
    }

    @Test
    public void candidateHeightOverloadMatchesScalarOverloadWhenHeightsAreEqual() {
        int aNr = findNodeNrById("A");
        int bNr = findNodeNrById("B");
        int cNr = findNodeNrById("C");
        double toHeight = 0.2;

        double[] scalarWeights = weights.getTargetWeightsInteger(aNr, List.of(bNr, cNr), toHeight);
        double[] vectorWeights = weights.getTargetWeightsInteger(aNr, List.of(bNr, cNr), Arrays.asList(toHeight, toHeight));

        assertArrayEquals(scalarWeights, vectorWeights, 1e-12);
    }

    @Test
    public void largerProposalScaleSharpensNormalizedSquaredDistribution() {
        int aNr = findNodeNrById("A");
        int bNr = findNodeNrById("B");
        int cNr = findNodeNrById("C");
        double toHeight = 0.2;

        weights.setProposalLogLikelihoodScale(0.5);
        double[] lowScale = normalizeAfterOperatorSquare(
                weights.getTargetWeightsInteger(aNr, List.of(bNr, cNr), toHeight));

        weights.setProposalLogLikelihoodScale(2.0);
        double[] highScale = normalizeAfterOperatorSquare(
                weights.getTargetWeightsInteger(aNr, List.of(bNr, cNr), toHeight));

        assertTrue(lowScale[0] > lowScale[1]);
        assertTrue(highScale[0] > highScale[1]);
        assertTrue(highScale[0] > lowScale[0]);
        assertTrue(highScale[1] < lowScale[1]);
    }

    @Test
    public void strictClockRateChangeUpdatesLikelihoodAfterInitialization() {
        RealScalarParam<PositiveReal> clockRate =
                new RealScalarParam<>(1.0, PositiveReal.INSTANCE);
        StrictClockModel strictClock = new StrictClockModel();
        strictClock.initByName("clock.rate", clockRate);

        PCAWeightsBrownianFullLikelihood rateAwareWeights = new PCAWeightsBrownianFullLikelihood();
        rateAwareWeights.initByName(
                "tree", tree,
                "data", alignment,
                "dimension", 1,
                "value", "A=0.0,B=2.0,C=4.0",
                "brownianRate", 100.0,
                "branchRateModel", strictClock,
                "offset", 0.01);

            PCAWeightsBrownianFullLikelihood fixedRateOneWeights = new PCAWeightsBrownianFullLikelihood();
            fixedRateOneWeights.initByName(
                "tree", tree,
                "data", alignment,
                "dimension", 1,
                "value", "A=0.0,B=2.0,C=4.0",
                "brownianRate", 1.0,
                "offset", 0.01);

        double initialLogLikelihood = rateAwareWeights.getApproximateTreeLogLikelihood();
            assertEquals(fixedRateOneWeights.getApproximateTreeLogLikelihood(), initialLogLikelihood, 1e-12);

        clockRate.set(2.0);
        strictClock.requiresRecalculation();
        rateAwareWeights.requiresRecalculation();
        double updatedLogLikelihood = rateAwareWeights.getApproximateTreeLogLikelihood();

        PCAWeightsBrownianFullLikelihood fixedRateTwoWeights = new PCAWeightsBrownianFullLikelihood();
        fixedRateTwoWeights.initByName(
            "tree", tree,
            "data", alignment,
            "dimension", 1,
            "value", "A=0.0,B=2.0,C=4.0",
            "brownianRate", 2.0,
            "offset", 0.01);

        assertEquals(fixedRateTwoWeights.getApproximateTreeLogLikelihood(), updatedLogLikelihood, 1e-12);
        assertTrue(Math.abs(updatedLogLikelihood - initialLogLikelihood) > 1e-12);
    }

    @Test
    public void negativeProposalScaleIsAcceptedAndReversesCandidatePreference() {
        int aNr = findNodeNrById("A");
        int bNr = findNodeNrById("B");
        int cNr = findNodeNrById("C");
        double toHeight = 0.2;

        weights.setProposalLogLikelihoodScale(1.0);
        double[] positiveScale = normalizeAfterOperatorSquare(
                weights.getTargetWeightsInteger(aNr, List.of(bNr, cNr), toHeight));

        weights.setProposalLogLikelihoodScale(-1.0);
        double[] negativeScale = normalizeAfterOperatorSquare(
                weights.getTargetWeightsInteger(aNr, List.of(bNr, cNr), toHeight));

        assertTrue(positiveScale[0] > positiveScale[1]);
        assertTrue(negativeScale[0] < negativeScale[1]);
    }

    @Test
    public void proposalScoreBreakdownReflectsCurrentScalingMode() {
    int aNr = findNodeNrById("A");
    int bNr = findNodeNrById("B");
    double toHeight = 0.2;

    weights.setProposalLogLikelihoodScale(-1.5);
    PCAWeightsBrownianFullLikelihood.ProposalScoreBreakdown defaultBreakdown =
        weights.getProposalScoreBreakdown(aNr, bNr, toHeight);

    assertEquals(defaultBreakdown.rawBaseContribution + defaultBreakdown.rawQuadraticContribution,
        defaultBreakdown.rawTotalContribution, 1e-12);
    assertEquals(-1.5 * defaultBreakdown.rawBaseContribution,
        defaultBreakdown.scaledBaseContribution, 1e-12);
    assertEquals(-1.5 * defaultBreakdown.rawQuadraticContribution,
        defaultBreakdown.scaledQuadraticContribution, 1e-12);
    assertEquals(defaultBreakdown.scaledBaseContribution + defaultBreakdown.scaledQuadraticContribution,
        defaultBreakdown.scaledTotalContribution, 1e-12);

    PCAWeightsBrownianFullLikelihood quadraticOnlyWeights = new PCAWeightsBrownianFullLikelihood();
    quadraticOnlyWeights.initByName(
        "tree", tree,
        "data", alignment,
        "dimension", 1,
        "scaleQuadraticPartOnly", true,
        "value", "A=0.0,B=2.0,C=4.0",
        "offset", 0.01);
    quadraticOnlyWeights.setProposalLogLikelihoodScale(-1.5);
    PCAWeightsBrownianFullLikelihood.ProposalScoreBreakdown quadraticOnlyBreakdown =
        quadraticOnlyWeights.getProposalScoreBreakdown(aNr, bNr, toHeight);

    assertEquals(quadraticOnlyBreakdown.rawBaseContribution,
        quadraticOnlyBreakdown.scaledBaseContribution, 1e-12);
    assertEquals(-1.5 * quadraticOnlyBreakdown.rawQuadraticContribution,
        quadraticOnlyBreakdown.scaledQuadraticContribution, 1e-12);
    assertEquals(quadraticOnlyBreakdown.scaledBaseContribution + quadraticOnlyBreakdown.scaledQuadraticContribution,
        quadraticOnlyBreakdown.scaledTotalContribution, 1e-12);
    }

    @Test
    public void proposalScoreBreakdownSupportsSeparateBaseAndQuadraticCalibration() {
    int aNr = findNodeNrById("A");
    int bNr = findNodeNrById("B");
    double toHeight = 0.2;

    PCAWeightsBrownianFullLikelihood splitWeights = new PCAWeightsBrownianFullLikelihood();
    splitWeights.initByName(
        "tree", tree,
        "data", alignment,
        "dimension", 1,
        "separateBaseAndQuadraticCalibration", true,
        "value", "A=0.0,B=2.0,C=4.0",
        "offset", 0.01);
    splitWeights.setProposalBaseLogLikelihoodScales(new double[] {0.25});
    splitWeights.setProposalLogLikelihoodScales(new double[] {-1.5});

    PCAWeightsBrownianFullLikelihood.ProposalScoreBreakdown breakdown =
        splitWeights.getProposalScoreBreakdown(aNr, bNr, toHeight);

    assertEquals(0.25 * breakdown.rawBaseContribution,
        breakdown.scaledBaseContribution, 1e-12);
    assertEquals(-1.5 * breakdown.rawQuadraticContribution,
        breakdown.scaledQuadraticContribution, 1e-12);
    assertEquals(breakdown.scaledBaseContribution + breakdown.scaledQuadraticContribution,
        breakdown.scaledTotalContribution, 1e-12);
    }

        @Test
        public void scaleQuadraticPartOnly_makesPureNormalizationScoresInvariantToScale() {
        PCAWeightsBrownianFullLikelihood normalizationOnlyWeights = new PCAWeightsBrownianFullLikelihood();
        normalizationOnlyWeights.initByName(
            "tree", tree,
            "data", alignment,
            "dimension", 1,
            "scaleQuadraticPartOnly", true,
            "value", "A=0.0,B=0.0,C=0.0",
            "offset", 0.01);

        int aNr = findNodeNrById("A");
        int bNr = findNodeNrById("B");
        int cNr = findNodeNrById("C");
        double toHeight = 0.2;

        normalizationOnlyWeights.setProposalLogLikelihoodScale(0.5);
        double[] lowScale = normalizeAfterOperatorSquare(
            normalizationOnlyWeights.getTargetWeightsInteger(aNr, List.of(bNr, cNr), toHeight));

        normalizationOnlyWeights.setProposalLogLikelihoodScale(2.0);
        double[] highScale = normalizeAfterOperatorSquare(
            normalizationOnlyWeights.getTargetWeightsInteger(aNr, List.of(bNr, cNr), toHeight));

        assertArrayEquals(lowScale, highScale, 1e-12);
        }

        @Test
        public void largerBrownianRatePenalizesMatchingLeavesThroughLargerPropagationVariance() {
        Alignment twoTaxonAlignment = new Alignment();
        twoTaxonAlignment.initByName(
            "sequence", new Sequence("A", "ACGTACGTAA"),
            "sequence", new Sequence("B", "ACGTACGTAA"),
            "dataType", "nucleotide");

        Tree twoTaxonTree = new TreeParser();
        ((TreeParser) twoTaxonTree).initByName(
            "newick", "(A:0.5,B:0.5);",
            "IsLabelledNewick", true,
            "taxa", twoTaxonAlignment,
            "adjustTipHeights", false);

        PCAWeightsBrownianFullLikelihood lowRateWeights = new PCAWeightsBrownianFullLikelihood();
        lowRateWeights.initByName(
            "tree", twoTaxonTree,
            "data", twoTaxonAlignment,
            "dimension", 1,
            "brownianRate", 0.5,
            "value", "A=0.0,B=0.0",
            "offset", 0.01);

        PCAWeightsBrownianFullLikelihood highRateWeights = new PCAWeightsBrownianFullLikelihood();
        highRateWeights.initByName(
            "tree", twoTaxonTree,
            "data", twoTaxonAlignment,
            "dimension", 1,
            "brownianRate", 2.0,
            "value", "A=0.0,B=0.0",
            "offset", 0.01);

        assertTrue(lowRateWeights.getApproximateTreeLogLikelihood() > highRateWeights.getApproximateTreeLogLikelihood());
        }

    @Test
    public void uniformTraitSpecificBrownianRatesMatchScalarBrownianRate() {
        PCAWeightsBrownianFullLikelihood scalarWeights = new PCAWeightsBrownianFullLikelihood();
        scalarWeights.initByName(
            "tree", tree,
            "data", alignment,
            "dimension", 2,
            "brownianRate", 1.5,
            "value", "A=0.0 0.0,B=2.0 0.5,C=4.0 3.0",
            "offset", 0.01);

        PCAWeightsBrownianFullLikelihood traitRateWeights = new PCAWeightsBrownianFullLikelihood();
        traitRateWeights.initByName(
            "tree", tree,
            "data", alignment,
            "dimension", 2,
            "brownianTraitRates", "1.5 1.5",
            "value", "A=0.0 0.0,B=2.0 0.5,C=4.0 3.0",
            "offset", 0.01);

        assertEquals(scalarWeights.getApproximateTreeLogLikelihood(),
            traitRateWeights.getApproximateTreeLogLikelihood(), 1e-12);
        assertArrayEquals(new double[] {1.5, 1.5}, traitRateWeights.getBrownianTraitRates(), 1e-12);
    }

    @Test
    public void parseValueAssignsEmbeddingsByLeafNodeIdInsteadOfTaxonOrder() {
        Alignment fourTaxonAlignment = new Alignment();
        fourTaxonAlignment.initByName(
                "sequence", new Sequence("A", "ACGTACGTAA"),
                "sequence", new Sequence("B", "ACGTACGTAT"),
                "sequence", new Sequence("C", "ACGTACGTGA"),
                "sequence", new Sequence("D", "ACGTACGTGT"),
                "dataType", "nucleotide");

        Tree reorderedTree = new TreeParser();
        ((TreeParser) reorderedTree).initByName(
                "newick", "((D:0.1,B:0.1):0.2,(C:0.1,A:0.1):0.2);",
                "IsLabelledNewick", true,
                "taxa", fourTaxonAlignment,
                "adjustTipHeights", false);

        PCAWeightsBrownianFullLikelihood reorderedWeights = new PCAWeightsBrownianFullLikelihood();
        reorderedWeights.initByName(
                "tree", reorderedTree,
                "data", fourTaxonAlignment,
                "dimension", 1,
                "value", "A=0.0,B=2.0,C=4.0,D=6.0",
                "offset", 0.01);

        assertEquals(0.0, reorderedWeights.getPoints(findNodeNrById(reorderedTree, "A"))[0], 1e-12);
        assertEquals(2.0, reorderedWeights.getPoints(findNodeNrById(reorderedTree, "B"))[0], 1e-12);
        assertEquals(4.0, reorderedWeights.getPoints(findNodeNrById(reorderedTree, "C"))[0], 1e-12);
        assertEquals(6.0, reorderedWeights.getPoints(findNodeNrById(reorderedTree, "D"))[0], 1e-12);
    }

        @Test
        public void proposedTreeLogLikelihoodMatchesFreshRebuildOnExplicitMove() {
        Alignment fourTaxonAlignment = new Alignment();
        fourTaxonAlignment.initByName(
            "sequence", new Sequence("A", "ACGTACGTAA"),
            "sequence", new Sequence("B", "ACGTACGTAT"),
            "sequence", new Sequence("C", "ACGTACGTGA"),
            "sequence", new Sequence("D", "ACGTACGTGT"),
            "dataType", "nucleotide");

        Tree originalTree = new TreeParser();
        ((TreeParser) originalTree).initByName(
            "newick", "((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);",
            "IsLabelledNewick", true,
            "taxa", fourTaxonAlignment,
            "adjustTipHeights", false);

        PCAWeightsBrownianFullLikelihood originalWeights = new PCAWeightsBrownianFullLikelihood();
        originalWeights.initByName(
            "tree", originalTree,
            "data", fourTaxonAlignment,
            "dimension", 1,
            "value", "A=0.0,B=2.0,C=4.0,D=6.0",
            "offset", 0.01);

        int aNr = findNodeNrById(originalTree, "A");
        int cNr = findNodeNrById(originalTree, "C");
        double toHeight = 0.05;

        originalWeights.updateByOperatorWithoutNode(aNr, Collections.emptyList());
        PCAWeightsBrownianFullLikelihood.GaussianMessage fromInside = invokeGaussianMessage(originalWeights, "getInsideMessage", aNr);
        PCAWeightsBrownianFullLikelihood.GaussianMessage toInsideWithout = invokeGaussianMessage(originalWeights, "getInsideWithoutMessage", cNr);

        double incrementalScore = originalWeights.computeProposedTreeLogLikelihood(
            originalTree.getNode(aNr),
            originalTree.getNode(cNr),
            fromInside,
            toInsideWithout,
            toHeight);

        Tree movedTree = new TreeParser();
        ((TreeParser) movedTree).initByName(
            "newick", "(B:0.3,((A:0.05,C:0.05):0.05,D:0.1):0.2);",
            "IsLabelledNewick", true,
            "taxa", fourTaxonAlignment,
            "adjustTipHeights", false);

        PCAWeightsBrownianFullLikelihood rebuiltWeights = new PCAWeightsBrownianFullLikelihood();
        rebuiltWeights.initByName(
            "tree", movedTree,
            "data", fourTaxonAlignment,
            "dimension", 1,
            "value", "A=0.0,B=2.0,C=4.0,D=6.0",
            "offset", 0.01);

        assertEquals(rebuiltWeights.getApproximateTreeLogLikelihood(), incrementalScore, 1e-12);
        }

    private double[] normalizeAfterOperatorSquare(double[] operatorReadyWeights) {
        double[] normalized = new double[operatorReadyWeights.length];
        double sum = 0.0;
        for (int i = 0; i < operatorReadyWeights.length; i++) {
            normalized[i] = operatorReadyWeights[i] * operatorReadyWeights[i];
            sum += normalized[i];
        }
        for (int i = 0; i < normalized.length; i++) {
            normalized[i] /= sum;
        }
        return normalized;
    }

    private static PCAWeightsBrownianFullLikelihood.GaussianMessage invokeGaussianMessage(
            PCAWeightsBrownianFullLikelihood weights,
            String methodName,
            int nodeNr) {
        try {
            Method method = PCAWeightsBrownianFullLikelihood.class.getDeclaredMethod(methodName, int.class);
            method.setAccessible(true);
            return (PCAWeightsBrownianFullLikelihood.GaussianMessage) method.invoke(weights, nodeNr);
        } catch (ReflectiveOperationException e) {
            throw new AssertionError("Failed to access " + methodName, e);
        }
    }

    private Node findInternalNodeWithChildren(String leftId, String rightId) {
        for (Node node : tree.getNodesAsArray()) {
            if (node.isLeaf()) {
                continue;
            }
            if (node.getLeft().isLeaf() && node.getRight().isLeaf()) {
                String left = node.getLeft().getID();
                String right = node.getRight().getID();
                if ((leftId.equals(left) && rightId.equals(right)) || (leftId.equals(right) && rightId.equals(left))) {
                    return node;
                }
            }
        }
        throw new IllegalArgumentException("Could not find internal node for children " + leftId + " and " + rightId);
    }

    private int findNodeNrById(String id) {
        return findNodeNrById(tree, id);
    }

    private static int findNodeNrById(Tree currentTree, String id) {
        for (Node node : currentTree.getNodesAsArray()) {
            if (id.equals(node.getID())) {
                return node.getNr();
            }
        }
        throw new IllegalArgumentException("Could not find node with id " + id);
    }
}
