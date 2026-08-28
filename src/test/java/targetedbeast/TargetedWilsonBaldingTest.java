package targetedbeast;

import org.junit.Assert;
import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.operator.SubtreeSlide;
import beast.base.evolution.speciation.YuleModel;
import beast.base.evolution.tree.Tree;
import beast.base.inference.DirectSimulator;
import targetedbeast.edgeweights.PCAWeights;
import targetedbeast.operators.TargetedWilsonBalding;
import targetedbeast.operators.TargetedWilsonBaldingHeightDelta;

public class TargetedWilsonBaldingTest {

	private static final int NUM_TAXA = 5;
	private static final int NUM_PROPOSALS = 25;

	@Test
	public void testProposalWithMaxHeightRatioAndEdgeWeights() throws Exception {
		assertProposalRuns(false, true, true);
	}

	@Test
	public void testProposalWithMaxHeightRatioAndEdgeLengths() throws Exception {
		assertProposalRuns(true, true, true);
	}

	@Test
	public void testProposalWithoutMaxHeightRatioStillRuns() throws Exception {
		assertProposalRuns(false, false, true);
	}

	@Test
	public void testProposalWithoutAttachmentIntervalLengthWeightingStillRuns() throws Exception {
		assertProposalRuns(false, true, false);
	}

	@Test
	public void testHeightDeltaProposalWithEdgeWeights() throws Exception {
		assertHeightDeltaProposalRuns(false, false);
	}

	@Test
	public void testHeightDeltaProposalWithEdgeLengths() throws Exception {
		assertHeightDeltaProposalRuns(true, false);
	}

	@Test
	public void testHeightDeltaProposalWithSharedSubtreeSlideStillRuns() throws Exception {
		assertHeightDeltaProposalRuns(false, true);
	}

	@Test
	public void testHeightDeltaProposalWithForcedTopologyChangeStillRuns() throws Exception {
		assertHeightDeltaProposalRuns(false, false, true);
	}

	private void assertProposalRuns(boolean useEdgeLength, boolean useMaxHeightRatio,
			boolean useAttachmentIntervalLength) throws Exception {
		Alignment alignment = TargetedOperatorDetailedBalanceTest.createDummyAlignment();
		Tree tree = new Tree();
		tree.initByName("taxonset", createTaxonSet(NUM_TAXA));

		YuleModel prior = new YuleModel();
		prior.initByName("tree", tree, "birthDiffRate", "2.0");

		DirectSimulator simulator = new DirectSimulator();
		simulator.initByName("distribution", prior, "nSamples", 1);

		PCAWeights edgeWeights = new PCAWeights();
		edgeWeights.initByName("tree", tree, "data", alignment, "dimension", 3);

		TargetedWilsonBalding operator = new TargetedWilsonBalding();
		if (useMaxHeightRatio) {
			operator.initByName(
					"tree", tree,
					"weight", 1.0,
					"edgeWeights", edgeWeights,
					"useEdgeLength", useEdgeLength,
					"useAttachmentIntervalLength", useAttachmentIntervalLength,
					"maxHeightRatio", 2.0);
		} else {
			operator.initByName(
					"tree", tree,
					"weight", 1.0,
					"edgeWeights", edgeWeights,
					"useEdgeLength", useEdgeLength,
					"useAttachmentIntervalLength", useAttachmentIntervalLength);
		}

		for (int iteration = 0; iteration < NUM_PROPOSALS; iteration++) {
			simulator.run();
			double logPBefore = prior.calculateLogP();
			Assert.assertFalse(Double.isNaN(logPBefore));

			double logHR = operator.proposal();
			Assert.assertFalse(Double.isNaN(logHR));

			double logPAfter = prior.calculateLogP();
			Assert.assertFalse(Double.isNaN(logPAfter));
		}
	}

	private void assertHeightDeltaProposalRuns(boolean useEdgeLength, boolean useSharedSubtreeSlide) throws Exception {
		assertHeightDeltaProposalRuns(useEdgeLength, useSharedSubtreeSlide, false);
	}

	private void assertHeightDeltaProposalRuns(boolean useEdgeLength, boolean useSharedSubtreeSlide,
			boolean forceTopologyChange) throws Exception {
		Alignment alignment = TargetedOperatorDetailedBalanceTest.createDummyAlignment();
		Tree tree = new Tree();
		tree.initByName("taxonset", createTaxonSet(NUM_TAXA));

		YuleModel prior = new YuleModel();
		prior.initByName("tree", tree, "birthDiffRate", "2.0");

		DirectSimulator simulator = new DirectSimulator();
		simulator.initByName("distribution", prior, "nSamples", 1);

		PCAWeights edgeWeights = new PCAWeights();
		edgeWeights.initByName("tree", tree, "data", alignment, "dimension", 3);

		TargetedWilsonBaldingHeightDelta operator = new TargetedWilsonBaldingHeightDelta();
		if (useSharedSubtreeSlide) {
			SubtreeSlide subtreeSlide = new SubtreeSlide();
			subtreeSlide.initByName(
					"tree", tree,
					"weight", 1.0,
					"size", 0.5,
					"gaussian", false);
			operator.initByName(
					"tree", tree,
					"weight", 1.0,
					"edgeWeights", edgeWeights,
					"useEdgeLength", useEdgeLength,
					"subtreeSlide", subtreeSlide,
					"shrinkage", 0.5,
					"forceTopologyChange", forceTopologyChange);
		} else {
			operator.initByName(
					"tree", tree,
					"weight", 1.0,
					"edgeWeights", edgeWeights,
					"useEdgeLength", useEdgeLength,
					"size", 0.25,
					"forceTopologyChange", forceTopologyChange);
		}

		for (int iteration = 0; iteration < NUM_PROPOSALS; iteration++) {
			simulator.run();
			double logPBefore = prior.calculateLogP();
			Assert.assertFalse(Double.isNaN(logPBefore));

			double logHR = operator.proposal();
			Assert.assertFalse(Double.isNaN(logHR));

			double logPAfter = prior.calculateLogP();
			Assert.assertFalse(Double.isNaN(logPAfter));
		}
	}

	private TaxonSet createTaxonSet(int numTaxa) {
		TaxonSet taxonSet = new TaxonSet();
		for (int taxonIdx = 0; taxonIdx < numTaxa; taxonIdx++) {
			taxonSet.initByName("taxon", new Taxon(String.valueOf(taxonIdx)));
		}
		return taxonSet;
	}
}