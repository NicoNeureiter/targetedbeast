package targetedbeast;

import org.junit.Assert;
import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.spec.evolution.speciation.YuleModel;
import beast.base.evolution.tree.Tree;
import beast.base.inference.DirectSimulator;
import targetedbeast.edgeweights.CombinedEdgeWeights;
import targetedbeast.edgeweights.ConstantWeights;
import targetedbeast.edgeweights.PriorWeights;
import targetedbeast.edgeweights.PCAWeights;
import targetedbeast.operators.TargetedWilsonBaldingFixedHeight;

public class TargetedWilsonBaldingFixedHeightTest {

	private static final int NUM_TAXA = 5;
	private static final int NUM_PROPOSALS = 25;

	@Test
	public void testProposalWithEdgeWeights() throws Exception {
		assertProposalRuns(false);
	}

	@Test
	public void testProposalWithEdgeLengths() throws Exception {
		assertProposalRuns(true);
	}

	@Test
	public void testProposalWithCombinedPriorWeights() throws Exception {
		assertProposalRunsWithCombinedPriorWeights();
	}

	private void assertProposalRuns(boolean useEdgeLength) throws Exception {
		Alignment alignment = TargetedOperatorDetailedBalanceTest.createDummyAlignment();
		Tree tree = new Tree();
		tree.initByName("taxonset", createTaxonSet(NUM_TAXA));

		YuleModel prior = new YuleModel();
		prior.initByName("tree", tree, "birthDiffRate", "2.0");

		DirectSimulator simulator = new DirectSimulator();
		simulator.initByName("distribution", prior, "nSamples", 1);

		PCAWeights edgeWeights = new PCAWeights();
		edgeWeights.initByName("tree", tree, "data", alignment, "dimension", 3);

		TargetedWilsonBaldingFixedHeight operator = new TargetedWilsonBaldingFixedHeight();
		operator.initByName(
				"tree", tree,
				"weight", 1.0,
				"edgeWeights", edgeWeights,
				"useEdgeLength", useEdgeLength);

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

	private void assertProposalRunsWithCombinedPriorWeights() throws Exception {
		Tree tree = new Tree();
		tree.initByName("taxonset", createTaxonSet(NUM_TAXA));

		YuleModel prior = new YuleModel();
		prior.initByName("tree", tree, "birthDiffRate", "2.0");

		DirectSimulator simulator = new DirectSimulator();
		simulator.initByName("distribution", prior, "nSamples", 1);

		ConstantWeights constantWeights = new ConstantWeights();
		constantWeights.initByName("tree", tree);

		PriorWeights priorWeights = new PriorWeights();
		priorWeights.initByName("tree", tree, "prior", prior, "temperature", 0.2, "minWeight", 1e-6, "maxWeight", 1e6);

		CombinedEdgeWeights combinedWeights = new CombinedEdgeWeights();
		combinedWeights.initByName("edgeWeights", constantWeights, "edgeWeights", priorWeights);

		TargetedWilsonBaldingFixedHeight operator = new TargetedWilsonBaldingFixedHeight();
		operator.initByName(
				"tree", tree,
				"weight", 1.0,
				"edgeWeights", combinedWeights,
				"useEdgeLength", false);

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