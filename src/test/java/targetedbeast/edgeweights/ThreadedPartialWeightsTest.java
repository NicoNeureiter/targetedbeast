package targetedbeast.edgeweights;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import beast.base.core.ProgramStatus;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeDistribution;
import beast.base.inference.DirectSimulator;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.likelihood.ThreadedTreeLikelihood;
import beast.base.spec.evolution.likelihood.TreeLikelihood;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.speciation.YuleModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.inference.parameter.RealScalarParam;

/** ThreadedPartialWeights must give exactly the weights of PartialWeights on the whole alignment. */
public class ThreadedPartialWeightsTest {

	private static final int NUM_TAXA = 12;
	private static final int SITES = 300;
	private static final int THREADS = 3;

	static Alignment randomAlignment(Random random) {
		String bases = "ACGT";
		char[] ancestor = new char[SITES];
		for (int s = 0; s < SITES; s++) {
			ancestor[s] = bases.charAt(random.nextInt(4));
		}
		List<Sequence> sequences = new ArrayList<>();
		for (int t = 0; t < NUM_TAXA; t++) {
			char[] seq = ancestor.clone();
			for (int s = 0; s < SITES; s++) {
				if (random.nextDouble() < 0.3) {
					seq[s] = bases.charAt(random.nextInt(4));
				}
			}
			sequences.add(new Sequence(String.valueOf(t), new String(seq)));
		}
		Alignment data = new Alignment();
		data.initByName("sequence", sequences, "dataType", "nucleotide");
		return data;
	}

	static SiteModel siteModel() {
		SiteModel siteModel = new SiteModel();
		siteModel.initByName("substModel", new JukesCantor(), "gammaCategoryCount", 4,
				"shape", new RealScalarParam<>(0.4, PositiveReal.INSTANCE));
		return siteModel;
	}

	@Test
	public void testMatchesSingleThreaded() throws Exception {
		System.setProperty("java.only", "true");
		int savedThreads = ProgramStatus.m_nThreads;
		ProgramStatus.m_nThreads = THREADS;
		try {
			Random random = new Random(3);
			Alignment data = randomAlignment(random);
			TaxonSet taxa = new TaxonSet();
			for (int t = 0; t < NUM_TAXA; t++) {
				taxa.initByName("taxon", new Taxon(String.valueOf(t)));
			}
			Tree tree = new Tree();
			tree.initByName("taxonset", taxa);
			YuleModel yule = new YuleModel();
			yule.initByName("tree", tree, "birthDiffRate", "2.0");
			DirectSimulator simulator = new DirectSimulator();
			simulator.initByName("distribution", (TreeDistribution) yule, "nSamples", 1);

			TreeLikelihood single = new TreeLikelihood();
			single.initByName("tree", tree, "siteModel", siteModel(), "data", data,
					"implementation", "BeerLikelihoodCore4");
			ThreadedTreeLikelihood threaded = new ThreadedTreeLikelihood();
			threaded.initByName("tree", tree, "siteModel", siteModel(), "data", data, "threads", THREADS);

			PartialWeights reference = new PartialWeights();
			reference.initByName("tree", tree, "likelihood", single, "data", data);
			ThreadedPartialWeights weights = new ThreadedPartialWeights();
			weights.initByName("tree", tree, "likelihood", threaded);
			assertEquals("slices", THREADS, weights.getSliceCount());
			ThreadedPartialWeights logWeights = new ThreadedPartialWeights();
			logWeights.initByName("tree", tree, "likelihood", threaded, "logSpace", true);
			PartialWeights logReference = new PartialWeights();
			logReference.initByName("tree", tree, "likelihood", single, "data", data, "logSpace", true);

			int checked = 0;
			for (int sample = 0; sample < 15; sample++) {
				simulator.run();
				tree.setEverythingDirty(true);
				double singleLogP = single.calculateLogP();
				double threadedLogP = threaded.calculateLogP();
				tree.setEverythingDirty(false);
				assertEquals("likelihoods", singleLogP, threadedLogP, 1e-9 * Math.abs(singleLogP));

				for (int k = 0; k < 5; k++) {
					Node from = tree.getNode(random.nextInt(tree.getNodeCount()));
					if (from.isRoot() || from.getParent().isRoot()) {
						continue;
					}
					List<Integer> ancestors = new ArrayList<>();
					for (Node a = from.getParent(); a != null; a = a.getParent()) {
						ancestors.add(a.getNr());
					}
					List<Integer> candidates = new ArrayList<>();
					for (Node c : tree.getNodesAsArray()) {
						if (c != from && c != from.getParent() && !c.isRoot()) {
							candidates.add(c.getNr());
						}
					}
					double sharedHeight = from.getParent().getHeight();
					for (EdgeWeights[] pair : new EdgeWeights[][] { { reference, weights }, { logReference, logWeights } }) {
						pair[0].updateByOperatorWithoutNode(from.getNr(), ancestors);
						pair[1].updateByOperatorWithoutNode(from.getNr(), ancestors);
						assertClose(pair[0].getTargetWeightsInteger(from.getNr(), candidates),
								pair[1].getTargetWeightsInteger(from.getNr(), candidates));
						assertClose(pair[0].getTargetWeightsInteger(from.getNr(), candidates, sharedHeight),
								pair[1].getTargetWeightsInteger(from.getNr(), candidates, sharedHeight));
						checked++;
					}
					assertEquals(reference.getEdgeWeights(from.getNr()), weights.getEdgeWeights(from.getNr()), 0.0);
				}
			}
			assertTrue(checked > 50);
		} finally {
			ProgramStatus.m_nThreads = savedThreads;
		}
	}

	static void assertClose(double[] expected, double[] actual) {
		assertEquals(expected.length, actual.length);
		for (int k = 0; k < expected.length; k++) {
			if (expected[k] == Double.NEGATIVE_INFINITY) {
				assertEquals(expected[k], actual[k], 0.0);
			} else {
				assertEquals("weight " + k, expected[k], actual[k], 1e-9 * Math.max(1.0, Math.abs(expected[k])));
			}
		}
	}
}
