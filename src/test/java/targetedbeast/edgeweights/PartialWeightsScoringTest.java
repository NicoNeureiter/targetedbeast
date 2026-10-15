package targetedbeast.edgeweights;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeDistribution;
import beast.base.inference.DirectSimulator;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.likelihood.TreeLikelihood;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.speciation.YuleModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.inference.parameter.RealScalarParam;

/**
 * PartialWeights' candidate score must equal a frozen copy of the original, straightforward
 * implementation, so that performance work on the scoring cannot change the weights.
 */
public class PartialWeightsScoringTest {

	private static final int NUM_TAXA = 12;
	private static final int SITES = 150;
	private static final double EPS = 1E-8;

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

	@Test
	public void testScoreMatchesReference() throws Exception {
		System.setProperty("java.only", "true");
		Random random = new Random(7);
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

		SiteModel siteModel = new SiteModel();
		siteModel.initByName("substModel", new JukesCantor(), "gammaCategoryCount", 4,
				"shape", new RealScalarParam<>(0.4, PositiveReal.INSTANCE));
		TreeLikelihood likelihood = new TreeLikelihood();
		likelihood.initByName("tree", tree, "siteModel", siteModel, "data", data,
				"implementation", "BeerLikelihoodCore4");
		PartialWeights weights = new PartialWeights();
		weights.initByName("tree", tree, "likelihood", likelihood, "data", data);

		int checked = 0;
		for (int sample = 0; sample < 20; sample++) {
			simulator.run();
			tree.setEverythingDirty(true);
			likelihood.calculateLogP();
			tree.setEverythingDirty(false);
			weights.ensureInitialized();
			for (int k = 0; k < 20; k++) {
				Node from = tree.getNode(random.nextInt(tree.getNodeCount() - 1));
				Node to = tree.getNode(random.nextInt(tree.getNodeCount()));
				if (from == to || from.isRoot()) {
					continue;
				}
				double toHeight = Math.max(from.getHeight(), to.getHeight()) + random.nextDouble();
				double[] fromPartials = weights.getPartials(from.getNr());
				double[] toPartials = weights.getPartials(to.getNr());
				double expected = referenceScore(weights, from, to, fromPartials, toPartials, toHeight);
				double actual = weights.computePartialsAgreement(from, to, fromPartials, toPartials, toHeight);
				assertEquals("score for " + from.getNr() + " -> " + to.getNr(), expected, actual,
						1e-12 * Math.abs(expected));
				checked++;
			}
		}
		assertTrue(checked > 100);
	}

	/** Frozen copy of the original computePartialsAgreement, computeLikelihoodFromPartials and helpers. */
	static double referenceScore(PartialWeights w, Node fromNode, Node toNode, double[] fromPartials,
			double[] toPartials, double toHeight) {
		int matrixCount = w.matrixCount, patternCount = w.patternCount, stateCount = w.stateCount;
		double[] fromMatrix = w.calculateTransitionMatrix(fromNode, toHeight);
		double[] toMatrix = w.calculateTransitionMatrix(toNode, toHeight);
		double[] fromAtParent = referenceApply(w, fromMatrix, fromPartials);
		double[] toAtParent = referenceApply(w, toMatrix, toPartials);

		double[] perCat = new double[toPartials.length];
		for (int i = 0; i < matrixCount * patternCount; i++) {
			int offset = i * stateCount;
			double toSum = 0;
			double fromSum = 0;
			for (int j = offset; j < offset + stateCount; j++) {
				toSum += toAtParent[j];
				fromSum += fromAtParent[j];
			}
			if (toSum <= EPS || fromSum <= EPS) {
				for (int j = offset; j < offset + stateCount; j++) {
					perCat[j] = 1.0 / stateCount / stateCount;
				}
			} else {
				for (int j = offset; j < offset + stateCount; j++) {
					perCat[j] = (toAtParent[j] / toSum) * (fromAtParent[j] / fromSum);
				}
			}
		}

		double[] proportions = w.getSiteModel().getCategoryProportions(toNode);
		double[] integrated = new double[patternCount * stateCount];
		int v = 0;
		for (int l = 0; l < matrixCount; l++) {
			int u = 0;
			for (int k = 0; k < patternCount; k++) {
				for (int s = 0; s < stateCount; s++) {
					integrated[u] = (l == 0 ? 0 : integrated[u]) + perCat[v] * proportions[l];
					u++;
					v++;
				}
			}
		}
		double[] patternLogL = new double[patternCount];
		w.likelihoodCore.calculateLogLikelihoods(integrated, w.getSubstitutionModel().getFrequencies(), patternLogL);
		Alignment alignment = w.treelikelihood.dataInput.get();
		double sum = 0.0;
		for (int p = 0; p < patternCount; p++) {
			sum += alignment.getPatternWeight(p) * patternLogL[p];
		}
		return sum;
	}

	static double[] referenceApply(PartialWeights w, double[] matrix, double[] partials) {
		double[] out = new double[partials.length];
		for (int category = 0; category < w.matrixCount; category++) {
			for (int pattern = 0; pattern < w.patternCount; pattern++) {
				int m = category * w.matrixSize;
				int offset = (category * w.patternCount + pattern) * w.stateCount;
				for (int i = 0; i < w.stateCount; i++) {
					double sum = 0.0;
					for (int j = 0; j < w.stateCount; j++) {
						sum += matrix[m] * partials[offset + j];
						m++;
					}
					out[offset + i] = sum;
				}
			}
		}
		return out;
	}
}
