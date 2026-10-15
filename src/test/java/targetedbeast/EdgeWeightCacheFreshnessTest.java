package targetedbeast;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;

import org.junit.Assert;
import org.junit.Test;

import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.operator.kernel.BactrianSubtreeSlide;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeDistribution;
import beast.base.inference.CompoundDistribution;
import beast.base.inference.DirectSimulator;
import beast.base.inference.Logger;
import beast.base.inference.MCMC;
import beast.base.inference.Operator;
import beast.base.inference.State;
import beast.base.spec.evolution.speciation.YuleModel;
import targetedbeast.edgeweights.PCAWeights2;
import targetedbeast.operators.RangeSlide;
import targetedbeast.operators.TargetedWilsonBalding;
import targetedbeast.operators.TargetedWilsonBaldingFixedHeight;
import targetedbeast.operators.WeightedWideOperator;

/**
 * Runs a real MCMC and, after every step, checks that PCAWeights2's cache equals a freshly built
 * instance on the current tree. Operators read the cache directly at the start of the next
 * proposal, so a stale value there makes the proposal depend on the chain's history rather than
 * on the current state alone. The detailed-balance tests cannot see this: they rebuild the cache
 * for every sample.
 */
public class EdgeWeightCacheFreshnessTest {

	private static final int NUM_TAXA = 14;
	private static final int SITES = 120;
	private static final long CHAIN_LENGTH = 5000;

	/** MCMC that compares the cache with a fresh instance after every step. */
	public static class FreshnessCheckingMCMC extends MCMC {
		PCAWeights2 weights;
		Tree tree;
		int checks = 0;
		String firstStale = null;

		@Override
		protected Operator propagateState(final long sampleNr) {
			Operator op = super.propagateState(sampleNr);
			checks++;
			if (firstStale == null) {
				int stale = staleNodes(weights, tree);
				if (stale > 0) {
					firstStale = "step " + sampleNr + " after " + op.getID() + ": " + stale + " of "
							+ tree.getNodeCount() + " nodes differ from a fresh cache";
				}
			}
			return op;
		}
	}

	static int staleNodes(PCAWeights2 cached, Tree tree) {
		PCAWeights2 fresh = new PCAWeights2();
		for (Input<?> input : cached.listInputs()) {
			if (input.get() != null && !(input.get() instanceof List<?> list && list.isEmpty())) {
				fresh.setInputValue(input.getName(), input.get());
			}
		}
		fresh.initAndValidate();
		int stale = 0;
		for (int nr = 0; nr < tree.getNodeCount(); nr++) {
			boolean differs = !tree.getNode(nr).isRoot()
					&& Math.abs(cached.getEdgeWeights(nr) - fresh.getEdgeWeights(nr)) > 1e-9;
			double[] a = cached.getPoints(nr);
			double[] b = fresh.getPoints(nr);
			for (int d = 0; d < a.length && !differs; d++) {
				differs = Math.abs(a[d] - b[d]) > 1e-9 * Math.max(1.0, Math.abs(b[d]));
			}
			if (differs) {
				stale++;
			}
		}
		return stale;
	}

	static Alignment randomAlignment() {
		Random random = new Random(42);
		String bases = "ACGT";
		List<Sequence> sequences = new ArrayList<>();
		char[] ancestor = new char[SITES];
		for (int s = 0; s < SITES; s++) {
			ancestor[s] = bases.charAt(random.nextInt(4));
		}
		for (int t = 0; t < NUM_TAXA; t++) {
			char[] seq = ancestor.clone();
			for (int s = 0; s < SITES; s++) {
				if (random.nextDouble() < 0.25) {
					seq[s] = bases.charAt(random.nextInt(4));
				}
			}
			sequences.add(new Sequence(String.valueOf(t), new String(seq)));
		}
		Alignment data = new Alignment();
		data.initByName("sequence", sequences, "dataType", "nucleotide");
		return data;
	}

	/** One operator plus a plain subtree slide, so moves by a non-targeted operator are covered too. */
	private void assertCacheStaysFresh(BiFunction<Tree, PCAWeights2, Operator> operatorFactory) throws Exception {
		assertCacheStaysFresh(CHAIN_LENGTH, (tree, w) -> {
			BactrianSubtreeSlide subtreeSlide = new BactrianSubtreeSlide();
			subtreeSlide.initByName("tree", tree, "weight", 1.0, "size", 0.5);
			return List.of(operatorFactory.apply(tree, w), subtreeSlide);
		});
	}

	private void assertCacheStaysFresh(long chainLength,
			BiFunction<Tree, PCAWeights2, List<Operator>> operatorsFactory) throws Exception {
		Alignment data = randomAlignment();
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
		simulator.run();

		PCAWeights2 weights = new PCAWeights2();
		weights.initByName("tree", tree, "data", data, "dimension", 3);

		// As in the example XMLs: the weights sit in the posterior, so the State manages them as a
		// CalculationNode through every store/restore/accept cycle.
		CompoundDistribution posterior = new CompoundDistribution();
		posterior.initByName("distribution", yule, "distribution", weights);

		List<Operator> operators = operatorsFactory.apply(tree, weights);

		// BEAST's own reject validation: the weights' getChecksum() must match after every reject.
		System.setProperty("beast.debug", "true");
		State state = new State();
		state.initByName("stateNode", tree);
		Logger.FILE_MODE = Logger.LogFileMode.overwrite;
		Logger screenlog = new Logger();
		screenlog.initByName("log", posterior, "logEvery", 1_000_000);

		FreshnessCheckingMCMC mcmc = new FreshnessCheckingMCMC();
		mcmc.weights = weights;
		mcmc.tree = tree;
		mcmc.initByName("chainLength", chainLength, "state", state, "distribution", posterior,
				"operator", operators, "logger", screenlog);
		mcmc.run();

		Assert.assertTrue("no steps were checked", mcmc.checks > 0);
		Assert.assertNull("edge-weight cache went stale: " + mcmc.firstStale, mcmc.firstStale);
	}

	@Test
	public void testTargetedWilsonBalding() throws Exception {
		assertCacheStaysFresh((tree, w) -> {
			TargetedWilsonBalding op = new TargetedWilsonBalding();
			op.initByName("tree", tree, "weight", 3.0, "edgeWeights", w, "maxHeightRatio", 1.2);
			op.setID("targetedWilsonBalding");
			return op;
		});
	}

	@Test
	public void testTargetedWilsonBaldingFixedHeight() throws Exception {
		assertCacheStaysFresh((tree, w) -> {
			TargetedWilsonBaldingFixedHeight op = new TargetedWilsonBaldingFixedHeight();
			op.initByName("tree", tree, "weight", 3.0, "edgeWeights", w);
			op.setID("targetedWilsonBaldingFixedHeight");
			return op;
		});
	}

	@Test
	public void testTargetedWide() throws Exception {
		assertCacheStaysFresh((tree, w) -> {
			WeightedWideOperator op = new WeightedWideOperator();
			op.initByName("tree", tree, "weight", 3.0, "edgeWeights", w);
			op.setID("TargetedWide");
			return op;
		});
	}

	@Test
	public void testRangeSlideWeighted() throws Exception {
		assertCacheStaysFresh((tree, w) -> {
			RangeSlide op = new RangeSlide();
			op.initByName("tree", tree, "weight", 3.0, "edgeWeights", w, "useWeightedStep", true,
					"size", 0.5, "optimise", false);
			op.setID("targetedRangeSlide");
			return op;
		});
	}

	/** The infB benchmark's tree operators together, over a longer chain. */
	@Test
	public void testAllTreeOperators() throws Exception {
		assertCacheStaysFresh(20_000, (tree, w) -> {
			TargetedWilsonBalding twb = new TargetedWilsonBalding();
			twb.initByName("tree", tree, "weight", 5.0, "edgeWeights", w, "maxHeightRatio", 1.2);
			TargetedWilsonBaldingFixedHeight fixedHeight = new TargetedWilsonBaldingFixedHeight();
			fixedHeight.initByName("tree", tree, "weight", 5.0, "edgeWeights", w);
			WeightedWideOperator wide = new WeightedWideOperator();
			wide.initByName("tree", tree, "weight", 5.0, "edgeWeights", w);
			RangeSlide weightedRange = new RangeSlide();
			weightedRange.initByName("tree", tree, "weight", 5.0, "edgeWeights", w, "useWeightedStep", true,
					"size", 2.0, "optimise", false);
			RangeSlide range = new RangeSlide();
			range.initByName("tree", tree, "weight", 6.0, "edgeWeights", w, "size", 0.5);
			targetedbeast.operators.IntervalScaleOperator scaleAll = new targetedbeast.operators.IntervalScaleOperator();
			scaleAll.initByName("tree", tree, "weight", 1.0, "scaleFactor", 0.9);
			targetedbeast.operators.IntervalScaleOperator scaleRandomly = new targetedbeast.operators.IntervalScaleOperator();
			scaleRandomly.initByName("tree", tree, "weight", 1.0, "scaleFactor", 0.9, "scaleAllNodesIndependently", true);
			BactrianSubtreeSlide subtreeSlide = new BactrianSubtreeSlide();
			subtreeSlide.initByName("tree", tree, "weight", 4.0, "size", 2.0);
			List<Operator> ops = List.of(twb, fixedHeight, wide, weightedRange, range, scaleAll, scaleRandomly, subtreeSlide);
			String[] ids = {"targetedWilsonBalding", "targetedWilsonBaldingFixedHeight", "TargetedWide",
					"targetedRangeSlide", "Range", "ScaleAll", "ScaleIntervalsRandomly", "subtreeSlide"};
			for (int k = 0; k < ops.size(); k++) {
				ops.get(k).setID(ids[k]);
			}
			return ops;
		});
	}

	@Test
	public void testSubtreeSlideOnly() throws Exception {
		assertCacheStaysFresh((tree, w) -> {
			BactrianSubtreeSlide op = new BactrianSubtreeSlide();
			op.initByName("tree", tree, "weight", 3.0, "size", 0.5);
			op.setID("subtreeSlide2");
			return op;
		});
	}
}
