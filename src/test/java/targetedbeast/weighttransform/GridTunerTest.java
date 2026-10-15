package targetedbeast.weighttransform;

import java.io.PrintWriter;
import java.io.StringWriter;

import beast.base.internal.json.JSONException;
import beast.base.internal.json.JSONObject;
import org.junit.Assert;
import org.junit.Test;

import beast.base.core.Function;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;
import targetedbeast.edgeweights.ConstantWeights;
import targetedbeast.operators.TargetedWilsonBalding;

/**
 * The grid tuner against synthetic acceptance surfaces with a known optimum.
 *
 * <p>The tuner never sees a surface directly: it only ever receives one logAlpha per proposal,
 * exactly as it would from MCMC.optimize().
 */
public class GridTunerTest {

	private static final String GRID = "1 2 4 8 16";
	private static final double TOL = 1e-12;

	/** Peak acceptance at exponent 4, which is a grid point, smooth and unimodal in log space. */
	private static double acceptanceFor(double exponent, double peak) {
		double d = Math.log(exponent) - Math.log(4.0);
		return peak * Math.exp(-(d * d) / (2 * 0.5 * 0.5));
	}

	private static PowerOffsetTransform transform(Object... tunerArgs) {
		Object[] args = new Object[tunerArgs.length + 2];
		args[0] = "exponents";
		args[1] = GRID;
		System.arraycopy(tunerArgs, 0, args, 2, tunerArgs.length);

		GridTuner tuner = new GridTuner();
		tuner.initByName(args);

		PowerOffsetTransform t = new PowerOffsetTransform();
		t.initByName("exponent", 2.0, "offset", 0.0, "tuner", tuner);
		return t;
	}

	private static GridTuner tunerOf(WeightTransform t) {
		return (GridTuner) t.getTuner();
	}

	/** Drive proposals, reporting an accept/reject outcome as the MH decision does. */
	private static void driveNoisy(WeightTransform t, int iterations, double peak) {
		double[] dummyWeights = { 1.0, 2.0, 3.0 };
		for (int i = 0; i < iterations; i++) {
			t.startProposal();
			t.apply(dummyWeights);
			double p = acceptanceFor(t.getEffectiveCoefficients()[0], peak);
			t.optimize(Randomizer.nextDouble() < p ? 0.0 : Double.NEGATIVE_INFINITY);
		}
	}

	/** Every proposal must be made at one of the configured grid points, never between them. */
	@Test
	public void proposalsOnlyEverUseGridPoints() {
		Randomizer.setSeed(11);
		PowerOffsetTransform t = transform();

		for (int i = 0; i < 2_000; i++) {
			t.startProposal();
			double b = t.getEffectiveCoefficients()[0];
			boolean onGrid = false;
			for (double point : tunerOf(t).getExponents()) {
				onGrid |= point == b;
			}
			Assert.assertTrue("proposed off the grid at " + b, onGrid);
			t.apply(new double[] { 1.0, 2.0, 3.0 });
			t.optimize(Math.log(0.5));
		}
	}

	/**
	 * The point at the optimum must end up with the highest measured reward, at an acceptance
	 * rate where a single proposal carries almost no information.
	 */
	@Test
	public void theRewardCurvePeaksAtTheOptimum() {
		Randomizer.setSeed(12);
		// Pure measurement: uniform over the grid forever, so every point gets the same
		// number of samples and their rewards are directly comparable.
		PowerOffsetTransform t = transform("explorationProb", 1.0, "burnin", 0);

		driveNoisy(t, 200_000, 0.02);

		GridTuner tuner = tunerOf(t);
		Assert.assertEquals("reward should peak at the surface's optimum",
				4.0, tuner.getLeadingExponent(), TOL);
		for (int i = 0; i < 5; i++) {
			Assert.assertTrue("every grid point must be sampled under a uniform sweep",
					tuner.getSampleCount(i) > 30_000);
		}
	}

	/**
	 * With adaptation on, the sampling frequencies must track the reward means rather than
	 * collapsing onto the leader: a point that scores nearly as well is proposed nearly as often.
	 */
	@Test
	public void adaptationSamplesInProportionToTheReward() {
		Randomizer.setSeed(13);
		int iterations = 200_000;
		double exploration = 0.1;
		PowerOffsetTransform t = transform("explorationProb", exploration, "learnin", 200, "burnin", 0);

		driveNoisy(t, iterations, 0.02);

		GridTuner tuner = tunerOf(t);
		Assert.assertEquals(4.0, tuner.getLeadingExponent(), TOL);

		double total = 0.0;
		for (int i = 0; i < 5; i++) {
			total += tuner.getMeanReward(i);
		}
		for (int i = 0; i < 5; i++) {
			double expected = (1 - exploration) * tuner.getMeanReward(i) / total + exploration / 5;
			double actual = tuner.getSampleCount(i) / (double) iterations;
			// The absolute term absorbs the early phase, where the means are still noisy enough
			// that a weak point is sometimes the temporary leader. Those counts never go away.
			Assert.assertEquals("grid point " + i + " was not sampled in proportion to its reward",
					expected, actual, 0.15 * expected + 0.01);
		}
		// Proportional, not greedy: the leader takes about half, not everything outside the floor.
		Assert.assertTrue(tuner.getSampleCount(2) < 0.7 * iterations);
	}

	/**
	 * The exploration floor must keep every grid point sampled forever, so a point that looked
	 * bad early can still be revisited and the family of kernels stays non-degenerate.
	 */
	@Test
	public void everyGridPointKeepsBeingSampled() {
		Randomizer.setSeed(14);
		PowerOffsetTransform t = transform("explorationProb", 0.1, "learnin", 10, "burnin", 0);

		driveNoisy(t, 100_000, 0.02);

		GridTuner tuner = tunerOf(t);
		for (int i = 0; i < 5; i++) {
			// 100k proposals at a 0.1/5 = 2% floor is ~2000 apiece; well clear of zero.
			Assert.assertTrue("grid point " + i + " was abandoned",
					tuner.getSampleCount(i) > 500);
		}
	}

	/** Proposals during burn-in must not contribute any reward. */
	@Test
	public void burninProposalsAreDiscarded() {
		Randomizer.setSeed(15);
		PowerOffsetTransform t = transform("burnin", 1_000);

		driveNoisy(t, 1_000, 0.9);

		Assert.assertEquals("burn-in must leave nothing accumulated",
				0, tunerOf(t).getAdaptationCount());
	}

	/** A proposal that aborted before touching the weights says nothing and must be ignored. */
	@Test
	public void abortedProposalsAreIgnored() {
		Randomizer.setSeed(16);
		PowerOffsetTransform t = transform("burnin", 0);

		for (int i = 0; i < 1_000; i++) {
			t.startProposal();
			// no apply() - the operator returned NEGATIVE_INFINITY early
			t.optimize(Double.NEGATIVE_INFINITY);
		}

		Assert.assertEquals(0, tunerOf(t).getAdaptationCount());
	}

	/** A grid point above the transform's ceiling would be silently clamped onto its neighbour. */
	@Test
	public void gridPointsBeyondMaxExponentAreRejected() {
		GridTuner tuner = new GridTuner();
		tuner.initByName("exponents", "1 2 4 8 16 32 64");

		PowerOffsetTransform t = new PowerOffsetTransform();
		// initByName wraps whatever initAndValidate throws in a RuntimeException.
		RuntimeException e = Assert.assertThrows(RuntimeException.class,
				() -> t.initByName("exponent", 2.0, "tuner", tuner));
		Assert.assertTrue(e.getMessage(), e.getMessage().contains("raise maxExponent"));
	}

	/** Raising the ceiling is what makes the wide grid legal. */
	@Test
	public void aRaisedMaxExponentAdmitsTheWiderGrid() {
		GridTuner tuner = new GridTuner();
		tuner.initByName("exponents", "1 2 4 8 16 32 64");

		PowerOffsetTransform t = new PowerOffsetTransform();
		t.initByName("exponent", 2.0, "tuner", tuner, "maxExponent", 64.0);

		Assert.assertEquals(7, tunerOf(t).getExponents().length);
	}

	/** A single-point grid is not a grid, and would silently disable the tuning. */
	@Test
	public void aDegenerateGridIsRejected() {
		GridTuner tuner = new GridTuner();
		RuntimeException e = Assert.assertThrows(RuntimeException.class,
				() -> tuner.initByName("exponents", "4"));
		Assert.assertTrue(e.getMessage(), e.getMessage().contains("at least two values"));
	}

	/** An exploration probability of zero lets the tuner abandon grid points permanently. */
	@Test
	public void aZeroExplorationFloorIsRejected() {
		GridTuner tuner = new GridTuner();
		RuntimeException e = Assert.assertThrows(RuntimeException.class,
				() -> tuner.initByName("exponents", GRID, "explorationProb", 0.0));
		Assert.assertTrue(e.getMessage(), e.getMessage().contains("explorationProb"));
	}

	/** The accumulated reward curve must survive a resume, or the chain restarts its learning. */
	@Test
	public void theRewardCurveSurvivesAStoreRestoreRoundTrip() throws JSONException {
		Randomizer.setSeed(17);

		PowerOffsetTransform target = transform("burnin", 0);
		TargetedWilsonBalding source = operatorWith(target);
		driveNoisy(target, 20_000, 0.05);

		StringWriter buffer = new StringWriter();
		PrintWriter out = new PrintWriter(buffer);
		source.storeToFile(out);
		out.flush();

		PowerOffsetTransform restoredTarget = transform("burnin", 0);
		operatorWith(restoredTarget).restoreFromFile(new JSONObject(buffer.toString()));

		GridTuner before = tunerOf(target);
		GridTuner after = tunerOf(restoredTarget);
		Assert.assertEquals(before.getAdaptationCount(), after.getAdaptationCount());
		Assert.assertEquals(before.getLeadingExponent(), after.getLeadingExponent(), TOL);
		for (int i = 0; i < 5; i++) {
			Assert.assertEquals(before.getMeanReward(i), after.getMeanReward(i), TOL);
			Assert.assertEquals(before.getSampleCount(i), after.getSampleCount(i));
		}
	}

	/** A mutable scalar standing in for {@code <parameter idref="likelihood"/>}. */
	private static class Scalar implements Function {
		double value;

		@Override
		public int getDimension() {
			return 1;
		}

		@Override
		public double getArrayValue() {
			return value;
		}

		@Override
		public double getArrayValue(int dim) {
			return value;
		}
	}

	/**
	 * A surface where acceptance and ESJD disagree: acceptance climbs with the exponent all the
	 * way to the top of the grid, while the jump it buys shrinks fast enough that the expected
	 * squared jump peaks at 4. This is the real pathology - the benchmark run with the acceptance
	 * reward pinned the exponent at its ceiling of 64.
	 */
	private static double acceptanceOf(double b) {
		return Math.min(0.9, 0.05 * b);
	}

	private static double esjdOf(double b) {
		double d = Math.log(b) - Math.log(4.0);
		return Math.exp(-(d * d) / (2 * 0.5 * 0.5));
	}

	/** Drive proposals, moving the scalar only when the proposal is accepted. */
	private static void driveConflicting(WeightTransform t, Scalar reward, int iterations) {
		double[] dummyWeights = { 1.0, 2.0, 3.0 };
		for (int i = 0; i < iterations; i++) {
			t.startProposal();
			t.apply(dummyWeights);
			double b = t.getEffectiveCoefficients()[0];
			double p = acceptanceOf(b);
			if (Randomizer.nextDouble() < p) {
				// Accepted: the state moves by whatever jump gives this point its ESJD, and it
				// has already moved by the time MCMC calls optimize().
				reward.value += Math.sqrt(esjdOf(b) / p);
				t.optimize(0.0);
			} else {
				// Rejected: MCMC has already restored, so the scalar is unchanged.
				t.optimize(Double.NEGATIVE_INFINITY);
			}
		}
	}

	@Test
	public void theEsjdRewardFindsTheOptimumThatAcceptanceMisses() {
		Randomizer.setSeed(18);
		Scalar reward = new Scalar();
		PowerOffsetTransform t = transform("explorationProb", 1.0, "burnin", 0, "reward", reward);

		driveConflicting(t, reward, 200_000);

		Assert.assertEquals("ESJD peaks at 4 on this surface",
				4.0, tunerOf(t).getLeadingExponent(), TOL);
	}

	/** The same surface without a reward source: acceptance alone walks to the top of the grid. */
	@Test
	public void theAcceptanceFallbackWalksToTheCeiling() {
		Randomizer.setSeed(18);
		Scalar reward = new Scalar();
		PowerOffsetTransform t = transform("explorationProb", 1.0, "burnin", 0);

		driveConflicting(t, reward, 200_000);

		Assert.assertEquals("acceptance is monotone in the exponent, so it picks the top point",
				16.0, tunerOf(t).getLeadingExponent(), TOL);
	}

	/**
	 * The jump is measured across the whole proposal, so a rejection - where MCMC has already
	 * restored the state by the time optimize() runs - must register as zero rather than be
	 * skipped. Skipping them would measure the jump given acceptance, not the expected jump.
	 */
	@Test
	public void rejectionsCountAsZeroJump() {
		Scalar reward = new Scalar();
		PowerOffsetTransform t = transform("explorationProb", 1.0, "burnin", 0, "reward", reward);

		t.startProposal();
		t.apply(new double[] { 1.0, 2.0, 3.0 });
		t.optimize(Double.NEGATIVE_INFINITY);

		GridTuner tuner = tunerOf(t);
		Assert.assertEquals("a rejection is still an observation", 1, tuner.getAdaptationCount());
		double total = 0.0;
		for (int i = 0; i < 5; i++) {
			total += tuner.getMeanReward(i);
		}
		Assert.assertEquals("and its reward is zero", 0.0, total, TOL);
	}

	private static TargetedWilsonBalding operatorWith(PowerOffsetTransform target) {
		Tree tree = new Tree("((A:1.0,B:1.0):1.0,C:2.0):0.0;");
		ConstantWeights weights = new ConstantWeights();
		weights.initByName("tree", tree);

		TargetedWilsonBalding op = new TargetedWilsonBalding();
		op.initByName("tree", tree, "weight", 1.0, "edgeWeights", weights,
				"targetWeightTransform", target);
		op.setID("targetedWB");
		return op;
	}

	/** A ProposalReward is begun when a proposal starts and read once when its outcome is reported. */
	@Test
	public void proposalRewardIsBegunAndRead() {
		int[] calls = { 0, 0 };
		class CountingReward implements Function, ProposalReward {
			@Override
			public void beginProposal() {
				calls[0]++;
			}

			@Override
			public double jump() {
				calls[1]++;
				return 1.0;
			}

			@Override
			public int getDimension() {
				return 1;
			}

			@Override
			public double getArrayValue(int dim) {
				throw new UnsupportedOperationException();
			}
		}
		PowerOffsetTransform t = transform("reward", new CountingReward());
		t.startProposal();
		t.startProposal(); // MCMC skipped optimize() for the first proposal
		t.apply(new double[] { 1.0, 2.0 });
		t.optimize(0.0);
		Assert.assertArrayEquals(new int[] { 2, 1 }, calls);
	}
}
