package targetedbeast.weighttransform;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.Assert;
import org.junit.Test;

import beast.base.util.Randomizer;

/**
 * SPSA adaptation against synthetic acceptance surfaces with a known optimum.
 *
 * <p>The transform never sees a surface directly: it only ever receives one noisy logAlpha per
 * proposal, exactly as it would from MCMC.optimize().
 */
public class SpsaAdaptationTest {

	private static final double OPTIMUM = 6.0;

	/** Peak acceptance at exponent = 6.0, smooth and unimodal in log space. */
	private static double acceptanceFor(double exponent, double peak) {
		double d = Math.log(exponent) - Math.log(OPTIMUM);
		return peak * Math.exp(-(d * d) / (2 * 0.5 * 0.5));
	}

	/** Drive proposals against a noiseless surface, feeding back the exact acceptance. */
	private static void drive(WeightTransform t, int iterations) {
		double[] dummyWeights = { 1.0, 2.0, 3.0 };
		for (int i = 0; i < iterations; i++) {
			t.startProposal();
			t.apply(dummyWeights);
			t.optimize(Math.log(acceptanceFor(t.getEffectiveCoefficients()[0], 0.8)));
		}
	}

	/**
	 * Drive proposals against a low-acceptance surface, reporting an accept/reject outcome
	 * rather than the underlying probability. This is what the transform actually sees in an
	 * MCMC run, and it is the regime batching exists for.
	 */
	private static void driveNoisy(WeightTransform t, int iterations, double peak) {
		double[] dummyWeights = { 1.0, 2.0, 3.0 };
		for (int i = 0; i < iterations; i++) {
			t.startProposal();
			t.apply(dummyWeights);
			double p = acceptanceFor(t.getEffectiveCoefficients()[0], peak);
			// Accept outright or reject outright, as the Metropolis-Hastings decision does.
			t.optimize(Randomizer.nextDouble() < p ? 0.0 : Double.NEGATIVE_INFINITY);
		}
	}

	/** Inputs that live on the tuner rather than on the transform. */
	private static final Set<String> TUNER_INPUTS = new HashSet<>(Arrays.asList(
			"tuneOffset", "stepSize", "perturbation", "stabilityConstant", "baselineRate",
			"batchSize"));

	/**
	 * Build a transform with an SPSA tuner, splitting the arguments between the two by name so
	 * the tests can go on naming both in one list.
	 */
	private static PowerOffsetTransform transform(Object... args) {
		// Maps rather than lists, because initByName rejects a repeated input name and the tests
		// override the defaults set here.
		Map<String, Object> transformArgs = new LinkedHashMap<>();
		transformArgs.put("exponent", 2.0);
		transformArgs.put("offset", 0.01);
		Map<String, Object> tunerArgs = new LinkedHashMap<>();
		for (int i = 0; i < args.length; i += 2) {
			String name = (String) args[i];
			(TUNER_INPUTS.contains(name) ? tunerArgs : transformArgs).put(name, args[i + 1]);
		}

		SpsaTuner tuner = new SpsaTuner();
		tuner.initByName(flatten(tunerArgs));

		transformArgs.put("tuner", tuner);
		PowerOffsetTransform t = new PowerOffsetTransform();
		t.initByName(flatten(transformArgs));
		return t;
	}

	private static Object[] flatten(Map<String, Object> args) {
		List<Object> flat = new ArrayList<>();
		for (Map.Entry<String, Object> e : args.entrySet()) {
			flat.add(e.getKey());
			flat.add(e.getValue());
		}
		return flat.toArray();
	}

	private static SpsaTuner tunerOf(WeightTransform t) {
		return (SpsaTuner) t.getTuner();
	}

	@Test
	public void adaptationClimbsToTheOptimum() {
		Randomizer.setSeed(42);
		PowerOffsetTransform t = transform("optimise", true, "batchSize", 20);

		Assert.assertEquals(2.0, t.getCoefficients()[0], 1e-12);
		drive(t, 50_000);

		// Started at 2.0, optimum is 6.0. Loose tolerance: this is a stochastic estimator.
		Assert.assertEquals(OPTIMUM, t.getCoefficients()[0], 1.5);
	}

	/** Diminishing adaptation: the perturbation must shrink toward zero. */
	@Test
	public void perturbationDecays() {
		Randomizer.setSeed(43);
		PowerOffsetTransform t = transform("optimise", true, "batchSize", 20);

		double initial = tunerOf(t).getPerturbationSize();
		drive(t, 50_000);
		double later = tunerOf(t).getPerturbationSize();

		Assert.assertTrue("perturbation should shrink, was " + initial + " now " + later,
				later < initial);
		// One adaptation step per completed batch, not per proposal.
		Assert.assertEquals(50_000 / 20, t.getAdaptationCount());
	}

	/** With optimise=false the kernel must be completely fixed - required by the DB tests. */
	@Test
	public void optimiseFalseFreezesTheKernel() {
		Randomizer.setSeed(44);
		PowerOffsetTransform t = transform("optimise", false);

		drive(t, 5_000);

		Assert.assertEquals(2.0, t.getCoefficients()[0], 1e-12);
		Assert.assertEquals(0.01, t.getCoefficients()[1], 1e-12);
		Assert.assertEquals(2.0, t.getEffectiveCoefficients()[0], 1e-12);
		Assert.assertEquals(0, t.getAdaptationCount());
	}

	/**
	 * A proposal that aborted before touching the weights carries no information about the
	 * coefficients and must not move them, nor contribute to a batch.
	 */
	@Test
	public void abortedProposalsAreIgnored() {
		Randomizer.setSeed(45);
		PowerOffsetTransform t = transform("optimise", true, "batchSize", 10);

		for (int i = 0; i < 1_000; i++) {
			t.startProposal();
			// no apply() - the operator returned NEGATIVE_INFINITY early
			t.optimize(Double.NEGATIVE_INFINITY);
		}

		Assert.assertEquals(2.0, t.getCoefficients()[0], 1e-12);
		Assert.assertEquals(0, t.getAdaptationCount());
		Assert.assertEquals(0, tunerOf(t).getBatchCount());
	}

	/** logAlpha of negative infinity is a real rejection and must count as acceptance 0. */
	@Test
	public void negativeInfinityLogAlphaIsFinite() {
		Randomizer.setSeed(46);
		PowerOffsetTransform t = transform("optimise", true, "batchSize", 1);

		t.startProposal();
		t.apply(new double[] { 1.0, 2.0 });
		t.optimize(Double.NEGATIVE_INFINITY);

		for (double c : t.getCoefficients()) {
			Assert.assertTrue("coefficient went non-finite: " + c, Double.isFinite(c));
		}
		Assert.assertEquals(1, t.getAdaptationCount());
	}

	/** A batch must hold one perturbed point for all of its proposals. */
	@Test
	public void perturbationIsHeldForTheWholeBatch() {
		Randomizer.setSeed(47);
		PowerOffsetTransform t = transform("optimise", true, "batchSize", 50);
		double[] dummyWeights = { 1.0, 2.0, 3.0 };

		t.startProposal();
		t.apply(dummyWeights);
		double first = t.getEffectiveCoefficients()[0];
		t.optimize(Math.log(0.5));

		for (int i = 1; i < 50; i++) {
			t.startProposal();
			Assert.assertEquals("perturbation changed mid-batch at proposal " + i,
					first, t.getEffectiveCoefficients()[0], 1e-12);
			t.apply(dummyWeights);
			t.optimize(Math.log(0.5));
		}

		// The batch closed, so the next proposal draws a fresh perturbation.
		Assert.assertEquals(1, t.getAdaptationCount());
	}

	/** By default only the exponent is tuned; the offset stays where it was configured. */
	@Test
	public void offsetIsFixedUnlessTuneOffsetIsSet() {
		Randomizer.setSeed(48);
		PowerOffsetTransform t = transform("optimise", true, "batchSize", 20);

		drive(t, 20_000);

		Assert.assertEquals("offset must not move when tuneOffset is false",
				0.01, t.getCoefficients()[1], 1e-12);
		Assert.assertNotEquals("exponent should have moved", 2.0, t.getCoefficients()[0], 0.05);
	}

	@Test
	public void tuneOffsetAdaptsBothCoefficients() {
		Randomizer.setSeed(49);
		PowerOffsetTransform t = transform("optimise", true, "batchSize", 20, "tuneOffset", true);

		drive(t, 20_000);

		Assert.assertNotEquals("offset should have moved", 0.01, t.getCoefficients()[1], 1e-6);
	}

	/**
	 * An exact-zero offset is a sentinel, not a value to be clamped. It must survive adaptation
	 * of the exponent, which shares the clamping code path.
	 */
	@Test
	public void exactZeroOffsetSurvivesAdaptation() {
		Randomizer.setSeed(50);
		PowerOffsetTransform t = transform("optimise", true, "batchSize", 20, "offset", 0.0);

		drive(t, 20_000);

		Assert.assertEquals(0.0, t.getCoefficients()[1], 0.0);
		Assert.assertEquals(0.0, t.getEffectiveCoefficients()[1], 0.0);
	}

	/**
	 * The regime this machinery is for: ~2% acceptance, reported as a binary outcome. A single
	 * proposal carries almost no signal, so a batch is what makes the gradient visible.
	 */
	@Test
	public void batchingRecoversSignalAtLowAcceptance() {
		Randomizer.setSeed(51);
		PowerOffsetTransform batched = transform("optimise", true, "batchSize", 500);
		driveNoisy(batched, 400_000, 0.02);

		double moved = batched.getCoefficients()[0];
		Assert.assertTrue(
				"batched adaptation should climb from 2.0 toward " + OPTIMUM + ", got " + moved,
				moved > 3.0);
	}

	/**
	 * At a bound, half the perturbations point outward and clamp() truncates them away entirely.
	 * The batch is then measured at the <em>unperturbed</em> point, so it says nothing about the
	 * coefficient, and the update must not move it. Stepping along the nominal perturbation
	 * instead would inject a full-size step in a direction nothing was measured along.
	 */
	@Test
	public void aFullyTruncatedPerturbationTakesNoStep() {
		Randomizer.setSeed(52);
		// Pinned at the upper bound by a surface whose optimum lies above it.
		PowerOffsetTransform t = transform("optimise", true, "batchSize", 4, "exponent", 20.0);
		double[] dummyWeights = { 1.0, 2.0, 3.0 };

		int truncatedBatches = 0;
		for (int batch = 0; batch < 2_000; batch++) {
			double before = t.getCoefficients()[0];
			boolean truncated = false;
			for (int k = 0; k < 4; k++) {
				t.startProposal();
				if (k == 0) {
					// The perturbation was truncated to nothing iff the point being measured
					// is the current iterate itself.
					truncated = t.getEffectiveCoefficients()[0] == t.getCoefficients()[0];
				}
				t.apply(dummyWeights);
				// Acceptance increases with the exponent throughout the feasible range.
				t.optimize(Math.log(0.05 * t.getEffectiveCoefficients()[0] / 20.0));
			}
			if (truncated) {
				truncatedBatches++;
				Assert.assertEquals("a fully truncated perturbation must not move the coefficient",
						before, t.getCoefficients()[0], 0.0);
			}
		}

		Assert.assertTrue("the test never exercised a truncated perturbation", truncatedBatches > 20);
	}
}
