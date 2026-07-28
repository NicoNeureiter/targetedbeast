package targetedbeast.weighttransform;

import org.junit.Assert;
import org.junit.Test;

import beast.base.util.Randomizer;

/**
 * SPSA adaptation against a synthetic acceptance surface with a known optimum.
 *
 * <p>The transform never sees the surface directly: it only ever receives one noisy
 * logAlpha per proposal, exactly as it would from MCMC.optimize().
 */
public class SpsaAdaptationTest {

	/** Peak acceptance at exponent = 6.0, smooth and unimodal in log space. */
	private static double acceptanceFor(double exponent) {
		double d = Math.log(exponent) - Math.log(6.0);
		return 0.8 * Math.exp(-(d * d) / (2 * 0.5 * 0.5));
	}

	/** Drive n proposals through the transform against the synthetic surface. */
	private static void drive(WeightTransform t, int iterations) {
		double[] dummyWeights = { 1.0, 2.0, 3.0 };
		for (int i = 0; i < iterations; i++) {
			t.startProposal();
			t.apply(dummyWeights);
			double alpha = acceptanceFor(t.getEffectiveCoefficients()[0]);
			t.optimize(Math.log(alpha));
		}
	}

	private static PowerOffsetTransform transform(boolean optimise) {
		PowerOffsetTransform t = new PowerOffsetTransform();
		t.initByName("exponent", 2.0, "offset", 0.01, "optimise", optimise);
		return t;
	}

	@Test
	public void adaptationClimbsToTheOptimum() {
		Randomizer.setSeed(42);
		PowerOffsetTransform t = transform(true);

		Assert.assertEquals(2.0, t.getCoefficients()[0], 1e-12);
		drive(t, 50_000);

		// Started at 2.0, optimum is 6.0. Loose tolerance: this is a stochastic estimator.
		Assert.assertEquals(6.0, t.getCoefficients()[0], 1.5);
	}

	/** Diminishing adaptation: the perturbation must shrink toward zero. */
	@Test
	public void perturbationDecays() {
		Randomizer.setSeed(43);
		PowerOffsetTransform t = transform(true);

		double initial = t.getPerturbationSize();
		drive(t, 50_000);
		double later = t.getPerturbationSize();

		Assert.assertTrue("perturbation should shrink, was " + initial + " now " + later,
				later < initial);
		Assert.assertEquals(50_000, t.getAdaptationCount());
	}

	/** With optimise=false the kernel must be completely fixed - required by the DB tests. */
	@Test
	public void optimiseFalseFreezesTheKernel() {
		Randomizer.setSeed(44);
		PowerOffsetTransform t = transform(false);

		drive(t, 5_000);

		Assert.assertEquals(2.0, t.getCoefficients()[0], 1e-12);
		Assert.assertEquals(0.01, t.getCoefficients()[1], 1e-12);
		Assert.assertEquals(2.0, t.getEffectiveCoefficients()[0], 1e-12);
		Assert.assertEquals(0, t.getAdaptationCount());
	}

	/**
	 * A proposal that aborted before touching the weights carries no information about the
	 * coefficients and must not move them.
	 */
	@Test
	public void abortedProposalsAreIgnored() {
		Randomizer.setSeed(45);
		PowerOffsetTransform t = transform(true);

		for (int i = 0; i < 1_000; i++) {
			t.startProposal();
			// no apply() - the operator returned NEGATIVE_INFINITY early
			t.optimize(Double.NEGATIVE_INFINITY);
		}

		Assert.assertEquals(2.0, t.getCoefficients()[0], 1e-12);
		Assert.assertEquals(0, t.getAdaptationCount());
	}

	/** logAlpha of negative infinity is a real rejection and must count as acceptance 0. */
	@Test
	public void negativeInfinityLogAlphaIsFinite() {
		Randomizer.setSeed(46);
		PowerOffsetTransform t = transform(true);

		t.startProposal();
		t.apply(new double[] { 1.0, 2.0 });
		t.optimize(Double.NEGATIVE_INFINITY);

		for (double c : t.getCoefficients()) {
			Assert.assertTrue("coefficient went non-finite: " + c, Double.isFinite(c));
		}
		Assert.assertEquals(1, t.getAdaptationCount());
	}
}
