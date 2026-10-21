package targetedbeast.weighttransform;

import org.junit.Assert;
import org.junit.Test;

/** Math of the weight transforms: pointiness, mean-relative offset, normalization. */
public class WeightTransformTest {

	private static final double TOL = 1e-12;

	/** Normalize to sum 1. */
	private static double[] normalize(double[] w) {
		double sum = 0.0;
		for (double x : w) {
			sum += x;
		}
		double[] p = new double[w.length];
		for (int i = 0; i < w.length; i++) {
			p[i] = w[i] / sum;
		}
		return p;
	}

	private static PowerOffsetTransform power(double exponent, double offset) {
		PowerOffsetTransform t = new PowerOffsetTransform();
		t.initByName("exponent", exponent, "offset", offset, "optimise", false);
		return t;
	}

	/**
	 * With offset 0 the power transform must reproduce the hardcoded
	 * Math.pow(w, 2) at TargetedWilsonBalding.java:252 exactly.
	 */
	@Test
	public void exponentTwoWithoutOffsetReproducesSquaring() {
		double[] raw = { 0.4, 1.7, 0.05, 3.0 };
		double[] got = power(2.0, 0.0).apply(raw);

		Assert.assertEquals(raw.length, got.length);
		for (int k = 0; k < raw.length; k++) {
			Assert.assertEquals(raw[k] * raw[k], got[k], TOL);
		}
	}

	/**
	 * Normalizing a * mean(u) + u must equal the uniform/targeted mixture
	 * (a/(1+a)) * uniform + (1/(1+a)) * normalize(u).
	 */
	@Test
	public void offsetActsAsUniformMixtureProportion() {
		double[] raw = { 0.4, 1.7, 0.05, 3.0 };
		double a = 0.35;
		int k = raw.length;

		double[] u = power(1.0, 0.0).apply(raw);
		double[] expected = new double[k];
		double[] normU = normalize(u);
		for (int i = 0; i < k; i++) {
			expected[i] = (a / (1.0 + a)) * (1.0 / k) + (1.0 / (1.0 + a)) * normU[i];
		}

		double[] got = normalize(power(1.0, a).apply(raw));
		Assert.assertArrayEquals(expected, got, TOL);
	}

	/** With offsetBase="branchLength" the offset mixes towards the normalised lengths instead. */
	@Test
	public void branchLengthOffsetMixesTowardsLengths() {
		double[] raw = { 0.4, 1.7, 0.05, 3.0 };
		double[] lengths = { 2.0, 0.5, 1.0, 0.1 };
		double a = 0.35;
		PowerOffsetTransform t = new PowerOffsetTransform();
		t.initByName("exponent", 1.0, "offset", a, "offsetBase", "branchLength", "optimise", false);

		double[] got = normalize(t.apply(raw, lengths));
		double[] normU = normalize(raw);
		double[] normL = normalize(lengths);
		for (int i = 0; i < raw.length; i++) {
			Assert.assertEquals((a / (1.0 + a)) * normL[i] + (1.0 / (1.0 + a)) * normU[i], got[i], TOL);
		}
	}

	@Test(expected = IllegalStateException.class)
	public void branchLengthOffsetRequiresLengths() {
		PowerOffsetTransform t = new PowerOffsetTransform();
		t.initByName("exponent", 1.0, "offset", 1.0, "offsetBase", "branchLength", "optimise", false);
		t.apply(new double[] { 1.0, 2.0 });
	}

	/** apply() must not modify its argument. */
	@Test
	public void applyDoesNotMutateInput() {
		double[] raw = { 0.4, 1.7, 0.05, 3.0 };
		double[] copy = raw.clone();
		power(2.5, 0.1).apply(raw);
		Assert.assertArrayEquals(copy, raw, 0.0);
	}

	/** Coefficients are readable in natural space, in the fixed order {b, a}. */
	@Test
	public void coefficientsAreReportedInNaturalSpace() {
		double[] c = power(3.5, 0.25).getCoefficients();
		Assert.assertEquals(3.5, c[0], TOL);
		Assert.assertEquals(0.25, c[1], TOL);
	}

	/**
	 * Configured coefficients outside the declared bounds are clamped, not accepted.
	 *
	 * <p>Uses a genuinely tiny positive offset (not exactly 0.0) to exercise the lower-bound
	 * clamp: an exact 0.0 is the distinct "no offset" sentinel (see
	 * {@link #exponentTwoWithoutOffsetReproducesSquaring}) and passes through unclamped as
	 * exactly zero rather than being floored.
	 */
	@Test
	public void coefficientsAreClampedToBounds() {
		double[] c = power(1000.0, 1e-9).getCoefficients();
		Assert.assertEquals(20.0, c[0], TOL);
		Assert.assertEquals(1e-6, c[1], TOL);
	}

	/**
	 * maxExponent raises the exponent's ceiling without disturbing the offset's, and without
	 * changing the default. The default of 20 is a guess rather than a measured limit, so a
	 * sweep that finds the reward still climbing there needs to be able to look further out.
	 */
	@Test
	public void maxExponentRaisesOnlyTheExponentCeiling() {
		PowerOffsetTransform t = new PowerOffsetTransform();
		t.initByName("exponent", 40.0, "offset", 1000.0, "optimise", false, "maxExponent", 64.0);

		double[] c = t.getCoefficients();
		Assert.assertEquals("a starting exponent under the raised ceiling must survive init",
				40.0, c[0], TOL);
		Assert.assertEquals("the offset ceiling must be unaffected", 100.0, c[1], TOL);

		// Still clamped, just at the new bound.
		PowerOffsetTransform over = new PowerOffsetTransform();
		over.initByName("exponent", 1000.0, "optimise", false, "maxExponent", 64.0);
		Assert.assertEquals(64.0, over.getCoefficients()[0], TOL);
	}

	/** A ceiling at or below the lower bound leaves no feasible range and must be rejected. */
	@Test
	public void degenerateMaxExponentIsRejected() {
		PowerOffsetTransform t = new PowerOffsetTransform();
		// initByName wraps whatever initAndValidate throws in a RuntimeException, so match on
		// the message rather than the type.
		RuntimeException e = Assert.assertThrows(RuntimeException.class,
				() -> t.initByName("exponent", 2.0, "optimise", false, "maxExponent", 0.05));
		Assert.assertTrue(e.getMessage(), e.getMessage().contains("maxExponent must exceed"));
	}

	/** The real-space transform must reject being paired with log-space weights. */
	@Test
	public void powerTransformDeclaresRealSpace() {
		Assert.assertFalse(power(2.0, 0.0).requiresLogSpace());
	}

	private static ExpOffsetTransform exp(double exponent, double offset) {
		ExpOffsetTransform t = new ExpOffsetTransform();
		t.initByName("exponent", exponent, "offset", offset, "optimise", false);
		return t;
	}

	private static double[] softmax(double[] x) {
		double max = Double.NEGATIVE_INFINITY;
		for (double v : x) {
			max = Math.max(max, v);
		}
		double[] e = new double[x.length];
		double sum = 0.0;
		for (int i = 0; i < x.length; i++) {
			e[i] = Math.exp(x[i] - max);
			sum += e[i];
		}
		for (int i = 0; i < x.length; i++) {
			e[i] /= sum;
		}
		return e;
	}

	/**
	 * Log-space weights are only defined up to an additive constant, because every consumer
	 * normalizes. The transform must therefore be invariant to shifting them.
	 */
	@Test
	public void expTransformIsInvariantToShiftingLogWeights() {
		double[] logRaw = { -3.2, 0.0, -1.1, -7.4 };
		double[] shifted = new double[logRaw.length];
		for (int i = 0; i < logRaw.length; i++) {
			shifted[i] = logRaw[i] + 12.7;
		}

		double[] a = normalize(exp(1.6, 0.3).apply(logRaw));
		double[] b = normalize(exp(1.6, 0.3).apply(shifted));
		Assert.assertArrayEquals(a, b, TOL);
	}

	/**
	 * The power form on softmax output is exactly temperature scaling:
	 * normalize(softmax(x)^b) == softmax(b*x).
	 */
	@Test
	public void powerOnSoftmaxOutputIsTemperatureScaling() {
		double[] x = { -3.2, 0.0, -1.1, -7.4 };
		double b = 2.4;

		double[] scaled = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			scaled[i] = b * x[i];
		}
		double[] expected = softmax(scaled);

		double[] got = normalize(power(b, 0.0).apply(softmax(x)));
		Assert.assertArrayEquals(expected, got, 1e-10);
	}

	/** A log-weight of negative infinity means an impossible candidate and must stay at the offset. */
	@Test
	public void expTransformHandlesNegativeInfinity() {
		double[] logRaw = { -1.0, Double.NEGATIVE_INFINITY, 0.0 };
		double[] got = exp(1.0, 0.5).apply(logRaw);

		for (double v : got) {
			Assert.assertTrue("expected finite, got " + v, Double.isFinite(v));
		}
		// The impossible candidate carries only the offset, so it is the smallest entry.
		Assert.assertTrue(got[1] < got[0]);
		Assert.assertTrue(got[1] < got[2]);
	}

	/** An all-impossible candidate set must degrade to uniform rather than to NaN. */
	@Test
	public void expTransformOnAllNegativeInfinityIsUniform() {
		double[] logRaw = { Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY };
		double[] got = normalize(exp(1.0, 0.01).apply(logRaw));
		Assert.assertArrayEquals(new double[] { 0.5, 0.5 }, got, TOL);
	}

	@Test
	public void expTransformDeclaresLogSpace() {
		Assert.assertTrue(exp(1.0, 0.0).requiresLogSpace());
	}
}
