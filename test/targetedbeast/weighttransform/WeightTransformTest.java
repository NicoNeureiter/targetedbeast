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

	/** The real-space transform must reject being paired with log-space weights. */
	@Test
	public void powerTransformDeclaresRealSpace() {
		Assert.assertFalse(power(2.0, 0.0).requiresLogSpace());
	}
}
