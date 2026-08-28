package targetedbeast.weighttransform;

import beast.base.core.Description;

@Description("Weight transform for real-space weights: u_k = w_k^b, then the mean-relative offset. "
		+ "With exponent=2 and offset=0 this reproduces the squaring that TargetedWilsonBalding "
		+ "applies today. For EdgeWeights that already return a softmax, normalising w^b is exactly "
		+ "temperature scaling, because normalize(softmax(x)^b) == softmax(b*x).")
public class PowerOffsetTransform extends WeightTransform {

	@Override
	protected double[] pointiness(double[] raw, double b) {
		double[] u = new double[raw.length];
		for (int k = 0; k < raw.length; k++) {
			// Guard against negative weights, which would make Math.pow return NaN for
			// fractional exponents. No EdgeWeights implementation should produce them.
			u[k] = raw[k] <= 0.0 ? 0.0 : Math.pow(raw[k], b);
		}
		return u;
	}

	@Override
	public boolean requiresLogSpace() {
		return false;
	}
}
