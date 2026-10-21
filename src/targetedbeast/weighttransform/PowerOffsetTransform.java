package targetedbeast.weighttransform;

import beast.base.core.Description;

@Description("Weight transform for real-space weights: u_k = w_k^b, plus the offset.")
public class PowerOffsetTransform extends WeightTransform {

	@Override
	protected double[] pointiness(double[] raw, double b) {
		double[] u = new double[raw.length];
		for (int k = 0; k < raw.length; k++) {
			// Math.pow of a negative weight is NaN for fractional b.
			u[k] = raw[k] <= 0.0 ? 0.0 : Math.pow(raw[k], b);
		}
		return u;
	}

	@Override
	public boolean requiresLogSpace() {
		return false;
	}
}
