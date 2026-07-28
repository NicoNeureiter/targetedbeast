package targetedbeast.weighttransform;

import beast.base.core.Description;

@Description("Weight transform for log-space weights: u_k = exp(b * (w_k - max w)), then the "
		+ "mean-relative offset. The max subtraction is for numerical stability only and cancels "
		+ "in the normalization. Use with an EdgeWeights whose isLogSpace() returns true.")
public class ExpOffsetTransform extends WeightTransform {

	@Override
	protected double[] pointiness(double[] raw, double b) {
		double max = Double.NEGATIVE_INFINITY;
		for (double w : raw) {
			max = Math.max(max, w);
		}

		double[] u = new double[raw.length];
		if (!Double.isFinite(max)) {
			// Every candidate is impossible (or the weights are corrupt). Fall back to a
			// uniform vector so the caller normalizes to uniform instead of dividing by zero.
			java.util.Arrays.fill(u, 1.0);
			return u;
		}

		for (int k = 0; k < raw.length; k++) {
			u[k] = Math.exp(b * (raw[k] - max));
		}
		return u;
	}

	@Override
	public boolean requiresLogSpace() {
		return true;
	}
}
