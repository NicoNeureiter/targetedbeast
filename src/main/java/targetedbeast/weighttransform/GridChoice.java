package targetedbeast.weighttransform;

import beast.base.util.Randomizer;

/** Choosing a grid point with probability increasing in its mean reward. */
final class GridChoice {

	private GridChoice() {
	}

	/**
	 * A grid point drawn proportionally to its mean reward, or -1 if no point has a positive
	 * weight. Signed rewards (a displacement change can be negative) are shifted so the lowest
	 * mean gets weight zero; non-negative rewards are used as they are.
	 */
	static int proportional(double[] meanReward) {
		double min = Double.POSITIVE_INFINITY;
		for (double r : meanReward) {
			min = Math.min(min, r);
		}
		double shift = min < 0.0 ? -min : 0.0;
		double total = 0.0;
		for (double r : meanReward) {
			total += r + shift;
		}
		if (!(total > 0.0)) {
			return -1;
		}
		double u = Randomizer.nextDouble() * total;
		for (int i = 0; i < meanReward.length; i++) {
			u -= meanReward[i] + shift;
			if (u < 0.0) {
				return i;
			}
		}
		return meanReward.length - 1;
	}
}
