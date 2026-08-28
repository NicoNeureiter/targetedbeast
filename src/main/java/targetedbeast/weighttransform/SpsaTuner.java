package targetedbeast.weighttransform;

import beast.base.internal.json.JSONException;
import beast.base.internal.json.JSONObject;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.util.Randomizer;

@Description("Tunes the transform coefficients by simultaneous perturbation stochastic "
		+ "approximation: each batch of proposals is made at a randomly jittered point, and the "
		+ "coefficients step along the jitter in proportion to how much the batch's mean "
		+ "acceptance beat a running baseline. One batch buys one gradient estimate regardless of "
		+ "how many coefficients there are, which is what makes it attractive; the cost is that "
		+ "the estimate is local, so it needs a signal-to-noise ratio the operator may not have. "
		+ "See GridTuner for the alternative when there is only one coefficient that matters.")
public class SpsaTuner extends CoefficientTuner {

	final public Input<Boolean> tuneOffsetInput = new Input<>("tuneOffset",
			"if true, the offset coefficient is adapted alongside the exponent; if false (the "
					+ "default) it is held at its configured value and only the exponent is tuned. "
					+ "The exponent dominates the proposal shape, and tuning one coefficient instead "
					+ "of two roughly halves the noise in the gradient estimate", false);
	final public Input<Double> stepSizeInput = new Input<>("stepSize",
			"gain numerator A; the step taken per adaptation is A/(n+n0)^0.602 divided by the "
					+ "current perturbation", 0.03);
	final public Input<Double> perturbationInput = new Input<>("perturbation",
			"perturbation numerator C; coefficients are jittered by C/(n+n0)^0.101 in log space "
					+ "for each batch. The measurable effect of a perturbation grows with C while the "
					+ "measurement noise does not, so a larger C buys signal at the cost of gradient "
					+ "bias of order C^2", 0.4);
	final public Input<Integer> stabilityConstantInput = new Input<>("stabilityConstant",
			"stability constant n0; larger values make the early adaptation gentler", 500);
	final public Input<Double> baselineRateInput = new Input<>("baselineRate",
			"EWMA rate for the running acceptance baseline used as a control variate. The "
					+ "baseline enters the update as the subtrahend, so its variance adds to that of "
					+ "the batch mean: an EWMA at rate lambda averages 2/lambda effective samples, "
					+ "and unless that is well above batchSize the baseline is the dominant noise "
					+ "source rather than the batch", 0.001);
	final public Input<Integer> batchSizeInput = new Input<>("batchSize",
			"number of proposals to average over before taking one adaptation step. The acceptance "
					+ "signal is close to binary, so a single proposal carries very little "
					+ "information; averaging over a batch cuts the noise by sqrt(batchSize) while "
					+ "leaving the gradient unchanged. Raise it when the operator's acceptance rate "
					+ "is low", 200);

	/**
	 * Floor on the baseline used to rescale the update. Caps how far a near-zero baseline can
	 * amplify the step, which matters early on when the baseline has not yet converged.
	 */
	private static final double BASELINE_FLOOR = 1e-3;

	/** Rademacher perturbation directions drawn for the current batch. */
	private double[] signs;

	/**
	 * Perturbation actually realized for the current batch, in log space. Equal to
	 * {@code c_n * signs} except where {@link WeightTransform#clamp} truncated it against a bound.
	 */
	private double[] delta;

	/**
	 * Perturbed point in force for the current batch, in natural space. Held here rather than
	 * left in the transform's buffer, because the batch mean is one measurement of the objective
	 * at this point and every proposal in the batch has to be made at it.
	 */
	private double[] perturbed;

	/** Perturbation size c_n in force for the current batch, kept for the update. */
	private double batchC;

	/** Running mean acceptance, used as a control variate in the update. */
	private double baseline;

	/** Number of completed adaptation steps, i.e. of completed batches. */
	private long adaptationCount;

	/** Sum of the acceptance probabilities observed so far in the current batch. */
	private double batchAlphaSum;

	/** Number of proposals contributing to the current batch. */
	private int batchCount;

	/** Baseline as of the start of the current batch, so the batch does not rescale itself. */
	private double batchBaseline;

	@Override
	public void initAndValidate() {
		int dim = WeightTransform.COEFFICIENT_NAMES.length;
		signs = new double[dim];
		delta = new double[dim];
		perturbed = new double[dim];
		baseline = 0.0;
		adaptationCount = 0;
		batchC = getPerturbationSize();
		reset();
	}

	/**
	 * {@inheritDoc}
	 *
	 * <p>A fresh perturbation is drawn only at the start of a batch: the same perturbed point has
	 * to be held for every proposal in the batch, because the batch mean is one measurement of
	 * the objective at that point.
	 */
	@Override
	public void startProposal(double[] theta, double[] effective) {
		if (batchCount > 0) {
			// Mid-batch: re-propose at the point this batch is measuring.
			System.arraycopy(perturbed, 0, effective, 0, perturbed.length);
			return;
		}

		batchBaseline = baseline;
		batchC = getPerturbationSize();
		for (int i = 0; i < theta.length; i++) {
			signs[i] = perturbedCoefficient(i) ? (Randomizer.nextBoolean() ? 1.0 : -1.0) : 0.0;
			if (signs[i] == 0.0) {
				// An unperturbed coefficient must not be pushed through clamp(), which would
				// floor the exact-zero sentinel (-Infinity in log space) up to the lower bound.
				delta[i] = 0.0;
				perturbed[i] = Math.exp(theta[i]);
				effective[i] = perturbed[i];
				continue;
			}
			double perturbedLog = transform.clamp(theta[i] + batchC * signs[i], i);
			double realized = perturbedLog - theta[i];
			// theta = -Infinity is the exact-zero sentinel, where the log-space difference is
			// infinite and carries no scale. Tuning was explicitly requested for this
			// coefficient, so it has to be able to leave zero: fall back to the nominal
			// perturbation, whose first update clamps the coefficient onto its lower bound.
			delta[i] = Double.isFinite(realized) ? realized : batchC * signs[i];
			perturbed[i] = Math.exp(perturbedLog);
			effective[i] = perturbed[i];
		}
	}

	/**
	 * {@inheritDoc}
	 *
	 * <p>Because the perturbation directions are symmetric, the expectation of
	 * {@code (alpha - baseline) * signs} is proportional to the gradient of the expected
	 * acceptance with respect to the log coefficients. Two corrections make that estimate
	 * usable at the acceptance rates targeted tree operators actually run at:
	 *
	 * <ul>
	 * <li><b>Batching.</b> A single alpha is nearly binary, so its variance swamps the drift.
	 * Averaging over {@code batchSize} proposals divides the noise by {@code sqrt(batchSize)}
	 * and leaves the gradient untouched.</li>
	 * <li><b>Relative scaling.</b> Dividing by the baseline makes the step depend on the
	 * <em>fractional</em> change in acceptance rather than the absolute one. Without it the
	 * step size scales with the acceptance rate itself, so a value of {@code stepSize} tuned
	 * at 50% acceptance is two orders of magnitude too small at 0.5%.</li>
	 * </ul>
	 */
	@Override
	public void optimize(double logAlpha, boolean used, double[] theta) {
		// A proposal that aborted before touching the weights says nothing about the
		// coefficients; counting it would inject pure noise.
		if (!used) {
			return;
		}
		double alpha = Math.exp(Math.min(logAlpha, 0.0));
		if (Double.isNaN(alpha)) {
			return;
		}

		batchAlphaSum += alpha;
		batchCount++;
		baseline += baselineRateInput.get() * (alpha - baseline);

		if (batchCount < batchSizeInput.get()) {
			return;
		}

		double meanAlpha = batchAlphaSum / batchCount;
		double relativeGain = (meanAlpha - batchBaseline) / Math.max(batchBaseline, BASELINE_FLOOR);
		// Step along the perturbation that was actually realized. With the nominal
		// delta = c_n * s this reduces exactly to the textbook a_n/c_n * relativeGain * s,
		// since 1/s == s for a Rademacher sign. Where clamp() truncated the perturbation the
		// step shrinks in proportion, and a coefficient that could not move at all takes no
		// step rather than a full-size one in a direction nothing was measured along.
		double gain = getStepSize() / (batchC * batchC);
		for (int i = 0; i < theta.length; i++) {
			if (delta[i] == 0.0) {
				continue;
			}
			theta[i] = transform.clamp(theta[i] + gain * relativeGain * delta[i], i);
		}

		adaptationCount++;
		reset();
		transform.refreshEffective();
	}

	/** Whether coefficient {@code i} takes part in the perturbation and the update. */
	private boolean perturbedCoefficient(int i) {
		return i != WeightTransform.OFFSET || tuneOffsetInput.get();
	}

	@Override
	public void reset() {
		batchAlphaSum = 0.0;
		batchCount = 0;
		batchBaseline = baseline;
	}

	/** Current step gain a_n. */
	protected double getStepSize() {
		return stepSizeInput.get() / Math.pow(adaptationCount + stabilityConstantInput.get(), 0.602);
	}

	/**
	 * Current perturbation size c_n.
	 *
	 * <p>Both a_n/c_n and c_n tend to zero, which is what gives diminishing adaptation.
	 */
	public double getPerturbationSize() {
		return perturbationInput.get() / Math.pow(adaptationCount + stabilityConstantInput.get(), 0.101);
	}

	@Override
	public long getAdaptationCount() {
		return adaptationCount;
	}

	/** Running mean acceptance used as the control variate. */
	public double getAcceptanceBaseline() {
		return baseline;
	}

	/** Number of proposals accumulated into the batch currently in progress. */
	public int getBatchCount() {
		return batchCount;
	}

	@Override
	public void storeState(JSONObject entry) throws JSONException {
		// The iteration count matters as much as the coefficients: without it a resumed chain
		// resets the gain sequences to their maximum and takes a burst of large steps mid-run.
		entry.put("n", adaptationCount);
		entry.put("baseline", baseline);
	}

	@Override
	public void restoreState(JSONObject entry) throws JSONException {
		if (entry.has("n")) {
			adaptationCount = entry.getLong("n");
		}
		if (entry.has("baseline")) {
			baseline = entry.getDouble("baseline");
		}
		reset();
	}
}
