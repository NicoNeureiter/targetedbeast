package targetedbeast.weighttransform;

import org.json.JSONException;
import org.json.JSONObject;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.util.Randomizer;

@Description("Tunes the coefficients by simultaneous perturbation stochastic approximation (SPSA): "
		+ "each batch of proposals runs at a jittered point, and the coefficients step along the "
		+ "jitter by how much the batch's mean acceptance beat a running baseline.")
public class SpsaTuner extends CoefficientTuner {

	final public Input<Boolean> tuneOffsetInput = new Input<>("tuneOffset",
			"whether to tune the offset as well as the exponent", false);
	final public Input<Double> stepSizeInput = new Input<>("stepSize",
			"gain numerator A; step gain a_n = A/(n+n0)^0.602", 0.03);
	final public Input<Double> perturbationInput = new Input<>("perturbation",
			"perturbation numerator C; log-space jitter c_n = C/(n+n0)^0.101", 0.4);
	final public Input<Integer> stabilityConstantInput = new Input<>("stabilityConstant",
			"stability constant n0; larger values make early adaptation gentler", 500);
	final public Input<Double> baselineRateInput = new Input<>("baselineRate",
			"EWMA rate of the acceptance baseline", 0.001);
	final public Input<Integer> batchSizeInput = new Input<>("batchSize",
			"proposals averaged per adaptation step; raise it at low acceptance", 200);

	/** Floor on the baseline the step is divided by. */
	private static final double BASELINE_FLOOR = 1e-3;

	/** Rademacher perturbation directions drawn for the current batch. */
	private double[] signs;

	/** Realized log-space perturbation: {@code c_n * signs} unless clamped. */
	private double[] delta;

	/** Perturbed point for the current batch, in natural space. */
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

	/** {@inheritDoc} A new perturbation is drawn only at the start of a batch. */
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
				// Bypass clamp(), which would lift an exact zero to the lower bound.
				delta[i] = 0.0;
				perturbed[i] = Math.exp(theta[i]);
				effective[i] = perturbed[i];
				continue;
			}
			double perturbedLog = transform.clamp(theta[i] + batchC * signs[i], i);
			double realized = perturbedLog - theta[i];
			// From an exact zero the difference is infinite: use the nominal perturbation.
			delta[i] = Double.isFinite(realized) ? realized : batchC * signs[i];
			perturbed[i] = Math.exp(perturbedLog);
			effective[i] = perturbed[i];
		}
	}

	/**
	 * {@inheritDoc} Steps once per batch, by the batch's acceptance relative to the baseline, so
	 * the step size does not scale with the acceptance rate.
	 */
	@Override
	public void optimize(double logAlpha, boolean used, double[] theta) {
		// An aborted proposal says nothing about the coefficients.
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
		// Step along the realized perturbation: a clamped coefficient steps less, or not at all.
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

	/** Current perturbation size c_n. */
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
		// Without n, a resumed chain would restart the gain sequences at their largest.
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
