package targetedbeast.weighttransform;

import beast.base.internal.json.JSONArray;
import beast.base.internal.json.JSONException;
import beast.base.internal.json.JSONObject;

import beast.base.core.Description;
import beast.base.core.Function;
import beast.base.core.Input;
import beast.base.util.Randomizer;

@Description("Tunes the pointiness coefficient by picking each proposal's value from a fixed "
		+ "grid and keeping a running reward per grid point. Unlike SPSA this makes no local "
		+ "gradient estimate: the grid points are far enough apart that the difference between "
		+ "them is large compared to the measurement noise, there is no baseline to subtract and "
		+ "no gain schedule to decay, so the estimate only improves with more samples. That "
		+ "trades badly against many coefficients - the grid grows exponentially - and well "
		+ "against one, which is the situation here. Only the exponent is gridded; the offset "
		+ "stays at the value configured on the transform.")
public class GridTuner extends CoefficientTuner {

	final public Input<String> exponentsInput = new Input<>("exponents",
			"the grid of pointiness values to choose among, whitespace- or comma-separated. A "
					+ "geometric grid is the natural choice, since the exponent acts "
					+ "multiplicatively on the log weights. Values above the transform's "
					+ "maxExponent are rejected rather than silently clamped onto each other",
			"1 2 4 8 16", Input.Validate.REQUIRED);
	final public Input<Double> explorationProbInput = new Input<>("explorationProb",
			"probability of ignoring what has been learned and picking a grid point uniformly. "
					+ "A floor above zero keeps every grid point sampled forever, so a point that "
					+ "looked bad early can still be revisited and the family of kernels stays "
					+ "non-degenerate. Set it to 1.0 to disable the adaptation entirely and just "
					+ "measure the reward curve: equal sample sizes per point give an unbiased "
					+ "comparison, which is what says whether the surface has an interior optimum "
					+ "at all", 0.1);
	final public Input<Integer> learninInput = new Input<>("learnin",
			"proposals to collect on every grid point before the choice starts following the "
					+ "reward. Until then the choice is uniform. This is what stops one lucky "
					+ "early acceptance from locking the sampler onto a grid point", 200);
	final public Input<Function> rewardInput = new Input<>("reward",
			"quantity whose squared change across a proposal is the reward, typically "
					+ "<parameter idref=\"likelihood\"/>. Since MCMC calls optimize() after "
					+ "accepting or restoring, this measures the realized jump and its mean is "
					+ "the expected squared jump distance. Omitting it falls back to acceptance "
					+ "probability, which rises with pointiness long past the point where the "
					+ "chain stops exploring");
	final public Input<Integer> burninInput = new Input<>("burnin",
			"proposals to discard entirely before any reward is accumulated. The early chain is "
					+ "still travelling to stationarity, where acceptance says more about the "
					+ "starting tree than about the coefficient", 1000);

	/** Grid points in natural space, as configured. */
	private double[] exponents;

	/** Grid points in log space, for reporting the leader back through theta. */
	private double[] logExponents;

	/** Mean reward per grid point, and the number of observations behind each mean. */
	private double[] meanReward;
	private long[] samples;

	/** Grid point chosen for the proposal currently in flight. */
	private int current;

	/** Reward source, and its value as of the start of the proposal in flight. Null = acceptance. */
	private Function rewardFunction;
	private double rewardBefore;

	/** Proposals seen so far, including those discarded during burn-in. */
	private long proposalCount;

	@Override
	public void initAndValidate() {
		String[] fields = exponentsInput.get().trim().split("[,\\s]+");
		if (fields.length < 2) {
			throw new IllegalArgumentException(
					"exponents must list at least two values, got " + exponentsInput.get());
		}
		exponents = new double[fields.length];
		logExponents = new double[fields.length];
		for (int i = 0; i < fields.length; i++) {
			exponents[i] = Double.parseDouble(fields[i]);
			if (exponents[i] <= 0.0) {
				throw new IllegalArgumentException("exponents must be positive, got " + fields[i]);
			}
			logExponents[i] = Math.log(exponents[i]);
		}
		meanReward = new double[fields.length];
		samples = new long[fields.length];
		rewardFunction = rewardInput.get();
		current = 0;
		proposalCount = 0;
		if (explorationProbInput.get() <= 0.0 || explorationProbInput.get() > 1.0) {
			throw new IllegalArgumentException(
					"explorationProb must lie in (0, 1]; a floor of zero would let the sampler "
							+ "abandon grid points permanently. Got " + explorationProbInput.get());
		}
	}

	@Override
	void setTransform(WeightTransform transform) {
		super.setTransform(transform);
		// Deferred to here because the bounds live on the transform, which does not exist yet
		// when this tuner is initialised. Clamping instead would silently collapse the top of
		// the grid onto a single value and quietly turn a 7-point sweep into a 5-point one.
		for (double b : exponents) {
			if (transform.clamp(Math.log(b), WeightTransform.EXPONENT) != Math.log(b)) {
				throw new IllegalArgumentException("grid exponent " + b + " lies outside the "
						+ "transform's bounds; raise maxExponent to at least " + b);
			}
		}
	}

	@Override
	public void startProposal(double[] theta, double[] effective) {
		current = chooseGridPoint();
		if (rewardFunction != null) {
			rewardBefore = rewardFunction.getArrayValue();
		}
		effective[WeightTransform.EXPONENT] = exponents[current];
		// The offset is not gridded, so it proposes at whatever the transform holds. The
		// exact-zero sentinel is -Infinity in log space, and exp of that is exactly 0.0.
		effective[WeightTransform.OFFSET] = Math.exp(theta[WeightTransform.OFFSET]);
	}

	/** Samples proportionally to the reward means, uniform until every point has met learnin. */
	private int chooseGridPoint() {
		if (Randomizer.nextDouble() < explorationProbInput.get() || !hasLearned()) {
			return Randomizer.nextInt(exponents.length);
		}
		double total = 0.0;
		for (double r : meanReward) {
			total += r;
		}
		if (!(total > 0.0)) {
			return Randomizer.nextInt(exponents.length);
		}
		double u = Randomizer.nextDouble() * total;
		for (int i = 0; i < exponents.length; i++) {
			u -= meanReward[i];
			if (u < 0.0) {
				return i;
			}
		}
		return exponents.length - 1;
	}

	private boolean hasLearned() {
		for (long n : samples) {
			if (n < learninInput.get()) {
				return false;
			}
		}
		return true;
	}

	@Override
	public void optimize(double logAlpha, boolean used, double[] theta) {
		// A proposal that aborted before touching the weights says nothing about the
		// coefficients; counting it would inject pure noise.
		if (!used) {
			return;
		}
		double reward;
		if (rewardFunction != null) {
			double jump = rewardFunction.getArrayValue() - rewardBefore;
			reward = jump * jump;
		} else {
			reward = Math.exp(Math.min(logAlpha, 0.0));
		}
		if (!Double.isFinite(reward)) {
			return;
		}

		proposalCount++;
		if (proposalCount <= burninInput.get()) {
			return;
		}

		samples[current]++;
		meanReward[current] += (reward - meanReward[current]) / samples[current];

		// Report the current leader as the coefficient, so the trace log and the state file say
		// which grid point the sampler is actually favouring. This is a readout, not an iterate:
		// nothing here reads theta back, and the grid points themselves never move.
		theta[WeightTransform.EXPONENT] = logExponents[leader()];
		transform.refreshEffective();
	}

	/** Index of the grid point with the highest mean reward. */
	public int leader() {
		int best = 0;
		for (int i = 1; i < exponents.length; i++) {
			if (meanReward[i] > meanReward[best]) {
				best = i;
			}
		}
		return best;
	}

	/** Grid point with the highest mean reward, in natural space. */
	public double getLeadingExponent() {
		return exponents[leader()];
	}

	/** Mean reward accumulated at grid point {@code i}. */
	public double getMeanReward(int i) {
		return meanReward[i];
	}

	/** Observations behind {@link #getMeanReward}. */
	public long getSampleCount(int i) {
		return samples[i];
	}

	public double[] getExponents() {
		return exponents.clone();
	}

	@Override
	public long getAdaptationCount() {
		// One observation is one adaptation: unlike SPSA there is no batch to complete first.
		long total = 0;
		for (long n : samples) {
			total += n;
		}
		return total;
	}

	@Override
	public void reset() {
		// Nothing is accumulated between proposals; every observation is folded in immediately.
	}

	@Override
	public void storeState(JSONObject entry) throws JSONException {
		JSONArray grid = new JSONArray();
		for (int i = 0; i < exponents.length; i++) {
			JSONObject point = new JSONObject();
			point.put("exponent", exponents[i]);
			point.put("reward", meanReward[i]);
			point.put("n", samples[i]);
			grid.put(point);
		}
		entry.put("grid", grid);
		entry.put("proposals", proposalCount);
	}

	@Override
	public void restoreState(JSONObject entry) throws JSONException {
		if (!entry.has("grid")) {
			return;
		}
		JSONArray grid = entry.getJSONArray("grid");
		if (grid.length() != exponents.length) {
			// The grid was reconfigured between runs, so the stored rewards are not comparable
			// to the current points. Starting over is the only honest option.
			throw new JSONException("state file holds " + grid.length() + " grid points but the "
					+ "tuner is configured with " + exponents.length);
		}
		for (int i = 0; i < grid.length(); i++) {
			JSONObject point = grid.getJSONObject(i);
			if (point.getDouble("exponent") != exponents[i]) {
				throw new JSONException("state file grid point " + i + " is "
						+ point.getDouble("exponent") + " but the tuner is configured with "
						+ exponents[i]);
			}
			meanReward[i] = point.getDouble("reward");
			samples[i] = point.getLong("n");
		}
		proposalCount = entry.optLong("proposals", getAdaptationCount());
	}

	@Override
	public String[] getLogNames() {
		String[] names = new String[exponents.length + 1];
		names[0] = "leadingExponent";
		for (int i = 0; i < exponents.length; i++) {
			names[i + 1] = "reward." + exponents[i];
		}
		return names;
	}

	@Override
	public double[] getLogValues() {
		double[] values = new double[exponents.length + 1];
		values[0] = getLeadingExponent();
		System.arraycopy(meanReward, 0, values, 1, exponents.length);
		return values;
	}
}
