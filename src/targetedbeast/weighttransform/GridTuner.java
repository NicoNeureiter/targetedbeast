package targetedbeast.weighttransform;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import beast.base.core.Description;
import beast.base.core.Function;
import beast.base.core.Input;
import beast.base.util.Randomizer;

@Description("Tunes the exponent by drawing it from a fixed grid, proportionally to each grid "
		+ "point's mean reward. The offset is not tuned.")
public class GridTuner extends CoefficientTuner {

	final public Input<String> exponentsInput = new Input<>("exponents",
			"grid of exponents, whitespace- or comma-separated",
			"1 2 4 8 16", Input.Validate.REQUIRED);
	final public Input<Double> explorationProbInput = new Input<>("explorationProb",
			"probability of picking a grid point uniformly; 1.0 disables adaptation", 0.1);
	final public Input<Integer> learninInput = new Input<>("learnin",
			"observations per grid point before the choice follows the reward", 200);
	final public Input<Function> rewardInput = new Input<>("reward",
			"quantity whose squared change across a proposal is the reward, e.g. the likelihood; "
					+ "defaults to the acceptance probability");
	final public Input<Integer> burninInput = new Input<>("burnin",
			"proposals to discard before accumulating rewards", 1000);
	final public Input<Integer> blockLengthInput = new Input<>("blockLength",
			"proposals per grid-point choice; the reward is measured over the whole block", 1);

	/** Grid points in natural space, as configured. */
	private double[] exponents;

	/** Grid points in log space, for reporting the leader back through theta. */
	private double[] logExponents;

	/** Mean reward per grid point, and the number of observations behind each mean. */
	private double[] meanReward;
	private long[] samples;

	/** Grid point chosen for the proposal currently in flight. */
	private int current;

	/** Reward source (null: acceptance) and its value at the start of the block. */
	private Function rewardFunction;
	private double rewardBefore;

	/** Proposals seen so far, including those discarded during burn-in. */
	private long proposalCount;

	/** Whether a block is open, the proposals counted in it, and its summed acceptance. */
	private boolean blockOpen;
	private int inBlock;
	private double blockAcceptance;

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
		if (blockLengthInput.get() < 1) {
			throw new IllegalArgumentException("blockLength must be at least 1, got " + blockLengthInput.get());
		}
		if (explorationProbInput.get() <= 0.0 || explorationProbInput.get() > 1.0) {
			throw new IllegalArgumentException(
					"explorationProb must lie in (0, 1]; a floor of zero would let the sampler "
							+ "abandon grid points permanently. Got " + explorationProbInput.get());
		}
	}

	@Override
	void setTransform(WeightTransform transform) {
		super.setTransform(transform);
		// Checked here since the bounds live on the transform. Clamping would merge grid points.
		for (double b : exponents) {
			if (transform.clamp(Math.log(b), WeightTransform.EXPONENT) != Math.log(b)) {
				throw new IllegalArgumentException("grid exponent " + b + " lies outside the "
						+ "transform's bounds; raise maxExponent to at least " + b);
			}
		}
	}

	@Override
	public void startProposal(double[] theta, double[] effective) {
		// A proposal whose outcome was not reported extends the open block.
		if (!blockOpen || blockLengthInput.get() == 1) {
			current = chooseGridPoint();
			if (rewardFunction != null) {
				rewardBefore = rewardFunction.getArrayValue();
			}
			blockOpen = true;
			inBlock = 0;
			blockAcceptance = 0.0;
		}
		effective[WeightTransform.EXPONENT] = exponents[current];
		// The offset is not gridded.
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
		// An aborted proposal says nothing about the coefficients.
		if (!used) {
			return;
		}
		inBlock++;
		blockAcceptance += Math.exp(Math.min(logAlpha, 0.0));
		if (inBlock < blockLengthInput.get()) {
			return;
		}
		blockOpen = false;
		double reward;
		if (rewardFunction != null) {
			double jump = rewardFunction.getArrayValue() - rewardBefore;
			reward = jump * jump;
		} else {
			reward = blockAcceptance / inBlock;
		}
		proposalCount += inBlock;
		if (!Double.isFinite(reward) || proposalCount <= burninInput.get()) {
			return;
		}

		samples[current]++;
		meanReward[current] += (reward - meanReward[current]) / samples[current];

		// Report the leader through theta, for the log and the state file.
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
		long total = 0;
		for (long n : samples) {
			total += n;
		}
		return total;
	}

	@Override
	public void reset() {
		// Every observation is folded in immediately.
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
			// Rewards from a different grid are not comparable.
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
