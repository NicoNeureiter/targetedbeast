package targetedbeast.weighttransform;

import beast.base.internal.json.JSONArray;
import beast.base.internal.json.JSONException;
import beast.base.internal.json.JSONObject;

import beast.base.core.Description;
import beast.base.core.Function;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.util.Randomizer;

@Description("Tunes the pointiness coefficient as a multiplier on another transform's live, "
        + "per-proposal exponent, rather than as an absolute grid value like GridTuner. The grid "
        + "holds ratios, not exponents: the effective exponent for a proposal is "
        + "reference.getEffectiveCoefficients()[EXPONENT] * ratios[chosen]. A narrow ratio grid "
        + "centred on 1.0 regularises this transform toward the reference's own tuned value while "
        + "bounding how far the two can diverge, instead of tuning a fully independent exponent "
        + "from scratch (see GridTuner) or adding an explicit penalty term to the reward. Bandit "
        + "mechanics - exploration, learnin, burnin, reward tracking - are identical to GridTuner; "
        + "only the mapping from a chosen grid point to the effective exponent differs.")
public class RatioGridTuner extends CoefficientTuner {

    final public Input<WeightTransform> referenceInput = new Input<>("reference",
            "transform whose live per-proposal exponent this tuner's ratio multiplies. Its "
                    + "startProposal() must run before this tuner's owner's - "
                    + "TargetedWeightOperator.startTransformProposal() runs edgeWeightTransform "
                    + "before targetWeightTransform for exactly this reason",
            Validate.REQUIRED);
    final public Input<String> ratiosInput = new Input<>("ratios",
            "grid of multipliers applied to the reference's live exponent, whitespace- or "
                    + "comma-separated. Centre it on 1.0 to regularise toward the reference's own "
                    + "tuned value while still allowing a bounded asymmetry",
            "0.7 1.0 1.4", Validate.REQUIRED);
    final public Input<Double> explorationProbInput = new Input<>("explorationProb",
            "probability of ignoring what has been learned and picking a grid point uniformly. "
                    + "A floor above zero keeps every grid point sampled forever", 0.1);
    final public Input<Integer> learninInput = new Input<>("learnin",
            "proposals to collect on every grid point before the choice starts following the "
                    + "reward. Until then the choice is uniform", 200);
    final public Input<Function> rewardInput = new Input<>("reward",
            "quantity whose squared change across a proposal is the reward. Omitting it falls "
                    + "back to acceptance probability");
    final public Input<Integer> burninInput = new Input<>("burnin",
            "proposals to discard entirely before any reward is accumulated", 1000);

    private double[] ratios;
    private double[] meanReward;
    private long[] samples;
    private int current;
    private WeightTransform reference;
    private Function rewardFunction;
    private double rewardBefore;
    private long proposalCount;

    @Override
    public void initAndValidate() {
        String[] fields = ratiosInput.get().trim().split("[,\\s]+");
        if (fields.length < 2) {
            throw new IllegalArgumentException(
                    "ratios must list at least two values, got " + ratiosInput.get());
        }
        ratios = new double[fields.length];
        for (int i = 0; i < fields.length; i++) {
            ratios[i] = Double.parseDouble(fields[i]);
            if (ratios[i] <= 0.0) {
                throw new IllegalArgumentException("ratios must be positive, got " + fields[i]);
            }
        }
        meanReward = new double[fields.length];
        samples = new long[fields.length];
        rewardFunction = rewardInput.get();
        reference = referenceInput.get();
        current = 0;
        proposalCount = 0;
        if (explorationProbInput.get() <= 0.0 || explorationProbInput.get() > 1.0) {
            throw new IllegalArgumentException(
                    "explorationProb must lie in (0, 1]; a floor of zero would let the sampler "
                            + "abandon grid points permanently. Got " + explorationProbInput.get());
        }
    }

    @Override
    public void startProposal(double[] theta, double[] effective) {
        current = chooseGridPoint();
        if (rewardFunction != null) {
            rewardBefore = ProposalReward.begin(rewardFunction);
        }
        double baseExponent = reference.getEffectiveCoefficients()[WeightTransform.EXPONENT];
        double exponent = baseExponent * ratios[current];
        // Defensive clamp: the reference's own leader moves at runtime, so a ratio that was fine
        // against the reference's exponent at one point in the chain can push the product outside
        // this transform's own bounds later, even though nothing here is misconfigured.
        effective[WeightTransform.EXPONENT] =
                Math.exp(transform.clamp(Math.log(exponent), WeightTransform.EXPONENT));
        effective[WeightTransform.OFFSET] = Math.exp(theta[WeightTransform.OFFSET]);
    }

    private int chooseGridPoint() {
        if (Randomizer.nextDouble() < explorationProbInput.get() || !hasLearned()) {
            return Randomizer.nextInt(ratios.length);
        }
        int choice = GridChoice.proportional(meanReward);
        return choice < 0 ? Randomizer.nextInt(ratios.length) : choice;
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
        if (!used) {
            return;
        }
        double reward;
        if (rewardFunction != null) {
            reward = ProposalReward.reward(rewardFunction, rewardBefore);
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

        // Report the resulting effective exponent (not the bare ratio) as the coefficient, so the
        // trace log's "exponent" column means the same thing it does for GridTuner. Recomputed
        // from the reference's current leader, which may have moved since startProposal() chose
        // this ratio - a one-step lag on a pure readout, not something read back as an iterate.
        double baseExponent = reference.getEffectiveCoefficients()[WeightTransform.EXPONENT];
        double exponent = baseExponent * ratios[leader()];
        theta[WeightTransform.EXPONENT] = transform.clamp(Math.log(exponent), WeightTransform.EXPONENT);
        transform.refreshEffective();
    }

    /** Index of the grid point with the highest mean reward. */
    public int leader() {
        int best = 0;
        for (int i = 1; i < ratios.length; i++) {
            if (meanReward[i] > meanReward[best]) {
                best = i;
            }
        }
        return best;
    }

    /** Grid point with the highest mean reward, as a ratio. */
    public double getLeadingRatio() {
        return ratios[leader()];
    }

    public double getMeanReward(int i) {
        return meanReward[i];
    }

    public long getSampleCount(int i) {
        return samples[i];
    }

    public double[] getRatios() {
        return ratios.clone();
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
    }

    @Override
    public void storeState(JSONObject entry) throws JSONException {
        JSONArray grid = new JSONArray();
        for (int i = 0; i < ratios.length; i++) {
            JSONObject point = new JSONObject();
            point.put("ratio", ratios[i]);
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
        if (grid.length() != ratios.length) {
            throw new JSONException("state file holds " + grid.length() + " grid points but the "
                    + "tuner is configured with " + ratios.length);
        }
        for (int i = 0; i < grid.length(); i++) {
            JSONObject point = grid.getJSONObject(i);
            if (point.getDouble("ratio") != ratios[i]) {
                throw new JSONException("state file grid point " + i + " is "
                        + point.getDouble("ratio") + " but the tuner is configured with "
                        + ratios[i]);
            }
            meanReward[i] = point.getDouble("reward");
            samples[i] = point.getLong("n");
        }
        proposalCount = entry.optLong("proposals", getAdaptationCount());
    }

    @Override
    public String[] getLogNames() {
        String[] names = new String[ratios.length + 1];
        names[0] = "leadingRatio";
        for (int i = 0; i < ratios.length; i++) {
            names[i + 1] = "reward." + ratios[i];
        }
        return names;
    }

    @Override
    public double[] getLogValues() {
        double[] values = new double[ratios.length + 1];
        values[0] = getLeadingRatio();
        System.arraycopy(meanReward, 0, values, 1, ratios.length);
        return values;
    }
}
