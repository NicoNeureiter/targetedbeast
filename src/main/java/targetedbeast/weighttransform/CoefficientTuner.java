package targetedbeast.weighttransform;

import beast.base.internal.json.JSONException;
import beast.base.internal.json.JSONObject;

import beast.base.core.BEASTObject;
import beast.base.core.Description;

@Description("Strategy for choosing the coefficients a WeightTransform proposes with, and for "
		+ "updating them from the acceptance feedback the operator reports. Separated from the "
		+ "transform itself because the pointiness function (power, exp) and the way its "
		+ "coefficients are tuned (SPSA, grid) are independent choices: without this split every "
		+ "combination would need its own subclass.")
public abstract class CoefficientTuner extends BEASTObject {

	/**
	 * The transform this tuner belongs to. Set once by {@link WeightTransform#initAndValidate()},
	 * because the tuner is constructed by the XML parser before its owner exists. Tuners use it
	 * for {@link WeightTransform#clamp} and the coefficient bounds.
	 */
	protected WeightTransform transform;

	void setTransform(WeightTransform transform) {
		this.transform = transform;
	}

	/**
	 * Choose the coefficients for the proposal about to be made.
	 *
	 * <p>They are frozen for the whole proposal, which is what keeps its forward and reverse
	 * probabilities mutually consistent so the Hastings ratio stays correct.
	 *
	 * <p>Returning natural values rather than logs for the transform to exponentiate matters to a
	 * tuner proposing from a fixed set: {@code exp(log(16))} is 15.999999999999998, so a round
	 * trip would propose at a value the tuner never declared.
	 *
	 * @param theta     the current iterate, owned by the transform, in log space as everywhere
	 *                  else on {@link WeightTransform}; must not be modified here
	 * @param effective written in place, in natural space. Every entry must be written on every
	 *                  call: the transform does not reset the buffer between proposals
	 */
	public abstract void startProposal(double[] theta, double[] effective);

	/**
	 * Report the outcome of one proposal.
	 *
	 * @param logAlpha difference in posterior between previous and proposed state, plus the
	 *                 Hastings ratio
	 * @param used     false if the operator aborted before touching the weights, in which case
	 *                 the outcome says nothing about the coefficients
	 * @param theta    the current iterate; may be updated in place
	 */
	public abstract void optimize(double logAlpha, boolean used, double[] theta);

	/** Number of completed adaptation steps. Reported in the state file and by the operator. */
	public abstract long getAdaptationCount();

	/** Discard any partially accumulated observations, e.g. after restoring from a state file. */
	public abstract void reset();

	/** Add this tuner's state to the operator's state-file entry. */
	public void storeState(JSONObject entry) throws JSONException {
	}

	/** Read back what {@link #storeState} wrote. Missing fields must leave the defaults. */
	public void restoreState(JSONObject entry) throws JSONException {
	}

	/** Names of extra trace-log columns, appended after the coefficients. */
	public String[] getLogNames() {
		return new String[0];
	}

	/** Values for {@link #getLogNames}, in the same order. */
	public double[] getLogValues() {
		return new double[0];
	}
}
