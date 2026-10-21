package targetedbeast.weighttransform;

import org.json.JSONException;
import org.json.JSONObject;

import beast.base.core.BEASTObject;
import beast.base.core.Description;

@Description("Strategy for choosing and adapting a WeightTransform's coefficients.")
public abstract class CoefficientTuner extends BEASTObject {

	/** The owning transform, set by {@link WeightTransform#initAndValidate()}. */
	protected WeightTransform transform;

	void setTransform(WeightTransform transform) {
		this.transform = transform;
	}

	/**
	 * Choose the coefficients for the next proposal.
	 *
	 * @param theta     current coefficients in log space; read-only
	 * @param effective output in natural space; every entry must be written
	 */
	public abstract void startProposal(double[] theta, double[] effective);

	/**
	 * Report one proposal's outcome.
	 *
	 * @param used  false if the operator aborted before using the weights
	 * @param theta current coefficients in log space; may be updated in place
	 */
	public abstract void optimize(double logAlpha, boolean used, double[] theta);

	/** Number of completed adaptation steps. */
	public abstract long getAdaptationCount();

	/** Discard partially accumulated observations. */
	public abstract void reset();

	/** Add this tuner's state to the operator's state-file entry. */
	public void storeState(JSONObject entry) throws JSONException {
	}

	/** Read back what {@link #storeState} wrote; missing fields keep their defaults. */
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
