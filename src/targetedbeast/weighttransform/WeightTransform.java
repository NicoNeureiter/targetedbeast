package targetedbeast.weighttransform;

import java.io.PrintStream;

import org.json.JSONException;
import org.json.JSONObject;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Loggable;

@Description("Reshapes raw edge or target weights before a targeted operator normalizes them. "
		+ "The exponent sets how greedy the proposal is, the offset mixes in a base distribution. "
		+ "A CoefficientTuner adapts both.")
public abstract class WeightTransform extends BEASTObject implements Loggable {

	/** Fixed coefficient order, used by the tuners, the log and the state file. */
	protected static final int EXPONENT = 0;
	protected static final int OFFSET = 1;
	protected static final String[] COEFFICIENT_NAMES = { "exponent", "offset" };

	protected static final double[] LOWER_BOUND = { 0.1, 1e-6 };
	protected static final double[] DEFAULT_UPPER_BOUND = { 20.0, 100.0 };

	/** Per-instance upper bounds; only the exponent's is configurable. */
	private final double[] upperBound = DEFAULT_UPPER_BOUND.clone();

	final public Input<Double> exponentInput = new Input<>("exponent",
			"starting value of the exponent b", 2.0);
	final public Input<Double> offsetInput = new Input<>("offset",
			"offset a; the proposal is a mixture with weight a/(1+a) on the base distribution", 0.0);
	final public Input<String> offsetBaseInput = new Input<>("offsetBase",
			"distribution the offset mixes towards: 'uniform' or 'branchLength' (source weights only)",
			"uniform");
	final public Input<Double> maxExponentInput = new Input<>("maxExponent",
			"upper bound on the exponent b", 20.0);
	final public Input<Boolean> optimiseInput = new Input<>("optimise",
			"whether the tuner adapts the coefficients", true);
	final public Input<CoefficientTuner> tunerInput = new Input<>("tuner",
			"strategy for adapting the coefficients (default: SpsaTuner)");

	/** Coefficients in log space. Updated by the tuner; never read directly by apply(). */
	protected double[] theta;

	/** Coefficients in natural space, frozen for the duration of one proposal. */
	protected double[] effective;

	/** True once apply() has been called since the last startProposal(). */
	private boolean used;

	private CoefficientTuner tuner;

	@Override
	public void initAndValidate() {
		theta = new double[COEFFICIENT_NAMES.length];
		effective = new double[COEFFICIENT_NAMES.length];

		if (maxExponentInput.get() <= LOWER_BOUND[EXPONENT]) {
			throw new IllegalArgumentException("maxExponent must exceed the lower bound of "
					+ LOWER_BOUND[EXPONENT] + ", got " + maxExponentInput.get());
		}
		upperBound[EXPONENT] = maxExponentInput.get();
		if (!offsetBaseInput.get().equals("uniform") && !offsetBaseInput.get().equals("branchLength")) {
			throw new IllegalArgumentException("offsetBase must be 'uniform' or 'branchLength', got '"
					+ offsetBaseInput.get() + "'");
		}

		// After the bounds, since setCoefficients clamps.
		setCoefficients(new double[] { exponentInput.get(), offsetInput.get() });

		tuner = tunerInput.get();
		if (tuner == null) {
			tuner = new SpsaTuner();
			tuner.initAndValidate();
		}
		tuner.setTransform(this);
	}

	/** True if {@link #apply(double[], double[])} needs branch lengths. */
	public boolean offsetTowardsBranchLength() {
		return offsetBaseInput.get().equals("branchLength");
	}

	/** Reshape one weight vector, mixing towards uniform. */
	public final double[] apply(double[] raw) {
		return apply(raw, null);
	}

	/**
	 * Reshape one weight vector. Drop infeasible candidates first: every entry gets a share of the offset.
	 *
	 * @param base the candidates' branch lengths if {@link #offsetTowardsBranchLength()}, else null
	 * @return a new array
	 */
	public final double[] apply(double[] raw, double[] base) {
		used = true;
		if (offsetTowardsBranchLength() != (base != null)) {
			throw new IllegalStateException("offsetBase is '" + offsetBaseInput.get() + "' but branch lengths were "
					+ (base == null ? "not " : "") + "supplied");
		}
		double[] u = pointiness(raw, effective[EXPONENT]);

		double sum = 0.0;
		for (double x : u) {
			sum += x;
		}
		// Relative to the total, so it is free of the weights' scale.
		double offset = effective[OFFSET] * sum;
		if (base == null) {
			for (int k = 0; k < u.length; k++) {
				u[k] += offset / u.length;
			}
		} else {
			double baseSum = 0.0;
			for (double x : base) {
				baseSum += x;
			}
			for (int k = 0; k < u.length; k++) {
				u[k] += offset * base[k] / baseSum;
			}
		}
		return u;
	}

	/** Apply the pointiness function elementwise. Implementations must return a new array. */
	protected abstract double[] pointiness(double[] raw, double b);

	/** True if this transform expects log-space input weights. */
	public abstract boolean requiresLogSpace();

	/** Fix the coefficients for one proposal, so its forward and reverse moves use the same kernel. */
	public void startProposal() {
		used = false;
		if (!optimiseInput.get()) {
			return;
		}
		tuner.startProposal(theta, effective);
	}

	/** Report one proposal's log acceptance ratio to the tuner. */
	public void optimize(double logAlpha) {
		if (!optimiseInput.get()) {
			return;
		}
		tuner.optimize(logAlpha, used, theta);
	}

	/** The tuner adapting this transform's coefficients. */
	public CoefficientTuner getTuner() {
		return tuner;
	}

	/** Coefficients in use for the current proposal, in natural space. */
	public double[] getEffectiveCoefficients() {
		return effective.clone();
	}

	/** Number of completed adaptation steps. */
	public long getAdaptationCount() {
		return tuner.getAdaptationCount();
	}

	/** True if apply() has been called since the last startProposal(). */
	protected boolean wasUsed() {
		return used;
	}

	/** Current coefficients in natural space, in the order {exponent, offset}. */
	public double[] getCoefficients() {
		double[] c = new double[theta.length];
		for (int i = 0; i < theta.length; i++) {
			c[i] = Math.exp(theta[i]);
		}
		return c;
	}

	/** Set the coefficients in natural space, clamped to their bounds. An exact 0 stays 0 (-Infinity in log space). */
	public void setCoefficients(double[] natural) {
		for (int i = 0; i < theta.length; i++) {
			double logValue = Math.log(natural[i]);
			theta[i] = Double.isInfinite(logValue) ? logValue : clamp(logValue, i);
		}
		refreshEffective();
	}

	/** Write the adaptation state into the owning operator's state-file entry. */
	public void storeState(JSONObject entry) throws JSONException {
		double[] coefficients = getCoefficients();
		entry.put("exponent", coefficients[EXPONENT]);
		entry.put("offset", coefficients[OFFSET]);
		tuner.storeState(entry);
	}

	/** Read back what {@link #storeState} wrote. */
	public void restoreState(JSONObject entry) throws JSONException {
		setCoefficients(new double[] { entry.getDouble("exponent"), entry.getDouble("offset") });
		tuner.restoreState(entry);
	}

	/** Recompute the natural-space coefficients used by apply() from theta. */
	protected void refreshEffective() {
		for (int i = 0; i < theta.length; i++) {
			effective[i] = Math.exp(theta[i]);
		}
	}

	/** Clamp a log-space coefficient to the bounds declared for index i. */
	protected double clamp(double logValue, int i) {
		return Math.max(Math.log(LOWER_BOUND[i]), Math.min(Math.log(upperBound[i]), logValue));
	}

	@Override
	public void init(PrintStream out) {
		String prefix = getID() == null ? "weightTransform" : getID();
		for (String name : COEFFICIENT_NAMES) {
			out.print(prefix + "." + name + "\t");
		}
		for (String name : tuner.getLogNames()) {
			out.print(prefix + "." + name + "\t");
		}
	}

	@Override
	public void log(long sample, PrintStream out) {
		for (double c : getCoefficients()) {
			out.print(c + "\t");
		}
		for (double v : tuner.getLogValues()) {
			out.print(v + "\t");
		}
	}

	@Override
	public void close(PrintStream out) {
	}
}
