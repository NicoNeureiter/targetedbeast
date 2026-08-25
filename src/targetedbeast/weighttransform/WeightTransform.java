package targetedbeast.weighttransform;

import java.io.PrintStream;

import org.json.JSONException;
import org.json.JSONObject;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Loggable;

@Description("Reshapes raw edge or target weights before a targeted operator normalizes them. "
		+ "The pointiness coefficient controls how greedy the resulting proposal is; the offset "
		+ "coefficient mixes in a uniform distribution and is an adaptive replacement for the "
		+ "minWeight input of the EdgeWeights implementations. How the coefficients are tuned is "
		+ "delegated to a CoefficientTuner, so the shape of the transform and the search strategy "
		+ "can be chosen independently.")
public abstract class WeightTransform extends BEASTObject implements Loggable {

	/** Fixed coefficient order, used by the tuners, the log and the state file. */
	protected static final int EXPONENT = 0;
	protected static final int OFFSET = 1;
	protected static final String[] COEFFICIENT_NAMES = { "exponent", "offset" };

	protected static final double[] LOWER_BOUND = { 0.1, 1e-6 };
	protected static final double[] DEFAULT_UPPER_BOUND = { 20.0, 100.0 };

	/**
	 * Per-instance upper bounds, so the exponent ceiling can be raised without touching the
	 * offset's. Only the exponent entry is configurable; see {@link #maxExponentInput}.
	 */
	private final double[] upperBound = DEFAULT_UPPER_BOUND.clone();

	final public Input<Double> exponentInput = new Input<>("exponent",
			"starting value of the pointiness coefficient b; 2.0 reproduces the exponent "
					+ "hardcoded in TargetedWilsonBalding today", 2.0);
	final public Input<Double> offsetInput = new Input<>("offset",
			"value of the offset coefficient a, as a proportion of the mean transformed weight; the "
					+ "resulting proposal is a mixture with weight a/(1+a) on the uniform. The "
					+ "default of 0 means no uniform mixture at all",
			0.0);
	final public Input<Double> maxExponentInput = new Input<>("maxExponent",
			"upper bound on the pointiness coefficient b. This is what keeps the family of "
					+ "fixed-coefficient kernels compact, which the diminishing-adaptation "
					+ "argument needs, and it also guards against a proposal so pointy that it "
					+ "oscillates between a handful of topologies at high acceptance. The default "
					+ "of 20 is a conservative guess rather than a measured limit; raise it if a "
					+ "sweep shows the reward still climbing at the bound", 20.0);
	final public Input<Boolean> optimiseInput = new Input<>("optimise",
			"if true, the coefficients are adapted during the MCMC by the tuner. Set it to false "
					+ "to pin the transform at its configured coefficients", true);
	final public Input<CoefficientTuner> tunerInput = new Input<>("tuner",
			"strategy for adapting the coefficients. Defaults to SPSA with its own defaults, "
					+ "which is what this transform did before tuners were separable");

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

		// Must follow the bound assignment, since setCoefficients clamps: a starting exponent
		// above the default ceiling has to be honoured, not silently pulled back to 20.
		setCoefficients(new double[] { exponentInput.get(), offsetInput.get() });

		tuner = tunerInput.get();
		if (tuner == null) {
			tuner = new SpsaTuner();
			tuner.initAndValidate();
		}
		// The tuner needs the bounds, which live here, so it can only be finished off once this
		// object is valid.
		tuner.setTransform(this);
	}

	/**
	 * Reshape one feasible weight vector.
	 *
	 * <p>Every entry of {@code raw} must be a genuinely selectable candidate. Infeasible
	 * candidates must be dropped by the caller beforehand: the offset is added to every
	 * entry, so passing an infeasible candidate in as a zero weight would give it positive
	 * selection probability.
	 *
	 * @return a new array; {@code raw} is not modified
	 */
	public final double[] apply(double[] raw) {
		used = true;
		double[] u = pointiness(raw, effective[EXPONENT]);

		double sum = 0.0;
		for (double x : u) {
			sum += x;
		}
		// Offset relative to the mean, so that it is free of the scale of the raw weights
		// and invariant to the arbitrary additive constant in log-space weights.
		double offset = effective[OFFSET] * sum / u.length;
		for (int k = 0; k < u.length; k++) {
			u[k] += offset;
		}
		return u;
	}

	/** Apply the pointiness function elementwise. Implementations must return a new array. */
	protected abstract double[] pointiness(double[] raw, double b);

	/** True if this transform expects log-space input weights. */
	public abstract boolean requiresLogSpace();

	/**
	 * Called by the operator at the start of every proposal, before any call to apply().
	 *
	 * <p>Delegates the choice of coefficients to the tuner and freezes the result for the whole
	 * proposal, which is what keeps the forward and reverse probabilities of a single proposal
	 * mutually consistent so the Hastings ratio stays correct.
	 */
	public void startProposal() {
		used = false;
		if (!optimiseInput.get()) {
			return;
		}
		tuner.startProposal(theta, effective);
	}

	/**
	 * Report the outcome of one proposal to the tuner.
	 *
	 * @param logAlpha difference in posterior between previous and proposed state, plus the
	 *                 Hastings ratio
	 */
	public void optimize(double logAlpha) {
		if (!optimiseInput.get()) {
			return;
		}
		tuner.optimize(logAlpha, used, theta);
	}

	/** The tuner adapting this transform's coefficients. Never null after initAndValidate(). */
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

	/**
	 * Set the coefficients from natural-space values, clamping each to its bounds.
	 *
	 * <p>An exact {@code 0.0} is a distinct case from a small positive value: it is the
	 * sentinel for "no offset at all" (as used to reproduce the pre-transform hardcoded
	 * squaring, {@code exponent=2, offset=0}) and is passed through as {@code -Infinity} in
	 * log space rather than floored to the lower bound. {@code Math.exp(-Infinity) == 0.0}
	 * exactly, so the effective coefficient is exactly zero. Only a finite, merely
	 * out-of-bounds value gets clamped.
	 */
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
