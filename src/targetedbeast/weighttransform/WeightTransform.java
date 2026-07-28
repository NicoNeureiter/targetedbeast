package targetedbeast.weighttransform;

import java.io.PrintStream;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Loggable;

@Description("Reshapes raw edge or target weights before a targeted operator normalizes them. "
		+ "The pointiness coefficient controls how greedy the resulting proposal is; the offset "
		+ "coefficient mixes in a uniform distribution and is an adaptive replacement for the "
		+ "minWeight input of the EdgeWeights implementations.")
public abstract class WeightTransform extends BEASTObject implements Loggable {

	/** Fixed coefficient order, used by the SPSA vector, the log and the state file. */
	protected static final int EXPONENT = 0;
	protected static final int OFFSET = 1;
	protected static final String[] COEFFICIENT_NAMES = { "exponent", "offset" };

	protected static final double[] LOWER_BOUND = { 0.1, 1e-6 };
	protected static final double[] UPPER_BOUND = { 20.0, 100.0 };

	final public Input<Double> exponentInput = new Input<>("exponent",
			"starting value of the pointiness coefficient b; 2.0 reproduces the exponent "
					+ "hardcoded in TargetedWilsonBalding today", 2.0);
	final public Input<Double> offsetInput = new Input<>("offset",
			"starting value of the offset coefficient a, as a proportion of the mean transformed "
					+ "weight; the resulting proposal is a mixture with weight a/(1+a) on the uniform",
			0.01);
	final public Input<Boolean> optimiseInput = new Input<>("optimise",
			"if true, the coefficients are adapted during the MCMC to raise the acceptance rate", true);

	/** Coefficients in log space. Adapted by optimize(); never read directly by apply(). */
	protected double[] theta;

	/** Coefficients in natural space, frozen for the duration of one proposal. */
	protected double[] effective;

	/** True once apply() has been called since the last startProposal(). */
	private boolean used;

	@Override
	public void initAndValidate() {
		theta = new double[COEFFICIENT_NAMES.length];
		effective = new double[COEFFICIENT_NAMES.length];
		setCoefficients(new double[] { exponentInput.get(), offsetInput.get() });
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

	/** Called by the operator at the start of every proposal, before any call to apply(). */
	public void startProposal() {
		used = false;
	}

	/**
	 * One adaptation step, called from the operator's optimize().
	 *
	 * @param logAlpha difference in posterior between previous and proposed state, plus the
	 *                 Hastings ratio
	 */
	public void optimize(double logAlpha) {
		// Adaptation is added in a later task.
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

	/** Recompute the natural-space coefficients used by apply() from theta. */
	protected void refreshEffective() {
		for (int i = 0; i < theta.length; i++) {
			effective[i] = Math.exp(theta[i]);
		}
	}

	/** Clamp a log-space coefficient to the bounds declared for index i. */
	protected static double clamp(double logValue, int i) {
		return Math.max(Math.log(LOWER_BOUND[i]), Math.min(Math.log(UPPER_BOUND[i]), logValue));
	}

	@Override
	public void init(PrintStream out) {
		String prefix = getID() == null ? "weightTransform" : getID();
		for (String name : COEFFICIENT_NAMES) {
			out.print(prefix + "." + name + "\t");
		}
	}

	@Override
	public void log(long sample, PrintStream out) {
		for (double c : getCoefficients()) {
			out.print(c + "\t");
		}
	}

	@Override
	public void close(PrintStream out) {
	}
}
