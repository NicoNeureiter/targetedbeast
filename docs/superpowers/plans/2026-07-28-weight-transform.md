# WeightTransform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an adaptive `WeightTransform` layer that reshapes `EdgeWeights` output before a targeted operator normalizes it, tuning its own coefficients during the MCMC to raise the acceptance rate.

**Architecture:** A new `targetedbeast.weighttransform` package holds an abstract `WeightTransform` (BEASTObject, not a CalculationNode — it carries tuning state like `ScaleOperator.scaleFactor`, not MCMC state) plus two concrete forms. `TargetedWilsonBalding` gains two optional transform inputs; when null, every existing code path is bit-identical to today. Adaptation uses SPSA, because BEAST's `calcDelta` targets a fixed acceptance rate and cannot maximize one.

**Tech Stack:** Java 17, BEAST 2.7, Ant, JUnit 3-style (`junit.framework`) and JUnit 4 (`org.junit`) both present in this repo, Snakemake + Python for the benchmark harness.

Design spec: `docs/superpowers/specs/2026-07-28-weight-transform-design.md`. Read it first.

## Global Constraints

- Package for new classes: `targetedbeast.weighttransform`.
- **Null transform inputs must leave `TargetedWilsonBalding` bit-identical to today.** This is the single most important constraint; a reviewer will check it.
- Coefficient order is fixed everywhere: index `0` = `exponent` (`b`), index `1` = `offset` (`a`). Defaults `b = 2.0`, `a = 0.01`. Bounds `b ∈ [0.1, 20.0]`, `a ∈ [1e-6, 100.0]`.
- Coefficients are adapted in **log space**; all bounds above are natural-space and get log-transformed internally.
- SPSA gains: `a_n = A/(n+n0)^0.602`, `c_n = C/(n+n0)^0.101`, defaults `A = 0.05`, `C = 0.2`, `n0 = 100`, baseline EWMA rate `λ = 0.01`.
- The offset is **relative to the mean** of the pointiness output: `w'_k = a·mean(u) + u_k`. Never an absolute offset — see the spec for why.
- A transform receives only the **feasible subvector**. Infeasible candidates must be excluded by the caller *before* `apply()`, never passed in as zeros, because the offset would resurrect them.
- The transform must be applied **structurally identically in the forward and reverse blocks** of a proposal: same masking order, same interval-weight multiplication, same candidate-set membership.
- Use tabs for indentation in `src/targetedbeast/**` — that is the existing convention in the files being modified.
- **`optimize()` is never called under `-Dbeast.debug=true`** (`MCMC.java:482-485` calls it only in the non-debug branch). Any run that is supposed to exercise adaptation must not set that flag. This does not affect `EdgeWeightRejectConsistencyTest`, which sets the flag deliberately and does not care about adaptation.

## Build and test commands

Run everything from `/Users/nneureiter/Projects/BEAST/targetedbeast`.

```bash
# Run one test class (compiles incrementally first, includes -ea):
ant test-single -Dtest.class=targetedbeast.weighttransform.WeightTransformTest

# MANDATORY before trusting any result after touching more than one file.
# `ant compile` is incremental WITHOUT dependency tracking and silently leaves
# stale classes behind; this has produced phantom failures in this repo before.
rm -rf build build-test && ant compile compile-test

# Raise detailed-balance statistical power (default 100_000):
ant test-single -Dtest.class=targetedbeast.TargetedOperatorDetailedBalanceTest \
                -Ddb.numSamples=360000
```

## Known-failing baseline — do NOT try to fix these

These fail on `main` before this work starts. If you see them, you have not broken anything.

- `TargetedOperatorDetailedBalanceTest`: 12 pass / 4 fail — `testTargWilsonBaldingHeightDelta`, `testTargWilsonBaldingFixedHeight`, `testTargWilsonBaldingMaxHeightRatio` (degenerate `maxHeightRatio=1.0`, zero acceptance), plus one that rotates between `testIntervalScaleOperator` and `testTargWilsonBaldingBrownianFullLikelihoodWeights`. The suite is mildly flaky by multiple-comparison construction (~40 pairs × 3 mappers at 3.2σ).
- `targetedbeast.edgeweights.PriorWeightsTest`: 2 failures.
- `EdgeWeightRejectConsistencyTest.testFelsensteinWeights2`: `@Ignore`d for a pre-existing bug.

**Capture the baseline yourself before Task 5** by running the DB suite once and recording which tests fail. Compare against it afterwards rather than against this list.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/targetedbeast/weighttransform/WeightTransform.java` | Abstract base: coefficient inputs, mean-relative offset, SPSA adaptation, JSON persistence helpers, `Loggable`. |
| `src/targetedbeast/weighttransform/PowerOffsetTransform.java` | `u_k = w_k^b` for real-space weights. |
| `src/targetedbeast/weighttransform/ExpOffsetTransform.java` | `u_k = exp(b·(w_k − max w))` for log-space weights. |
| `src/targetedbeast/edgeweights/EdgeWeights.java` | Gains `default boolean isLogSpace()`. |
| `src/targetedbeast/edgeweights/PartialWeights.java` | Gains a `logSpace` input that skips the final softmax. |
| `src/targetedbeast/operators/TargetedWilsonBalding.java` | Two transform inputs; target-path and source-path integration; JSON persistence override. |
| `test/targetedbeast/weighttransform/WeightTransformTest.java` | Transform math: softmax equivalence, shift invariance, mixture identity, squaring equivalence. |
| `test/targetedbeast/weighttransform/SpsaAdaptationTest.java` | SPSA climbs a synthetic unimodal acceptance surface; gains decay. |
| `test/targetedbeast/TargetedOperatorDetailedBalanceTest.java` | Two new detailed-balance cases with a fixed (non-adapting) transform. |
| `docs/WeightTransformAdaptation.md` | Prose theory write-up. |
| `scripts/generate_xmls.py` | `WEIGHT_TRANSFORMS` axis, `--chain-length` argument. |
| `scripts/Snakefile` | `quick=1` profile. |
| `examples/lassaL_kli22/template.xml` | `{target_weight_transform}` slot. |

---

## Task 1: `WeightTransform` base + `PowerOffsetTransform` (math only)

Adaptation is stubbed out in this task — `optimize()` does nothing yet, coefficients stay at their configured values. Task 3 adds SPSA. This split exists so the transform *math* gets its own review gate before the stochastic machinery lands on top of it.

**Files:**
- Create: `src/targetedbeast/weighttransform/WeightTransform.java`
- Create: `src/targetedbeast/weighttransform/PowerOffsetTransform.java`
- Test: `test/targetedbeast/weighttransform/WeightTransformTest.java`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `public abstract class WeightTransform extends BEASTObject implements Loggable`
  - `public void startProposal()` — resets the `used` flag and recomputes effective coefficients.
  - `public double[] apply(double[] raw)` — returns a **new** array; does not modify `raw`.
  - `public void optimize(double logAlpha)` — no-op until Task 3.
  - `public double[] getCoefficients()` — natural-space `{b, a}`, a defensive copy.
  - `public boolean requiresLogSpace()`
  - `protected abstract double[] pointiness(double[] raw, double b)`
  - `public class PowerOffsetTransform extends WeightTransform`

- [ ] **Step 1: Write the failing test**

Create `test/targetedbeast/weighttransform/WeightTransformTest.java`:

```java
package targetedbeast.weighttransform;

import org.junit.Assert;
import org.junit.Test;

/** Math of the weight transforms: pointiness, mean-relative offset, normalization. */
public class WeightTransformTest {

	private static final double TOL = 1e-12;

	/** Normalize to sum 1. */
	private static double[] normalize(double[] w) {
		double sum = 0.0;
		for (double x : w) {
			sum += x;
		}
		double[] p = new double[w.length];
		for (int i = 0; i < w.length; i++) {
			p[i] = w[i] / sum;
		}
		return p;
	}

	private static PowerOffsetTransform power(double exponent, double offset) {
		PowerOffsetTransform t = new PowerOffsetTransform();
		t.initByName("exponent", exponent, "offset", offset, "optimise", false);
		return t;
	}

	/**
	 * With offset 0 the power transform must reproduce the hardcoded
	 * Math.pow(w, 2) at TargetedWilsonBalding.java:252 exactly.
	 */
	@Test
	public void exponentTwoWithoutOffsetReproducesSquaring() {
		double[] raw = { 0.4, 1.7, 0.05, 3.0 };
		double[] got = power(2.0, 0.0).apply(raw);

		Assert.assertEquals(raw.length, got.length);
		for (int k = 0; k < raw.length; k++) {
			Assert.assertEquals(raw[k] * raw[k], got[k], TOL);
		}
	}

	/**
	 * Normalizing a * mean(u) + u must equal the uniform/targeted mixture
	 * (a/(1+a)) * uniform + (1/(1+a)) * normalize(u).
	 */
	@Test
	public void offsetActsAsUniformMixtureProportion() {
		double[] raw = { 0.4, 1.7, 0.05, 3.0 };
		double a = 0.35;
		int k = raw.length;

		double[] u = power(1.0, 0.0).apply(raw);
		double[] expected = new double[k];
		double[] normU = normalize(u);
		for (int i = 0; i < k; i++) {
			expected[i] = (a / (1.0 + a)) * (1.0 / k) + (1.0 / (1.0 + a)) * normU[i];
		}

		double[] got = normalize(power(1.0, a).apply(raw));
		Assert.assertArrayEquals(expected, got, TOL);
	}

	/** apply() must not modify its argument. */
	@Test
	public void applyDoesNotMutateInput() {
		double[] raw = { 0.4, 1.7, 0.05, 3.0 };
		double[] copy = raw.clone();
		power(2.5, 0.1).apply(raw);
		Assert.assertArrayEquals(copy, raw, 0.0);
	}

	/** Coefficients are readable in natural space, in the fixed order {b, a}. */
	@Test
	public void coefficientsAreReportedInNaturalSpace() {
		double[] c = power(3.5, 0.25).getCoefficients();
		Assert.assertEquals(3.5, c[0], TOL);
		Assert.assertEquals(0.25, c[1], TOL);
	}

	/** Configured coefficients outside the declared bounds are clamped, not accepted. */
	@Test
	public void coefficientsAreClampedToBounds() {
		double[] c = power(1000.0, 0.0).getCoefficients();
		Assert.assertEquals(20.0, c[0], TOL);
		Assert.assertEquals(1e-6, c[1], TOL);
	}

	/** The real-space transform must reject being paired with log-space weights. */
	@Test
	public void powerTransformDeclaresRealSpace() {
		Assert.assertFalse(power(2.0, 0.0).requiresLogSpace());
	}
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `ant test-single -Dtest.class=targetedbeast.weighttransform.WeightTransformTest`

Expected: compilation failure — `package targetedbeast.weighttransform does not exist`.

- [ ] **Step 3: Write `WeightTransform`**

Create `src/targetedbeast/weighttransform/WeightTransform.java`:

```java
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

	/** Set the coefficients from natural-space values, clamping each to its bounds. */
	public void setCoefficients(double[] natural) {
		for (int i = 0; i < theta.length; i++) {
			theta[i] = clamp(Math.log(natural[i]), i);
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
```

- [ ] **Step 4: Write `PowerOffsetTransform`**

Create `src/targetedbeast/weighttransform/PowerOffsetTransform.java`:

```java
package targetedbeast.weighttransform;

import beast.base.core.Description;

@Description("Weight transform for real-space weights: u_k = w_k^b, then the mean-relative offset. "
		+ "With exponent=2 and offset=0 this reproduces the squaring that TargetedWilsonBalding "
		+ "applies today. For EdgeWeights that already return a softmax, normalising w^b is exactly "
		+ "temperature scaling, because normalize(softmax(x)^b) == softmax(b*x).")
public class PowerOffsetTransform extends WeightTransform {

	@Override
	protected double[] pointiness(double[] raw, double b) {
		double[] u = new double[raw.length];
		for (int k = 0; k < raw.length; k++) {
			// Guard against negative weights, which would make Math.pow return NaN for
			// fractional exponents. No EdgeWeights implementation should produce them.
			u[k] = raw[k] <= 0.0 ? 0.0 : Math.pow(raw[k], b);
		}
		return u;
	}

	@Override
	public boolean requiresLogSpace() {
		return false;
	}
}
```

- [ ] **Step 5: Run the test and verify it passes**

Run: `ant test-single -Dtest.class=targetedbeast.weighttransform.WeightTransformTest`

Expected: all 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/targetedbeast/weighttransform test/targetedbeast/weighttransform
git commit -m "feat: add WeightTransform base and PowerOffsetTransform"
```

---

## Task 2: `ExpOffsetTransform`

**Files:**
- Create: `src/targetedbeast/weighttransform/ExpOffsetTransform.java`
- Modify: `test/targetedbeast/weighttransform/WeightTransformTest.java` (append tests)

**Interfaces:**
- Consumes: `WeightTransform.pointiness(double[], double)`, `WeightTransform.apply(double[])` from Task 1.
- Produces: `public class ExpOffsetTransform extends WeightTransform`, `requiresLogSpace()` returning `true`.

- [ ] **Step 1: Write the failing tests**

Append to `test/targetedbeast/weighttransform/WeightTransformTest.java`, inside the class:

```java
	private static ExpOffsetTransform exp(double exponent, double offset) {
		ExpOffsetTransform t = new ExpOffsetTransform();
		t.initByName("exponent", exponent, "offset", offset, "optimise", false);
		return t;
	}

	private static double[] softmax(double[] x) {
		double max = Double.NEGATIVE_INFINITY;
		for (double v : x) {
			max = Math.max(max, v);
		}
		double[] e = new double[x.length];
		double sum = 0.0;
		for (int i = 0; i < x.length; i++) {
			e[i] = Math.exp(x[i] - max);
			sum += e[i];
		}
		for (int i = 0; i < x.length; i++) {
			e[i] /= sum;
		}
		return e;
	}

	/**
	 * Log-space weights are only defined up to an additive constant, because every consumer
	 * normalizes. The transform must therefore be invariant to shifting them.
	 */
	@Test
	public void expTransformIsInvariantToShiftingLogWeights() {
		double[] logRaw = { -3.2, 0.0, -1.1, -7.4 };
		double[] shifted = new double[logRaw.length];
		for (int i = 0; i < logRaw.length; i++) {
			shifted[i] = logRaw[i] + 12.7;
		}

		double[] a = normalize(exp(1.6, 0.3).apply(logRaw));
		double[] b = normalize(exp(1.6, 0.3).apply(shifted));
		Assert.assertArrayEquals(a, b, TOL);
	}

	/**
	 * The power form on softmax output is exactly temperature scaling:
	 * normalize(softmax(x)^b) == softmax(b*x).
	 */
	@Test
	public void powerOnSoftmaxOutputIsTemperatureScaling() {
		double[] x = { -3.2, 0.0, -1.1, -7.4 };
		double b = 2.4;

		double[] scaled = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			scaled[i] = b * x[i];
		}
		double[] expected = softmax(scaled);

		double[] got = normalize(power(b, 0.0).apply(softmax(x)));
		Assert.assertArrayEquals(expected, got, 1e-10);
	}

	/** A log-weight of negative infinity means an impossible candidate and must stay at the offset. */
	@Test
	public void expTransformHandlesNegativeInfinity() {
		double[] logRaw = { -1.0, Double.NEGATIVE_INFINITY, 0.0 };
		double[] got = exp(1.0, 0.5).apply(logRaw);

		for (double v : got) {
			Assert.assertTrue("expected finite, got " + v, Double.isFinite(v));
		}
		// The impossible candidate carries only the offset, so it is the smallest entry.
		Assert.assertTrue(got[1] < got[0]);
		Assert.assertTrue(got[1] < got[2]);
	}

	/** An all-impossible candidate set must degrade to uniform rather than to NaN. */
	@Test
	public void expTransformOnAllNegativeInfinityIsUniform() {
		double[] logRaw = { Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY };
		double[] got = normalize(exp(1.0, 0.01).apply(logRaw));
		Assert.assertArrayEquals(new double[] { 0.5, 0.5 }, got, TOL);
	}

	@Test
	public void expTransformDeclaresLogSpace() {
		Assert.assertTrue(exp(1.0, 0.0).requiresLogSpace());
	}
```

- [ ] **Step 2: Run and verify failure**

Run: `ant test-single -Dtest.class=targetedbeast.weighttransform.WeightTransformTest`

Expected: compilation failure — `cannot find symbol: class ExpOffsetTransform`.

- [ ] **Step 3: Write `ExpOffsetTransform`**

Create `src/targetedbeast/weighttransform/ExpOffsetTransform.java`:

```java
package targetedbeast.weighttransform;

import beast.base.core.Description;

@Description("Weight transform for log-space weights: u_k = exp(b * (w_k - max w)), then the "
		+ "mean-relative offset. The max subtraction is for numerical stability only and cancels "
		+ "in the normalization. Use with an EdgeWeights whose isLogSpace() returns true.")
public class ExpOffsetTransform extends WeightTransform {

	@Override
	protected double[] pointiness(double[] raw, double b) {
		double max = Double.NEGATIVE_INFINITY;
		for (double w : raw) {
			max = Math.max(max, w);
		}

		double[] u = new double[raw.length];
		if (!Double.isFinite(max)) {
			// Every candidate is impossible (or the weights are corrupt). Fall back to a
			// uniform vector so the caller normalizes to uniform instead of dividing by zero.
			java.util.Arrays.fill(u, 1.0);
			return u;
		}

		for (int k = 0; k < raw.length; k++) {
			u[k] = Math.exp(b * (raw[k] - max));
		}
		return u;
	}

	@Override
	public boolean requiresLogSpace() {
		return true;
	}
}
```

- [ ] **Step 4: Run and verify pass**

Run: `ant test-single -Dtest.class=targetedbeast.weighttransform.WeightTransformTest`

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/targetedbeast/weighttransform/ExpOffsetTransform.java test/targetedbeast/weighttransform/WeightTransformTest.java
git commit -m "feat: add ExpOffsetTransform for log-space edge weights"
```

---

## Task 3: SPSA adaptation

Replaces the no-op `optimize()` from Task 1 with simultaneous-perturbation stochastic approximation. Read the "Adaptation: SPSA" section of the spec before starting — in particular *why* BEAST's `calcDelta` cannot be used here.

**Files:**
- Modify: `src/targetedbeast/weighttransform/WeightTransform.java`
- Test: `test/targetedbeast/weighttransform/SpsaAdaptationTest.java`

**Interfaces:**
- Consumes: `theta`, `effective`, `wasUsed()`, `clamp(double, int)`, `refreshEffective()` from Task 1.
- Produces:
  - `public double[] getEffectiveCoefficients()` — the perturbed values in use for the current proposal.
  - `public long getAdaptationCount()` — the SPSA iteration counter `n`.
  - `public double getPerturbationSize()` — the current `c_n`.
  - `public double getAcceptanceBaseline()` — the current EWMA baseline.
  - `public void setAdaptationState(double[] naturalCoefficients, long n, double baseline)` — used by Task 7's state-file restore.

- [ ] **Step 1: Write the failing test**

Create `test/targetedbeast/weighttransform/SpsaAdaptationTest.java`:

```java
package targetedbeast.weighttransform;

import org.junit.Assert;
import org.junit.Test;

import beast.base.util.Randomizer;

/**
 * SPSA adaptation against a synthetic acceptance surface with a known optimum.
 *
 * <p>The transform never sees the surface directly: it only ever receives one noisy
 * logAlpha per proposal, exactly as it would from MCMC.optimize().
 */
public class SpsaAdaptationTest {

	/** Peak acceptance at exponent = 6.0, smooth and unimodal in log space. */
	private static double acceptanceFor(double exponent) {
		double d = Math.log(exponent) - Math.log(6.0);
		return 0.8 * Math.exp(-(d * d) / (2 * 0.5 * 0.5));
	}

	/** Drive n proposals through the transform against the synthetic surface. */
	private static void drive(WeightTransform t, int iterations) {
		double[] dummyWeights = { 1.0, 2.0, 3.0 };
		for (int i = 0; i < iterations; i++) {
			t.startProposal();
			t.apply(dummyWeights);
			double alpha = acceptanceFor(t.getEffectiveCoefficients()[0]);
			t.optimize(Math.log(alpha));
		}
	}

	private static PowerOffsetTransform transform(boolean optimise) {
		PowerOffsetTransform t = new PowerOffsetTransform();
		t.initByName("exponent", 2.0, "offset", 0.01, "optimise", optimise);
		return t;
	}

	@Test
	public void adaptationClimbsToTheOptimum() {
		Randomizer.setSeed(42);
		PowerOffsetTransform t = transform(true);

		Assert.assertEquals(2.0, t.getCoefficients()[0], 1e-12);
		drive(t, 50_000);

		// Started at 2.0, optimum is 6.0. Loose tolerance: this is a stochastic estimator.
		Assert.assertEquals(6.0, t.getCoefficients()[0], 1.5);
	}

	/** Diminishing adaptation: the perturbation must shrink toward zero. */
	@Test
	public void perturbationDecays() {
		Randomizer.setSeed(43);
		PowerOffsetTransform t = transform(true);

		double initial = t.getPerturbationSize();
		drive(t, 50_000);
		double later = t.getPerturbationSize();

		Assert.assertTrue("perturbation should shrink, was " + initial + " now " + later,
				later < initial);
		Assert.assertEquals(50_000, t.getAdaptationCount());
	}

	/** With optimise=false the kernel must be completely fixed - required by the DB tests. */
	@Test
	public void optimiseFalseFreezesTheKernel() {
		Randomizer.setSeed(44);
		PowerOffsetTransform t = transform(false);

		drive(t, 5_000);

		Assert.assertEquals(2.0, t.getCoefficients()[0], 1e-12);
		Assert.assertEquals(0.01, t.getCoefficients()[1], 1e-12);
		Assert.assertEquals(2.0, t.getEffectiveCoefficients()[0], 1e-12);
		Assert.assertEquals(0, t.getAdaptationCount());
	}

	/**
	 * A proposal that aborted before touching the weights carries no information about the
	 * coefficients and must not move them.
	 */
	@Test
	public void abortedProposalsAreIgnored() {
		Randomizer.setSeed(45);
		PowerOffsetTransform t = transform(true);

		for (int i = 0; i < 1_000; i++) {
			t.startProposal();
			// no apply() - the operator returned NEGATIVE_INFINITY early
			t.optimize(Double.NEGATIVE_INFINITY);
		}

		Assert.assertEquals(2.0, t.getCoefficients()[0], 1e-12);
		Assert.assertEquals(0, t.getAdaptationCount());
	}

	/** logAlpha of negative infinity is a real rejection and must count as acceptance 0. */
	@Test
	public void negativeInfinityLogAlphaIsFinite() {
		Randomizer.setSeed(46);
		PowerOffsetTransform t = transform(true);

		t.startProposal();
		t.apply(new double[] { 1.0, 2.0 });
		t.optimize(Double.NEGATIVE_INFINITY);

		for (double c : t.getCoefficients()) {
			Assert.assertTrue("coefficient went non-finite: " + c, Double.isFinite(c));
		}
		Assert.assertEquals(1, t.getAdaptationCount());
	}
}
```

- [ ] **Step 2: Run and verify failure**

Run: `ant test-single -Dtest.class=targetedbeast.weighttransform.SpsaAdaptationTest`

Expected: compilation failure — `cannot find symbol: method getEffectiveCoefficients()`.

- [ ] **Step 3: Add the SPSA inputs and state fields**

In `src/targetedbeast/weighttransform/WeightTransform.java`, add after `optimiseInput`:

```java
	final public Input<Double> stepSizeInput = new Input<>("stepSize",
			"SPSA gain numerator A; the step taken per proposal is A/(n+n0)^0.602 divided by the "
					+ "current perturbation", 0.05);
	final public Input<Double> perturbationInput = new Input<>("perturbation",
			"SPSA perturbation numerator C; coefficients are jittered by C/(n+n0)^0.101 in log space "
					+ "each proposal", 0.2);
	final public Input<Integer> stabilityConstantInput = new Input<>("stabilityConstant",
			"SPSA stability constant n0; larger values make the early adaptation gentler", 100);
	final public Input<Double> baselineRateInput = new Input<>("baselineRate",
			"EWMA rate for the running acceptance baseline used as an SPSA control variate", 0.01);
```

and add after the `used` field:

```java
	/** Rademacher perturbation directions drawn for the current proposal. */
	private double[] signs;

	/** Running mean acceptance, used as a control variate in the SPSA update. */
	private double baseline;

	/** Number of completed adaptation steps. */
	private long adaptationCount;
```

- [ ] **Step 4: Initialise the new state**

Replace the body of `initAndValidate()`:

```java
	@Override
	public void initAndValidate() {
		theta = new double[COEFFICIENT_NAMES.length];
		effective = new double[COEFFICIENT_NAMES.length];
		signs = new double[COEFFICIENT_NAMES.length];
		baseline = 0.0;
		adaptationCount = 0;
		setCoefficients(new double[] { exponentInput.get(), offsetInput.get() });
	}
```

- [ ] **Step 5: Implement perturbation and the update**

Replace `startProposal()` and `optimize(double)`:

```java
	/**
	 * Called by the operator at the start of every proposal, before any call to apply().
	 *
	 * <p>Draws the SPSA perturbation and freezes the resulting coefficients for the whole
	 * proposal. Freezing is what keeps the forward and reverse probabilities of a single
	 * proposal mutually consistent, so the Hastings ratio stays correct.
	 */
	public void startProposal() {
		used = false;
		if (!optimiseInput.get()) {
			return;
		}
		double c = getPerturbationSize();
		for (int i = 0; i < theta.length; i++) {
			signs[i] = Randomizer.nextBoolean() ? 1.0 : -1.0;
			effective[i] = Math.exp(clamp(theta[i] + c * signs[i], i));
		}
	}

	/**
	 * One SPSA step toward higher acceptance.
	 *
	 * <p>Because the perturbation directions are symmetric, the expectation of
	 * {@code (alpha - baseline) * signs} is proportional to the gradient of the expected
	 * acceptance with respect to the log coefficients. The baseline is a control variate:
	 * without it the update is dominated by the magnitude of alpha rather than by its
	 * dependence on the coefficients.
	 *
	 * @param logAlpha difference in posterior between previous and proposed state, plus the
	 *                 Hastings ratio
	 */
	public void optimize(double logAlpha) {
		// A proposal that aborted before touching the weights says nothing about the
		// coefficients; counting it would inject pure noise.
		if (!optimiseInput.get() || !wasUsed()) {
			return;
		}
		double alpha = Math.exp(Math.min(logAlpha, 0.0));
		if (Double.isNaN(alpha)) {
			return;
		}

		double gain = getStepSize() / getPerturbationSize();
		for (int i = 0; i < theta.length; i++) {
			theta[i] = clamp(theta[i] + gain * (alpha - baseline) * signs[i], i);
		}
		baseline += baselineRateInput.get() * (alpha - baseline);
		adaptationCount++;
		refreshEffective();
	}

	/** Current SPSA step gain a_n. */
	protected double getStepSize() {
		return stepSizeInput.get() / Math.pow(adaptationCount + stabilityConstantInput.get(), 0.602);
	}

	/**
	 * Current SPSA perturbation size c_n.
	 *
	 * <p>Both a_n/c_n and c_n tend to zero, which is what gives diminishing adaptation.
	 */
	public double getPerturbationSize() {
		return perturbationInput.get() / Math.pow(adaptationCount + stabilityConstantInput.get(), 0.101);
	}

	/** Perturbed coefficients in use for the current proposal, in natural space. */
	public double[] getEffectiveCoefficients() {
		return effective.clone();
	}

	/** Number of completed adaptation steps. */
	public long getAdaptationCount() {
		return adaptationCount;
	}

	/** Running mean acceptance used as the SPSA control variate. */
	public double getAcceptanceBaseline() {
		return baseline;
	}

	/** Restore a previously persisted adaptation state (see the operator's state file). */
	public void setAdaptationState(double[] naturalCoefficients, long n, double acceptanceBaseline) {
		setCoefficients(naturalCoefficients);
		this.adaptationCount = n;
		this.baseline = acceptanceBaseline;
	}
```

Add the import `beast.base.util.Randomizer`.

- [ ] **Step 6: Run and verify pass**

Run: `ant test-single -Dtest.class=targetedbeast.weighttransform.SpsaAdaptationTest`

Expected: all 6 tests PASS. If `adaptationClimbsToTheOptimum` misses by a little, do **not** widen the tolerance without first checking that the sign of the update is right — a sign error still converges to *something*, just not the optimum. Verify by asserting the final exponent is above 4.0 and below 8.0 with a couple of different seeds.

- [ ] **Step 7: Verify Task 1's tests still pass**

Run: `ant test-single -Dtest.class=targetedbeast.weighttransform.WeightTransformTest`

Expected: all 11 PASS. `optimise=false` in those fixtures means the perturbation path is skipped entirely.

- [ ] **Step 8: Commit**

```bash
git add src/targetedbeast/weighttransform/WeightTransform.java test/targetedbeast/weighttransform/SpsaAdaptationTest.java
git commit -m "feat: adapt WeightTransform coefficients by SPSA on acceptance"
```

---

## Task 4: `EdgeWeights.isLogSpace()` and the `PartialWeights` log-space mode

`isLogSpace()` describes the **target weights only**. Every implementation returns a
real-space scalar from `getEdgeWeights` (an edge-mutation count or a branch length); this
task makes that contract explicit in the javadoc.

**Files:**
- Modify: `src/targetedbeast/edgeweights/EdgeWeights.java`
- Modify: `src/targetedbeast/edgeweights/PartialWeights.java:544-546` and `:640-642`
- Test: `test/targetedbeast/weighttransform/PartialWeightsLogSpaceTest.java`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `EdgeWeights.isLogSpace()` returning `false` by default; `PartialWeights` `logSpace` input.

- [ ] **Step 1: Write the failing test**

Create `test/targetedbeast/weighttransform/PartialWeightsLogSpaceTest.java`:

```java
package targetedbeast.weighttransform;

import org.junit.Assert;
import org.junit.Test;

import targetedbeast.edgeweights.ConstantWeights;
import targetedbeast.edgeweights.PartialWeights;

/** The logSpace switch on PartialWeights, and the isLogSpace default on EdgeWeights. */
public class PartialWeightsLogSpaceTest {

	@Test
	public void edgeWeightsDefaultToRealSpace() {
		ConstantWeights w = new ConstantWeights();
		Assert.assertFalse(w.isLogSpace());
	}

	@Test
	public void partialWeightsDefaultToRealSpace() {
		PartialWeights w = new PartialWeights();
		// Reading the input directly - initAndValidate needs a full alignment and tree.
		Assert.assertFalse(w.logSpaceInput.get());
	}

	@Test
	public void partialWeightsReportLogSpaceWhenEnabled() {
		PartialWeights w = new PartialWeights();
		w.logSpaceInput.setValue(true, w);
		Assert.assertTrue(w.isLogSpace());
	}
}
```

- [ ] **Step 2: Run and verify failure**

Run: `ant test-single -Dtest.class=targetedbeast.weighttransform.PartialWeightsLogSpaceTest`

Expected: compilation failure — `cannot find symbol: method isLogSpace()`.

- [ ] **Step 3: Add `isLogSpace()` to the interface**

In `src/targetedbeast/edgeweights/EdgeWeights.java`, add before `double minEdgeWeight();`:

```java
	/**
	 * Whether {@link #getTargetWeights} and {@link #getTargetWeightsInteger} return weights
	 * in log space.
	 *
	 * <p>This describes the <em>target</em> weights only. {@link #getEdgeWeights} returns a
	 * real-space, non-negative scalar in every implementation and must continue to do so.
	 *
	 * <p>A {@code targetWeightTransform} on an operator must agree with this flag: a
	 * transform written for real-space input applied to log-space weights, or the reverse,
	 * silently produces a wrong proposal distribution rather than an error.
	 */
	default boolean isLogSpace() {
		return false;
	}
```

- [ ] **Step 4: Add the `logSpace` input to `PartialWeights`**

In `src/targetedbeast/edgeweights/PartialWeights.java`, add after `minWeightInput`:

```java
	final public Input<Boolean> logSpaceInput = new Input<>("logSpace",
			"if true, the target weights are returned as raw log-weights instead of being passed "
					+ "through a softmax, so that a log-space WeightTransform can do the conversion "
					+ "to real space with an adaptable temperature", false);
```

and add near `minEdgeWeight()`:

```java
	@Override
	public boolean isLogSpace() {
		return logSpaceInput.get();
	}
```

- [ ] **Step 5: Make the two target-weight methods honour it**

In `getTargetWeightsInteger(int, List<Integer>)`, replace:

```java
        // // Normalise probabilities
        weights = clipMaxToSecondMaxDiff(weights, 2.);
        double[] p = softmax(weights);
        return p;
```

with:

```java
        weights = clipMaxToSecondMaxDiff(weights, 2.);
        // In log-space mode a WeightTransform performs the exponentiation instead, with an
        // adaptable temperature; normalize(softmax(x)^b) == softmax(b*x), so the two modes
        // differ only in who chooses b.
        return logSpaceInput.get() ? weights : softmax(weights);
```

Apply the identical replacement in `getTargetWeightsInteger(int, List<Integer>, double newHeight)`.

- [ ] **Step 6: Run and verify pass**

Run: `ant test-single -Dtest.class=targetedbeast.weighttransform.PartialWeightsLogSpaceTest`

Expected: 3 tests PASS.

- [ ] **Step 7: Verify no existing behaviour changed**

Run: `ant test-single -Dtest.class=targetedbeast.TargetedOperatorDetailedBalanceTest`

Expected: identical pass/fail set to the baseline you recorded. `logSpace` defaults to
`false`, so `testTargWilsonBaldingPartial` must behave exactly as before.

- [ ] **Step 8: Commit**

```bash
git add src/targetedbeast/edgeweights/EdgeWeights.java \
        src/targetedbeast/edgeweights/PartialWeights.java \
        test/targetedbeast/weighttransform/PartialWeightsLogSpaceTest.java
git commit -m "feat: add isLogSpace to EdgeWeights and a logSpace mode to PartialWeights"
```

---

## Task 5: Target-path integration in `TargetedWilsonBalding`

The riskiest task in the plan. `TargetedWilsonBalding`'s Hastings ratio was recently repaired
and is delicate; the detailed-balance suite is the only thing that will catch a mistake here.

**Files:**
- Modify: `src/targetedbeast/operators/TargetedWilsonBalding.java`
- Test: `test/targetedbeast/TargetedOperatorDetailedBalanceTest.java`

**Interfaces:**
- Consumes: `WeightTransform.startProposal()`, `apply(double[])`, `optimize(double)`, `requiresLogSpace()` from Tasks 1–3; `EdgeWeights.isLogSpace()` from Task 4.
- Produces: `targetWeightTransformInput`, `edgeWeightTransformInput` on the operator; `applyTargetTransform(double[])` helper.

- [ ] **Step 1: Record the baseline**

```bash
ant test-single -Dtest.class=targetedbeast.TargetedOperatorDetailedBalanceTest 2>&1 | tee /tmp/db-baseline.txt
grep -c FAILED /tmp/db-baseline.txt
```

Write down which test names failed. You will compare against this exact set, not against
the list in this plan.

- [ ] **Step 2: Write the failing test**

Append to `test/targetedbeast/TargetedOperatorDetailedBalanceTest.java`:

```java
    /**
     * Detailed balance with a target-weight transform whose coefficients are deliberately
     * far from the defaults. The transform must be applied identically in the forward and
     * reverse blocks; if it is not, the forward and backward acceptance flows diverge.
     *
     * <p>optimise="false" is required: the harness accumulates flow across samples, which is
     * only meaningful for a fixed proposal kernel.
     */
    @Test
    public void testTargWilsonBaldingTargetTransform() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 3.7, "offset", 0.3, "optimise", false);

            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getPCAWeights(tree),
                    "useEdgeLength", false,
                    "targetWeightTransform", transform);
            return operator;
        }));
    }

    /**
     * The same, with an offset large enough that the proposal is close to uniform over the
     * candidate set. Catches normalization errors that a small offset would mask.
     */
    @Test
    public void testTargWilsonBaldingTargetTransformHeavyOffset() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 1.0, "offset", 8.0, "optimise", false);

            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getPCAWeights(tree),
                    "useEdgeLength", false,
                    "targetWeightTransform", transform);
            return operator;
        }));
    }
```

- [ ] **Step 3: Run and verify failure**

Run: `ant test-single -Dtest.class=targetedbeast.TargetedOperatorDetailedBalanceTest`

Expected: compilation failure — `targetWeightTransform` is not an input of `TargetedWilsonBalding`.

- [ ] **Step 4: Add the inputs and validation**

In `src/targetedbeast/operators/TargetedWilsonBalding.java`, add after `edgeWeightsInput`:

```java
	public Input<WeightTransform> targetWeightTransformInput = new Input<>("targetWeightTransform",
			"optional transform applied to the target weights before they are normalized; "
					+ "when absent the target weights are squared, as they always have been");

	public Input<WeightTransform> edgeWeightTransformInput = new Input<>("edgeWeightTransform",
			"optional transform applied to the source-edge weights before they are normalized");
```

add the fields next to `EdgeWeights edgeWeights;`:

```java
	WeightTransform targetWeightTransform;
	WeightTransform edgeWeightTransform;
```

and extend `initAndValidate()`:

```java
		targetWeightTransform = targetWeightTransformInput.get();
		edgeWeightTransform = edgeWeightTransformInput.get();

		if (targetWeightTransform != null
				&& targetWeightTransform.requiresLogSpace() != edgeWeights.isLogSpace()) {
			throw new IllegalArgumentException("targetWeightTransform expects "
					+ (targetWeightTransform.requiresLogSpace() ? "log" : "real")
					+ "-space weights but " + edgeWeights.getClass().getSimpleName()
					+ " returns " + (edgeWeights.isLogSpace() ? "log" : "real") + "-space weights");
		}
		// getEdgeWeights is real-space by contract in every implementation, and with
		// useEdgeLength=true the source weights are branch lengths rather than EdgeWeights
		// output, so a log-space transform is never correct here.
		if (edgeWeightTransform != null && edgeWeightTransform.requiresLogSpace()) {
			throw new IllegalArgumentException(
					"edgeWeightTransform must be a real-space transform; source edge weights are "
							+ "never in log space");
		}
```

Add the import `targetedbeast.weighttransform.WeightTransform`.

- [ ] **Step 5: Add the transform helper and call `startProposal`**

Add a private helper to the operator:

```java
	/**
	 * Reshape a target-weight vector.
	 *
	 * <p>Without a transform this is the squaring that has always been applied. With a
	 * transform, the transform's own exponent replaces it, so the two must never both run.
	 *
	 * <p>Every entry must be a genuinely selectable candidate: the transform's offset is
	 * added to all entries, so an infeasible candidate passed in as zero would acquire
	 * positive selection probability.
	 */
	private double[] applyTargetTransform(double[] rawWeights) {
		if (targetWeightTransform == null) {
			double[] out = new double[rawWeights.length];
			for (int k = 0; k < rawWeights.length; k++) {
				out[k] = rawWeights[k] * rawWeights[k];
			}
			return out;
		}
		return targetWeightTransform.apply(rawWeights);
	}
```

At the very top of `proposal()`, before anything else:

```java
	@Override
	public double proposal() {
		// Freeze the transform coefficients for this proposal, so the forward and reverse
		// probabilities below are computed with identical weights.
		if (targetWeightTransform != null) {
			targetWeightTransform.startProposal();
		}
		if (edgeWeightTransform != null) {
			edgeWeightTransform.startProposal();
		}

		boolean useEdgeLength = useEdgeLengthInput.get();
		...
```

- [ ] **Step 6: Rewrite the forward target block**

Replace `TargetedWilsonBalding.java:249-255`:

```java
		double[] distance = edgeWeights.getTargetWeights(i.getNr(), candidateNodes, candidateAges);
		double[] forwardMass = new double[distance.length];
        for (int k=0; k<distance.length; k++) {
			distance[k] = Math.pow(distance[k], 2);
			double intervalWeight = getAttachmentSelectionWeight(candidateIntervals.get(k), useAttachmentIntervalLength);
			forwardMass[k] = distance[k] * intervalWeight;
        }
```

with:

```java
		double[] rawTargetWeights = edgeWeights.getTargetWeights(i.getNr(), candidateNodes, candidateAges);
		double[] distance = applyTargetTransform(rawTargetWeights);
		double[] forwardMass = new double[distance.length];
		for (int k = 0; k < distance.length; k++) {
			double intervalWeight = getAttachmentSelectionWeight(candidateIntervals.get(k), useAttachmentIntervalLength);
			forwardMass[k] = distance[k] * intervalWeight;
		}
```

- [ ] **Step 7: Rewrite the reverse target block identically**

Replace `TargetedWilsonBalding.java:383-389`:

```java
		double[] reverseDistance = edgeWeights.getTargetWeights(i.getNr(), reverseCandidateNodes, reverseCandidateAges);
		double[] reverseMass = new double[reverseDistance.length];
		for (int k = 0; k < reverseDistance.length; k++) {
			reverseDistance[k] = Math.pow(reverseDistance[k], 2);
			double intervalWeight = getAttachmentSelectionWeight(reverseCandidateIntervals.get(k), useAttachmentIntervalLength);
			reverseMass[k] = reverseDistance[k] * intervalWeight;
		}
```

with:

```java
		double[] rawReverseWeights = edgeWeights.getTargetWeights(i.getNr(), reverseCandidateNodes, reverseCandidateAges);
		double[] reverseDistance = applyTargetTransform(rawReverseWeights);
		double[] reverseMass = new double[reverseDistance.length];
		for (int k = 0; k < reverseDistance.length; k++) {
			double intervalWeight = getAttachmentSelectionWeight(reverseCandidateIntervals.get(k), useAttachmentIntervalLength);
			reverseMass[k] = reverseDistance[k] * intervalWeight;
		}
```

The two blocks are now structurally identical: transform the full candidate vector, multiply
by the interval weight, and only then zero the excluded element (which the existing code
below already does, via `selectionMass` and `reverseSelectionMass`). Do not move the zeroing.

- [ ] **Step 8: Add `optimize()`**

Add to the operator:

```java
	@Override
	public void optimize(final double logAlpha) {
		if (targetWeightTransform != null) {
			targetWeightTransform.optimize(logAlpha);
		}
		// Guard against the same object being wired to both inputs, which would otherwise
		// take two adaptation steps from one observation.
		if (edgeWeightTransform != null && edgeWeightTransform != targetWeightTransform) {
			edgeWeightTransform.optimize(logAlpha);
		}
	}
```

- [ ] **Step 9: Clean rebuild and run the detailed-balance suite**

```bash
rm -rf build build-test && ant compile compile-test
ant test-single -Dtest.class=targetedbeast.TargetedOperatorDetailedBalanceTest 2>&1 | tee /tmp/db-after.txt
```

Expected: the same failures as `/tmp/db-baseline.txt`, plus **zero** failures among
`testTargWilsonBaldingTargetTransform` and `testTargWilsonBaldingTargetTransformHeavyOffset`.

If either new test fails, re-run at higher power before concluding anything — this suite is
stochastic and a single failure at 3.2σ is not evidence:

```bash
ant test-single -Dtest.class=targetedbeast.TargetedOperatorDetailedBalanceTest -Ddb.numSamples=360000
```

A genuine bias grows as `diff/tol ∝ √N`; noise does not. If `nsigma` roughly triples when
you raise the sample count 12-fold, it is a real bug. If it stays flat, it was noise.

- [ ] **Step 10: Commit**

```bash
git add src/targetedbeast/operators/TargetedWilsonBalding.java \
        test/targetedbeast/TargetedOperatorDetailedBalanceTest.java
git commit -m "feat: apply an optional WeightTransform to TargetedWilsonBalding target weights"
```

---

## Task 6: Source-path integration in `TargetedWilsonBalding`

`selectSourceNode` currently accumulates weights on the fly, which a transform cannot hook
into — the mean-relative offset needs the whole vector before any entry is final. This task
restructures it into build → transform → sample, and mirrors that in the reverse block.

**Root masking is the correctness trap.** `getSourceWeight` returns `0.0` for the root. If
the root were handed to the transform as a zero, the offset would give it positive weight,
`selectSourceNode` could return the root, and `p = i.getParent()` would dereference null. The
root must be *excluded from the vector*, not zeroed.

**Files:**
- Modify: `src/targetedbeast/operators/TargetedWilsonBalding.java:155-183` and `:418-440`
- Test: `test/targetedbeast/TargetedOperatorDetailedBalanceTest.java`

**Interfaces:**
- Consumes: `edgeWeightTransform` field and `startProposal()` wiring from Task 5.
- Produces: private `SourceWeights` record and `getSourceWeights(Tree, boolean)` on the operator.

- [ ] **Step 1: Write the failing test**

Append to `test/targetedbeast/TargetedOperatorDetailedBalanceTest.java`:

```java
    /**
     * Detailed balance for the source-selection path under a transform.
     *
     * <p>StructuralSourceWeights gives uniform target weights (so the target term cancels)
     * and strongly depth-dependent, uncached source weights, so a mistake in the reverse
     * source term cannot be blamed on stale caches.
     */
    @Test
    public void testTargWilsonBaldingSourceTransform() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.edgeweights.StructuralSourceWeights edgeWeights =
                    new targetedbeast.edgeweights.StructuralSourceWeights();
            edgeWeights.initByName("tree", tree);

            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 0.6, "offset", 0.4, "optimise", false);

            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", edgeWeights,
                    "useEdgeLength", false,
                    "edgeWeightTransform", transform);
            return operator;
        }));
    }

    /**
     * The source transform with useEdgeLength=true. This path used to update the reverse
     * total incrementally, which is only valid when the total is a plain sum; a transform
     * makes it nonlinear, so the whole vector must be recomputed.
     */
    @Test
    public void testTargWilsonBaldingSourceTransformEdgeLength() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            transform.initByName("exponent", 2.5, "offset", 0.2, "optimise", false);

            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getPCAWeights(tree),
                    "useEdgeLength", true,
                    "edgeWeightTransform", transform);
            return operator;
        }));
    }

    /** The transform must never make the root selectable as a source edge. */
    @Test
    public void testTargWilsonBaldingSourceTransformNeverSelectsRoot() throws Exception {
        testDetailedBalance(new TreeTrial(tree -> {
            targetedbeast.weighttransform.PowerOffsetTransform transform =
                    new targetedbeast.weighttransform.PowerOffsetTransform();
            // A huge offset makes the proposal nearly uniform over source edges, which is
            // when a wrongly-included root would be picked most often.
            transform.initByName("exponent", 1.0, "offset", 50.0, "optimise", false);

            TreeOperator operator = new TargetedWilsonBalding();
            operator.initByName(
                    "tree", tree,
                    "weight", 1.0,
                    "edgeWeights", getPCAWeights(tree),
                    "useEdgeLength", false,
                    "edgeWeightTransform", transform);
            return operator;
        }));
    }
```

- [ ] **Step 2: Run and verify failure**

Run: `ant test-single -Dtest.class=targetedbeast.TargetedOperatorDetailedBalanceTest`

Expected: `testTargWilsonBaldingSourceTransform*` fail on detailed balance (the transform is
accepted as an input after Task 5, but nothing applies it to the source weights yet), and
`...NeverSelectsRoot` may instead throw `NullPointerException` or
`IllegalStateException: TargetedWilsonBalding has no selectable source edges`.

- [ ] **Step 3: Add the `SourceWeights` holder**

In `src/targetedbeast/operators/TargetedWilsonBalding.java`, add next to `SourceSelection`:

```java
	/**
	 * Transformed source weights over the feasible (non-root) nodes.
	 *
	 * <p>The root is left out of the vector entirely rather than given weight zero: a
	 * transform adds its offset to every entry it receives, so a zero-weight root would
	 * become selectable and break the {@code p = i.getParent()} that follows.
	 */
	private static final class SourceWeights {
		final int[] nodeNrs;
		final double[] weights;
		final double total;

		SourceWeights(int[] nodeNrs, double[] weights) {
			this.nodeNrs = nodeNrs;
			this.weights = weights;
			double sum = 0.0;
			for (double w : weights) {
				sum += w;
			}
			this.total = sum;
		}

		double weightFor(int nodeNr) {
			for (int m = 0; m < nodeNrs.length; m++) {
				if (nodeNrs[m] == nodeNr) {
					return weights[m];
				}
			}
			throw new IllegalArgumentException("node " + nodeNr + " is not a feasible source edge");
		}
	}

	/** Build and transform the source weights of every non-root node, in node-number order. */
	private SourceWeights getSourceWeights(Tree tree, boolean useEdgeLength) {
		int nodeCount = tree.getNodeCount();
		int[] nodeNrs = new int[nodeCount - 1];
		double[] raw = new double[nodeCount - 1];

		int m = 0;
		for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {
			Node node = tree.getNode(nodeNr);
			if (node.isRoot()) {
				continue;
			}
			nodeNrs[m] = nodeNr;
			raw[m] = useEdgeLength ? node.getLength() : edgeWeights.getEdgeWeights(nodeNr);
			m++;
		}

		double[] weights = edgeWeightTransform == null ? raw : edgeWeightTransform.apply(raw);
		return new SourceWeights(nodeNrs, weights);
	}
```

- [ ] **Step 4: Rewrite `selectSourceNode`**

Replace the whole body of `selectSourceNode` (`TargetedWilsonBalding.java:162-183`):

```java
	private SourceSelection selectSourceNode(Tree tree, boolean useEdgeLength) {
		SourceWeights sourceWeights = getSourceWeights(tree, useEdgeLength);
		if (sourceWeights.total < 1E-12) {
			throw new IllegalStateException("TargetedWilsonBalding has no selectable source edges");
		}

		double scaler = Randomizer.nextDouble() * sourceWeights.total;
		double cumulativeWeight = 0.0;
		for (int m = 0; m < sourceWeights.weights.length; m++) {
			cumulativeWeight += sourceWeights.weights[m];
			if (cumulativeWeight > scaler) {
				return new SourceSelection(
						tree.getNode(sourceWeights.nodeNrs[m]),
						sourceWeights.total,
						sourceWeights.weights[m]);
			}
		}

		throw new IllegalStateException("Failed to select source node despite positive total weight");
	}
```

Leave `getSourceWeight(Node, boolean)` in place; the reverse block still uses it in the
shortcut path.

- [ ] **Step 5: Rewrite the reverse source block**

Replace `TargetedWilsonBalding.java:418-436` (the block starting at the comment
`// Reverse source-selection probability, evaluated on the moved tree T'.`):

```java
		// Reverse source-selection probability, evaluated on the moved tree T'.
		double iWeight;
		double reverseTotalWeight;
		if (useEdgeLength && edgeWeightTransform == null) {
			// A Wilson-Balding move only changes the moved subtree's parent edge length, and
			// without a transform the total is a plain sum, so one term can be swapped out.
			iWeight = getSourceWeight(i, useEdgeLength);
			reverseTotalWeight = sourceSelection.totalWeight - sourceSelection.selectedWeight + iWeight;
		} else {
			// Either many edge weights changed, or the transform makes the total a nonlinear
			// function of the whole vector. Recompute it in full.
			if (!useEdgeLength) {
				// Sync the cached weights to T' before reading them.
				edgeWeights.prestore();
				edgeWeights.updateByOperator();
			}
			SourceWeights reverseWeights = getSourceWeights(tree, useEdgeLength);
			iWeight = reverseWeights.weightFor(i.getNr());
			reverseTotalWeight = reverseWeights.total;
		}
		if (reverseTotalWeight < 1E-12) {
			return Double.NEGATIVE_INFINITY;
		}
		logHastingsRatio += Math.log(iWeight / reverseTotalWeight);
```

- [ ] **Step 6: Clean rebuild and run the suite**

```bash
rm -rf build build-test && ant compile compile-test
ant test-single -Dtest.class=targetedbeast.TargetedOperatorDetailedBalanceTest 2>&1 | tee /tmp/db-task6.txt
```

Expected: the baseline failures from `/tmp/db-baseline.txt` and nothing else. All five new
transform tests pass.

- [ ] **Step 7: Confirm the no-transform path is untouched**

This is the constraint a reviewer will check first. With no transform inputs the operator
must behave exactly as it did before Task 5.

```bash
git stash
ant test-single -Dtest.class=targetedbeast.TargetedOperatorDetailedBalanceTest -Ddb.numSamples=200000 2>&1 | grep nsigma > /tmp/nsigma-before.txt
git stash pop
rm -rf build build-test && ant compile compile-test
ant test-single -Dtest.class=targetedbeast.TargetedOperatorDetailedBalanceTest -Ddb.numSamples=200000 2>&1 | grep nsigma > /tmp/nsigma-after.txt
```

The numbers will not match exactly — the tests draw from a shared random stream and the new
test methods consume from it. What must hold is that no *pre-existing* test moved from
passing to failing.

- [ ] **Step 8: Commit**

```bash
git add src/targetedbeast/operators/TargetedWilsonBalding.java \
        test/targetedbeast/TargetedOperatorDetailedBalanceTest.java
git commit -m "feat: apply an optional WeightTransform to TargetedWilsonBalding source weights"
```

---

## Task 7: State-file persistence

Without this, a resumed chain restarts adaptation from the configured values with `n = 0`,
so the gain sequences jump back to maximum and the chain takes a burst of large adaptation
steps mid-run.

**Files:**
- Modify: `src/targetedbeast/operators/TargetedWilsonBalding.java`
- Test: `test/targetedbeast/weighttransform/WeightTransformPersistenceTest.java`

**Interfaces:**
- Consumes: `getCoefficients()`, `getAdaptationCount()`, `getAcceptanceBaseline()`, `setAdaptationState(double[], long, double)` from Task 3.
- Produces: `storeToFile(PrintWriter)` / `restoreFromFile(JSONObject)` overrides on the operator.

- [ ] **Step 1: Write the failing test**

Create `test/targetedbeast/weighttransform/WeightTransformPersistenceTest.java`:

```java
package targetedbeast.weighttransform;

import java.io.PrintWriter;
import java.io.StringWriter;

import org.json.JSONObject;
import org.junit.Assert;
import org.junit.Test;

import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;
import targetedbeast.edgeweights.ConstantWeights;
import targetedbeast.operators.TargetedWilsonBalding;

/** Round-tripping the transform tuning through the operator's state-file entry. */
public class WeightTransformPersistenceTest {

	private static TargetedWilsonBalding operatorWithTransforms(
			PowerOffsetTransform target, PowerOffsetTransform edge) {
		Tree tree = new Tree("((A:1.0,B:1.0):1.0,C:2.0):0.0;");
		ConstantWeights weights = new ConstantWeights();
		weights.initByName("tree", tree);

		TargetedWilsonBalding op = new TargetedWilsonBalding();
		op.initByName("tree", tree, "weight", 1.0, "edgeWeights", weights,
				"targetWeightTransform", target, "edgeWeightTransform", edge);
		op.setID("targetedWB");
		return op;
	}

	private static PowerOffsetTransform transform(double exponent, double offset) {
		PowerOffsetTransform t = new PowerOffsetTransform();
		t.initByName("exponent", exponent, "offset", offset, "optimise", true);
		return t;
	}

	@Test
	public void adaptationStateSurvivesAStoreRestoreRoundTrip() {
		Randomizer.setSeed(7);

		PowerOffsetTransform target = transform(2.0, 0.01);
		PowerOffsetTransform edge = transform(2.0, 0.01);
		TargetedWilsonBalding source = operatorWithTransforms(target, edge);

		// Drive some adaptation so the stored values differ from the defaults.
		for (int i = 0; i < 500; i++) {
			target.startProposal();
			target.apply(new double[] { 1.0, 2.0, 3.0 });
			target.optimize(Math.log(0.5));
		}

		StringWriter buffer = new StringWriter();
		PrintWriter out = new PrintWriter(buffer);
		source.storeToFile(out);
		out.flush();

		PowerOffsetTransform restoredTarget = transform(2.0, 0.01);
		PowerOffsetTransform restoredEdge = transform(2.0, 0.01);
		TargetedWilsonBalding restored = operatorWithTransforms(restoredTarget, restoredEdge);
		restored.restoreFromFile(new JSONObject(buffer.toString()));

		Assert.assertArrayEquals(target.getCoefficients(), restoredTarget.getCoefficients(), 1e-12);
		Assert.assertEquals(target.getAdaptationCount(), restoredTarget.getAdaptationCount());
		Assert.assertEquals(target.getAcceptanceBaseline(), restoredTarget.getAcceptanceBaseline(), 1e-12);
		Assert.assertEquals(edge.getAdaptationCount(), restoredEdge.getAdaptationCount());
	}

	/** An operator without transforms must still produce a state entry BEAST can read back. */
	@Test
	public void operatorWithoutTransformsStillRoundTrips() {
		Tree tree = new Tree("((A:1.0,B:1.0):1.0,C:2.0):0.0;");
		ConstantWeights weights = new ConstantWeights();
		weights.initByName("tree", tree);

		TargetedWilsonBalding op = new TargetedWilsonBalding();
		op.initByName("tree", tree, "weight", 1.0, "edgeWeights", weights);
		op.setID("plainWB");

		StringWriter buffer = new StringWriter();
		PrintWriter out = new PrintWriter(buffer);
		op.storeToFile(out);
		out.flush();

		JSONObject json = new JSONObject(buffer.toString());
		Assert.assertEquals("plainWB", json.getString("id"));
		Assert.assertFalse(json.has("transforms"));

		op.restoreFromFile(json);
	}
}
```

- [ ] **Step 2: Run and verify failure**

Run: `ant test-single -Dtest.class=targetedbeast.weighttransform.WeightTransformPersistenceTest`

Expected: `adaptationStateSurvivesAStoreRestoreRoundTrip` fails — the restored transform is
still at the defaults, because `Operator.storeToFile` writes only `p`.

- [ ] **Step 3: Implement the overrides**

Add to `src/targetedbeast/operators/TargetedWilsonBalding.java`:

```java
	/**
	 * Persist the transform tuning alongside the standard operator statistics, using the
	 * nested-JSON convention documented on {@link beast.base.inference.Operator#storeToFile}.
	 *
	 * <p>The iteration count matters as much as the coefficients: without it a resumed chain
	 * resets the SPSA gain sequences to their maximum and takes a burst of large adaptation
	 * steps mid-run.
	 */
	@Override
	public void storeToFile(final PrintWriter out) {
		if (targetWeightTransform == null && edgeWeightTransform == null) {
			super.storeToFile(out);
			return;
		}

		StringWriter buffer = new StringWriter();
		PrintWriter bufferedOut = new PrintWriter(buffer);
		super.storeToFile(bufferedOut);
		bufferedOut.flush();

		try {
			JSONObject json = new JSONObject(buffer.toString());
			JSONArray transforms = new JSONArray();
			appendTransform(transforms, "target", targetWeightTransform);
			appendTransform(transforms, "edge", edgeWeightTransform);
			json.put("transforms", transforms);
			out.print(json.toString());
		} catch (JSONException e) {
			// Match the base class: report and carry on rather than killing the chain.
			e.printStackTrace();
			out.print(buffer.toString());
		}
	}

	private void appendTransform(JSONArray transforms, String role, WeightTransform transform)
			throws JSONException {
		if (transform == null) {
			return;
		}
		double[] coefficients = transform.getCoefficients();
		JSONObject entry = new JSONObject();
		entry.put("role", role);
		entry.put("exponent", coefficients[0]);
		entry.put("offset", coefficients[1]);
		entry.put("n", transform.getAdaptationCount());
		entry.put("baseline", transform.getAcceptanceBaseline());
		transforms.put(entry);
	}

	@Override
	public void restoreFromFile(JSONObject o) {
		super.restoreFromFile(o);
		if (!o.has("transforms")) {
			return;
		}
		try {
			JSONArray transforms = o.getJSONArray("transforms");
			for (int k = 0; k < transforms.length(); k++) {
				JSONObject entry = transforms.getJSONObject(k);
				WeightTransform transform = "target".equals(entry.getString("role"))
						? targetWeightTransform
						: edgeWeightTransform;
				if (transform == null) {
					continue;
				}
				transform.setAdaptationState(
						new double[] { entry.getDouble("exponent"), entry.getDouble("offset") },
						entry.getLong("n"),
						entry.getDouble("baseline"));
			}
		} catch (JSONException e) {
			// Failed to restore tuning; report and continue with the configured defaults.
			e.printStackTrace();
		}
	}
```

Add the imports: `java.io.PrintWriter`, `java.io.StringWriter`, `org.json.JSONArray`,
`org.json.JSONException`, `org.json.JSONObject`.

- [ ] **Step 4: Run and verify pass**

Run: `ant test-single -Dtest.class=targetedbeast.weighttransform.WeightTransformPersistenceTest`

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/targetedbeast/operators/TargetedWilsonBalding.java \
        test/targetedbeast/weighttransform/WeightTransformPersistenceTest.java
git commit -m "feat: persist WeightTransform tuning in the operator state file"
```

---

## Task 8: Theory write-up

**Files:**
- Create: `docs/WeightTransformAdaptation.md`

**Interfaces:** none — documentation only.

- [ ] **Step 1: Write the document**

Create `docs/WeightTransformAdaptation.md` with exactly this content:

````markdown
# Adapting weight-transform coefficients

## What is being optimized

A targeted operator picks a source edge and a target edge from distributions built out of
`EdgeWeights` output. A `WeightTransform` reshapes those weights before normalization:

```
u_k  = f(w_k, b)                pointiness
w'_k = a · mean(u) + u_k        offset
q_k  = w'_k / Σ_j w'_j          selection probability
```

`b` controls how greedy the proposal is; `a` mixes in a uniform distribution, since

```
q = (a/(1+a)) · uniform + (1/(1+a)) · normalize(u)
```

The objective is the expected Metropolis-Hastings acceptance probability of the operator,

```
J(θ) = E[ min(1, exp(logAlpha)) ]
```

as a function of θ = (log b, log a). We want the argmax.

## Why BEAST's standard machinery cannot do this

Every adaptable operator in BEAST tunes through `OperatorSchedule.calcDelta`:

```
delta = (1/count) · (exp(min(logAlpha, 0)) − target)
```

with `target = 0.234`. This is Robbins-Monro applied to the equation `J(θ) = target`. It
finds a coefficient at which the acceptance rate *equals* a preset value; it has no notion
of a maximum. For a step-size parameter that is the right thing to do, because acceptance
decreases monotonically in step size and 0.234 is the known optimal trade-off between
acceptance and distance travelled.

A weight transform is not a step size. Making the proposal pointier raises acceptance
without shortening the move, so there is no trade-off for 0.234 to balance. Feeding `J` into
`calcDelta` would drive `b` *downward* whenever acceptance exceeded 0.234 — away from the
optimum. A maximizer is needed instead.

## SPSA

Simultaneous perturbation stochastic approximation estimates a gradient from a single noisy
measurement per iteration, regardless of the dimension of θ.

At iteration `n`, draw a Rademacher vector `s ∈ {−1,+1}^d` and evaluate the objective at the
perturbed point:

```
θ_eff = θ_n + c_n · s
α     = min(1, exp(logAlpha))       measured at θ_eff
```

then update

```
θ_{n+1} = θ_n + (a_n / c_n) · (α − ᾱ_n) · s
ᾱ_{n+1} = ᾱ_n + λ · (α − ᾱ_n)
```

Why this estimates a gradient: expand `α` around `θ_n`. Its expectation at the perturbed
point is `J(θ_n) + c_n · sᵀ∇J(θ_n) + O(c_n²)`. Multiplying by `s` and taking expectations,
the cross terms `E[s_i s_j] = 0` for `i ≠ j` and `E[s_i²] = 1`, so

```
E[(α − ᾱ) · s]  =  c_n · ∇J(θ_n)  +  O(c_n²)
```

Dividing by `c_n` leaves an estimate of `∇J` biased only at order `c_n`, which vanishes as
`c_n → 0`. The cost is one measurement per iteration in any dimension — as opposed to `2d`
for coordinate-wise finite differences.

### The baseline

`ᾱ` is an exponentially weighted running mean of the acceptance probability, used as a
control variate. It does not change the expectation, because `E[ᾱ · s] = 0` for `ᾱ`
independent of the current `s`. It does change the variance: without it, each update has
magnitude proportional to `α` itself rather than to how `α` responded to the perturbation,
and the signal is swamped. With acceptance around 0.4 and a perturbation effect of a few
percent, the baseline is the difference between converging and random-walking.

### Gain sequences

```
a_n = A / (n + n₀)^0.602
c_n = C / (n + n₀)^0.101
```

These are Spall's recommended exponents. The requirements they satisfy:

- `Σ a_n/c_n = ∞` — the iterate can still travel an unbounded distance, so a bad starting
  point is recoverable.
- `Σ (a_n/c_n)² < ∞` — accumulated noise has bounded variance, so the iterate settles.
- `c_n → 0` — the gradient bias vanishes.

`n₀` is a stability constant that damps the first few hundred steps, when `ᾱ` has not yet
converged and the updates would otherwise be large and uninformed.

## Why this is a valid MCMC

Two separate claims.

**Each proposal is a correct Metropolis-Hastings move.** The coefficients are frozen at
`θ_eff` for the entire duration of one `proposal()` call. The forward probability of the
proposed move and the reverse probability of its inverse are therefore computed from the
same distribution family, and the Hastings ratio is exact. Adaptation happens only in
`optimize()`, which runs after the accept/reject decision.

**The adaptive chain has the right stationary distribution.** This is the standard
diminishing-adaptation argument (Roberts & Rosenthal 2007). Two conditions:

- *Diminishing adaptation.* The kernel used at step `n+1` differs from the one at step `n`
  by an amount tending to zero in probability. Here the coefficients move by
  `O(a_n/c_n) = O(n^−0.501)` and the perturbation jitter is `O(c_n) = O(n^−0.101)`, both
  vanishing.
- *Bounded convergence.* Each fixed-coefficient kernel converges, uniformly over the
  reachable coefficient range. The coefficients are clamped to `b ∈ [0.1, 20]` and
  `a ∈ [1e-6, 100]`, so the family is compact and no member degenerates: `a > 0` keeps
  every feasible candidate reachable, and `b < ∞` keeps the proposal from collapsing onto a
  single target.

The clamping is not cosmetic — it is what makes the second condition hold.

## The limitation worth knowing about

Maximizing acceptance is not the same as maximizing mixing. A proposal that is extremely
pointy will keep proposing the same target and be accepted almost every time, while the
chain oscillates between two topologies and explores nothing. Acceptance rate is blind to
this.

`TargetedWilsonBalding` partially defends itself: it zeroes the sibling's selection mass, so
a proposal always changes the topology, and a pure no-op is impossible. But the degenerate
two-state oscillation is not excluded. The upper bound on `b` exists for exactly this
reason, and the benchmark reports ESS per hour rather than acceptance alone, because ESS is
the quantity that actually matters.

If the adapted `b` is sitting at its upper bound in a real run, treat that as a signal to
investigate rather than as a converged answer.

## References

- Spall, J. C. (1992). Multivariate stochastic approximation using a simultaneous
  perturbation gradient approximation. *IEEE Transactions on Automatic Control* 37(3).
- Roberts, G. O. and Rosenthal, J. S. (2007). Coupling and ergodicity of adaptive MCMC.
  *Journal of Applied Probability* 44(2).
````

- [ ] **Step 2: Verify the claims against the implementation**

Read `src/targetedbeast/weighttransform/WeightTransform.java` and check line by line that the
gain exponents (0.602, 0.101), the bounds (`b ∈ [0.1, 20]`, `a ∈ [1e-6, 100]`), the update
sign, and the freeze-during-proposal behaviour all match what the document claims. Fix
whichever is wrong. A theory document that describes different code than the code is worse
than none.

- [ ] **Step 3: Commit**

```bash
git add docs/WeightTransformAdaptation.md
git commit -m "docs: explain the SPSA adaptation of weight-transform coefficients"
```

---

## Task 9: Benchmark harness

**Files:**
- Modify: `scripts/generate_xmls.py`
- Modify: `scripts/Snakefile`
- Modify: `examples/lassaL_kli22/template.xml`

**Interfaces:**
- Consumes: `PowerOffsetTransform` XML surface (`exponent`, `offset`, `optimise` attributes) from Tasks 1 and 3; the `targetWeightTransform` operator input from Task 5.
- Produces: `generate_xmls.SUBDIRS` (list of `{edgeweights}__{transform}` names) imported by the Snakefile.

- [ ] **Step 1: Add the template slots**

In `examples/lassaL_kli22/template.xml`, replace the `targetedWilsonBalding` operator line:

```xml
        <operator id="targetedWilsonBalding" spec="targetedbeast.operators.TargetedWilsonBalding" tree="@tree" weight="5" edgeWeights="@edgeweights" maxHeightRatio="1.2"/>
```

with:

```xml
        <operator id="targetedWilsonBalding" spec="targetedbeast.operators.TargetedWilsonBalding" tree="@tree" weight="5" edgeWeights="@edgeweights" maxHeightRatio="1.2">
            {target_weight_transform}
        </operator>
```

and add this line inside the `tracelog` logger, next to the other `<log idref=.../>` entries:

```xml
            {target_weight_transform_log}
```

- [ ] **Step 2: Add the transform axis to `generate_xmls.py`**

Replace the configuration block and `render_xml` in `scripts/generate_xmls.py`:

```python
N_REPS = 4
CHAIN_LENGTH = 10_000_000
EXAMPLE_NAME = "lassaL_kli22"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_ROOT = PROJECT_ROOT / "examples" / EXAMPLE_NAME
TEMPLATE_PATH = EXAMPLE_ROOT / "template.xml"


EDGE_WEIGHTS = {
	'untargeted': '<distribution id="edgeweights" spec="targetedbeast.edgeweights.ConstantWeights" tree="@tree"/>',
	'parsimony': '<distribution id="edgeweights" spec="targetedbeast.edgeweights.ParsimonyWeights2" tree="@tree" maxWeight="20" data="@alignment"/>',
} | {
	f'brownian_{d}': f'<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeightsBrownian" tree="@tree" maxWeight="20" data="@alignment" dimension="{d}"/>'
	for d in [3, 6]
}

# Each entry is (operator-nested XML, tracelog XML). The 'none' entry keeps the operator
# on its historical hardcoded squaring, which is the baseline every comparison is against.
WEIGHT_TRANSFORMS = {
	'none': ('', ''),
	'power': (
		'<targetWeightTransform id="targetWeightTransform"'
		' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
		' exponent="2.0" offset="0.01"/>',
		'<log idref="targetWeightTransform"/>',
	),
}

# One subdirectory per (edge weights, transform) combination.
SUBDIRS = [f"{w}__{t}" for w in EDGE_WEIGHTS for t in WEIGHT_TRANSFORMS]


def render_xml(template_text: str, weights_string: str, transform: tuple[str, str], chain_length: int) -> str:
	transform_xml, transform_log = transform
	return template_text.format(
		edge_weights=weights_string,
		target_weight_transform=transform_xml,
		target_weight_transform_log=transform_log,
		chain_length=str(chain_length),
	)
```

and replace the body of `main()`:

```python
def main() -> None:
	parser = argparse.ArgumentParser(description="Render the BEAST XML files from the template.")
	parser.add_argument(
		"--n-reps",
		type=int,
		default=N_REPS,
		help="number of replicate run?.xml files to generate per subdirectory",
	)
	parser.add_argument(
		"--chain-length",
		type=int,
		default=CHAIN_LENGTH,
		help="MCMC chain length; lower this for a quick development comparison",
	)
	args = parser.parse_args()

	template_text = TEMPLATE_PATH.read_text(encoding="utf-8")

	for w, weights_string in EDGE_WEIGHTS.items():
		for t, transform in WEIGHT_TRANSFORMS.items():
			rendered_xml = render_xml(template_text, weights_string, transform, args.chain_length)

			output_dir = EXAMPLE_ROOT / f"{w}__{t}"
			output_dir.mkdir(parents=True, exist_ok=True)

			for i in range(args.n_reps):
				output_path = output_dir / f"run{i}.xml"
				output_path.write_text(rendered_xml, encoding="utf-8")
				print(f"Wrote {output_path}")
```

- [ ] **Step 3: Verify XML rendering before involving BEAST**

```bash
python scripts/generate_xmls.py --n-reps 1 --chain-length 200000
grep -A2 'id="targetedWilsonBalding"' examples/lassaL_kli22/parsimony__power/run0.xml
grep -A2 'id="targetedWilsonBalding"' examples/lassaL_kli22/parsimony__none/run0.xml
```

Expected: the `__power` file contains a `<targetWeightTransform ... />` element inside the
operator; the `__none` file has an empty operator body. Both must contain
`chainLength="200000"`.

- [ ] **Step 4: Add the quick profile to the Snakefile**

In `scripts/Snakefile`, replace the import and the `N_REPS` block:

```python
from generate_xmls import EXAMPLE_ROOT, SUBDIRS  # noqa: E402

# `--config quick=1` gives a short comparison suitable for the development loop.
QUICK = bool(int(config.get("quick", 0)))

N_REPS = int(config.get("n_reps", 2 if QUICK else 4))
CHAIN_LENGTH = int(config.get("chain_length", 200_000 if QUICK else 10_000_000))
```

delete the now-unused `SUBDIRS = list(EDGE_WEIGHTS.keys())` line, and change the
`generate_xmls` rule's shell command to pass the chain length:

```python
rule generate_xmls:
    output:
        expand(
            str(EXAMPLE_ROOT / "{subdir}" / "run{rep}.xml"),
            subdir=SUBDIRS,
            rep=REPS,
        ),
    shell:
        "python {SCRIPT_DIR}/generate_xmls.py --n-reps {N_REPS} --chain-length {CHAIN_LENGTH}"
```

- [ ] **Step 5: Run the quick profile end to end**

```bash
cd scripts && snakemake -j4 --config quick=1
```

Expected: `examples/lassaL_kli22/mcmc_stats.tsv` and `operator_acceptance.tsv` appear, with
one row per `{edgeweights}__{transform}` × replicate. Completion in minutes, not hours.

Then compare the target operator's acceptance rate between the `__none` and `__power`
subdirectories:

```bash
python - <<'PY'
import csv, collections
rows = list(csv.DictReader(open("examples/lassaL_kli22/operator_acceptance.tsv"), delimiter="\t"))
acc = collections.defaultdict(list)
for r in rows:
    if "targetedWilsonBalding" in r["operator"]:
        acc[r["subdir"]].append(float(r["accept_rate"]))
for k in sorted(acc):
    print(f"{k:28s} {sum(acc[k])/len(acc[k]):.4f}")
PY
```

**Interpreting this:** 200k iterations with `n₀ = 100` gives the adaptation plenty of steps,
but two replicates is not enough to distinguish a small improvement from noise. Treat the
quick profile as a smoke test — it tells you the transform is wired up and not catastrophic.
Do not report an improvement from it; use the full profile for that.

(The column names `subdir`, `operator` and `accept_rate` are the ones `collect_stats.py`
writes at line 129.)

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_xmls.py scripts/Snakefile examples/lassaL_kli22/template.xml
git commit -m "test: add a weight-transform axis and quick profile to the benchmark harness"
```

---

## Final verification

- [ ] **Clean rebuild and run everything relevant**

```bash
rm -rf build build-test && ant compile compile-test
for c in targetedbeast.weighttransform.WeightTransformTest \
         targetedbeast.weighttransform.SpsaAdaptationTest \
         targetedbeast.weighttransform.PartialWeightsLogSpaceTest \
         targetedbeast.weighttransform.WeightTransformPersistenceTest \
         targetedbeast.EdgeWeightRejectConsistencyTest \
         targetedbeast.TargetedOperatorDetailedBalanceTest; do
  echo "=== $c ==="
  ant test-single -Dtest.class=$c 2>&1 | grep -E "Tests run|FAILED"
done
```

- [ ] **Confirm against the baseline**

The only acceptable failures are the ones in `/tmp/db-baseline.txt` plus the known-failing
list at the top of this plan. Every new test must pass.

- [ ] **Confirm the no-transform path is untouched**

`git diff main -- src/targetedbeast/operators/TargetedWilsonBalding.java` and read it. Every
behavioural change must be inside an `if (transform != null)` branch or equivalent. If you
find a change that affects the null path, that is a bug regardless of what the tests say.
