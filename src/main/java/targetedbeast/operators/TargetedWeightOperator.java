package targetedbeast.operators;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Arrays;
import java.util.function.Predicate;
import java.util.function.ToDoubleFunction;

import beast.base.internal.json.JSONArray;
import beast.base.internal.json.JSONException;
import beast.base.internal.json.JSONObject;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;
import targetedbeast.edgeweights.EdgeWeights;
import targetedbeast.weighttransform.WeightTransform;

/**
 * A tree operator whose proposal is driven by {@link EdgeWeights}, with optional adaptive
 * {@link WeightTransform}s on the two weight vectors it uses.
 *
 * <p>Subclasses pick a <em>source</em> edge to move, weighted by per-edge weights, and a
 * <em>target</em> edge to move it to, weighted by per-candidate weights. Either vector may be
 * reshaped before it is normalized, and the transform coefficients are adapted during the run
 * to raise the operator's acceptance rate.
 *
 * <p>Two invariants are easy to get wrong and are handled here rather than in each subclass:
 *
 * <ul>
 * <li>A transform adds its offset to <em>every</em> entry it is handed, so an infeasible
 * candidate has to be left out of the vector rather than passed in with weight zero — a zero
 * entry would acquire positive selection probability. {@link #sourceWeights} builds the source
 * vector that way; target vectors are the subclass's responsibility.</li>
 * <li>When one transform object is wired to both inputs it must be driven once per proposal,
 * not twice. A second {@link WeightTransform#startProposal()} at the head of a batch would
 * redraw the perturbation, and a second {@link WeightTransform#optimize(double)} would take two
 * adaptation steps from a single observation.</li>
 * </ul>
 *
 * <p>Subclasses must call {@link #initWeightTransforms} from their {@code initAndValidate}, and
 * {@link #startTransformProposal()} at the top of {@code proposal()} before any weight is read.
 */
@Description("Base class for tree operators that propose from EdgeWeights, with optional "
		+ "adaptive transforms on the target and source-edge weight vectors.")
public abstract class TargetedWeightOperator extends TreeOperator {

	/** Below this a normalizing total is treated as degenerate rather than divided by. */
	protected static final double MIN_TOTAL_WEIGHT = 1E-12;

	public Input<WeightTransform> targetWeightTransformInput = new Input<>("targetWeightTransform",
			"optional transform applied to the target weights before they are normalized");

	public Input<WeightTransform> edgeWeightTransformInput = new Input<>("edgeWeightTransform",
			"optional transform applied to the source-edge weights before they are normalized");

	protected WeightTransform targetWeightTransform;
	protected WeightTransform edgeWeightTransform;

	/** What {@link #applyTargetTransform} does when no target transform is configured. */
	private boolean squareTargetsWhenAbsent;

	/**
	 * Resolve and check the configured transforms. Call from {@code initAndValidate}.
	 *
	 * @param edgeWeights             the weights this operator proposes from
	 * @param squareTargetsWhenAbsent whether target weights are squared in the absence of a
	 *                                transform. True for the operators that have always done
	 *                                so, false for those that use the weights as they come. A
	 *                                transform's own exponent replaces the squaring, so the two
	 *                                must never both run.
	 */
	protected void initWeightTransforms(EdgeWeights edgeWeights, boolean squareTargetsWhenAbsent) {
		this.squareTargetsWhenAbsent = squareTargetsWhenAbsent;
		targetWeightTransform = targetWeightTransformInput.get();
		edgeWeightTransform = edgeWeightTransformInput.get();

		// A log-space transform applied to real-space weights, or the reverse, produces a
		// silently wrong proposal distribution rather than an error, so catch it at startup.
		if (targetWeightTransform != null
				&& targetWeightTransform.requiresLogSpace() != edgeWeights.isLogSpace()) {
			throw new IllegalArgumentException(describe() + ": targetWeightTransform expects "
					+ (targetWeightTransform.requiresLogSpace() ? "log" : "real")
					+ "-space weights but " + edgeWeights.getClass().getSimpleName()
					+ " returns " + (edgeWeights.isLogSpace() ? "log" : "real") + "-space weights");
		}
		// getEdgeWeights is real-space by contract in every implementation, and where the
		// source weights are branch lengths they are real-space too, so a log-space transform
		// is never correct on this side.
		if (edgeWeightTransform != null && edgeWeightTransform.requiresLogSpace()) {
			throw new IllegalArgumentException(describe() + ": edgeWeightTransform must be a "
					+ "real-space transform; source edge weights are never in log space");
		}
	}

	private String describe() {
		return getID() == null ? getClass().getSimpleName() : getID();
	}

	/** True if neither transform was configured, i.e. the operator behaves as it did before. */
	protected boolean hasNoWeightTransform() {
		return targetWeightTransform == null && edgeWeightTransform == null;
	}

	/**
	 * Freeze the transform coefficients for the duration of one proposal.
	 *
	 * <p>This is what keeps the forward and reverse probabilities of a single proposal mutually
	 * consistent, so the Hastings ratio stays exact.
	 */
	protected void startTransformProposal() {
		// Edge before target: a target transform's tuner may read the edge transform's live
		// per-proposal exponent (see RatioGridTuner), which must already be current for this
		// proposal by the time target reads it. Harmless for every other combination, since
		// independent transforms don't read each other's state.
		if (edgeWeightTransform != null) {
			edgeWeightTransform.startProposal();
		}
		if (targetWeightTransform != null && targetWeightTransform != edgeWeightTransform) {
			targetWeightTransform.startProposal();
		}
	}

	/**
	 * Reshape a target-weight vector.
	 *
	 * <p>Every entry must be a genuinely selectable candidate; see the class comment.
	 *
	 * @return a new array; {@code raw} is not modified
	 */
	protected double[] applyTargetTransform(double[] raw) {
		if (targetWeightTransform != null) {
			return targetWeightTransform.apply(raw);
		}
		if (!squareTargetsWhenAbsent) {
			return raw.clone();
		}
		double[] out = new double[raw.length];
		for (int k = 0; k < raw.length; k++) {
			out[k] = raw[k] * raw[k];
		}
		return out;
	}

	/**
	 * Reshape a source-edge weight vector. Without a transform the weights are used as they
	 * come, which is what every operator has always done on this side.
	 *
	 * @return a new array; {@code raw} is not modified
	 */
	protected double[] applyEdgeTransform(double[] raw) {
		return edgeWeightTransform == null ? raw.clone() : edgeWeightTransform.apply(raw);
	}

	/**
	 * Transformed source weights over the feasible nodes, in node-number order.
	 *
	 * @param feasible  which nodes may be chosen as the source edge; infeasible nodes are left
	 *                  out of the vector entirely rather than given weight zero
	 * @param rawWeight the untransformed weight of a feasible node
	 */
	protected SourceWeights sourceWeights(Tree tree, Predicate<Node> feasible,
			ToDoubleFunction<Node> rawWeight) {
		int nodeCount = tree.getNodeCount();
		int[] nodeNrs = new int[nodeCount];
		double[] raw = new double[nodeCount];

		int m = 0;
		for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {
			Node node = tree.getNode(nodeNr);
			if (!feasible.test(node)) {
				continue;
			}
			nodeNrs[m] = nodeNr;
			raw[m] = rawWeight.applyAsDouble(node);
			m++;
		}

		return new SourceWeights(Arrays.copyOf(nodeNrs, m), applyEdgeTransform(Arrays.copyOf(raw, m)));
	}

	/** A source-weight vector over the feasible nodes only. */
	public static final class SourceWeights {
		public final int[] nodeNrs;
		public final double[] weights;
		public final double total;

		SourceWeights(int[] nodeNrs, double[] weights) {
			this.nodeNrs = nodeNrs;
			this.weights = weights;
			double sum = 0.0;
			for (double w : weights) {
				sum += w;
			}
			this.total = sum;
		}

		/** True if the total is too small to normalize by. */
		public boolean isDegenerate() {
			return !(total > MIN_TOTAL_WEIGHT);
		}

		/** Position of a node in this vector, or -1 if it is not a feasible source. */
		public int indexOf(int nodeNr) {
			for (int m = 0; m < nodeNrs.length; m++) {
				if (nodeNrs[m] == nodeNr) {
					return m;
				}
			}
			return -1;
		}

		/** Weight of a node, or 0.0 if it is not a feasible source. */
		public double weightFor(int nodeNr) {
			int m = indexOf(nodeNr);
			return m < 0 ? 0.0 : weights[m];
		}

		/**
		 * Draw an index proportional to the weights, or -1 if the total is degenerate.
		 *
		 * <p>Falls back to the last entry when accumulated rounding leaves the cumulative sum
		 * just short of the scaler, rather than reporting a failure the caller cannot act on.
		 */
		public int sampleIndex() {
			if (isDegenerate()) {
				return -1;
			}
			double scaler = Randomizer.nextDouble() * total;
			double cumulative = 0.0;
			for (int m = 0; m < weights.length; m++) {
				cumulative += weights[m];
				if (cumulative > scaler) {
					return m;
				}
			}
			return weights.length - 1;
		}
	}

	/** Feed one acceptance observation back to the transforms. */
	@Override
	public void optimize(final double logAlpha) {
		if (targetWeightTransform != null) {
			targetWeightTransform.optimize(logAlpha);
		}
		if (edgeWeightTransform != null && edgeWeightTransform != targetWeightTransform) {
			edgeWeightTransform.optimize(logAlpha);
		}
	}

	/**
	 * Persist the transform tuning alongside the operator's own statistics, using the nested
	 * JSON convention documented on {@link beast.base.inference.Operator#storeToFile}.
	 *
	 * <p>Only the role and the coefficients are written here; everything else is whatever the
	 * transform's tuner needs to resume, which varies by tuner and is therefore its own business.
	 */
	@Override
	public void storeToFile(final PrintWriter out) {
		if (hasNoWeightTransform()) {
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
			// One shared object is one piece of state; writing it twice would restore it twice.
			if (edgeWeightTransform != targetWeightTransform) {
				appendTransform(transforms, "edge", edgeWeightTransform);
			}
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
		JSONObject entry = new JSONObject();
		entry.put("role", role);
		transform.storeState(entry);
		transforms.put(entry);
	}

	/** Restore tuning written by {@link #storeToFile}. Missing entries leave the defaults. */
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
				transform.restoreState(entry);
			}
		} catch (JSONException e) {
			// Failed to restore tuning; report and continue with the configured defaults.
			e.printStackTrace();
		}
	}
}
