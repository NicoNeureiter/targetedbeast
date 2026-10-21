package targetedbeast.operators;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.function.Predicate;
import java.util.function.ToDoubleFunction;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;
import targetedbeast.edgeweights.EdgeWeights;
import targetedbeast.weighttransform.WeightTransform;

/**
 * Subclasses call {@link #initWeightTransforms} from {@code initAndValidate} and
 * {@link #startTransformProposal()} before reading any weight. Infeasible target candidates must
 * be left out of the vector, not given weight zero, since the offset makes every entry positive.
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
	 * Resolve and check the configured transforms.
	 *
	 * @param squareTargetsWhenAbsent whether target weights are squared when there is no target transform
	 */
	protected void initWeightTransforms(EdgeWeights edgeWeights, boolean squareTargetsWhenAbsent) {
		this.squareTargetsWhenAbsent = squareTargetsWhenAbsent;
		targetWeightTransform = targetWeightTransformInput.get();
		edgeWeightTransform = edgeWeightTransformInput.get();

		if (targetWeightTransform != null
				&& targetWeightTransform.requiresLogSpace() != edgeWeights.isLogSpace()) {
			throw new IllegalArgumentException(describe() + ": targetWeightTransform expects "
					+ (targetWeightTransform.requiresLogSpace() ? "log" : "real")
					+ "-space weights but " + edgeWeights.getClass().getSimpleName()
					+ " returns " + (edgeWeights.isLogSpace() ? "log" : "real") + "-space weights");
		}
		// Target vectors reach the transform without their nodes, so it cannot be given lengths.
		if (targetWeightTransform != null && targetWeightTransform.offsetTowardsBranchLength()) {
			throw new IllegalArgumentException(describe() + ": offsetBase='branchLength' is only "
					+ "supported on edgeWeightTransform");
		}
		// Source weights are always real-space.
		if (edgeWeightTransform != null && edgeWeightTransform.requiresLogSpace()) {
			throw new IllegalArgumentException(describe() + ": edgeWeightTransform must be a "
					+ "real-space transform; source edge weights are never in log space");
		}
	}

	private String describe() {
		return getID() == null ? getClass().getSimpleName() : getID();
	}

	/** True if neither transform was configured. */
	protected boolean hasNoWeightTransform() {
		return targetWeightTransform == null && edgeWeightTransform == null;
	}

	/** Fix the transform coefficients for one proposal; a shared transform is started once. */
	protected void startTransformProposal() {
		if (targetWeightTransform != null) {
			targetWeightTransform.startProposal();
		}
		if (edgeWeightTransform != null && edgeWeightTransform != targetWeightTransform) {
			edgeWeightTransform.startProposal();
		}
	}

	/**
	 * i's ancestors up to the highest target, whose "without i" values the target weights need.
	 * Non-target ancestors in between are included, since each value is built from its children's.
	 */
	protected static List<Integer> ancestorsUpToHighestTarget(Node i, Collection<Node> targets) {
		List<Integer> path = new ArrayList<>();
		int needed = 0;
		for (Node ancestor = i.getParent(); ancestor != null; ancestor = ancestor.getParent()) {
			path.add(ancestor.getNr());
			if (targets.contains(ancestor)) {
				needed = path.size();
			}
		}
		return new ArrayList<>(path.subList(0, needed));
	}

	/** Reshape a target-weight vector of feasible candidates. Returns a new array. */
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

	/** Reshape a source-edge weight vector; unchanged without a transform. Returns a new array. */
	protected double[] applyEdgeTransform(double[] raw) {
		return applyEdgeTransform(raw, null);
	}

	/** As {@link #applyEdgeTransform(double[])}, with branch lengths for a branch-length offset. */
	private double[] applyEdgeTransform(double[] raw, double[] lengths) {
		if (edgeWeightTransform == null) {
			return raw.clone();
		}
		return edgeWeightTransform.apply(raw, edgeWeightTransform.offsetTowardsBranchLength() ? lengths : null);
	}

	/** Transformed source weights over the feasible nodes, in node-number order. */
	protected SourceWeights sourceWeights(Tree tree, Predicate<Node> feasible,
			ToDoubleFunction<Node> rawWeight) {
		int nodeCount = tree.getNodeCount();
		int[] nodeNrs = new int[nodeCount];
		double[] raw = new double[nodeCount];

		double[] lengths = new double[nodeCount];

		int m = 0;
		for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {
			Node node = tree.getNode(nodeNr);
			if (!feasible.test(node)) {
				continue;
			}
			nodeNrs[m] = nodeNr;
			raw[m] = rawWeight.applyAsDouble(node);
			lengths[m] = node.getLength();
			m++;
		}

		return new SourceWeights(Arrays.copyOf(nodeNrs, m),
				applyEdgeTransform(Arrays.copyOf(raw, m), Arrays.copyOf(lengths, m)));
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

		/** Draw an index proportional to the weights, or -1 if the total is degenerate. */
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

	/** Store the transforms' tuning alongside the operator's own statistics. */
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
