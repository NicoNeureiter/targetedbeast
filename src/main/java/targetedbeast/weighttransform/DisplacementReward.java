package targetedbeast.weighttransform;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Function;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeMetric;

@Description("Tuner reward combining several measures of how far the chain moved: tree metrics and "
		+ "scalar parameters, each normalised by its own typical squared displacement so that all count "
		+ "equally. With references = 0 the reward is the normalised squared displacement since the "
		+ "proposal (or, under a block-mode tuner, the block) began. With references = K > 0 it keeps K "
		+ "past states, one every referenceSpacing measured proposals, and rewards the proposal by how "
		+ "much it increased the squared distance to them, so that moves undone soon after earn "
		+ "nothing. Tuners driven by the same proposals can share an instance.")
public class DisplacementReward extends BEASTObject implements Function, ProposalReward {

	final public Input<Tree> treeInput = new Input<>("tree", "tree the tree metrics are measured on");
	final public Input<List<TreeMetric>> treeMetricInput = new Input<>("treeMetric",
			"tree metrics, each a component of the reward", new ArrayList<>());
	final public Input<List<Function>> parameterInput = new Input<>("parameter",
			"scalar or vector quantities, e.g. the likelihood, each a component of the reward", new ArrayList<>());
	final public Input<Integer> referencesInput = new Input<>("references",
			"number of past reference states; 0 measures displacement since the proposal began", 0);
	final public Input<Integer> referenceSpacingInput = new Input<>("referenceSpacing",
			"measured proposals between consecutive reference states", 100);
	final public Input<Integer> normalisationInput = new Input<>("normalisation",
			"measurements over which each component's scale is estimated before it is frozen", 1000);

	/** A copy of the state the reward depends on, with a cheap signature to compare it by. */
	private static final class Snapshot {
		Tree tree;
		int[] parents;
		double[] heights;
		double[] scalars;
	}

	private Tree tree;
	private List<TreeMetric> metrics;
	private List<Function> parameters;
	private int components;
	private int references;

	private Snapshot snapshot;
	private final Deque<Snapshot> referenceStates = new ArrayDeque<>();
	private long referenceVersion;

	/** Squared distances from the last measured state to each reference, for reuse as the next "before". */
	private double[][] lastToReference;
	private Snapshot lastMeasured;
	private long lastReferenceVersion = -1;

	private double[] scaleSum;
	private double[] scale;
	private long scaleCount;
	private long measured;
	private Double reward;

	@Override
	public void initAndValidate() {
		tree = treeInput.get();
		metrics = treeMetricInput.get();
		parameters = parameterInput.get();
		if (!metrics.isEmpty() && tree == null) {
			throw new IllegalArgumentException("tree metrics need a tree");
		}
		components = metrics.size() + parameters.size();
		if (components == 0) {
			throw new IllegalArgumentException("DisplacementReward needs at least one tree metric or parameter");
		}
		references = referencesInput.get();
		scaleSum = new double[components];
		scale = new double[components];
	}

	@Override
	public void beginProposal() {
		reward = null;
		if (snapshot != null && matches(snapshot)) {
			return;
		}
		snapshot = snapshot(true);
	}

	@Override
	public double reward() {
		if (snapshot == null) {
			throw new IllegalStateException("reward() before beginProposal()");
		}
		if (reward != null) {
			return reward;
		}
		double[] raw = new double[components];
		double[] level = new double[components];
		if (references == 0) {
			for (int c = 0; c < components; c++) {
				raw[c] = squaredToLive(c, snapshot);
				level[c] = raw[c];
			}
		} else if (!referenceStates.isEmpty()) {
			double[][] before = beforeToReference();
			double[][] after = new double[components][referenceStates.size()];
			int k = 0;
			for (Snapshot ref : referenceStates) {
				for (int c = 0; c < components; c++) {
					after[c][k] = squaredToLive(c, ref);
					raw[c] += (after[c][k] - before[c][k]) / referenceStates.size();
					level[c] += after[c][k] / referenceStates.size();
				}
				k++;
			}
			lastToReference = after;
			lastMeasured = snapshot(false);
			lastReferenceVersion = referenceVersion;
		}
		// Before the first reference exists there is nothing to measure, so no scale information either.
		if (references == 0 || !referenceStates.isEmpty()) {
			updateScale(level);
		}
		double total = 0.0;
		for (int c = 0; c < components; c++) {
			if (scale[c] > 0.0) {
				total += raw[c] / scale[c];
			}
		}
		measured++;
		if (references > 0 && measured % referenceSpacingInput.get() == 0) {
			referenceStates.addLast(snapshot(true));
			if (referenceStates.size() > references) {
				referenceStates.removeFirst();
			}
			referenceVersion++;
		}
		reward = total;
		return total;
	}

	/** The reward as a non-negative size, for callers that square it. */
	@Override
	public double jump() {
		return Math.sqrt(Math.abs(reward()));
	}

	/** Squared distances from the proposal's starting state to each reference. */
	private double[][] beforeToReference() {
		if (lastMeasured != null && lastReferenceVersion == referenceVersion && sameState(lastMeasured, snapshot)) {
			return lastToReference;
		}
		double[][] before = new double[components][referenceStates.size()];
		int k = 0;
		for (Snapshot ref : referenceStates) {
			for (int c = 0; c < components; c++) {
				before[c][k] = squared(c, snapshot, ref);
			}
			k++;
		}
		return before;
	}

	/** Mean squared displacement per component over the first measurements, then frozen. */
	private void updateScale(double[] level) {
		if (scaleCount >= normalisationInput.get()) {
			return;
		}
		scaleCount++;
		for (int c = 0; c < components; c++) {
			scaleSum[c] += level[c];
			scale[c] = scaleSum[c] / scaleCount;
		}
	}

	private double squaredToLive(int c, Snapshot from) {
		if (c < metrics.size()) {
			double d = metrics.get(c).distance(from.tree, tree);
			return d * d;
		}
		Function f = parameters.get(c - metrics.size());
		double[] values = from.scalars;
		int offset = scalarOffset(c);
		double sum = 0.0;
		for (int i = 0; i < f.getDimension(); i++) {
			double d = f.getArrayValue(i) - values[offset + i];
			sum += d * d;
		}
		return sum;
	}

	private double squared(int c, Snapshot a, Snapshot b) {
		if (c < metrics.size()) {
			double d = metrics.get(c).distance(a.tree, b.tree);
			return d * d;
		}
		int offset = scalarOffset(c);
		double sum = 0.0;
		for (int i = 0; i < parameters.get(c - metrics.size()).getDimension(); i++) {
			double d = a.scalars[offset + i] - b.scalars[offset + i];
			sum += d * d;
		}
		return sum;
	}

	private int scalarOffset(int c) {
		int offset = 0;
		for (int p = 0; p < c - metrics.size(); p++) {
			offset += parameters.get(p).getDimension();
		}
		return offset;
	}

	/** The current state; the tree is copied only if {@code copyTree}, the signature always. */
	private Snapshot snapshot(boolean copyTree) {
		Snapshot s = new Snapshot();
		if (tree != null) {
			if (copyTree) {
				s.tree = tree.copy();
			}
			int n = tree.getNodeCount();
			s.parents = new int[n];
			s.heights = new double[n];
			for (int nr = 0; nr < n; nr++) {
				Node node = tree.getNode(nr);
				s.parents[nr] = node.isRoot() ? -1 : node.getParent().getNr();
				s.heights[nr] = node.getHeight();
			}
		}
		int dims = 0;
		for (Function f : parameters) {
			dims += f.getDimension();
		}
		s.scalars = new double[dims];
		int i = 0;
		for (Function f : parameters) {
			for (int d = 0; d < f.getDimension(); d++) {
				s.scalars[i++] = f.getArrayValue(d);
			}
		}
		return s;
	}

	private boolean matches(Snapshot s) {
		return sameState(s, snapshot(false));
	}

	private static boolean sameState(Snapshot a, Snapshot b) {
		return java.util.Arrays.equals(a.parents, b.parents) && java.util.Arrays.equals(a.heights, b.heights)
				&& java.util.Arrays.equals(a.scalars, b.scalars);
	}

	@Override
	public int getDimension() {
		return 1;
	}

	/** Not a value of the state; tuners read it through {@link ProposalReward}. */
	@Override
	public double getArrayValue(int dim) {
		throw new UnsupportedOperationException("DisplacementReward is a ProposalReward; read it with beginProposal()/reward()");
	}
}
