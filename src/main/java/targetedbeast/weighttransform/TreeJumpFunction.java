package targetedbeast.weighttransform;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Function;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeMetric;

@Description("A live scalar over a tree metric, for wiring GridTuner's reward input to a "
		+ "tree-distance jump instead of a likelihood jump. Mimics a Function that changes "
		+ "value between two calls the way a likelihood does, so GridTuner's before/after "
		+ "squared-difference reward becomes the squared tree-metric jump across the proposal. "
		+ "Must be called in strictly alternating pairs (before the move, then after) and never "
		+ "shared between more than one consumer: the first call of each pair snapshots the tree "
		+ "and returns 0, the second call returns the distance from that snapshot to the tree's "
		+ "current (post-move) state.")
public class TreeJumpFunction extends BEASTObject implements Function {

	final public Input<Tree> treeInput = new Input<>("tree",
			"tree whose movement across a proposal is measured", Input.Validate.REQUIRED);
	final public Input<TreeMetric> treeMetricInput = new Input<>("treeMetric",
			"metric used to measure the distance between the pre- and post-proposal tree",
			Input.Validate.REQUIRED);

	private Tree tree;
	private TreeMetric treeMetric;

	/** Snapshot taken on the first call of each before/after pair. Mutable, so a copy. */
	private Tree snapshot;

	/** True when the next call starts a new pair and must snapshot rather than measure. */
	private boolean awaitingSnapshot = true;

	@Override
	public void initAndValidate() {
		tree = treeInput.get();
		treeMetric = treeMetricInput.get();
	}

	@Override
	public int getDimension() {
		return 1;
	}

	@Override
	public double getArrayValue(int dim) {
		if (awaitingSnapshot) {
			snapshot = tree.copy();
			awaitingSnapshot = false;
			return 0.0;
		}
		awaitingSnapshot = true;
		return treeMetric.distance(snapshot, tree);
	}
}
