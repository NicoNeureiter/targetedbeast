package targetedbeast.weighttransform;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Function;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeMetric;

@Description("Tree-distance jump across a proposal, for a tuner's reward input instead of a "
		+ "likelihood jump: the tuner snapshots the tree when a proposal starts and measures the "
		+ "distance to the tree when MCMC reports the outcome. Tuners driven by the same proposals "
		+ "can share an instance: the snapshot and the distance are then computed once per proposal.")
public class TreeJumpFunction extends BEASTObject implements Function, ProposalReward {

	final public Input<Tree> treeInput = new Input<>("tree",
			"tree whose movement across a proposal is measured", Input.Validate.REQUIRED);
	final public Input<TreeMetric> treeMetricInput = new Input<>("treeMetric",
			"metric used to measure the distance between the pre- and post-proposal tree",
			Input.Validate.REQUIRED);

	private Tree tree;
	private TreeMetric treeMetric;

	/** The tree when the current proposal started. Mutable, so a copy. */
	private Tree snapshot;

	/** Parent number and height of every node of the tree when {@link #snapshot} was taken. */
	private int[] snapshotParents;
	private double[] snapshotHeights;

	/** The jump since the snapshot, once measured; null until then. */
	private Double jump;

	@Override
	public void initAndValidate() {
		tree = treeInput.get();
		treeMetric = treeMetricInput.get();
	}

	/**
	 * Takes a new snapshot unless the tree is still the one already snapshotted, which is the case
	 * when a second tuner sharing this instance starts the same proposal. Comparing the tree rather
	 * than counting calls keeps this right when MCMC skipped reporting an earlier proposal.
	 */
	@Override
	public void beginProposal() {
		jump = null;
		if (snapshot != null && treeUnchanged()) {
			return;
		}
		snapshot = tree.copy();
		int n = tree.getNodeCount();
		if (snapshotParents == null || snapshotParents.length != n) {
			snapshotParents = new int[n];
			snapshotHeights = new double[n];
		}
		for (int nr = 0; nr < n; nr++) {
			Node node = tree.getNode(nr);
			snapshotParents[nr] = node.isRoot() ? -1 : node.getParent().getNr();
			snapshotHeights[nr] = node.getHeight();
		}
	}

	private boolean treeUnchanged() {
		if (snapshotParents.length != tree.getNodeCount()) {
			return false;
		}
		for (int nr = 0; nr < snapshotParents.length; nr++) {
			Node node = tree.getNode(nr);
			if (snapshotParents[nr] != (node.isRoot() ? -1 : node.getParent().getNr())
					|| snapshotHeights[nr] != node.getHeight()) {
				return false;
			}
		}
		return true;
	}

	/** Measured once per proposal; tuners sharing this instance get the same value. */
	@Override
	public double jump() {
		if (snapshot == null) {
			throw new IllegalStateException("jump() before beginProposal()");
		}
		if (jump == null) {
			jump = treeMetric.distance(snapshot, tree);
		}
		return jump;
	}

	@Override
	public int getDimension() {
		return 1;
	}

	/** Not a value of the state; tuners read it through {@link ProposalReward}. */
	@Override
	public double getArrayValue(int dim) {
		throw new UnsupportedOperationException("TreeJumpFunction is a ProposalReward; read it with beginProposal()/jump()");
	}
}
