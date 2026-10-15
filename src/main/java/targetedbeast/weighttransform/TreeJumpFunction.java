package targetedbeast.weighttransform;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Function;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeMetric;

@Description("Tree-distance jump across a proposal, for a tuner's reward input instead of a "
		+ "likelihood jump: the tuner snapshots the tree when a proposal starts and measures the "
		+ "distance to the tree when MCMC reports the outcome. Give each tuner its own instance.")
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

	@Override
	public void initAndValidate() {
		tree = treeInput.get();
		treeMetric = treeMetricInput.get();
	}

	@Override
	public void beginProposal() {
		snapshot = tree.copy();
	}

	@Override
	public double jump() {
		if (snapshot == null) {
			throw new IllegalStateException("jump() before beginProposal()");
		}
		return treeMetric.distance(snapshot, tree);
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
