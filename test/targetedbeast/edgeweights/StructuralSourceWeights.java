package targetedbeast.edgeweights;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Distribution;
import beast.base.inference.State;

/**
 * Diagnostic edge weights that isolate SOURCE selection: target weights are uniform (so the
 * target term cancels exactly), while the per-edge source weight is a strong, deterministic
 * function of the node's depth in the current tree, computed fresh on every call (no cache).
 *
 * <p>Because a Wilson-Balding move changes the depth of the moved subtree and its neighbours,
 * the source-selection distribution genuinely differs before and after the move. If the
 * operator omits the source-selection probability from its Hastings ratio, detailed balance
 * must break here -- and it cannot be blamed on any caching, since there is none.
 */
public class StructuralSourceWeights extends Distribution implements EdgeWeights {

	final public Input<TreeInterface> treeInput = new Input<>("tree", "tree", Validate.REQUIRED);

	@Override
	public void initAndValidate() {
	}

	@Override
	public void updateByOperator() {
	}

	@Override
	public void updateByOperatorWithoutNode(int ignore, List<Integer> nodes) {
	}

	@Override
	public void fakeUpdateByOperator() {
	}

	@Override
	protected boolean requiresRecalculation() {
		return true;
	}

	@Override
	public void prestore() {
	}

	@Override
	public void reset() {
	}

	/** Strongly depth-dependent source weight, recomputed from the current tree every call. */
	@Override
	public double getEdgeWeights(int nodeNr) {
		Node node = ((Tree) treeInput.get()).getNode(nodeNr);
		int depth = 0;
		while (!node.isRoot()) {
			node = node.getParent();
			depth++;
		}
		// Spread weights over a wide range so an omitted source term is easy to detect.
		return Math.pow(3.0, depth);
	}

	@Override
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes) {
		double[] w = new double[toNodes.size()];
		Arrays.fill(w, 1.0);
		return w;
	}

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
		double[] w = new double[toNodeNrs.size()];
		Arrays.fill(w, 1.0);
		return w;
	}

	@Override
	public double minEdgeWeight() {
		return 1.0;
	}

	@Override
	public List<String> getArguments() {
		return null;
	}

	@Override
	public List<String> getConditions() {
		return null;
	}

	@Override
	public void sample(State state, Random random) {
	}

	@Override
	public void init(PrintStream out) {
	}
}
