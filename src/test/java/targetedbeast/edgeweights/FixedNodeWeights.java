package targetedbeast.edgeweights;

import java.io.PrintStream;
import java.util.List;
import java.util.Random;

import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Distribution;
import beast.base.inference.State;

/**
 * Diagnostic edge weights that are deterministic, non-uniform, structure-independent,
 * and completely stateless (no cache, no incremental update). The target weight for a
 * candidate depends only on the candidate node number, so it is identical on the tree
 * before and after any move. Source weights are uniform (so the source term cancels).
 *
 * <p>Used to separate a genuine non-uniform-selection Hastings-ratio bug from a
 * caching/incremental-update bug: if detailed balance holds with these weights but not
 * with the cached implementations, the fault is in the cache, not the selection math.
 */
public class FixedNodeWeights extends Distribution implements EdgeWeights {

	final public Input<TreeInterface> treeInput = new Input<>("tree", "tree", Validate.REQUIRED);

	private double weightForNode(int nodeNr) {
		// Deterministic, non-uniform, strictly positive, structure-independent.
		return 1.0 + ((nodeNr * 2654435761L) & 0x7fff) / 32768.0; // in [1, 2)
	}

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

	@Override
	public double getEdgeWeights(int nodeNr) {
		return 1.0;
	}

	@Override
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes) {
		double[] distances = new double[toNodes.size()];
		for (int k = 0; k < toNodes.size(); k++) {
			distances[k] = weightForNode(toNodes.get(k).getNr());
		}
		return distances;
	}

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
		double[] distances = new double[toNodeNrs.size()];
		for (int k = 0; k < toNodeNrs.size(); k++) {
			distances[k] = weightForNode(toNodeNrs.get(k));
		}
		return distances;
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
