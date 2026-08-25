package targetedbeast.edgeweights;

import java.io.PrintStream;
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
 * Diagnostic edge weights that are structure-DEPENDENT but computed fresh from the
 * current tree on every call (no cache, no incremental "without node" bookkeeping).
 * The target weight for candidate c is a deterministic function of the topological
 * distance between the source node and c in the current tree, so it genuinely differs
 * before and after a move -- exactly like the cached implementations -- but without any
 * caching.
 *
 * <p>If detailed balance holds with these weights but fails with the cached
 * implementations, the fault is in the incremental cache, not in the operator's handling
 * of state-dependent target weights.
 */
public class StructuralWeights extends Distribution implements EdgeWeights {

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

	@Override
	public double getEdgeWeights(int nodeNr) {
		return 1.0;
	}

	private int depth(Node node) {
		int d = 0;
		while (!node.isRoot()) {
			node = node.getParent();
			d++;
		}
		return d;
	}

	/** Number of edges on the path between two nodes in the current tree. */
	private int topologicalDistance(Node a, Node b) {
		int da = depth(a);
		int db = depth(b);
		int dist = 0;
		while (da > db) {
			a = a.getParent();
			da--;
			dist++;
		}
		while (db > da) {
			b = b.getParent();
			db--;
			dist++;
		}
		while (a != b) {
			a = a.getParent();
			b = b.getParent();
			dist += 2;
		}
		return dist;
	}

	private double[] weights(int fromNodeNr, List<Integer> toNodeNrs) {
		Tree tree = (Tree) treeInput.get();
		Node from = tree.getNode(fromNodeNr);
		double[] distances = new double[toNodeNrs.size()];
		for (int k = 0; k < toNodeNrs.size(); k++) {
			int d = topologicalDistance(from, tree.getNode(toNodeNrs.get(k)));
			// Strictly positive, non-uniform, and genuinely structure-dependent.
			distances[k] = 1.0 / (1.0 + d);
		}
		return distances;
	}

	@Override
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes) {
		return weights(fromNodeNr, toNodes.stream().map(Node::getNr).toList());
	}

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
		return weights(fromNodeNr, toNodeNrs);
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
