package targetedbeast;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Function;
import java.util.function.Supplier;

import beast.base.core.BEASTObject;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import targetedbeast.edgeweights.EdgeWeights;

/**
 * Wraps the EdgeWeights an operator reads and checks every value read against a fresh instance
 * built from scratch on a snapshot of the current tree. Weights must be a function of the current
 * state alone: a proposal's forward weights must equal what a fresh instance on T gives, and its
 * reverse weights what a fresh instance on T' gives, i.e. what a proposal starting from T' would
 * read. Any cache that lags behind the tree (its own, or the likelihood's partials) shows up as a
 * mismatch at the first read that depends on it.
 *
 * <p>The fresh side always rebuilds the "without i" values for all of i's ancestors, so the truth
 * does not depend on the node list the operator passes.
 */
public class CheckedEdgeWeights extends BEASTObject implements EdgeWeights {

	private static final double TOLERANCE = 1e-9;

	private final EdgeWeights delegate;
	private final Tree tree;
	private final Function<Tree, EdgeWeights> freshFactory;
	private final Supplier<String> modelKey;

	/** -1 unless the delegate is between updateByOperatorWithoutNode and the next update or reset. */
	private int ignore = -1;

	private String freshKey;
	private EdgeWeights fresh;
	private Tree snapshot;

	int checks = 0;
	int mismatches = 0;
	final Map<String, Integer> mismatchesBySite = new TreeMap<>();
	String firstMismatch;

	/**
	 * @param freshFactory builds a new, independent instance (including any likelihood it reads)
	 *                     on the tree it is given
	 * @param modelKey     identifies the current values of every non-tree parameter the weights
	 *                     depend on
	 */
	public CheckedEdgeWeights(EdgeWeights delegate, Tree tree, Function<Tree, EdgeWeights> freshFactory,
			Supplier<String> modelKey) {
		this.delegate = delegate;
		this.tree = tree;
		this.freshFactory = freshFactory;
		this.modelKey = modelKey;
	}

	@Override
	public void initAndValidate() {
	}

	@Override
	public void updateByOperator() {
		ignore = -1;
		delegate.updateByOperator();
	}

	@Override
	public void updateByOperatorWithoutNode(int ignore, List<Integer> nodes) {
		this.ignore = ignore;
		delegate.updateByOperatorWithoutNode(ignore, nodes);
	}

	@Override
	public void fakeUpdateByOperator() {
		delegate.fakeUpdateByOperator();
	}

	@Override
	public void prestore() {
		delegate.prestore();
	}

	@Override
	public void reset() {
		ignore = -1;
		delegate.reset();
	}

	@Override
	public double minEdgeWeight() {
		return delegate.minEdgeWeight();
	}

	@Override
	public boolean isLogSpace() {
		return delegate.isLogSpace();
	}

	@Override
	public double getEdgeWeights(int nodeNr) {
		double actual = delegate.getEdgeWeights(nodeNr);
		compare("getEdgeWeights", new double[] { actual }, new double[] { oracle().getEdgeWeights(nodeNr) });
		return actual;
	}

	@Override
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes) {
		double[] actual = delegate.getTargetWeights(fromNodeNr, toNodes);
		EdgeWeights o = oracle();
		compare("getTargetWeights", actual, o.getTargetWeights(fromNodeNr, onSnapshot(toNodes)));
		return actual;
	}

	@Override
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes, double toHeight) {
		double[] actual = delegate.getTargetWeights(fromNodeNr, toNodes, toHeight);
		EdgeWeights o = oracle();
		compare("getTargetWeights(height)", actual, o.getTargetWeights(fromNodeNr, onSnapshot(toNodes), toHeight));
		return actual;
	}

	@Override
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes, List<Double> toHeights) {
		double[] actual = delegate.getTargetWeights(fromNodeNr, toNodes, toHeights);
		EdgeWeights o = oracle();
		compare("getTargetWeights(heights)", actual, o.getTargetWeights(fromNodeNr, onSnapshot(toNodes), toHeights));
		return actual;
	}

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
		double[] actual = delegate.getTargetWeightsInteger(fromNodeNr, toNodeNrs);
		compare("getTargetWeightsInteger", actual, oracle().getTargetWeightsInteger(fromNodeNr, toNodeNrs));
		return actual;
	}

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, double toHeight) {
		double[] actual = delegate.getTargetWeightsInteger(fromNodeNr, toNodeNrs, toHeight);
		compare("getTargetWeightsInteger(height)", actual,
				oracle().getTargetWeightsInteger(fromNodeNr, toNodeNrs, toHeight));
		return actual;
	}

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, List<Double> toHeights) {
		double[] actual = delegate.getTargetWeightsInteger(fromNodeNr, toNodeNrs, toHeights);
		compare("getTargetWeightsInteger(heights)", actual,
				oracle().getTargetWeightsInteger(fromNodeNr, toNodeNrs, toHeights));
		return actual;
	}

	/** A fresh instance on a snapshot of the current state, rebuilt only when the state changed. */
	private EdgeWeights oracle() {
		String key = treeKey() + "|" + modelKey.get() + "|" + ignore;
		if (!key.equals(freshKey)) {
			snapshot = tree.copy();
			snapshot.initArrays();
			fresh = freshFactory.apply(snapshot);
			if (ignore >= 0) {
				List<Integer> ancestors = new ArrayList<>();
				for (Node a = snapshot.getNode(ignore).getParent(); a != null; a = a.getParent()) {
					ancestors.add(a.getNr());
				}
				fresh.prestore();
				fresh.updateByOperatorWithoutNode(ignore, ancestors);
			}
			freshKey = key;
		}
		return fresh;
	}

	private String treeKey() {
		StringBuilder b = new StringBuilder();
		for (Node n : tree.getNodesAsArray()) {
			b.append(n.isRoot() ? -1 : n.getParent().getNr()).append(':')
					.append(Double.doubleToLongBits(n.getHeight())).append(',');
		}
		return b.toString();
	}

	private List<Node> onSnapshot(List<Node> nodes) {
		List<Node> mapped = new ArrayList<>(nodes.size());
		for (Node n : nodes) {
			mapped.add(snapshot.getNode(n.getNr()));
		}
		return mapped;
	}

	private void compare(String method, double[] actual, double[] expected) {
		checks++;
		int worst = -1;
		double worstDiff = 0;
		for (int k = 0; k < expected.length; k++) {
			double a = actual[k], e = expected[k];
			if (a == e || (Double.isNaN(a) && Double.isNaN(e))) {
				continue;
			}
			double diff = Math.abs(a - e) / Math.max(1.0, Math.abs(e));
			if (!(diff <= TOLERANCE) && (worst < 0 || !(diff <= worstDiff))) {
				worst = k;
				worstDiff = diff;
			}
		}
		if (worst < 0) {
			return;
		}
		mismatches++;
		String site = method + " from " + caller() + (ignore >= 0 ? " (without-node)" : "");
		mismatchesBySite.merge(site, 1, Integer::sum);
		if (firstMismatch == null) {
			firstMismatch = site + ", without " + ignore + ": entry " + worst + " of " + expected.length + " is " + actual[worst]
					+ ", fresh gives " + expected[worst];
		}
	}

	/** The operator line that made the read. */
	private static String caller() {
		for (StackTraceElement e : Thread.currentThread().getStackTrace()) {
			if (e.getClassName().startsWith("targetedbeast.operators.")) {
				return e.getClassName().substring("targetedbeast.operators.".length()) + ":" + e.getLineNumber();
			}
		}
		return "?";
	}

	String report() {
		StringBuilder b = new StringBuilder(checks + " reads checked, " + mismatches + " mismatched");
		mismatchesBySite.forEach((site, n) -> b.append("\n    ").append(n).append("x ").append(site));
		if (firstMismatch != null) {
			b.append("\n    first: ").append(firstMismatch);
		}
		return b.toString();
	}
}
