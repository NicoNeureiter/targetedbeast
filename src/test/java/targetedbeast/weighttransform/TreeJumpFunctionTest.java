package targetedbeast.weighttransform;

import org.junit.Assert;
import org.junit.Test;

import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beastlabs.evolution.tree.RNNIMetric;

/**
 * {@link TreeJumpFunction} against BEASTLabs' {@link RNNIMetric}, the metric GridTuner's RNNI
 * reward wiring actually uses.
 */
public class TreeJumpFunctionTest {

	private static final String[] TAXA = { "A", "B", "C", "D", "E" };
	private static final String NEWICK_BEFORE = "(((A:1,B:1):1,C:2):1,(D:1,E:1):2);";
	private static final String NEWICK_AFTER = "(((A:1,C:1):1,B:2):1,(D:1,E:1):2);";

	/** First call of a pair only snapshots: it must not report a jump against the initial tree. */
	@Test
	public void firstCallReturnsZero() {
		Tree tree = new TreeParser(NEWICK_BEFORE, false, true, true, 1);
		TreeJumpFunction fn = newFunction(tree);

		Assert.assertEquals(0.0, fn.getArrayValue(), 0.0);
	}

	/**
	 * Proves the snapshot is a frozen copy, not a live-mutating reference: mutate the tree between
	 * the two calls of a pair and check the reported jump matches the true distance between the
	 * pre-mutation topology and the post-mutation one - not zero, which is what a reference-aliasing
	 * bug would produce (both "before" and "after" pointing at the same, already-mutated tree).
	 */
	@Test
	public void secondCallMeasuresJumpFromFrozenSnapshot() {
		Tree tree = new TreeParser(NEWICK_BEFORE, false, true, true, 1);
		Tree original = tree.copy();
		TreeJumpFunction fn = newFunction(tree);

		fn.getArrayValue(); // snapshot taken here

		Tree mutated = new TreeParser(NEWICK_AFTER, false, true, true, 1);
		tree.assignFrom(mutated); // mutate the *same* tree object in place

		double reported = fn.getArrayValue();

		RNNIMetric metric = new RNNIMetric(TAXA);
		double expected = metric.distance(original, tree);

		Assert.assertNotEquals("a reference-aliasing bug would report zero here", 0.0, reported, 0.0);
		Assert.assertEquals(expected, reported, 1e-12);
	}

	/** Calls alternate strictly: after a measuring call, the next call snapshots again. */
	@Test
	public void callsAlternateBetweenSnapshotAndMeasurement() {
		Tree tree = new TreeParser(NEWICK_BEFORE, false, true, true, 1);
		TreeJumpFunction fn = newFunction(tree);

		Assert.assertEquals(0.0, fn.getArrayValue(), 0.0); // pair 1: snapshot
		Assert.assertEquals(0.0, fn.getArrayValue(), 0.0); // pair 1: measure (tree unchanged)
		Assert.assertEquals(0.0, fn.getArrayValue(), 0.0); // pair 2: snapshot again
	}

	private static TreeJumpFunction newFunction(Tree tree) {
		RNNIMetric metric = new RNNIMetric(TAXA);
		TreeJumpFunction fn = new TreeJumpFunction();
		fn.initByName("tree", tree, "treeMetric", metric);
		return fn;
	}
}
