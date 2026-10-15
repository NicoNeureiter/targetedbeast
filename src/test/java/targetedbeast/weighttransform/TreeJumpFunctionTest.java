package targetedbeast.weighttransform;

import org.junit.Assert;
import org.junit.Test;

import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beastlabs.evolution.tree.RNNIMetric;

/**
 * {@link TreeJumpFunction} against BEASTLabs' {@link RNNIMetric} and the in-chain KF metric.
 */
public class TreeJumpFunctionTest {

	private static final String[] TAXA = { "A", "B", "C", "D", "E" };
	private static final String NEWICK_BEFORE = "(((A:1,B:1):1,C:2):1,(D:1,E:1):2);";
	private static final String NEWICK_AFTER = "(((A:1,C:1):1,B:2):1,(D:1,E:1):2);";

	@Test
	public void noJumpWithoutAMove() {
		Tree tree = new TreeParser(NEWICK_BEFORE, false, true, true, 1);
		TreeJumpFunction fn = newFunction(tree);
		fn.beginProposal();
		Assert.assertEquals(0.0, fn.jump(), 0.0);
	}

	/**
	 * The snapshot is a frozen copy, not a live reference: mutate the tree after beginProposal()
	 * and the jump must be the true distance, not zero, which a reference-aliasing bug would give.
	 */
	@Test
	public void jumpIsMeasuredFromAFrozenSnapshot() {
		Tree tree = new TreeParser(NEWICK_BEFORE, false, true, true, 1);
		Tree original = tree.copy();
		TreeJumpFunction fn = newFunction(tree);
		fn.beginProposal();
		tree.assignFrom(new TreeParser(NEWICK_AFTER, false, true, true, 1));

		double expected = new RNNIMetric(TAXA).distance(original, tree);
		Assert.assertNotEquals("a reference-aliasing bug would report zero here", 0.0, expected, 0.0);
		Assert.assertEquals(expected, fn.jump(), 1e-12);
	}

	/** MCMC skips optimize() on some steps: a new beginProposal() must start over, not pair with a stale one. */
	@Test
	public void beginProposalStartsOver() {
		Tree tree = new TreeParser(NEWICK_BEFORE, false, true, true, 1);
		TreeJumpFunction fn = newFunction(tree);
		fn.beginProposal();
		tree.assignFrom(new TreeParser(NEWICK_AFTER, false, true, true, 1));
		fn.beginProposal(); // the previous proposal's outcome was never reported
		Assert.assertEquals(0.0, fn.jump(), 0.0);
	}

	/** The in-chain KF metric works on the snapshot, a Tree.copy() without a node array. */
	@Test
	public void kfJumpOnSnapshot() {
		Tree tree = new TreeParser(NEWICK_BEFORE, false, true, true, 1);
		TreeJumpFunction fn = new TreeJumpFunction();
		fn.initByName("tree", tree, "treeMetric", new KuhnerFelsensteinMetric());
		fn.beginProposal();
		tree.assignFrom(new TreeParser(NEWICK_AFTER, false, true, true, 1));
		Assert.assertEquals(2.0, fn.jump(), 1e-12);
	}

	private static TreeJumpFunction newFunction(Tree tree) {
		RNNIMetric metric = new RNNIMetric(TAXA);
		TreeJumpFunction fn = new TreeJumpFunction();
		fn.initByName("tree", tree, "treeMetric", metric);
		return fn;
	}
}
