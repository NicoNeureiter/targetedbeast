package targetedbeast.weighttransform;

import org.junit.Assert;
import org.junit.Test;

import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;

public class KuhnerFelsensteinMetricTest {

	private static final double TOL = 1e-12;

	private static Tree tree(String newick) {
		return new TreeParser(newick, false, true, true, 1);
	}

	@Test
	public void identicalTreesAreAtZero() {
		String newick = "(((A:1,B:1):1,C:2):1,(D:1,E:1):2);";
		Assert.assertEquals(0.0, new KuhnerFelsensteinMetric().distance(tree(newick), tree(newick)), TOL);
	}

	/** Lengthening AB's branch by 0.5, and C's and DE's with it to keep the tips level, changes three clades. */
	@Test
	public void branchLengthChangeCountsOnce() {
		Tree a = tree("(((A:1,B:1):1,C:2):1,(D:1,E:1):2);");
		Tree b = tree("(((A:1,B:1):1.5,C:2.5):1,(D:1,E:1):2.5);");
		// clades AB (+0.5), C (+0.5), DE (+0.5)
		Assert.assertEquals(Math.sqrt(3 * 0.25), new KuhnerFelsensteinMetric().distance(a, b), TOL);
	}

	/** Swapping B and C: B and C change length, clades AB and AC each exist in one tree only. */
	@Test
	public void topologyChangeCountsUnmatchedClades() {
		Tree a = tree("(((A:1,B:1):1,C:2):1,(D:1,E:1):2);");
		Tree b = tree("(((A:1,C:1):1,B:2):1,(D:1,E:1):2);");
		Assert.assertEquals(2.0, new KuhnerFelsensteinMetric().distance(a, b), TOL);
	}

	@Test
	public void referenceMatchesPairwise() {
		Tree a = tree("(((A:1,B:1):1,C:2):1,(D:1,E:1):2);");
		Tree b = tree("(((A:1,C:1):1,B:2):1,(D:1,E:1):2);");
		KuhnerFelsensteinMetric metric = new KuhnerFelsensteinMetric();
		metric.setReference(a);
		Assert.assertEquals(2.0, metric.distance(b), TOL);
	}

	@Test
	public void robinsonFouldsCountsUnmatchedClades() {
		Tree a = tree("(((A:1,B:1):1,C:2):1,(D:1,E:1):2);");
		Tree b = tree("(((A:1,C:1):1,B:2):1,(D:1,E:1):2);");
		Assert.assertEquals(0.0, new RobinsonFouldsMetric().distance(a, tree("(((A:1,B:1):1.5,C:2.5):1,(D:1,E:1):2.5);")), 0.0);
		Assert.assertEquals(2.0, new RobinsonFouldsMetric().distance(a, b), 0.0);
	}

	@Test
	public void treeHeightIsRootHeightDifference() {
		Tree a = tree("(((A:1,B:1):1,C:2):1,(D:1,E:1):2);");
		Tree b = tree("(((A:1,B:1):1.5,C:2.5):1,(D:1,E:1):2.5);");
		Assert.assertEquals(0.5, new TreeHeightMetric().distance(a, b), TOL);
	}
}
