package targetedbeast.weighttransform;

import org.junit.Assert;
import org.junit.Test;

import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.core.BEASTObject;
import beast.base.core.Function;

public class DisplacementRewardTest {

	private static final double TOL = 1e-12;

	/** A settable scalar, standing in for e.g. the likelihood. */
	public static class Value extends BEASTObject implements Function {
		double v;

		Value(double v) {
			this.v = v;
		}

		void set(double v) {
			this.v = v;
		}

		@Override
		public void initAndValidate() {
		}

		@Override
		public int getDimension() {
			return 1;
		}

		@Override
		public double getArrayValue(int dim) {
			return v;
		}
	}

	/** Each component is normalised by its own scale, so on the first measurement each contributes 1. */
	@Test
	public void componentsCountEqually() {
		Tree tree = new TreeParser("(((A:1,B:1):1,C:2):1,(D:1,E:1):2);", false, true, true, 1);
		Value x = new Value(0.0);
		DisplacementReward r = new DisplacementReward();
		r.initByName("tree", tree, "treeMetric", new KuhnerFelsensteinMetric(), "parameter", x, "normalisation", 1);
		r.beginProposal();
		tree.assignFrom(new TreeParser("(((A:1,C:1):1,B:2):1,(D:1,E:1):2);", false, true, true, 1));
		x.set(10.0);
		Assert.assertEquals(2.0, r.reward(), TOL);
		Assert.assertEquals(2.0, r.reward(), TOL); // cached for a second tuner
	}

	/** In reference mode a move back towards where the chain was is penalised. */
	@Test
	public void undoingAMoveIsNegative() {
		Value x = new Value(0.0);
		DisplacementReward r = new DisplacementReward();
		r.initByName("parameter", x, "references", 1, "referenceSpacing", 2, "normalisation", 1);
		double[] path = { 1, 3, 5, 3 };
		double[] expected = { 0, 0, 1, -1 };
		for (int i = 0; i < path.length; i++) {
			r.beginProposal();
			x.set(path[i]);
			Assert.assertEquals("step " + i, expected[i], r.reward(), TOL);
		}
	}

	/** A rejected proposal (state unchanged) earns nothing in either mode. */
	@Test
	public void rejectedProposalEarnsNothing() {
		Value x = new Value(2.0);
		DisplacementReward r = new DisplacementReward();
		r.initByName("parameter", x, "normalisation", 1);
		r.beginProposal();
		x.set(3.0);
		r.reward();
		r.beginProposal();
		Assert.assertEquals(0.0, r.reward(), TOL);
	}
}
