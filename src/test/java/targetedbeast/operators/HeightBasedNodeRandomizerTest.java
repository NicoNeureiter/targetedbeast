package targetedbeast.operators;

import org.junit.Assert;
import org.junit.Test;

import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.OperatorSchedule;

public class HeightBasedNodeRandomizerTest {

	private static HeightBasedNodeRandomizer operator(String newick, double percentage) {
		Tree tree = new TreeParser(newick, false, true, true, 1);
		HeightBasedNodeRandomizer op = new HeightBasedNodeRandomizer();
		op.initByName("tree", tree, "weight", 1.0, "percentage", percentage);
		OperatorSchedule schedule = new OperatorSchedule();
		schedule.initAndValidate();
		op.setOperatorSchedule(schedule);
		return op;
	}

	/** A proposal that found no candidate must make percentage grow, not shrink towards a dead end. */
	@Test
	public void noCandidatesGrowsPercentage() {
		HeightBasedNodeRandomizer op = operator("((A:1,B:1):1,C:2);", 0.1);
		Assert.assertEquals(Double.NEGATIVE_INFINITY, op.proposal(), 0.0);
		double before = op.getCoercableParameterValue();
		op.optimize(Double.NEGATIVE_INFINITY);
		Assert.assertTrue(op.getCoercableParameterValue() > before);
	}

	/** Rejections cannot push percentage below one internal node's worth. */
	@Test
	public void percentageStaysAboveFloor() {
		HeightBasedNodeRandomizer op = operator("(((A:1,B:1):1,C:2):1,(D:1,E:1):2);", 0.25);
		op.setCoercableParameterValue(1e-6);
		op.optimize(-100.0);
		Assert.assertTrue(op.getCoercableParameterValue() >= 1.0 / 4 - 1e-12);
	}
}
