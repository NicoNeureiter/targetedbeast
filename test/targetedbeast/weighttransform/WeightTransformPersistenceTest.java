package targetedbeast.weighttransform;

import java.io.PrintWriter;
import java.io.StringWriter;

import org.json.JSONException;
import org.json.JSONObject;
import org.junit.Assert;
import org.junit.Test;

import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;
import targetedbeast.edgeweights.ConstantWeights;
import targetedbeast.operators.TargetedWilsonBalding;

/** Round-tripping the transform tuning through the operator's state-file entry. */
public class WeightTransformPersistenceTest {

	private static TargetedWilsonBalding operatorWithTransforms(
			PowerOffsetTransform target, PowerOffsetTransform edge) {
		Tree tree = new Tree("((A:1.0,B:1.0):1.0,C:2.0):0.0;");
		ConstantWeights weights = new ConstantWeights();
		weights.initByName("tree", tree);

		TargetedWilsonBalding op = new TargetedWilsonBalding();
		op.initByName("tree", tree, "weight", 1.0, "edgeWeights", weights,
				"targetWeightTransform", target, "edgeWeightTransform", edge);
		op.setID("targetedWB");
		return op;
	}

	private static PowerOffsetTransform transform(double exponent, double offset) {
		PowerOffsetTransform t = new PowerOffsetTransform();
		t.initByName("exponent", exponent, "offset", offset, "optimise", true);
		return t;
	}

	@Test
	public void adaptationStateSurvivesAStoreRestoreRoundTrip() throws JSONException {
		Randomizer.setSeed(7);

		PowerOffsetTransform target = transform(2.0, 0.01);
		PowerOffsetTransform edge = transform(2.0, 0.01);
		TargetedWilsonBalding source = operatorWithTransforms(target, edge);

		// Drive some adaptation so the stored values differ from the defaults.
		for (int i = 0; i < 500; i++) {
			target.startProposal();
			target.apply(new double[] { 1.0, 2.0, 3.0 });
			target.optimize(Math.log(0.5));
		}

		StringWriter buffer = new StringWriter();
		PrintWriter out = new PrintWriter(buffer);
		source.storeToFile(out);
		out.flush();

		PowerOffsetTransform restoredTarget = transform(2.0, 0.01);
		PowerOffsetTransform restoredEdge = transform(2.0, 0.01);
		TargetedWilsonBalding restored = operatorWithTransforms(restoredTarget, restoredEdge);
		restored.restoreFromFile(new JSONObject(buffer.toString()));

		Assert.assertArrayEquals(target.getCoefficients(), restoredTarget.getCoefficients(), 1e-12);
		Assert.assertEquals(target.getAdaptationCount(), restoredTarget.getAdaptationCount());
		Assert.assertEquals(target.getAcceptanceBaseline(), restoredTarget.getAcceptanceBaseline(), 1e-12);
		Assert.assertEquals(edge.getAdaptationCount(), restoredEdge.getAdaptationCount());
	}

	/** An operator without transforms must still produce a state entry BEAST can read back. */
	@Test
	public void operatorWithoutTransformsStillRoundTrips() throws JSONException {
		Tree tree = new Tree("((A:1.0,B:1.0):1.0,C:2.0):0.0;");
		ConstantWeights weights = new ConstantWeights();
		weights.initByName("tree", tree);

		TargetedWilsonBalding op = new TargetedWilsonBalding();
		op.initByName("tree", tree, "weight", 1.0, "edgeWeights", weights);
		op.setID("plainWB");

		StringWriter buffer = new StringWriter();
		PrintWriter out = new PrintWriter(buffer);
		op.storeToFile(out);
		out.flush();

		JSONObject json = new JSONObject(buffer.toString());
		Assert.assertEquals("plainWB", json.getString("id"));
		Assert.assertFalse(json.has("transforms"));

		op.restoreFromFile(json);
	}
}
