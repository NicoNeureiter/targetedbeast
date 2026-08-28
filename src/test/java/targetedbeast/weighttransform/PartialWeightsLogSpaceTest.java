package targetedbeast.weighttransform;

import org.junit.Assert;
import org.junit.Test;

import targetedbeast.edgeweights.ConstantWeights;
import targetedbeast.edgeweights.PartialWeights;

/** The logSpace switch on PartialWeights, and the isLogSpace default on EdgeWeights. */
public class PartialWeightsLogSpaceTest {

	@Test
	public void edgeWeightsDefaultToRealSpace() {
		ConstantWeights w = new ConstantWeights();
		Assert.assertFalse(w.isLogSpace());
	}

	@Test
	public void partialWeightsDefaultToRealSpace() {
		PartialWeights w = new PartialWeights();
		// Reading the input directly - initAndValidate needs a full alignment and tree.
		Assert.assertFalse(w.logSpaceInput.get());
	}

	@Test
	public void partialWeightsReportLogSpaceWhenEnabled() {
		PartialWeights w = new PartialWeights();
		w.logSpaceInput.setValue(true, w);
		Assert.assertTrue(w.isLogSpace());
	}
}
