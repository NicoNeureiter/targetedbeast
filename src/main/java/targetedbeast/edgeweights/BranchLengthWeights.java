package targetedbeast.edgeweights;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Distribution;
import beast.base.inference.State;

@Description("Source-edge weights equal to the branch lengths, for use as SourceTargetWeights' source; "
		+ "an edgeWeightTransform's exponent c then gives weights proportional to length^c. "
		+ "It has no target weights.")
public class BranchLengthWeights extends Distribution implements EdgeWeights {

	final public Input<Tree> treeInput = new Input<>("tree", "the tree whose branch lengths are the weights",
			Validate.REQUIRED);

	@Override
	public void initAndValidate() {
		// Contributes nothing to the posterior, and is never recalculated, so it is never set otherwise.
		logP = 0;
	}

	@Override
	public double getEdgeWeights(int nodeNr) {
		return treeInput.get().getNode(nodeNr).getLength();
	}

	@Override
	public double minEdgeWeight() {
		return 0.0;
	}

	// Weights are read off the tree, so there is no cache to update.

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
	public void prestore() {
	}

	@Override
	public void reset() {
	}

	@Override
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes) {
		throw new UnsupportedOperationException("BranchLengthWeights has no target weights; use it as a source");
	}

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
		throw new UnsupportedOperationException("BranchLengthWeights has no target weights; use it as a source");
	}

	@Override
	protected boolean requiresRecalculation() {
		return false;
	}

	@Override
	public List<String> getArguments() {
		return new ArrayList<>();
	}

	@Override
	public List<String> getConditions() {
		return new ArrayList<>();
	}

	@Override
	public void sample(State state, Random random) {
	}
}
