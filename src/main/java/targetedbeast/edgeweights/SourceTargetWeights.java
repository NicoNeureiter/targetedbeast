package targetedbeast.edgeweights;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.tree.Node;
import beast.base.inference.Distribution;
import beast.base.inference.State;

@Description("Takes the source-edge weights from one EdgeWeights and the target weights from another, "
		+ "e.g. PCA distances to choose the edge to move and partial likelihoods to choose where it goes. "
		+ "Both must sit in the posterior themselves, so that MCMC stores and restores them.")
public class SourceTargetWeights extends Distribution implements EdgeWeights {

	final public Input<EdgeWeights> sourceInput = new Input<>("source",
			"weights for choosing the edge to move (getEdgeWeights)", Validate.REQUIRED);
	final public Input<EdgeWeights> targetInput = new Input<>("target",
			"weights for choosing where to move it (getTargetWeights)", Validate.REQUIRED);

	private EdgeWeights source;
	private EdgeWeights target;

	@Override
	public void initAndValidate() {
		source = sourceInput.get();
		target = targetInput.get();
		// Contributes nothing to the posterior, and is never recalculated, so it is never set otherwise.
		logP = 0;
	}

	// Cache protocol calls go to both, so each sees the sequence it would see on its own.

	@Override
	public void updateByOperator() {
		source.updateByOperator();
		target.updateByOperator();
	}

	@Override
	public void updateByOperatorWithoutNode(int ignore, List<Integer> nodes) {
		source.updateByOperatorWithoutNode(ignore, nodes);
		target.updateByOperatorWithoutNode(ignore, nodes);
	}

	@Override
	public void fakeUpdateByOperator() {
		source.fakeUpdateByOperator();
		target.fakeUpdateByOperator();
	}

	@Override
	public void prestore() {
		source.prestore();
		target.prestore();
	}

	@Override
	public void reset() {
		source.reset();
		target.reset();
	}

	@Override
	public double getEdgeWeights(int nodeNr) {
		return source.getEdgeWeights(nodeNr);
	}

	@Override
	public double minEdgeWeight() {
		return source.minEdgeWeight();
	}

	@Override
	public boolean isLogSpace() {
		return target.isLogSpace();
	}

	@Override
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes) {
		return target.getTargetWeights(fromNodeNr, toNodes);
	}

	@Override
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes, double toHeight) {
		return target.getTargetWeights(fromNodeNr, toNodes, toHeight);
	}

	@Override
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes, List<Double> toHeights) {
		return target.getTargetWeights(fromNodeNr, toNodes, toHeights);
	}

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
		return target.getTargetWeightsInteger(fromNodeNr, toNodeNrs);
	}

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, double toHeight) {
		return target.getTargetWeightsInteger(fromNodeNr, toNodeNrs, toHeight);
	}

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, List<Double> toHeights) {
		return target.getTargetWeightsInteger(fromNodeNr, toNodeNrs, toHeights);
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
