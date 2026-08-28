package targetedbeast.edgeweights;

import java.util.*;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.tree.Node;
import beast.base.inference.Distribution;
import beast.base.inference.State;

@Description("Combines multiple EdgeWeights by multiplying them together. " +
		"This is useful when you have multiple partitions/alignments for the same tree " +
		"and want to use all of them to guide tree operations.")
public class CombinedEdgeWeights extends Distribution implements EdgeWeights {
	
	final public Input<List<EdgeWeights>> edgeWeightsInput = new Input<>(
			"edgeWeights", 
			"List of EdgeWeights to combine by multiplication",
			new ArrayList<>(),
			Validate.REQUIRED);

	private List<EdgeWeights> edgeWeightsList;
	private double minWeight;

	@Override
	public void initAndValidate() {
		edgeWeightsList = edgeWeightsInput.get();
		
		if (edgeWeightsList.isEmpty()) {
			throw new IllegalArgumentException("CombinedEdgeWeights requires at least one EdgeWeights input.");
		}
		
		// Use the maximum of all minEdgeWeights as the combined minimum
		minWeight = 0.0;
		for (EdgeWeights ew : edgeWeightsList) {
			minWeight = Math.max(minWeight, ew.minEdgeWeight());
		}
	}

	@Override
	public void updateByOperator() {
		for (EdgeWeights ew : edgeWeightsList) {
			ew.updateByOperator();
		}
	}

	@Override
	public void updateByOperatorWithoutNode(int ignore, List<Integer> nodes) {
		for (EdgeWeights ew : edgeWeightsList) {
			ew.updateByOperatorWithoutNode(ignore, nodes);
		}
	}

	@Override
	public void fakeUpdateByOperator() {
		for (EdgeWeights ew : edgeWeightsList) {
			ew.fakeUpdateByOperator();
		}
	}

	@Override
	public void prestore() {
		for (EdgeWeights ew : edgeWeightsList) {
			ew.prestore();
		}
	}

	@Override
	public void reset() {
		for (EdgeWeights ew : edgeWeightsList) {
			ew.reset();
		}
	}

	@Override
	protected boolean requiresRecalculation() {
		// Requires recalculation if any child requires it
		for (EdgeWeights ew : edgeWeightsList) {
			if (ew instanceof Distribution) {
				if (((Distribution) ew).isDirtyCalculation()) {
					return true;
				}
			}
		}
		return false;
	}

	@Override
	public void store() {
		super.store();
		// Child EdgeWeights that are Distributions will have their store() called
		// automatically by the MCMC framework
	}

	@Override
	public void restore() {
		super.restore();
		// Child EdgeWeights that are Distributions will have their restore() called
		// automatically by the MCMC framework
	}

	@Override
	public double getEdgeWeights(int nodeNr) {
		// Multiply edge weights from all sources
		double combined = 1.0;
		for (EdgeWeights ew : edgeWeightsList) {
			combined *= ew.getEdgeWeights(nodeNr);
		}
		return combined;
	}

	@Override
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes) {
		if (toNodes.isEmpty()) {
			return new double[0];
		}
		return combineTargetWeights(fromNodeNr, toNodes, edgeWeights -> edgeWeights.getTargetWeights(fromNodeNr, toNodes));
	}

	@Override
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes, double toHeight) {
		if (toNodes.isEmpty()) {
			return new double[0];
		}
		return combineTargetWeights(fromNodeNr, toNodes,
				edgeWeights -> edgeWeights.getTargetWeights(fromNodeNr, toNodes, toHeight));
	}

	@Override
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes, List<Double> toHeights) {
		if (toNodes.isEmpty()) {
			return new double[0];
		}
		return combineTargetWeights(fromNodeNr, toNodes,
				edgeWeights -> edgeWeights.getTargetWeights(fromNodeNr, toNodes, toHeights));
	}

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
		if (toNodeNrs.isEmpty()) {
			return new double[0];
		}
		return combineTargetWeights(fromNodeNr, toNodeNrs,
				edgeWeights -> edgeWeights.getTargetWeightsInteger(fromNodeNr, toNodeNrs));
	}

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, double toHeight) {
		if (toNodeNrs.isEmpty()) {
			return new double[0];
		}
		return combineTargetWeights(fromNodeNr, toNodeNrs,
				edgeWeights -> edgeWeights.getTargetWeightsInteger(fromNodeNr, toNodeNrs, toHeight));
	}

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, List<Double> toHeights) {
		if (toNodeNrs.isEmpty()) {
			return new double[0];
		}
		return combineTargetWeights(fromNodeNr, toNodeNrs,
				edgeWeights -> edgeWeights.getTargetWeightsInteger(fromNodeNr, toNodeNrs, toHeights));
	}

	private double[] combineTargetWeights(int fromNodeNr, List<?> targets,
			java.util.function.Function<EdgeWeights, double[]> getter) {
		double[] combined = getter.apply(edgeWeightsList.get(0));
		for (int i = 1; i < edgeWeightsList.size(); i++) {
			double[] weights = getter.apply(edgeWeightsList.get(i));
			for (int k = 0; k < combined.length; k++) {
				combined[k] *= weights[k];
			}
		}
		renormalize(combined);
		return combined;
	}

	private void renormalize(double[] weights) {
		double sum = 0.0;
		for (double weight : weights) {
			sum += weight;
		}
		if (sum > 0.0) {
			for (int k = 0; k < weights.length; k++) {
				weights[k] /= sum;
			}
		}
	}

	@Override
	public double minEdgeWeight() {
		return minWeight;
	}

	@Override
	public List<String> getArguments() {
		return null;
	}

	@Override
	public List<String> getConditions() {
		return null;
	}

	@Override
	public void sample(State state, Random random) {
		// Not used for edge weights
	}

}
