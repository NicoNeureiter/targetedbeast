package targetedbeast.edgeweights;

import java.io.PrintStream;
import java.util.*;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Distribution;
import beast.base.inference.State;


@Description("Keeps track of the consensus sequences and the number of mutations between consensus sequences along edges"
		+ "Consensus weights is a distribution to ensure that it is updated correctly")
public class ConstantWeights extends Distribution implements EdgeWeights {
	
    final public Input<TreeInterface> treeInput = new Input<>("tree", "phylogenetic beast.tree with sequence data in the leafs", Validate.REQUIRED);
    int nodeCount;
    double[][] edgeMutations;

	@Override
	public void initAndValidate() {
        nodeCount = treeInput.get().getNodeCount();
		edgeMutations = new double[2][nodeCount];
        Arrays.fill(edgeMutations[0], 1.0);
        Arrays.fill(edgeMutations[1], 1.0);
	}

	
	public void updateWeights() {
	}

	public void updateByOperator() {
	}

	public void updateByOperatorWithoutNode(int ignore, List<Integer> nodes) {
	}

	public void fakeUpdateByOperator() {
	}

	/**
	 * check state for changed variables and update temp results if necessary *
	 */
	@Override
	protected boolean requiresRecalculation() {
        return true;
	}

	@Override
	public void store() {
		super.store();
	}

	public void prestore() {
	}

	public void reset() {}

	public void unstore() {}

	@Override
	public void restore() {
	}


	public double getEdgeMutations(int i) {
		return 1.0;
	}
	
	public boolean getChanged(int i) {
		return false;
	}

	@Override
	public double getEdgeWeights(int nodeNr) {
		return 1.0;
	}

	@Override
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodeNrs) {
		double[] distances = new double[toNodeNrs.size()];
        Arrays.fill(distances, 1.0);
		return distances;
	}
	
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
		double[] distances = new double[toNodeNrs.size()];
        Arrays.fill(distances, 1.0);
		return distances;
	}

	@Override
	public List<String> getArguments() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<String> getConditions() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void sample(State state, Random random) {
		// TODO Auto-generated method stub
	}
	
	@Override
	public void init(PrintStream out) {
	}

	@Override
	public double minEdgeWeight() {
		return 1.0;
	}


}
