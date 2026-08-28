package targetedbeast.edgeweights;

import java.util.List;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;

@Description("Interface for weights to be used to use weighted tree operations")
public interface EdgeWeights {
	
	
//	void updateWeights ();

    void updateByOperator();

	void updateByOperatorWithoutNode(int ignore, List<Integer> nodes);
	
	// byte[] getNodeConsensus(int NodeNo);

	
	void fakeUpdateByOperator();

	void prestore();

	void reset();

	/**
	 * Get the edge weights for a node for operations
	 * 
	 * @param nodeNr
	 * @return
	 */
	double getEdgeWeights(int nodeNr);
	
	/**
	 * Get the distance from one node to a list of nodes
	 * @param fromNodeNr
	 * @param toNodeNrs
	 * @return
	 */
	double[] getTargetWeights(int fromNodeNr, List<Node> toNodeNrs);

	default double[] getTargetWeights(int fromNodeNr, List<Node> toNodeNrs, double toHeight) {
        return getTargetWeights(fromNodeNr, toNodeNrs);
    }

	default double[] getTargetWeights(int fromNodeNr, List<Node> toNodeNrs, List<Double> toHeights) {
        return getTargetWeights(fromNodeNr, toNodeNrs);
    }
	
	/**
	 * Get the distance from one node to a list of nodes
	 * 
	 * @param fromNodeNr
	 * @param toNodeNrs
	 * @return
	 */
	double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs);

	default double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, double toHeight) {
        return getTargetWeightsInteger(fromNodeNr, toNodeNrs);
    }

	default double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, List<Double> toHeights) {
        return getTargetWeightsInteger(fromNodeNr, toNodeNrs);
    }

	double minEdgeWeight();

}
