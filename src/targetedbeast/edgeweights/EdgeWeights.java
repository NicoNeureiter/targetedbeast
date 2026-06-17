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
	 * Get per-candidate target weights for moving one lineage to a list of targets.
	 *
	 * Implementations must return intrinsic weights for each candidate move. In particular,
	 * a candidate weight must not depend on the current tree state's baseline score or on
	 * centering against the full candidate set, because targeted operators square and
	 * renormalize these values and may compare forward and backward masses built from
	 * different states or candidate lists.
	 *
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
	 * Get per-candidate target weights for moving one lineage to a list of targets.
	 *
	 * The same intrinsic-weight contract as above applies to these integer-indexed targets.
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
