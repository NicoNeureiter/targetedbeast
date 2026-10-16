package targetedbeast.weighttransform;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;

@Description("Absolute difference in total branch length between two trees.")
public class TreeLengthMetric extends BEASTObject implements PreparedTreeMetric {

	private double referenceLength;

	@Override
	public void initAndValidate() {
	}

	@Override
	public double distance(TreeInterface tree1, TreeInterface tree2) {
		return Math.abs(length(tree1.getRoot()) - length(tree2.getRoot()));
	}

	@Override
	public void setReference(TreeInterface ref) {
		referenceLength = length(ref.getRoot());
	}

	@Override
	public double distance(TreeInterface tree) {
		return Math.abs(length(tree.getRoot()) - referenceLength);
	}

	/** From the root rather than the node array: a Tree.copy() has none. */
	private static double length(Node node) {
		double sum = node.isRoot() ? 0.0 : node.getLength();
		for (Node child : node.getChildren()) {
			sum += length(child);
		}
		return sum;
	}

	@Override
	public Object prepare(TreeInterface tree) {
		return length(tree.getRoot());
	}

	@Override
	public double distance(Object prepared1, Object prepared2) {
		return Math.abs((Double) prepared1 - (Double) prepared2);
	}
}
