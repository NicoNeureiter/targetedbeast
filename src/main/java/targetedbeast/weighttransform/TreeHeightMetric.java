package targetedbeast.weighttransform;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.evolution.tree.TreeInterface;

@Description("Absolute difference in root height between two trees.")
public class TreeHeightMetric extends BEASTObject implements PreparedTreeMetric {

	private double referenceHeight;

	@Override
	public void initAndValidate() {
	}

	@Override
	public double distance(TreeInterface tree1, TreeInterface tree2) {
		return Math.abs(tree1.getRoot().getHeight() - tree2.getRoot().getHeight());
	}

	@Override
	public void setReference(TreeInterface ref) {
		referenceHeight = ref.getRoot().getHeight();
	}

	@Override
	public double distance(TreeInterface tree) {
		return Math.abs(tree.getRoot().getHeight() - referenceHeight);
	}

	@Override
	public Object prepare(TreeInterface tree) {
		return tree.getRoot().getHeight();
	}

	@Override
	public double distance(Object prepared1, Object prepared2) {
		return Math.abs((Double) prepared1 - (Double) prepared2);
	}
}
