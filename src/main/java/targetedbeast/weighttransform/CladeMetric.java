package targetedbeast.weighttransform;

import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;

import beast.base.core.BEASTObject;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.evolution.tree.TreeMetric;

/** A tree metric over rooted clades, matched by the taxa below them; leaves matched by taxon ID. */
public abstract class CladeMetric extends BEASTObject implements TreeMetric {

	private Map<String, Integer> leafIndex;
	private Map<BitSet, Double> reference;

	/** Distance between two trees' clades, each mapped to the length of the branch above it. */
	protected abstract double distance(Map<BitSet, Double> a, Map<BitSet, Double> b);

	@Override
	public void initAndValidate() {
	}

	@Override
	public double distance(TreeInterface tree1, TreeInterface tree2) {
		return distance(clades(tree1), clades(tree2));
	}

	@Override
	public void setReference(TreeInterface ref) {
		leafIndex = null;
		reference = clades(ref);
	}

	@Override
	public double distance(TreeInterface tree) {
		return distance(reference, clades(tree));
	}

	/** Branch length above each non-root node, keyed by the taxa below it. */
	private Map<BitSet, Double> clades(TreeInterface tree) {
		if (leafIndex == null) {
			leafIndex = new HashMap<>();
			// From the root rather than getExternalNodes(): a Tree.copy() has no node array.
			for (Node leaf : tree.getRoot().getAllLeafNodes()) {
				leafIndex.put(leaf.getID(), leafIndex.size());
			}
		}
		Map<BitSet, Double> clades = new HashMap<>();
		collect(tree.getRoot(), clades);
		return clades;
	}

	private BitSet collect(Node node, Map<BitSet, Double> clades) {
		BitSet taxa = new BitSet(leafIndex.size());
		if (node.isLeaf()) {
			Integer index = leafIndex.get(node.getID());
			if (index == null) {
				throw new IllegalArgumentException("taxon " + node.getID() + " is not in the first tree");
			}
			taxa.set(index);
		} else {
			for (Node child : node.getChildren()) {
				taxa.or(collect(child, clades));
			}
		}
		if (!node.isRoot()) {
			clades.put(taxa, node.getLength());
		}
		return taxa;
	}
}
