package targetedbeast.weighttransform;

import beast.base.evolution.tree.TreeInterface;
import beast.base.evolution.tree.TreeMetric;

/**
 * A tree metric that can precompute a tree's comparison form once, so that comparing one tree
 * against several others (or the same reference many times) does not redo that work.
 */
public interface PreparedTreeMetric extends TreeMetric {

	/** The tree's comparison form; valid until the tree changes. */
	Object prepare(TreeInterface tree);

	/** Distance between two prepared forms of this metric. */
	double distance(Object prepared1, Object prepared2);
}
