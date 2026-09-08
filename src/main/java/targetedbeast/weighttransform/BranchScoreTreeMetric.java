package targetedbeast.weighttransform;

import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.evolution.tree.TreeMetric;

@Description("Kuhner-Felsenstein / branch-score distance between two trees: sqrt of the sum of "
		+ "squared branch-length differences over the union of clades (leaf-set bipartitions), "
		+ "with a clade present in only one tree contributing its own squared length. "
		+ "Self-contained reimplementation of beastlabs.evolution.tree.BranchScoreMetric's "
		+ "algorithm directly against beast.base tree types: that class targets JEBL's "
		+ "RootedTree and doesn't implement TreeMetric, so isn't usable as a BenchmarkMCMC/"
		+ "GridTuner metric as-is. Clade identity uses leaf node numbers, which BEAST keeps "
		+ "stable across topology changes on the same tree, so no taxon-name Input is needed.")
public class BranchScoreTreeMetric extends BEASTObject implements TreeMetric {

	private Map<BitSet, Double> referenceClades;

	@Override
	public void initAndValidate() {
	}

	@Override
	public double distance(TreeInterface tree1, TreeInterface tree2) {
		return branchScoreDistance(getCladeLengths(tree1), getCladeLengths(tree2));
	}

	@Override
	public double distance(TreeInterface tree) {
		if (referenceClades == null) {
			throw new IllegalStateException(
					"BranchScoreTreeMetric: call setReference(tree) before distance(tree)");
		}
		return branchScoreDistance(getCladeLengths(tree), referenceClades);
	}

	@Override
	public void setReference(TreeInterface ref) {
		referenceClades = getCladeLengths(ref);
	}

	private static double branchScoreDistance(Map<BitSet, Double> a, Map<BitSet, Double> b) {
		Set<BitSet> allClades = new HashSet<>(a.keySet());
		allClades.addAll(b.keySet());
		double sumSq = 0.0;
		for (BitSet clade : allClades) {
			double diff = a.getOrDefault(clade, 0.0) - b.getOrDefault(clade, 0.0);
			sumSq += diff * diff;
		}
		return Math.sqrt(sumSq);
	}

	/** Every non-root node's clade (leaf-index bitset) mapped to its branch length. */
	private static Map<BitSet, Double> getCladeLengths(TreeInterface tree) {
		Map<BitSet, Double> clades = new HashMap<>();
		collectClades(tree.getRoot(), clades);
		return clades;
	}

	private static BitSet collectClades(Node node, Map<BitSet, Double> clades) {
		BitSet bits = new BitSet();
		if (node.isLeaf()) {
			bits.set(node.getNr());
		} else {
			for (Node child : node.getChildren()) {
				bits.or(collectClades(child, clades));
			}
		}
		if (!node.isRoot()) {
			clades.put(bits, node.getLength());
		}
		return bits;
	}
}
