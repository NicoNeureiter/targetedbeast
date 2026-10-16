package targetedbeast.weighttransform;

import java.util.BitSet;
import java.util.Map;

import beast.base.core.Description;

@Description("Robinson-Foulds distance between rooted trees: the number of clades in one tree but not "
		+ "the other. Leaves are matched by taxon ID.")
public class RobinsonFouldsMetric extends CladeMetric {

	@Override
	protected double distance(Map<BitSet, Double> a, Map<BitSet, Double> b) {
		int count = 0;
		for (BitSet clade : a.keySet()) {
			if (!b.containsKey(clade)) {
				count++;
			}
		}
		for (BitSet clade : b.keySet()) {
			if (!a.containsKey(clade)) {
				count++;
			}
		}
		return count;
	}
}
