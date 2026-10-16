package targetedbeast.weighttransform;

import java.util.BitSet;
import java.util.Map;

import beast.base.core.Description;

@Description("Kuhner-Felsenstein branch score distance between rooted trees: the square root of the "
		+ "summed squared differences in branch length over all clades, a clade missing from one tree "
		+ "counting with length 0. Leaves are matched by taxon ID.")
public class KuhnerFelsensteinMetric extends CladeMetric {

	@Override
	protected double distance(Map<BitSet, Double> a, Map<BitSet, Double> b) {
		double sum = 0.0;
		for (Map.Entry<BitSet, Double> e : a.entrySet()) {
			double d = e.getValue() - b.getOrDefault(e.getKey(), 0.0);
			sum += d * d;
		}
		for (Map.Entry<BitSet, Double> e : b.entrySet()) {
			if (!a.containsKey(e.getKey())) {
				sum += e.getValue() * e.getValue();
			}
		}
		return Math.sqrt(sum);
	}
}
