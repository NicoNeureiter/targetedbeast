package targetedbeast.edgeweights;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.junit.Assume;
import org.junit.Test;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.MCMC;
import beast.base.inference.Operator;
import beast.base.inference.State;
import beast.base.parser.XMLParser;

/**
 * Offline comparison of cheaper approximations of PartialWeights' candidate score against the
 * exact score, on real infB posterior states. Not a regression test: it only runs when
 * {@code -DpartialQuality=<dir>} names a directory of finished cells (rep*&#47;run.xml and
 * run.xml.state), and prints a table.
 *
 * <p>Proposals mimic TargetedWilsonBalding: a source node and every co-existing lineage whose
 * attachment interval is non-empty under maxHeightRatio. Each candidate is scored as
 * PartialWeights does (half the agreement log-likelihood, clipped so the top two differ by at
 * most 2), and the proposal distribution is softmax(b * score) for b = 2 (the operators' default
 * squaring) and b = 0.2 (the flattened setting). Approximations score with a subset of rate
 * categories and/or every k-th pattern (weights scaled by k).
 */
public class PartialWeightsApproximationQuality {

	private static final double EPS = 1E-8;
	private static final double MAX_HEIGHT_RATIO = 1.2;
	private static final int PROPOSALS_PER_STATE = 150;
	private static final double[] EXPONENTS = { 2.0, 0.2 };

	record Approx(String name, int[] categories, int stride) {}

	static final class Stats {
		final double[] kl = new double[EXPONENTS.length];
		final double[] tv = new double[EXPONENTS.length];
		final int[] top1 = new int[EXPONENTS.length];
		final double[] exactMassInTopK = new double[EXPONENTS.length];
		int proposals;
		long nanos;
		long candidates;
	}

	@Test
	public void report() throws Exception {
		String dir = System.getProperty("partialQuality");
		Assume.assumeTrue("set -DpartialQuality=<dir of finished partial cells>", dir != null);
		System.setProperty("java.only", "true");

		List<Approx> approxes = List.of(
				new Approx("exact", null, 1),
				new Approx("category 1 only", new int[] { 0 }, 1),
				new Approx("category 2 only", new int[] { 1 }, 1),
				new Approx("category 3 only", new int[] { 2 }, 1),
				new Approx("category 4 only", new int[] { 3 }, 1),
				new Approx("categories 3+4", new int[] { 2, 3 }, 1),
				new Approx("every 2nd pattern", null, 2),
				new Approx("every 4th pattern", null, 4),
				new Approx("every 8th pattern", null, 8),
				new Approx("cats 3+4, every 2nd pattern", new int[] { 2, 3 }, 2));
		// Two-stage: PCAWeights2's top K candidates share their combined PCAWeights2 mass in proportion
		// to PartialWeights' weights among those K; the rest keep their PCAWeights2 weight. K = 0 is
		// PCAWeights2 alone.
		int[] twoStage = { 0, 3, 5, 10, 20 };
		Map<String, Stats> stats = new LinkedHashMap<>();
		for (Approx a : approxes) {
			stats.put(a.name(), new Stats());
		}
		for (int k : twoStage) {
			stats.put(twoStageName(k), new Stats());
		}

		Random random = new Random(11);
		List<Path> cells;
		try (var stream = Files.list(Path.of(dir))) {
			cells = stream.filter(p -> p.getFileName().toString().startsWith("rep")).sorted().toList();
		}
		for (Path cell : cells) {
			MCMC mcmc = (MCMC) new XMLParser().parseFile(cell.resolve("run.xml").toFile());
			State state = mcmc.startStateInput.get();
			state.setStateFileName(cell.resolve("run.xml.state").toString());
			state.restoreFromFile();
			PartialWeights weights = findWeights(mcmc);
			Tree tree = (Tree) weights.treeInput.get();
			// As BEAST does on resume: every restored parameter, not just the tree, must reach the
			// substitution and site models before the partials are computed.
			state.initialise();
			state.setPosterior(mcmc.posteriorInput.get());
			state.robustlyCalcPosterior(mcmc.posteriorInput.get());
			weights.ensureInitialized();
			System.out.printf("%s: %d taxa, %d patterns, %d rate categories, tree likelihood %.4f%n",
					cell.getFileName(), tree.getLeafNodeCount(), weights.patternCount, weights.matrixCount,
					weights.treelikelihood.calculateLogP());

			PCAWeights2 filter = new PCAWeights2();
			filter.initByName("tree", tree, "data", weights.treelikelihood.dataInput.get(), "dimension", 10,
					"maxWeight", 20.0);
			Scorer scorer = new Scorer(weights);
			int done = 0;
			while (done < PROPOSALS_PER_STATE) {
				Node i = tree.getNode(random.nextInt(tree.getNodeCount()));
				if (i.isRoot() || i.getParent().isRoot()) {
					continue;
				}
				List<Node> candidates = candidates(tree, i);
				if (candidates.size() < 2) {
					continue;
				}
				evaluate(weights, scorer, filter, i, candidates, approxes, twoStage, stats);
				done++;
			}
		}

		System.out.printf("%n%-28s %10s %10s %8s %10s %10s %8s %12s%n", "approximation", "KL b=2", "TV b=2",
				"top1", "KL b=0.2", "TV b=0.2", "top1", "us/candidate");
		System.out.printf("%n%-34s %8s %8s %7s %8s %8s %7s %9s %9s %12s%n", "two-stage", "KL b=2", "TV b=2",
				"top1", "KL b=0.2", "TV b=0.2", "top1", "mass b=2", "mass b=.2", "us/candidate");
		for (int k : twoStage) {
			Stats s = stats.get(twoStageName(k));
			System.out.printf("%-34s %8.4f %8.4f %6.1f%% %8.5f %8.4f %6.1f%% %9.3f %9.3f %12.2f%n", twoStageName(k),
					s.kl[0] / s.proposals, s.tv[0] / s.proposals, 100.0 * s.top1[0] / s.proposals,
					s.kl[1] / s.proposals, s.tv[1] / s.proposals, 100.0 * s.top1[1] / s.proposals,
					s.exactMassInTopK[0] / s.proposals, s.exactMassInTopK[1] / s.proposals,
					s.nanos / 1000.0 / s.candidates);
		}
		System.out.println();
		for (Approx a : approxes) {
			Stats s = stats.get(a.name());
			System.out.printf("%-28s %10.4f %10.4f %7.1f%% %10.5f %10.4f %7.1f%% %12.2f%n", a.name(),
					s.kl[0] / s.proposals, s.tv[0] / s.proposals, 100.0 * s.top1[0] / s.proposals,
					s.kl[1] / s.proposals, s.tv[1] / s.proposals, 100.0 * s.top1[1] / s.proposals,
					s.nanos / 1000.0 / s.candidates);
		}
	}

	static PartialWeights findWeights(MCMC mcmc) {
		for (Operator op : mcmc.operatorsInput.get()) {
			for (Input<?> input : op.getInputs().values()) {
				if (input.get() instanceof PartialWeights w) {
					return w;
				}
			}
		}
		throw new IllegalStateException("no PartialWeights among the operators' inputs");
	}

	/** TargetedWilsonBalding's forward candidates for source i. */
	static List<Node> candidates(Tree tree, Node i) {
		Node p = i.getParent();
		double minAge = Math.max(i.getHeight(), p.getHeight() / MAX_HEIGHT_RATIO);
		double maxAge = p.getHeight() * MAX_HEIGHT_RATIO;
		List<Node> out = new ArrayList<>();
		for (Node c : tree.getNodesAsArray()) {
			if (c.isRoot() || c == i || c == p || c.getParent().getHeight() <= i.getHeight()) {
				continue;
			}
			double lo = Math.max(minAge, c.getHeight());
			double hi = Math.min(maxAge, c.getParent().getHeight());
			if (hi > lo) {
				out.add(c);
			}
		}
		return out;
	}

	static void evaluate(PartialWeights w, Scorer scorer, PCAWeights2 filter, Node i, List<Node> candidates,
			List<Approx> approxes, int[] twoStage, Map<String, Stats> stats) {
		List<Integer> ancestors = new ArrayList<>();
		for (Node a = i.getParent(); a != null; a = a.getParent()) {
			ancestors.add(a.getNr());
		}
		w.prestore();
		w.updateByOperatorWithoutNode(i.getNr(), ancestors);
		double[] fromPartials = w.getPartials(i.getNr());

		// Per-candidate inputs exactly as getTargetWeightsInteger(from, toNodes) builds them.
		List<Node> toNodes = new ArrayList<>();
		List<double[]> toPartials = new ArrayList<>();
		List<Double> heights = new ArrayList<>();
		for (Node c : candidates) {
			Node to = resolveScoringNode(c, i);
			double parentHeight = to.isRoot() ? Double.POSITIVE_INFINITY : to.getParent().getHeight();
			if (parentHeight <= i.getHeight()) {
				continue;
			}
			toNodes.add(to);
			toPartials.add(w.getPartialsWithoutNode(to.getNr()));
			heights.add(w.getToHeightWithoutNode(to, i));
		}
		if (toNodes.size() < 2) {
			w.reset();
			return;
		}

		// Sanity: the harness' exact scorer must reproduce PartialWeights' own score.
		double own = w.computePartialsAgreement(i, toNodes.get(0), fromPartials, toPartials.get(0), heights.get(0));
		double mine = scorer.score(i, toNodes.get(0), fromPartials, toPartials.get(0), heights.get(0), null, 1);
		if (Math.abs(own - mine) > 1e-9 * Math.abs(own)) {
			throw new IllegalStateException("harness exact score " + mine + " != PartialWeights " + own);
		}

		double[][] probs = new double[approxes.size()][];
		double[][][] dists = new double[approxes.size()][EXPONENTS.length][];
		for (int a = 0; a < approxes.size(); a++) {
			Approx approx = approxes.get(a);
			double[] scores = new double[toNodes.size()];
			long t0 = System.nanoTime();
			for (int k = 0; k < toNodes.size(); k++) {
				scores[k] = EPS + scorer.score(i, toNodes.get(k), fromPartials, toPartials.get(k), heights.get(k),
						approx.categories(), approx.stride()) / 2.0;
			}
			Stats s = stats.get(approx.name());
			s.nanos += System.nanoTime() - t0;
			s.candidates += toNodes.size();
			double[] clipped = w.clipMaxToSecondMaxDiff(scores, 2.0);
			for (int e = 0; e < EXPONENTS.length; e++) {
				dists[a][e] = softmax(clipped, EXPONENTS[e]);
			}
			probs[a] = scores;
		}
		filter.prestore();
		filter.updateByOperatorWithoutNode(i.getNr(), ancestors);
		long f0 = System.nanoTime();
		double[] filterWeights = filter.getTargetWeights(i.getNr(), toNodes);
		long filterNanos = System.nanoTime() - f0;
		filter.reset();
		Integer[] order = new Integer[toNodes.size()];
		for (int k = 0; k < order.length; k++) {
			order[k] = k;
		}
		java.util.Arrays.sort(order, (x, y) -> Double.compare(filterWeights[y], filterWeights[x]));
		double exactMeanCost = (double) stats.get("exact").nanos / Math.max(1, stats.get("exact").candidates);
		for (int K : twoStage) {
			int top = Math.min(K, toNodes.size());
			double[] combined = filterWeights.clone();
			if (top > 0) {
				double[] subsetScores = new double[top];
				double mass = 0;
				for (int r = 0; r < top; r++) {
					int k = order[r];
					subsetScores[r] = EPS + scorer.score(i, toNodes.get(k), fromPartials, toPartials.get(k), heights.get(k), null, 1) / 2.0;
					mass += filterWeights[k];
				}
				double[] q = softmax(w.clipMaxToSecondMaxDiff(subsetScores, 2.0), 1.0);
				for (int r = 0; r < top; r++) {
					combined[order[r]] = mass * q[r];
				}
			}
			Stats st = stats.get(twoStageName(K));
			st.nanos += filterNanos + (long) (top * exactMeanCost);
			st.candidates += toNodes.size();
			for (int e = 0; e < EXPONENTS.length; e++) {
				double[] approx = power(combined, EXPONENTS[e]);
				double[] exact = dists[0][e];
				double kl = 0, tv = 0;
				for (int k = 0; k < exact.length; k++) {
					if (exact[k] > 0) {
						kl += exact[k] * Math.log(exact[k] / Math.max(approx[k], 1e-300));
					}
					tv += Math.abs(exact[k] - approx[k]);
				}
				st.kl[e] += kl;
				st.tv[e] += tv / 2;
				if (argmax(exact) == argmax(approx)) {
					st.top1[e]++;
				}
				double inTop = 0;
				for (int r = 0; r < top; r++) {
					inTop += exact[order[r]];
				}
				st.exactMassInTopK[e] += inTop;
			}
			st.proposals++;
		}

		for (int a = 0; a < approxes.size(); a++) {
			Stats s = stats.get(approxes.get(a).name());
			for (int e = 0; e < EXPONENTS.length; e++) {
				double[] exact = dists[0][e];
				double[] approx = dists[a][e];
				double kl = 0, tv = 0;
				for (int k = 0; k < exact.length; k++) {
					if (exact[k] > 0) {
						kl += exact[k] * Math.log(exact[k] / Math.max(approx[k], 1e-300));
					}
					tv += Math.abs(exact[k] - approx[k]);
				}
				s.kl[e] += kl;
				s.tv[e] += tv / 2;
				if (argmax(exact) == argmax(approx)) {
					s.top1[e]++;
				}
			}
			s.proposals++;
		}
		w.reset();
	}

	static Node resolveScoringNode(Node toNode, Node fromNode) {
		Node parent = fromNode.getParent();
		if (toNode != parent) {
			return toNode;
		}
		return parent.getLeft() == fromNode ? parent.getRight() : parent.getLeft();
	}

	static double[] softmax(double[] x, double b) {
		double max = Double.NEGATIVE_INFINITY;
		for (double v : x) {
			max = Math.max(max, v);
		}
		double[] out = new double[x.length];
		double sum = 0;
		for (int k = 0; k < x.length; k++) {
			out[k] = Math.exp(b * (x[k] - max));
			sum += out[k];
		}
		for (int k = 0; k < x.length; k++) {
			out[k] /= sum;
		}
		return out;
	}

	static String twoStageName(int k) {
		return k == 0 ? "PCAWeights2 alone" : "PCA top " + k + " + PartialWeights";
	}

	/** Normalised x^b: what an operator's exponent b makes of raw weights x. */
	static double[] power(double[] x, double b) {
		double[] out = new double[x.length];
		double sum = 0;
		for (int k = 0; k < x.length; k++) {
			out[k] = Math.pow(x[k], b);
			sum += out[k];
		}
		for (int k = 0; k < x.length; k++) {
			out[k] /= sum;
		}
		return out;
	}

	static int argmax(double[] x) {
		int best = 0;
		for (int k = 1; k < x.length; k++) {
			if (x[k] > x[best]) {
				best = k;
			}
		}
		return best;
	}

	/** PartialWeights' fused agreement score, restricted to some categories and every k-th pattern. */
	static final class Scorer {
		final PartialWeights w;
		final int stateCount, patternCount, matrixCount, matrixSize;
		final double[] fm, tm, target;
		final double[] patternWeights;

		Scorer(PartialWeights w) {
			this.w = w;
			stateCount = w.stateCount;
			patternCount = w.patternCount;
			matrixCount = w.matrixCount;
			matrixSize = w.matrixSize;
			fm = new double[matrixCount * matrixSize];
			tm = new double[matrixCount * matrixSize];
			target = new double[patternCount * stateCount];
			patternWeights = new double[patternCount];
			for (int p = 0; p < patternCount; p++) {
				patternWeights[p] = w.treelikelihood.dataInput.get().getPatternWeight(p);
			}
		}

		double score(Node from, Node to, double[] fp, double[] tp, double toHeight, int[] categories, int stride) {
			if (stateCount != 4) {
				throw new IllegalStateException("harness assumes nucleotides");
			}
			w.calculateTransitionMatrix(from, toHeight, fm);
			w.calculateTransitionMatrix(to, toHeight, tm);
			double[] proportions = w.getSiteModel().getCategoryProportions(to);
			int[] cats = categories;
			if (cats == null) {
				cats = new int[matrixCount];
				for (int c = 0; c < matrixCount; c++) {
					cats[c] = c;
				}
			}
			double proportionSum = 0;
			for (int c : cats) {
				proportionSum += proportions[c];
			}
			java.util.Arrays.fill(target, 0.0);
			for (int c : cats) {
				int m = c * matrixSize;
				double pr = proportions[c] / proportionSum;
				for (int p = 0; p < patternCount; p += stride) {
					int in = (c * patternCount + p) * 4;
					int out = p * 4;
					double x0 = fp[in], x1 = fp[in + 1], x2 = fp[in + 2], x3 = fp[in + 3];
					double y0 = tp[in], y1 = tp[in + 1], y2 = tp[in + 2], y3 = tp[in + 3];
					double f0 = fm[m] * x0 + fm[m + 1] * x1 + fm[m + 2] * x2 + fm[m + 3] * x3;
					double f1 = fm[m + 4] * x0 + fm[m + 5] * x1 + fm[m + 6] * x2 + fm[m + 7] * x3;
					double f2 = fm[m + 8] * x0 + fm[m + 9] * x1 + fm[m + 10] * x2 + fm[m + 11] * x3;
					double f3 = fm[m + 12] * x0 + fm[m + 13] * x1 + fm[m + 14] * x2 + fm[m + 15] * x3;
					double t0 = tm[m] * y0 + tm[m + 1] * y1 + tm[m + 2] * y2 + tm[m + 3] * y3;
					double t1 = tm[m + 4] * y0 + tm[m + 5] * y1 + tm[m + 6] * y2 + tm[m + 7] * y3;
					double t2 = tm[m + 8] * y0 + tm[m + 9] * y1 + tm[m + 10] * y2 + tm[m + 11] * y3;
					double t3 = tm[m + 12] * y0 + tm[m + 13] * y1 + tm[m + 14] * y2 + tm[m + 15] * y3;
					double fs = f0 + f1 + f2 + f3, ts = t0 + t1 + t2 + t3;
					if (ts <= EPS || fs <= EPS) {
						double u = pr / 16.0;
						target[out] += u; target[out + 1] += u; target[out + 2] += u; target[out + 3] += u;
					} else {
						double s = pr / (fs * ts);
						target[out] += t0 * f0 * s; target[out + 1] += t1 * f1 * s;
						target[out + 2] += t2 * f2 * s; target[out + 3] += t3 * f3 * s;
					}
				}
			}
			double[] freqs = w.getSubstitutionModel().getFrequencies();
			double sum = 0;
			for (int p = 0; p < patternCount; p += stride) {
				int o = p * 4;
				double l = freqs[0] * target[o] + freqs[1] * target[o + 1] + freqs[2] * target[o + 2] + freqs[3] * target[o + 3];
				sum += patternWeights[p] * stride * Math.log(l);
			}
			return sum;
		}
	}
}
