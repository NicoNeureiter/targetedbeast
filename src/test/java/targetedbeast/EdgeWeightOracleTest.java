package targetedbeast;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;

import org.junit.Assert;
import org.junit.Test;

import beast.base.core.BEASTInterface;
import beast.base.core.ProgramStatus;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.operator.kernel.BactrianSubtreeSlide;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeDistribution;
import beast.base.inference.CompoundDistribution;
import beast.base.inference.DirectSimulator;
import beast.base.inference.Distribution;
import beast.base.inference.Logger;
import beast.base.inference.MCMC;
import beast.base.inference.Operator;
import beast.base.inference.State;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.likelihood.ThreadedTreeLikelihood;
import beast.base.spec.evolution.likelihood.TreeLikelihood;
import beast.base.spec.evolution.operator.UpDownOperator;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.speciation.YuleModel;
import beast.base.spec.evolution.substitutionmodel.Frequencies;
import beast.base.spec.evolution.substitutionmodel.HKY;
import beast.base.spec.inference.operator.ScaleOperator;
import beast.base.spec.inference.parameter.RealScalarParam;
import targetedbeast.edgeweights.BranchLengthWeights;
import targetedbeast.edgeweights.ConstantWeights;
import targetedbeast.edgeweights.EdgeWeights;
import targetedbeast.edgeweights.PCAWeights2;
import targetedbeast.edgeweights.PartialWeights;
import targetedbeast.edgeweights.SourceTargetWeights;
import targetedbeast.edgeweights.ThreadedPartialWeights;
import targetedbeast.operators.RangeSlide;
import targetedbeast.operators.TargetedWilsonBalding;
import targetedbeast.operators.TargetedWilsonBaldingFixedHeight;
import targetedbeast.operators.WeightedWideOperator;

/**
 * Runs a real MCMC, as in the benchmark (likelihood in the posterior, tree and substitution
 * parameters moving), and checks every edge weight a targeted operator reads against a fresh
 * instance on the current state (see {@link CheckedEdgeWeights}). One chain per targeted operator,
 * so a mismatch is attributed to the operator that read it.
 */
public class EdgeWeightOracleTest {

	private static final int NUM_TAXA = 14;
	private static final int SITES = 120;
	private static final long CHAIN_LENGTH = 3000;

	/** The substitution and clock parameters every likelihood here shares, and the data. */
	static class Model {
		final Alignment data = randomAlignment();
		final RealScalarParam<PositiveReal> kappa = new RealScalarParam<>(2.0, PositiveReal.INSTANCE);
		final RealScalarParam<PositiveReal> gammaShape = new RealScalarParam<>(0.5, PositiveReal.INSTANCE);
		final RealScalarParam<PositiveReal> clockRate = new RealScalarParam<>(1.0, PositiveReal.INSTANCE);

		Model() {
			kappa.setID("kappa");
			gammaShape.setID("gammaShape");
			clockRate.setID("clockRate");
		}

		/** Data outputs registered by the live chain; fresh instances' registrations are dropped. */
		private Set<BEASTInterface> liveDataOutputs;

		void sealLive() {
			liveDataOutputs = new HashSet<>(data.getOutputs());
		}

		void forgetFresh() {
			data.getOutputs().retainAll(liveDataOutputs);
		}

		/** Live models share the State's parameters; fresh ones get copies of their current values. */
		private RealScalarParam<PositiveReal> param(RealScalarParam<PositiveReal> p, boolean fresh) {
			return fresh ? new RealScalarParam<>(p.get(), PositiveReal.INSTANCE) : p;
		}

		SiteModel siteModel(boolean fresh) {
			Frequencies freqs = new Frequencies();
			freqs.initByName("data", data);
			HKY hky = new HKY();
			hky.initByName("kappa", param(kappa, fresh), "frequencies", freqs);
			SiteModel siteModel = new SiteModel();
			siteModel.initByName("substModel", hky, "gammaCategoryCount", 4, "shape", param(gammaShape, fresh));
			return siteModel;
		}

		StrictClockModel clock(boolean fresh) {
			StrictClockModel clock = new StrictClockModel();
			clock.initByName("clock.rate", param(clockRate, fresh));
			return clock;
		}

		TreeLikelihood likelihood(Tree tree, boolean fresh) {
			System.setProperty("java.only", "true");
			TreeLikelihood likelihood = new TreeLikelihood();
			likelihood.initByName("tree", tree, "siteModel", siteModel(fresh), "branchRateModel", clock(fresh),
					"data", data, "implementation", "BeerLikelihoodCore4");
			return likelihood;
		}

		ThreadedTreeLikelihood threadedLikelihood(Tree tree) {
			System.setProperty("java.only", "true");
			ThreadedTreeLikelihood likelihood = new ThreadedTreeLikelihood();
			likelihood.initByName("tree", tree, "siteModel", siteModel(false), "branchRateModel", clock(false),
					"data", data, "threads", 2);
			return likelihood;
		}

		String key() {
			return kappa.get() + "," + gammaShape.get() + "," + clockRate.get();
		}
	}

	/** How to build one weight class: the live instance reads the posterior's likelihood. */
	interface WeightsFactory {
		Distribution liveLikelihood(Model model, Tree tree);

		EdgeWeights live(Model model, Tree tree, Distribution likelihood);

		EdgeWeights fresh(Model model, Tree tree);
	}

	static final WeightsFactory PCA = new WeightsFactory() {
		public Distribution liveLikelihood(Model m, Tree t) {
			return m.likelihood(t, false);
		}

		public EdgeWeights live(Model m, Tree t, Distribution l) {
			return fresh(m, t);
		}

		public EdgeWeights fresh(Model m, Tree t) {
			PCAWeights2 w = new PCAWeights2();
			w.initByName("tree", t, "data", m.data, "dimension", 3);
			return w;
		}
	};

	static final WeightsFactory CONSTANT = new WeightsFactory() {
		public Distribution liveLikelihood(Model m, Tree t) {
			return m.likelihood(t, false);
		}

		public EdgeWeights live(Model m, Tree t, Distribution l) {
			return fresh(m, t);
		}

		public EdgeWeights fresh(Model m, Tree t) {
			ConstantWeights w = new ConstantWeights();
			w.initByName("tree", t);
			return w;
		}
	};

	static final WeightsFactory PARTIAL = new WeightsFactory() {
		public Distribution liveLikelihood(Model m, Tree t) {
			return m.likelihood(t, false);
		}

		public EdgeWeights live(Model m, Tree t, Distribution l) {
			PartialWeights w = new PartialWeights();
			w.initByName("tree", t, "likelihood", l, "data", m.data);
			return w;
		}

		public EdgeWeights fresh(Model m, Tree t) {
			return live(m, t, m.likelihood(t, true));
		}
	};

	/** Fresh side is a single-threaded PartialWeights, which ThreadedPartialWeightsTest shows is equal. */
	static final WeightsFactory THREADED_PARTIAL = new WeightsFactory() {
		public Distribution liveLikelihood(Model m, Tree t) {
			return m.threadedLikelihood(t);
		}

		public EdgeWeights live(Model m, Tree t, Distribution l) {
			ThreadedPartialWeights w = new ThreadedPartialWeights();
			w.initByName("tree", t, "likelihood", l);
			return w;
		}

		public EdgeWeights fresh(Model m, Tree t) {
			return PARTIAL.fresh(m, t);
		}
	};

	static final WeightsFactory LENGTH = new WeightsFactory() {
		public Distribution liveLikelihood(Model m, Tree t) {
			return m.likelihood(t, false);
		}

		public EdgeWeights live(Model m, Tree t, Distribution l) {
			return fresh(m, t);
		}

		public EdgeWeights fresh(Model m, Tree t) {
			BranchLengthWeights w = new BranchLengthWeights();
			w.initByName("tree", t);
			return w;
		}
	};

	/** Source and target weights from two classes, as in the source-weight benchmark settings. */
	static WeightsFactory sourceTarget(WeightsFactory source, WeightsFactory target) {
		return new WeightsFactory() {
			public Distribution liveLikelihood(Model m, Tree t) {
				return target.liveLikelihood(m, t);
			}

			public EdgeWeights live(Model m, Tree t, Distribution l) {
				SourceTargetWeights w = new SourceTargetWeights();
				w.initByName("source", source.live(m, t, l), "target", target.live(m, t, l));
				return w;
			}

			public EdgeWeights fresh(Model m, Tree t) {
				SourceTargetWeights w = new SourceTargetWeights();
				w.initByName("source", source.fresh(m, t), "target", target.fresh(m, t));
				return w;
			}
		};
	}

	/** The benchmark's targeted operators, each built on the weights it is given. */
	static Map<String, Function<Object[], Operator>> targetedOperators() {
		Map<String, Function<Object[], Operator>> ops = new LinkedHashMap<>();
		ops.put("TargetedWilsonBalding", a -> {
			TargetedWilsonBalding op = new TargetedWilsonBalding();
			op.initByName("tree", a[0], "weight", 3.0, "edgeWeights", a[1], "maxHeightRatio", 1.2);
			return op;
		});
		ops.put("TargetedWilsonBaldingFixedHeight", a -> {
			TargetedWilsonBaldingFixedHeight op = new TargetedWilsonBaldingFixedHeight();
			op.initByName("tree", a[0], "weight", 3.0, "edgeWeights", a[1]);
			return op;
		});
		ops.put("WeightedWide", a -> {
			WeightedWideOperator op = new WeightedWideOperator();
			op.initByName("tree", a[0], "weight", 3.0, "edgeWeights", a[1]);
			return op;
		});
		ops.put("RangeSlide(weighted)", a -> {
			RangeSlide op = new RangeSlide();
			op.initByName("tree", a[0], "weight", 3.0, "edgeWeights", a[1], "useWeightedStep", true, "size", 2.0,
					"optimise", false);
			return op;
		});
		ops.put("RangeSlide", a -> {
			RangeSlide op = new RangeSlide();
			op.initByName("tree", a[0], "weight", 3.0, "edgeWeights", a[1], "size", 0.5);
			return op;
		});
		return ops;
	}

	/** Runs one chain per targeted operator and returns each one's report, or null if clean. */
	static Map<String, String> runMatrix(WeightsFactory factory) throws Exception {
		Map<String, String> failures = new LinkedHashMap<>();
		int savedThreads = ProgramStatus.m_nThreads;
		ProgramStatus.m_nThreads = 2;
		try {
			for (Map.Entry<String, Function<Object[], Operator>> op : targetedOperators().entrySet()) {
				CheckedEdgeWeights checked = runChain(factory, op.getValue());
				System.out.println(op.getKey() + ": " + checked.report());
				Assert.assertTrue(op.getKey() + ": no reads were checked", checked.checks > 0);
				if (checked.mismatches > 0) {
					failures.put(op.getKey(), checked.report());
				}
			}
		} finally {
			ProgramStatus.m_nThreads = savedThreads;
		}
		return failures;
	}

	static CheckedEdgeWeights runChain(WeightsFactory factory, Function<Object[], Operator> targeted) throws Exception {
		Model model = new Model();
		TaxonSet taxa = new TaxonSet();
		for (int t = 0; t < NUM_TAXA; t++) {
			taxa.initByName("taxon", new Taxon(String.valueOf(t)));
		}
		Tree tree = new Tree();
		tree.initByName("taxonset", taxa);
		tree.setID("tree");
		YuleModel yule = new YuleModel();
		yule.initByName("tree", tree, "birthDiffRate", "2.0");
		DirectSimulator simulator = new DirectSimulator();
		simulator.initByName("distribution", (TreeDistribution) yule, "nSamples", 1);
		simulator.run();

		Distribution likelihood = factory.liveLikelihood(model, tree);
		EdgeWeights live = factory.live(model, tree, likelihood);
		model.sealLive();
		CheckedEdgeWeights checked = new CheckedEdgeWeights(live, tree, t -> {
			EdgeWeights fresh = factory.fresh(model, t);
			model.forgetFresh();
			return fresh;
		}, model::key);

		// As in the benchmark XML: the weights sit in the prior, the likelihood next to it.
		List<Distribution> priorParts = new ArrayList<>(List.of(yule, (Distribution) live));
		if (live instanceof SourceTargetWeights st) {
			priorParts.add((Distribution) st.sourceInput.get());
			priorParts.add((Distribution) st.targetInput.get());
		}
		CompoundDistribution prior = new CompoundDistribution();
		prior.initByName("distribution", priorParts);
		CompoundDistribution posterior = new CompoundDistribution();
		posterior.initByName("distribution", prior, "distribution", likelihood);

		Operator targetedOp = targeted.apply(new Object[] { tree, checked });
		targetedOp.setID("targeted");
		BactrianSubtreeSlide subtreeSlide = new BactrianSubtreeSlide();
		subtreeSlide.initByName("tree", tree, "weight", 1.0, "size", 0.5);
		subtreeSlide.setID("subtreeSlide");
		List<Operator> operators = new ArrayList<>(List.of(targetedOp, subtreeSlide));
		for (RealScalarParam<?> p : List.of(model.kappa, model.gammaShape, model.clockRate)) {
			ScaleOperator scale = new ScaleOperator();
			scale.initByName("parameter", p, "weight", 0.3, "scaleFactor", 0.5);
			scale.setID(p.getID() + "Scaler");
			operators.add(scale);
		}
		UpDownOperator upDown = new UpDownOperator();
		upDown.initByName("up", model.clockRate, "down", tree, "weight", 0.3, "scaleFactor", 0.75);
		upDown.setID("upDown");
		operators.add(upDown);

		State state = new State();
		state.initByName("stateNode", tree, "stateNode", model.kappa, "stateNode", model.gammaShape,
				"stateNode", model.clockRate);
		Logger.FILE_MODE = Logger.LogFileMode.overwrite;
		Logger screenlog = new Logger();
		screenlog.initByName("log", posterior, "logEvery", 1_000_000);

		MCMC mcmc = new MCMC();
		mcmc.initByName("chainLength", CHAIN_LENGTH, "state", state, "distribution", posterior,
				"operator", operators, "logger", screenlog);
		mcmc.run();
		return checked;
	}

	static Alignment randomAlignment() {
		Random random = new Random(42);
		String bases = "ACGT";
		List<Sequence> sequences = new ArrayList<>();
		char[] ancestor = new char[SITES];
		for (int s = 0; s < SITES; s++) {
			ancestor[s] = bases.charAt(random.nextInt(4));
		}
		for (int t = 0; t < NUM_TAXA; t++) {
			char[] seq = ancestor.clone();
			for (int s = 0; s < SITES; s++) {
				if (random.nextDouble() < 0.25) {
					seq[s] = bases.charAt(random.nextInt(4));
				}
			}
			sequences.add(new Sequence(String.valueOf(t), new String(seq)));
		}
		Alignment data = new Alignment();
		data.initByName("sequence", sequences, "dataType", "nucleotide");
		return data;
	}

	private static void assertClean(WeightsFactory factory) throws Exception {
		Map<String, String> failures = runMatrix(factory);
		Assert.assertTrue("weights read during proposals differ from a fresh instance:\n" + failures, failures.isEmpty());
	}

	@Test
	public void testPCAWeights2() throws Exception {
		assertClean(PCA);
	}

	@Test
	public void testConstantWeights() throws Exception {
		assertClean(CONSTANT);
	}

	@Test
	public void testPartialWeights() throws Exception {
		assertClean(PARTIAL);
	}

	@Test
	public void testThreadedPartialWeights() throws Exception {
		assertClean(THREADED_PARTIAL);
	}

	@Test
	public void testPCASourcePartialTarget() throws Exception {
		assertClean(sourceTarget(PCA, THREADED_PARTIAL));
	}

	@Test
	public void testLengthSourcePCATarget() throws Exception {
		assertClean(sourceTarget(LENGTH, PCA));
	}

	@Test
	public void testConstantSourcePartialTarget() throws Exception {
		assertClean(sourceTarget(CONSTANT, THREADED_PARTIAL));
	}
}
