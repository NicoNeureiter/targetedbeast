package targetedbeast.edgeweights;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Distribution;
import beast.base.inference.State;
import beast.base.spec.evolution.likelihood.ThreadedTreeLikelihood;
import beast.base.spec.evolution.likelihood.TreeLikelihood;

@Description("PartialWeights over a ThreadedTreeLikelihood: one PartialWeights per likelihood thread, "
		+ "each scoring that thread's slice of the sites, run in parallel. A candidate's score is a sum "
		+ "over patterns, so the slices' scores add up to the single-threaded score exactly.")
public class ThreadedPartialWeights extends Distribution implements EdgeWeights {

	final public Input<Tree> treeInput = new Input<>("tree", "phylogenetic tree with sequence data in the leafs",
			Validate.REQUIRED);
	final public Input<ThreadedTreeLikelihood> likelihoodInput = new Input<>("likelihood",
			"threaded tree likelihood whose per-thread likelihoods supply the partials; it must run on "
					+ "BEAST's Java likelihood core (-java), since BEAGLE does not expose partials",
			Validate.REQUIRED);
	final public Input<Double> maxWeightInput = new Input<>("maxWeight", "maximum weight for an edge",
			Double.POSITIVE_INFINITY);
	final public Input<Double> minWeightInput = new Input<>("minWeight", "minimum weight for an edge", 1E-10);
	final public Input<Boolean> logSpaceInput = new Input<>("logSpace",
			"if true, the target weights are returned as raw log-weights instead of being passed "
					+ "through a softmax, so that a log-space WeightTransform can do the conversion",
			false);

	private final List<PartialWeights> slices = new ArrayList<>();
	private ExecutorService pool;

	@Override
	public void initAndValidate() {
		ThreadedTreeLikelihood likelihood = likelihoodInput.get();
		// ThreadedTreeLikelihood keeps its per-thread likelihoods in an input named "*".
		Input<?> perThread = likelihood.getInputs().get("*");
		if (perThread == null || !(perThread.get() instanceof List<?> list) || list.isEmpty()) {
			throw new IllegalArgumentException("ThreadedPartialWeights could not find the per-thread "
					+ "likelihoods of " + likelihood.getID());
		}
		slices.clear();
		for (Object o : list) {
			TreeLikelihood sub = (TreeLikelihood) o;
			PartialWeights slice = new PartialWeights();
			slice.initByName("tree", treeInput.get(), "likelihood", sub, "data", sub.dataInput.get(),
					"maxWeight", maxWeightInput.get(), "minWeight", minWeightInput.get(),
					"logSpace", logSpaceInput.get());
			slices.add(slice);
		}
		if (slices.size() > 1) {
			pool = Executors.newFixedThreadPool(slices.size(), runnable -> {
				Thread thread = new Thread(runnable, "ThreadedPartialWeights");
				thread.setDaemon(true);
				return thread;
			});
		}
	}

	/** Number of slices, i.e. likelihood threads. */
	public int getSliceCount() {
		return slices.size();
	}

	/** Runs {@code task} on every slice, in parallel when there are several, and returns the results in order. */
	private <T> List<T> onSlices(Function<PartialWeights, T> task) {
		List<T> results = new ArrayList<>(slices.size());
		if (pool == null) {
			for (PartialWeights slice : slices) {
				results.add(task.apply(slice));
			}
			return results;
		}
		List<Future<T>> futures = new ArrayList<>(slices.size());
		for (PartialWeights slice : slices) {
			futures.add(pool.submit((Callable<T>) () -> task.apply(slice)));
		}
		try {
			for (Future<T> future : futures) {
				results.add(future.get());
			}
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new RuntimeException(e);
		} catch (ExecutionException e) {
			throw new RuntimeException(e.getCause());
		}
		return results;
	}

	/** Sums the slices' raw scores and finishes them as a single PartialWeights would. */
	private double[] combine(List<double[]> sliceScores) {
		double[] sum = sliceScores.get(0).clone();
		for (int s = 1; s < sliceScores.size(); s++) {
			double[] scores = sliceScores.get(s);
			for (int k = 0; k < sum.length; k++) {
				sum[k] += scores[k];
			}
		}
		return slices.get(0).finishWeights(sum);
	}

	@Override
	public double[] getTargetWeights(int fromNodeNr, List<Node> toNodes) {
		return getTargetWeightsInteger(fromNodeNr, toNodes.stream().map(Node::getNr).toList());
	}

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs) {
		return combine(onSlices(slice -> slice.rawScores(fromNodeNr, toNodeNrs)));
	}

	@Override
	public double[] getTargetWeightsInteger(int fromNodeNr, List<Integer> toNodeNrs, double newHeight) {
		return combine(onSlices(slice -> slice.rawScoresAtHeight(fromNodeNr, toNodeNrs, newHeight)));
	}

	@Override
	public void updateByOperatorWithoutNode(int ignore, List<Integer> nodes) {
		onSlices(slice -> {
			slice.updateByOperatorWithoutNode(ignore, nodes);
			return null;
		});
	}

	@Override
	public void updateByOperator() {
		for (PartialWeights slice : slices) {
			slice.updateByOperator();
		}
	}

	@Override
	public void fakeUpdateByOperator() {
		for (PartialWeights slice : slices) {
			slice.fakeUpdateByOperator();
		}
	}

	@Override
	public void prestore() {
		for (PartialWeights slice : slices) {
			slice.prestore();
		}
	}

	@Override
	public void reset() {
		for (PartialWeights slice : slices) {
			slice.reset();
		}
	}

	/** The slices are not in the posterior, so MCMC's store and restore reach them only from here. */
	@Override
	public void store() {
		for (PartialWeights slice : slices) {
			slice.forgetPartialsWithoutNode();
		}
		super.store();
	}

	@Override
	public void restore() {
		for (PartialWeights slice : slices) {
			slice.forgetPartialsWithoutNode();
		}
		super.restore();
	}

	/** Source weights depend only on branch lengths, so any slice gives them. */
	@Override
	public double getEdgeWeights(int nodeNr) {
		return slices.get(0).getEdgeWeights(nodeNr);
	}

	@Override
	public double minEdgeWeight() {
		return slices.get(0).minEdgeWeight();
	}

	@Override
	public boolean isLogSpace() {
		return logSpaceInput.get();
	}

	@Override
	protected boolean requiresRecalculation() {
		return treeInput.get().somethingIsDirty() || likelihoodInput.get().isDirtyCalculation();
	}

	@Override
	public List<String> getArguments() {
		return new ArrayList<>();
	}

	@Override
	public List<String> getConditions() {
		return new ArrayList<>();
	}

	@Override
	public void sample(State state, Random random) {
	}
}
