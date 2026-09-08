package targetedbeast.edgeweights.learnedweights;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;

/**
 * Demonstrates and verifies the fix for the embedding&lt;-&gt;Brownian-rate rescaling degeneracy
 * described in {@link LearnedWeightsTreeObjectiveFunction#canonicalizeParameters}: in the default
 * regression configuration, scaling embedding axis {@code a} by {@code c} and rate {@code a} by
 * {@code c^2} (with a compensating intercept shift) leaves the predicted log-likelihood of every
 * training sample exactly unchanged, so the fit is non-identifiable along that ridge unless something
 * picks a canonical point on it.
 */
public class LearnedWeightsTreeObjectiveFunctionCanonicalizationTest {

    private static final double RIDGE_SCALE_FACTOR = 3.0;

    @Test
    public void ridgeTransform_leavesObjectiveValueUnchanged() {
        Fixture fixture = new Fixture();

        double loss1 = fixture.objective.value(fixture.params1);
        double loss2 = fixture.objective.value(fixture.params2);

        assertEquals(
                "scaling one embedding axis by c and its rate by c^2 (with the matching intercept "
                        + "shift) must be an exact reparameterization of the same prediction",
                loss1, loss2, 1e-9);
    }

    @Test
    public void twoPointsOnTheSameRidge_canonicalizeToTheSamePoint() {
        Fixture fixture = new Fixture();

        double[] canonical1 = fixture.objective.canonicalizeParameters(fixture.params1);
        double[] canonical2 = fixture.objective.canonicalizeParameters(fixture.params2);

        assertArrayCloseTo(
                "two starting points on the same rescaling ridge must canonicalize to the same "
                        + "embeddings",
                fixture.objective.extractEmbeddings(canonical1),
                fixture.objective.extractEmbeddings(canonical2),
                1e-9);
        assertArrayCloseTo(
                "two starting points on the same rescaling ridge must canonicalize to the same rates",
                fixture.objective.extractBrownianRates(canonical1),
                fixture.objective.extractBrownianRates(canonical2),
                1e-9);
        assertEquals(
                fixture.objective.extractIntercept(canonical1),
                fixture.objective.extractIntercept(canonical2),
                1e-9);
    }

    @Test
    public void canonicalization_setsUnitRmsEmbeddingNormPerAxis() {
        Fixture fixture = new Fixture();

        double[] canonical = fixture.objective.canonicalizeParameters(fixture.params2);
        double[] embeddings = fixture.objective.extractEmbeddings(canonical);

        for (int axis = 0; axis < fixture.dim; axis++) {
            double sumSquares = 0.0;
            for (int taxon = 0; taxon < fixture.taxonCount; taxon++) {
                double e = embeddings[taxon * fixture.dim + axis];
                sumSquares += e * e;
            }
            double rms = Math.sqrt(sumSquares / fixture.taxonCount);
            assertEquals("axis " + axis + " RMS embedding norm must be canonicalized to 1", 1.0, rms, 1e-9);
        }
    }

    @Test
    public void canonicalization_preservesObjectiveValueExactly() {
        Fixture fixture = new Fixture();

        double lossBefore = fixture.objective.value(fixture.params2);
        double[] canonical = fixture.objective.canonicalizeParameters(fixture.params2);
        double lossAfter = fixture.objective.value(canonical);

        assertEquals("canonicalization must not change the predicted log-likelihoods", lossBefore, lossAfter, 1e-9);
    }

    @Test
    public void canonicalization_isIdempotent() {
        Fixture fixture = new Fixture();

        double[] once = fixture.objective.canonicalizeParameters(fixture.params2);
        double[] twice = fixture.objective.canonicalizeParameters(once);

        assertArrayCloseTo("canonicalizing an already-canonical point must be a no-op", once, twice, 1e-9);
    }

    private static void assertArrayCloseTo(String message, double[] expected, double[] actual, double tolerance) {
        assertTrue(message + " (length mismatch)", expected.length == actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(message + " (index " + i + ")", expected[i], actual[i], tolerance);
        }
    }

    /** Two parameter vectors on the same embedding&lt;-&gt;rate rescaling ridge (axis 0). */
    private static final class Fixture {
        final int dim = 2;
        final int taxonCount;
        final LearnedWeightsTreeObjectiveFunction objective;
        final double[] params1;
        final double[] params2;

        Fixture() {
            Alignment alignment = createAlignment();
            taxonCount = alignment.getTaxonCount();
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = createSamples(alignment);
            int logNormalizerDimension = samples.get(0).logNormalizerDimension;

            // pcaRegularizationWeight = slopeRegularizationWeight = 0: an unregularized objective
            // isolates the fit-loss invariance under the ridge transform from the (deliberately
            // weak, and by design not ridge-invariant) regularization terms.
            objective = new LearnedWeightsTreeObjectiveFunction(samples, taxonCount, dim, 1.0, null, 0.0, 0.0);

            double[] embeddings = new double[] {
                    0.3, 0.6,
                    -0.5, 0.1,
                    0.9, -0.4,
                    -0.2, 0.7
            };
            double[] rates = new double[] {1.0, 1.0};
            double[] slopes = new double[] {0.7, -0.3};
            double intercept = 0.05;

            params1 = objective.createParameters(embeddings, rates, null, null, slopes, intercept);

            double[] embeddings2 = embeddings.clone();
            for (int taxon = 0; taxon < taxonCount; taxon++) {
                embeddings2[taxon * dim] *= RIDGE_SCALE_FACTOR;
            }
            double[] rates2 = rates.clone();
            rates2[0] *= RIDGE_SCALE_FACTOR * RIDGE_SCALE_FACTOR;
            double intercept2 = intercept + slopes[0] * logNormalizerDimension * Math.log(RIDGE_SCALE_FACTOR);

            params2 = objective.createParameters(embeddings2, rates2, null, null, slopes, intercept2);
        }
    }

    private static List<LearnedWeightsTreeTrainer.TrainingTreeSample> createSamples(Alignment alignment) {
        LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(null, alignment, 0);
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = new ArrayList<>();
        for (Tree tree : createTrees(alignment)) {
            samples.add(trainer.buildTrainingSample(tree, 0.0));
        }
        return samples;
    }

    private static Alignment createAlignment() {
        Sequence seqA = new Sequence("A", "ACGTACGTAA");
        Sequence seqB = new Sequence("B", "ACGTACGTAT");
        Sequence seqC = new Sequence("C", "ACGTACGTGA");
        Sequence seqD = new Sequence("D", "ACGTACGTGT");

        Alignment alignment = new Alignment();
        alignment.initByName("sequence", seqA,
                "sequence", seqB,
                "sequence", seqC,
                "sequence", seqD);
        return alignment;
    }

    private static List<Tree> createTrees(Alignment alignment) {
        List<Tree> trees = new ArrayList<>();
        trees.add(parse("((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);", alignment));
        trees.add(parse("((A:0.12,C:0.12):0.18,(B:0.1,D:0.1):0.2);", alignment));
        trees.add(parse("(((A:0.08,B:0.08):0.1,C:0.18):0.12,D:0.3);", alignment));
        return trees;
    }

    private static Tree parse(String newick, Alignment alignment) {
        Tree tree = new TreeParser();
        ((TreeParser) tree).initByName(
                "newick", newick,
                "IsLabelledNewick", true,
                "taxa", alignment);
        return tree;
    }
}
