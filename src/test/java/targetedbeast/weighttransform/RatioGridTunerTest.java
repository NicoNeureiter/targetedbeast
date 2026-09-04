package targetedbeast.weighttransform;

import beast.base.internal.json.JSONException;
import beast.base.internal.json.JSONObject;
import org.junit.Assert;
import org.junit.Test;

import beast.base.util.Randomizer;

/**
 * The ratio tuner against a reference transform with a known, fixed exponent.
 *
 * <p>Unlike {@link GridTunerTest}, the quantity under test is the composition
 * {@code reference exponent * ratio}, not an independently learned absolute value.
 */
public class RatioGridTunerTest {

    private static final double TOL = 1e-9;
    private static final String RATIOS = "0.7 1.0 1.4";

    /** A reference transform pinned at a known exponent, never adapted. */
    private static PowerOffsetTransform pinnedReference(double exponent) {
        PowerOffsetTransform reference = new PowerOffsetTransform();
        reference.initByName("exponent", exponent, "offset", 0.0, "optimise", false);
        return reference;
    }

    private static PowerOffsetTransform withRatioTuner(WeightTransform reference,
            Object... tunerArgs) {
        Object[] args = new Object[tunerArgs.length + 4];
        args[0] = "ratios";
        args[1] = RATIOS;
        args[2] = "reference";
        args[3] = reference;
        System.arraycopy(tunerArgs, 0, args, 4, tunerArgs.length);

        RatioGridTuner tuner = new RatioGridTuner();
        tuner.initByName(args);

        PowerOffsetTransform t = new PowerOffsetTransform();
        t.initByName("exponent", 2.0, "offset", 0.0, "tuner", tuner, "maxExponent", 20.0);
        return t;
    }

    private static RatioGridTuner tunerOf(WeightTransform t) {
        return (RatioGridTuner) t.getTuner();
    }

    /** Every effective exponent must equal referenceExponent * one of the configured ratios. */
    @Test
    public void chosenRatioComposesWithTheReferencesLiveExponent() {
        Randomizer.setSeed(21);
        PowerOffsetTransform reference = pinnedReference(5.0);
        PowerOffsetTransform t = withRatioTuner(reference, "explorationProb", 1.0);
        double[] ratios = { 0.7, 1.0, 1.4 };

        for (int i = 0; i < 500; i++) {
            reference.startProposal();
            t.startProposal();
            double effective = t.getEffectiveCoefficients()[0];
            boolean matches = false;
            for (double r : ratios) {
                matches |= Math.abs(effective - 5.0 * r) < TOL;
            }
            Assert.assertTrue("effective exponent " + effective + " is not 5.0 * one of "
                    + RATIOS, matches);
            t.apply(new double[] { 1.0, 2.0, 3.0 });
            t.optimize(Math.log(0.5));
        }
    }

    /** When the reference's exponent moves, the owning transform's proposal tracks it. */
    @Test
    public void trackingHoldsAsTheReferenceExponentChanges() {
        Randomizer.setSeed(22);
        PowerOffsetTransform reference = pinnedReference(2.0);
        PowerOffsetTransform t = withRatioTuner(reference, "explorationProb", 1.0);

        t.startProposal();
        double first = t.getEffectiveCoefficients()[0];
        Assert.assertTrue(first == 2.0 * 0.7 || first == 2.0 * 1.0 || first == 2.0 * 1.4);

        // Reference re-pinned at a different exponent, as it would be after its own tuner moved.
        reference.setCoefficients(new double[] { 6.0, 1e-6 });
        t.startProposal();
        double second = t.getEffectiveCoefficients()[0];
        Assert.assertTrue("expected a ratio of the new reference exponent 6.0, got " + second,
                Math.abs(second - 6.0 * 0.7) < TOL || Math.abs(second - 6.0 * 1.0) < TOL
                        || Math.abs(second - 6.0 * 1.4) < TOL);
    }

    /** A ratio that would push the product above maxExponent must be clamped, not proposed raw. */
    @Test
    public void productsBeyondMaxExponentAreClamped() {
        Randomizer.setSeed(23);
        // Reference exponent 10, top ratio 1.4 -> product 14, above a maxExponent of 8.
        PowerOffsetTransform reference = pinnedReference(10.0);
        RatioGridTuner tuner = new RatioGridTuner();
        tuner.initByName("ratios", RATIOS, "reference", reference, "explorationProb", 1.0);
        PowerOffsetTransform t = new PowerOffsetTransform();
        t.initByName("exponent", 2.0, "offset", 0.0, "tuner", tuner, "maxExponent", 8.0);

        boolean sawClamp = false;
        for (int i = 0; i < 200; i++) {
            t.startProposal();
            double effective = t.getEffectiveCoefficients()[0];
            Assert.assertTrue("effective exponent must never exceed maxExponent",
                    effective <= 8.0 + TOL);
            if (Math.abs(effective - 8.0) < 1e-6) {
                sawClamp = true;
            }
        }
        Assert.assertTrue("the top ratio (1.4 * 10 = 14) should have been clamped to 8.0 at "
                + "least once over 200 draws", sawClamp);
    }

    /** The accumulated reward curve must survive a resume, mirroring GridTunerTest's coverage. */
    @Test
    public void theRewardCurveSurvivesAStoreRestoreRoundTrip() throws JSONException {
        Randomizer.setSeed(24);
        PowerOffsetTransform reference = pinnedReference(4.0);
        PowerOffsetTransform t = withRatioTuner(reference, "burnin", 0, "explorationProb", 0.3);

        double[] dummyWeights = { 1.0, 2.0, 3.0 };
        for (int i = 0; i < 5_000; i++) {
            reference.startProposal();
            t.startProposal();
            t.apply(dummyWeights);
            t.optimize(Randomizer.nextDouble() < 0.3 ? 0.0 : Double.NEGATIVE_INFINITY);
        }

        JSONObject entry = new JSONObject();
        t.storeState(entry);

        PowerOffsetTransform restoredReference = pinnedReference(4.0);
        PowerOffsetTransform restoredTransform = withRatioTuner(restoredReference, "burnin", 0,
                "explorationProb", 0.3);
        restoredTransform.restoreState(entry);

        RatioGridTuner before = tunerOf(t);
        RatioGridTuner after = tunerOf(restoredTransform);
        Assert.assertEquals(before.getAdaptationCount(), after.getAdaptationCount());
        Assert.assertEquals(before.getLeadingRatio(), after.getLeadingRatio(), TOL);
        for (int i = 0; i < 3; i++) {
            Assert.assertEquals(before.getMeanReward(i), after.getMeanReward(i), TOL);
            Assert.assertEquals(before.getSampleCount(i), after.getSampleCount(i));
        }
    }

    /** A state file whose grid size no longer matches the configured ratios must be rejected. */
    @Test
    public void aGridSizeMismatchOnRestoreIsRejected() throws JSONException {
        PowerOffsetTransform reference = pinnedReference(4.0);
        PowerOffsetTransform t = withRatioTuner(reference, "burnin", 0);

        JSONObject entry = new JSONObject();
        t.storeState(entry);

        PowerOffsetTransform otherReference = pinnedReference(4.0);
        RatioGridTuner smallerTuner = new RatioGridTuner();
        smallerTuner.initByName("ratios", "0.7 1.4", "reference", otherReference);
        PowerOffsetTransform smaller = new PowerOffsetTransform();
        smaller.initByName("exponent", 2.0, "tuner", smallerTuner, "maxExponent", 20.0);

        Assert.assertThrows(JSONException.class, () -> smaller.restoreState(entry));
    }

    /** A single-point grid is not a grid, and would silently disable the tuning. */
    @Test
    public void aDegenerateGridIsRejected() {
        PowerOffsetTransform reference = pinnedReference(4.0);
        RatioGridTuner tuner = new RatioGridTuner();
        RuntimeException e = Assert.assertThrows(RuntimeException.class,
                () -> tuner.initByName("ratios", "1.0", "reference", reference));
        Assert.assertTrue(e.getMessage(), e.getMessage().contains("at least two values"));
    }

    /** An exploration probability of zero lets the tuner abandon grid points permanently. */
    @Test
    public void aZeroExplorationFloorIsRejected() {
        PowerOffsetTransform reference = pinnedReference(4.0);
        RatioGridTuner tuner = new RatioGridTuner();
        RuntimeException e = Assert.assertThrows(RuntimeException.class,
                () -> tuner.initByName("ratios", RATIOS, "reference", reference,
                        "explorationProb", 0.0));
        Assert.assertTrue(e.getMessage(), e.getMessage().contains("explorationProb"));
    }
}
