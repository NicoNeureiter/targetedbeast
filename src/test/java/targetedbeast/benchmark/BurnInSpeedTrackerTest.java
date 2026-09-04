package targetedbeast.benchmark;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Fine-grained tests for {@link BurnInSpeedTracker} against synthetic log-posterior sequences -
 * no MCMC chain needed, so the "5 consecutive, not 5 total" requirement and the "never reached"
 * case can be pinned down exactly.
 */
public class BurnInSpeedTrackerTest {

    private static final double LOWER = -10.0;
    private static final double UPPER = -5.0;

    @Test
    public void reportsFirstProposalOfTheFirstQualifyingRunOfFive() {
        BurnInSpeedTracker tracker = new BurnInSpeedTracker(LOWER, UPPER);
        // proposals 1-3 out of band, 4-8 in band (5 consecutive) -> target is proposal 4.
        double[] values = {-20, -20, -20, -7, -7, -7, -7, -7, -7, -7};
        for (double v : values) {
            tracker.update(v);
        }
        assertTrue(tracker.isReached());
        assertEquals(4, tracker.getTarget());
    }

    @Test
    public void singleExcursionOutOfBandResetsTheStreak() {
        BurnInSpeedTracker tracker = new BurnInSpeedTracker(LOWER, UPPER);
        // 4 in-band, then 1 excursion out, then 5 in-band: the first run of 4 must NOT trigger -
        // only the second, later run of 5 (starting at proposal 6) should.
        double[] values = {-7, -7, -7, -7, -20, -7, -7, -7, -7, -7};
        for (int i = 0; i < values.length; i++) {
            tracker.update(values[i]);
            if (i < 8) {
                assertFalse("should not have reached target before the 9th proposal (index " + i + ")",
                        tracker.isReached());
            }
        }
        assertTrue(tracker.isReached());
        assertEquals(6, tracker.getTarget());
    }

    @Test
    public void fourInBandThenBackOutNeverTriggers() {
        BurnInSpeedTracker tracker = new BurnInSpeedTracker(LOWER, UPPER);
        double[] values = {-7, -7, -7, -7, -20};
        for (double v : values) {
            tracker.update(v);
        }
        assertFalse("4 consecutive in-band proposals must not satisfy a 5-consecutive requirement",
                tracker.isReached());
        assertEquals(-1, tracker.getTarget());
    }

    @Test
    public void unreachablyTightBandNeverReachesAndReportsMinusOne() {
        BurnInSpeedTracker tracker = new BurnInSpeedTracker(-6.0000001, -6.0);
        for (int i = 0; i < 1000; i++) {
            tracker.update(-7.0 + 0.01 * (i % 200)); // wanders between -7.0 and -5.0, never in band
        }
        assertFalse(tracker.isReached());
        assertEquals(-1, tracker.getTarget());
        assertEquals(1000, tracker.getProposalsSeen());
    }

    @Test
    public void boundsAreInclusive() {
        BurnInSpeedTracker tracker = new BurnInSpeedTracker(LOWER, UPPER);
        for (int i = 0; i < 5; i++) {
            tracker.update(UPPER); // exactly at the boundary
        }
        assertTrue("upper bound should be inclusive", tracker.isReached());
        assertEquals(1, tracker.getTarget());
    }

    @Test
    public void updatesAfterReachedAreNoOpsForTargetButStillCountProposalsSeen() {
        BurnInSpeedTracker tracker = new BurnInSpeedTracker(LOWER, UPPER);
        for (int i = 0; i < 5; i++) {
            tracker.update(-7);
        }
        assertTrue(tracker.isReached());
        long targetBefore = tracker.getTarget();
        tracker.update(-100); // way out of band, after target already reached
        assertEquals("target must not change once reached", targetBefore, tracker.getTarget());
        assertEquals(6, tracker.getProposalsSeen());
    }
}
