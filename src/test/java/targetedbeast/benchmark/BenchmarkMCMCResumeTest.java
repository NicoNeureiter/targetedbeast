package targetedbeast.benchmark;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.Test;

import beast.base.parser.XMLParser;
import beast.base.util.Randomizer;

/**
 * Specification test for {@link BenchmarkMCMC} checkpoint/resume transparency (Phase 3c).
 *
 * <p>BEAST's own {@code -resume} mechanism ({@code MCMC.run()}: {@code state.restoreFromFile()},
 * {@code burnIn} forced to 0, {@code sampleNr} restarting at 0 for the resumed segment) does
 * <em>not</em> persist {@link Randomizer}'s RNG state across a resume - {@code Randomizer.storeToFile}
 * is unimplemented in this BEAST build. So resuming twice from the same checkpoint is only
 * byte-reproducible if the RNG is deliberately put back into the same state both times; real-world
 * resumes get fresh entropy after the checkpoint by design (Phase 3c: "every setting's reps resume
 * from that *same* checkpoint" but each rep still needs its own randomness).
 *
 * <p>This test isolates exactly one question: given the <em>same</em> post-checkpoint RNG stream
 * (engineered here by never reseeding between the checkpoint run and the resumed run, since
 * {@code Randomizer} is a single JVM-wide generator per thread), does {@code BenchmarkMCMC}'s
 * instrumentation for the resumed segment match, proposal for proposal, what the identical segment
 * looks like inside one uninterrupted run? If resume corrupted the {@code sampleNr >= 0} postBurnin
 * gate, mis-attributed operators, or lost the accept/reject bookkeeping across the boundary, this
 * would show up as a mismatch here.
 */
public class BenchmarkMCMCResumeTest {

    private static final long SEED = 99L;
    private static final long CHECKPOINT_LENGTH = 800;
    private static final long RESUME_LENGTH = 400;
    private static final int JUMP_INTERVAL = 1;

    @Test
    public void resumedSegmentInstrumentationMatchesTheEquivalentContinuousTail() throws Exception {
        Path dir = Files.createTempDirectory("benchmark-resume");
        PrimatesFixture.primeRandomizerThreadEntry();
        Path statePath = dir.resolve("checkpoint.state.xml");

        // --- Run A: fresh, checkpointed at the end. ---
        Path traceA = dir.resolve("A.trace.tsv");
        String templateA = PrimatesFixture.loadTemplate();
        templateA = PrimatesFixture.withChainLength(templateA, CHECKPOINT_LENGTH);
        templateA = PrimatesFixture.withAbsoluteLogPaths(templateA, dir.resolve("A.log"), dir.resolve("A.trees"));
        String xmlA = PrimatesFixture.toBenchmarkVariant(templateA, dir.resolve("A"), JUMP_INTERVAL, traceA);
        Path xmlFileA = dir.resolve("A.xml");
        PrimatesFixture.write(xmlA, xmlFileA);

        Randomizer.setSeed(SEED);
        BenchmarkMCMC mcmcA = parse(xmlFileA);
        mcmcA.setStateFile(statePath.toString(), false);
        mcmcA.run();

        // --- Run B: resumes from A's checkpoint, RNG left exactly where A finished (no reseed). ---
        // Core's doLoop is `sampleNr = -burnIn; sampleNr <= chainLength; sampleNr++` - inclusive at
        // both ends - so a chainLength=X run performs X+1 proposals. To make B perform exactly
        // RESUME_LENGTH proposals (matching C's tail below), its own chainLength input must be
        // RESUME_LENGTH - 1.
        Path traceB = dir.resolve("B.trace.tsv");
        String templateB = PrimatesFixture.loadTemplate();
        templateB = PrimatesFixture.withChainLength(templateB, RESUME_LENGTH - 1);
        templateB = PrimatesFixture.withAbsoluteLogPaths(templateB, dir.resolve("B.log"), dir.resolve("B.trees"));
        String xmlB = PrimatesFixture.toBenchmarkVariant(templateB, dir.resolve("B"), JUMP_INTERVAL, traceB);
        Path xmlFileB = dir.resolve("B.xml");
        PrimatesFixture.write(xmlB, xmlFileB);

        BenchmarkMCMC mcmcB = parse(xmlFileB);
        mcmcB.setStateFile(statePath.toString(), true);
        mcmcB.run();

        assertEquals("resumed run should have tracked exactly RESUME_LENGTH post-burnin proposals",
                RESUME_LENGTH, mcmcB.getOperatorStatsForTesting().values().stream().mapToLong(s -> s.proposals).sum());

        // --- Run C: same seed, one uninterrupted run of CHECKPOINT_LENGTH + RESUME_LENGTH. ---
        Path traceC = dir.resolve("C.trace.tsv");
        String templateC = PrimatesFixture.loadTemplate();
        templateC = PrimatesFixture.withChainLength(templateC, CHECKPOINT_LENGTH + RESUME_LENGTH);
        templateC = PrimatesFixture.withAbsoluteLogPaths(templateC, dir.resolve("C.log"), dir.resolve("C.trees"));
        String xmlC = PrimatesFixture.toBenchmarkVariant(templateC, dir.resolve("C"), JUMP_INTERVAL, traceC);
        Path xmlFileC = dir.resolve("C.xml");
        PrimatesFixture.write(xmlC, xmlFileC);

        Randomizer.setSeed(SEED);
        BenchmarkMCMC mcmcC = parse(xmlFileC);
        mcmcC.run();

        // Sanity: A and C must agree over the shared prefix - otherwise the RNG-continuity setup
        // itself is broken and the tail comparison below would be meaningless.
        List<String> linesA = Files.readAllLines(traceA);
        List<String> linesC = Files.readAllLines(traceC);
        assertEquals("run A should have exactly CHECKPOINT_LENGTH+1 data rows (doLoop is inclusive "
                + "at both ends)", CHECKPOINT_LENGTH + 1, linesA.size() - 1);
        for (int i = 0; i < linesA.size(); i++) {
            assertEquals("run A and the shared prefix of run C must be identical at row " + i,
                    linesA.get(i), linesC.get(i));
        }

        // The actual claim: the resumed run B's own trace, proposal for proposal, matches the
        // corresponding tail of the continuous run C - proving the resume boundary perturbed
        // nothing in BenchmarkMCMC's bookkeeping.
        List<String> linesB = Files.readAllLines(traceB);
        assertEquals("run B should have exactly RESUME_LENGTH data rows", RESUME_LENGTH, linesB.size() - 1);
        assertEquals("header", linesB.get(0), linesC.get(0));

        // Data row for global sampleNr==CHECKPOINT_LENGTH (A's own last proposal, the checkpoint
        // itself) is at line index CHECKPOINT_LENGTH+1; the resumed segment's first proposal is the
        // next one, at CHECKPOINT_LENGTH+2.
        List<String> tailOfC = linesC.subList((int) CHECKPOINT_LENGTH + 2, linesC.size());
        assertEquals(RESUME_LENGTH, tailOfC.size());
        for (int i = 0; i < tailOfC.size(); i++) {
            // Column 0 is sampleNr, which legitimately differs (B restarts at 0, C continues from
            // CHECKPOINT_LENGTH) - compare every other column: operator, accepted, jump distance(s).
            String[] colsB = linesB.get(1 + i).split("\t", 2);
            String[] colsC = tailOfC.get(i).split("\t", 2);
            assertEquals("resumed proposal " + i + ": operator/accepted/jump columns must match "
                    + "the continuous run's tail exactly", colsC[1], colsB[1]);
        }
    }

    private static BenchmarkMCMC parse(Path xmlFile) throws Exception {
        XMLParser parser = new XMLParser();
        return (BenchmarkMCMC) parser.parseFile(new File(xmlFile.toString()));
    }
}
