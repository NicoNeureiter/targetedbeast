package targetedbeast.benchmark;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.Test;

import beast.base.parser.XMLParser;
import beast.base.util.Randomizer;

/**
 * Specification test for {@code BenchmarkMCMC}'s {@code checkpointOut} (Phase 3c, item 1).
 *
 * <p>The claim under test: the checkpoint is written exactly once, at the post-burnin/measured-
 * chain boundary (core's {@code sampleNr == 0}), independent of how long the measured chain
 * (chainLength) that follows turns out to be. Two runs from the same seed and the same preBurnin,
 * differing only in chainLength, must write byte-identical checkpoints - if the checkpoint were
 * instead written on every proposal (last-write-wins) or captured after the first proposal had
 * already been applied, the two contents would differ whenever chainLength differs.
 */
public class BenchmarkMCMCCheckpointTest {

    private static final long SEED = 7L;
    private static final long PRE_BURNIN = 300;

    @Test
    public void checkpointIsWrittenOnceAtTheBurninBoundaryIndependentOfChainLength() throws Exception {
        Path dir = Files.createTempDirectory("benchmark-checkpoint");
        PrimatesFixture.primeRandomizerThreadEntry();

        Path checkpointShort = dir.resolve("short.checkpoint.xml");
        String xmlShort = buildXml(dir, "short", PRE_BURNIN, 0, checkpointShort);
        Path xmlFileShort = dir.resolve("short.xml");
        PrimatesFixture.write(xmlShort, xmlFileShort);

        Path checkpointLong = dir.resolve("long.checkpoint.xml");
        String xmlLong = buildXml(dir, "long", PRE_BURNIN, 250, checkpointLong);
        Path xmlFileLong = dir.resolve("long.xml");
        PrimatesFixture.write(xmlLong, xmlFileLong);

        runToCompletion(xmlFileShort);
        runToCompletion(xmlFileLong);

        assertTrue("checkpoint (chainLength=0) was not written", Files.exists(checkpointShort));
        assertTrue("checkpoint (chainLength=250) was not written", Files.exists(checkpointLong));

        String contentShort = Files.readString(checkpointShort);
        String contentLong = Files.readString(checkpointLong);
        assertEquals("checkpoint content must not depend on chainLength - it is captured before "
                + "the first measured proposal, not after the chain finishes", contentShort, contentLong);

        // Sanity: it is a real, non-trivial state snapshot, not an empty/placeholder file.
        assertTrue("checkpoint should serialise the state nodes", contentShort.contains("tree"));
        assertTrue("checkpoint should serialise the state nodes", contentShort.contains("hky.kappa"));
        assertTrue("checkpoint should serialise the state nodes", contentShort.contains("popSize"));
    }

    private static String buildXml(Path dir, String label, long preBurnin, long chainLength,
            Path checkpointPath) throws Exception {
        String template = PrimatesFixture.loadTemplate();
        template = PrimatesFixture.withChainLength(template, chainLength);
        template = PrimatesFixture.withAbsoluteLogPaths(template, dir.resolve(label + ".log"),
                dir.resolve(label + ".trees"));
        String xml = PrimatesFixture.toBenchmarkVariant(template, dir.resolve(label), 5, null);
        xml = PrimatesFixture.withPreBurnin(xml, preBurnin);
        xml = PrimatesFixture.withCheckpointOut(xml, checkpointPath);
        return xml;
    }

    private static void runToCompletion(Path xmlFile) throws Exception {
        Randomizer.setSeed(SEED);
        XMLParser parser = new XMLParser();
        BenchmarkMCMC mcmc = (BenchmarkMCMC) parser.parseFile(new File(xmlFile.toString()));
        mcmc.run();
    }
}
