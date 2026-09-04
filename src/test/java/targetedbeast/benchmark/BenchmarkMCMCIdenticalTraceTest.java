package targetedbeast.benchmark;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.Test;

import beast.base.inference.Runnable;
import beast.base.parser.XMLParser;
import beast.base.util.Randomizer;

/**
 * Specification test for {@link BenchmarkMCMC}.
 *
 * <p>The whole point of subclassing {@code MCMC} to instrument it (rather than post-processing a
 * log, or wrapping operators) is that the instrumented chain must be the <em>same</em> chain: same
 * proposals, same accept/reject decisions, same sampled trace - only additional bookkeeping is
 * added on the side. This test is the proof: given the same seed and the same model
 * ({@code examples/primates.xml}, shortened to a few thousand states for test speed), a plain
 * {@code MCMC} run and a {@code BenchmarkMCMC} run must produce byte-identical trace logs.
 */
public class BenchmarkMCMCIdenticalTraceTest {

    private static final long SEED = 42L;
    private static final long CHAIN_LENGTH = 4000;

    @Test
    public void benchmarkMCMCProducesIdenticalTraceToPlainMCMC() throws Exception {
        Path dir = Files.createTempDirectory("benchmark-identical-trace");

        String template = PrimatesFixture.withChainLength(PrimatesFixture.loadTemplate(), CHAIN_LENGTH);

        Path plainLog = dir.resolve("plain.log");
        Path plainTrees = dir.resolve("plain.trees");
        String plainXml = PrimatesFixture.withAbsoluteLogPaths(template, plainLog, plainTrees);
        Path plainXmlFile = dir.resolve("plain.xml");
        PrimatesFixture.write(plainXml, plainXmlFile);

        Path benchLog = dir.resolve("bench.log");
        Path benchTrees = dir.resolve("bench.trees");
        String benchXml = PrimatesFixture.withAbsoluteLogPaths(template, benchLog, benchTrees);
        Path statsOut = dir.resolve("bench");
        benchXml = PrimatesFixture.toBenchmarkVariant(benchXml, statsOut, 5, null);
        Path benchXmlFile = dir.resolve("bench.xml");
        PrimatesFixture.write(benchXml, benchXmlFile);

        runToCompletion(plainXmlFile);
        runToCompletion(benchXmlFile);

        assertTrue("plain MCMC did not write a trace log", Files.exists(plainLog));
        assertTrue("BenchmarkMCMC did not write a trace log", Files.exists(benchLog));

        List<String> plainLines = Files.readAllLines(plainLog);
        List<String> benchLines = Files.readAllLines(benchLog);
        assertEquals("trace log line count differs", plainLines.size(), benchLines.size());
        for (int i = 0; i < plainLines.size(); i++) {
            assertEquals("trace log differs at line " + i, plainLines.get(i), benchLines.get(i));
        }

        // BenchmarkMCMC's own report is the other half of the proof: it must actually have run and
        // recorded something, not silently skipped its own instrumentation.
        Path opstats = Path.of(statsOut + ".opstats.tsv");
        assertTrue("opstats.tsv was not written", Files.exists(opstats));
        List<String> statsLines = Files.readAllLines(opstats);
        assertTrue("opstats.tsv has no data rows", statsLines.size() > 1);
        assertTrue("opstats.tsv missing chain-level row",
                statsLines.stream().anyMatch(l -> l.contains(BenchmarkMCMC.CHAIN_ROW_NAME)));
    }

    private static void runToCompletion(Path xmlFile) throws Exception {
        Randomizer.setSeed(SEED);
        XMLParser parser = new XMLParser();
        Runnable runnable = parser.parseFile(new File(xmlFile.toString()));
        runnable.run();
    }
}
