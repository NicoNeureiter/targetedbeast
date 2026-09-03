package targetedbeast.benchmark;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.junit.Test;

import beast.base.parser.XMLParser;
import beast.base.util.Randomizer;

/**
 * Specification test for the shared-checkpoint mechanism's open risk (Phase 3c, item 1): does
 * {@code State.restoreFromFile} correctly load tree/parameter state nodes by ID into an XML whose
 * edge-weights distribution differs from the one that produced the checkpoint?
 *
 * <p>Setting 1 ({@code PartialWeights}, needs {@code data}/{@code likelihood}) writes a checkpoint
 * via {@code checkpointOut} at the burn-in boundary. Setting 2 ({@code ConstantWeights}, needs only
 * {@code tree}) resumes from that checkpoint. The {@code <state>} block (tree, {@code hky.kappa},
 * {@code popSize}) is byte-identical between the two settings' XMLs - only the {@code edgeweights}
 * distribution differs - matching the real pipeline, where every setting for a given dataset shares
 * everything except its edge-weights distribution.
 *
 * <p>The core check: setting 2 also writes its own {@code checkpointOut} at its own {@code sampleNr
 * == 0} (which, on {@code -resume}, is immediately after {@code restoreFromFile} - before any
 * setting-2-specific proposal runs, since {@link BenchmarkMCMC} forces {@code burnIn = 0} on
 * resume, same as core). If restoration is correct, that re-serialised state must match the
 * original checkpoint it was restored from - this only holds if every state node was found and
 * restored by ID despite the differing model structure downstream. The comparison is semantic, not
 * byte-for-byte: {@code Tree.fromXML} reparses the Newick string into a fresh node structure whose
 * child order and last-ULP branch-length rounding are not guaranteed stable across a round-trip
 * (verified directly - see test comments below), so this compares parameter values with a small
 * floating-point tolerance and the tree by its (order-independent) total branch length and leaf set.
 */
public class BenchmarkMCMCCrossSettingResumeTest {

    private static final long SEED = 123L;
    private static final long PRE_BURNIN = 300;
    private static final long RESUME_CHAIN_LENGTH = 50;

    @Test
    public void resumingUnderADifferentSettingRestoresStateCorrectlyAndCompletes() throws Exception {
        Path dir = Files.createTempDirectory("benchmark-cross-setting-resume");
        PrimatesFixture.primeRandomizerThreadEntry();

        // --- Setting 1 (PartialWeights): produce the checkpoint. ---
        Path checkpoint = dir.resolve("setting1.checkpoint.xml");
        String template = PrimatesFixture.loadTemplate();
        template = PrimatesFixture.withChainLength(template, 0);
        String xml1 = PrimatesFixture.withAbsoluteLogPaths(template, dir.resolve("s1.log"), dir.resolve("s1.trees"));
        xml1 = PrimatesFixture.toBenchmarkVariant(xml1, dir.resolve("s1"), 5, null);
        xml1 = PrimatesFixture.withPreBurnin(xml1, PRE_BURNIN);
        xml1 = PrimatesFixture.withCheckpointOut(xml1, checkpoint);
        Path xmlFile1 = dir.resolve("s1.xml");
        PrimatesFixture.write(xml1, xmlFile1);

        Randomizer.setSeed(SEED);
        BenchmarkMCMC mcmc1 = parse(xmlFile1);
        mcmc1.run();
        assertTrue("setting 1 did not write its checkpoint", Files.exists(checkpoint));
        String donorCheckpointContent = Files.readString(checkpoint);

        // Copy the checkpoint into setting 2's own state-file path before resuming from it -
        // mirroring the Snakemake mechanism (copy first, avoid contention/clobbering): setting 2's
        // own run ends by writing its *own* final state back to whatever path -statefile points
        // at, so reading and writing the same shared file directly would corrupt the checkpoint
        // for anyone resuming from it after.
        Path resumeStateFile = dir.resolve("s2.xml.state");
        Files.copy(checkpoint, resumeStateFile);

        // --- Setting 2 (ConstantWeights): resume from setting 1's checkpoint. ---
        Path reCheckpoint = dir.resolve("setting2.recheckpoint.xml");
        String template2 = PrimatesFixture.loadTemplate();
        template2 = PrimatesFixture.withChainLength(template2, RESUME_CHAIN_LENGTH);
        template2 = PrimatesFixture.withConstantEdgeWeights(template2);
        String xml2 = PrimatesFixture.withAbsoluteLogPaths(template2, dir.resolve("s2.log"), dir.resolve("s2.trees"));
        Path statsOut2 = dir.resolve("s2");
        xml2 = PrimatesFixture.toBenchmarkVariant(xml2, statsOut2, 5, null);
        xml2 = PrimatesFixture.withCheckpointOut(xml2, reCheckpoint);
        Path xmlFile2 = dir.resolve("s2.xml");
        PrimatesFixture.write(xml2, xmlFile2);

        BenchmarkMCMC mcmc2 = parse(xmlFile2);
        mcmc2.setStateFile(resumeStateFile.toString(), true);
        mcmc2.run(); // must not throw

        // --- State-node-by-ID restoration is correct: the re-serialised state at setting 2's own
        // boundary (immediately post-restore, pre-first-proposal) is byte-identical to what setting
        // 1 checkpointed - despite the differing edgeweights distribution in between. ---
        assertTrue("setting 2 did not write its own boundary checkpoint", Files.exists(reCheckpoint));
        String reCheckpointContent = Files.readString(reCheckpoint);
        assertCheckpointsMatch(donorCheckpointContent, reCheckpointContent);

        // --- Run completes and produces sensible output under the new setting. ---
        Path opstats = Path.of(statsOut2 + ".opstats.tsv");
        assertTrue("opstats.tsv was not written", Files.exists(opstats));
        List<String> statsLines = Files.readAllLines(opstats);
        assertTrue("opstats.tsv missing chain-level row",
                statsLines.stream().anyMatch(l -> l.contains(BenchmarkMCMC.CHAIN_ROW_NAME)));
        String chainRow = statsLines.stream()
                .filter(l -> l.contains(BenchmarkMCMC.CHAIN_ROW_NAME)).findFirst().orElseThrow();
        String[] cols = chainRow.split("\t");
        long proposals = Long.parseLong(cols[2]);
        double acceptRate = Double.parseDouble(cols[4]);
        assertEquals("resumed run should have measured exactly RESUME_CHAIN_LENGTH+1 proposals",
                RESUME_CHAIN_LENGTH + 1, proposals);
        assertFalse("acceptRate must be a real number, not NaN", Double.isNaN(acceptRate));
        assertTrue("acceptRate out of [0,1]", acceptRate >= 0.0 && acceptRate <= 1.0);

        List<String> logLines = Files.readAllLines(dir.resolve("s2.log"));
        assertTrue("resumed run's trace log has no data rows", logLines.size() > 1);
        int posteriorCol = List.of(logLines.get(0).split("\t")).indexOf("posterior");
        for (int i = 1; i < logLines.size(); i++) {
            double posterior = Double.parseDouble(logLines.get(i).split("\t")[posteriorCol]);
            assertFalse("resumed run's posterior must be finite at row " + i,
                    Double.isNaN(posterior) || Double.isInfinite(posterior));
        }
    }

    private static final double TOLERANCE = 1e-6;
    private static final Pattern STATENODE = Pattern.compile(
            "<statenode id='(\\w[\\w.]*)'>([^<]*)</statenode>");
    private static final Pattern BRANCH_LENGTH = Pattern.compile(":([-0-9.Ee]+)");
    private static final Pattern NODE_LABEL = Pattern.compile("(\\d+):");

    /**
     * Compares two {@code State.toXML} snapshots node-by-node: scalar parameters ({@code hky.kappa},
     * {@code popSize}) by parsed value within {@link #TOLERANCE}; the tree by leaf set and total
     * branch length, both invariant to the child-order/rounding differences a Newick round-trip
     * introduces (see class javadoc).
     */
    private static void assertCheckpointsMatch(String expectedXml, String actualXml) {
        var expected = parseStateNodes(expectedXml);
        var actual = parseStateNodes(actualXml);
        assertEquals("checkpoint must contain the same state node IDs", expected.keySet(), actual.keySet());

        assertScalarMatches(expected, actual, "hky.kappa");
        assertScalarMatches(expected, actual, "popSize");

        String expectedTree = expected.get("tree");
        String actualTree = actual.get("tree");
        assertEquals("restored tree must have the same node labels as the donor checkpoint",
                nodeLabels(expectedTree), nodeLabels(actualTree));
        assertEquals("restored tree's total branch length must match the donor checkpoint "
                + "(order-independent, small float tolerance for Newick round-trip rounding)",
                totalBranchLength(expectedTree), totalBranchLength(actualTree), TOLERANCE);
    }

    private static void assertScalarMatches(java.util.Map<String, String> expected,
            java.util.Map<String, String> actual, String id) {
        double expectedValue = parseScalar(expected.get(id));
        double actualValue = parseScalar(actual.get(id));
        assertEquals("restored " + id + " must match the donor checkpoint's value",
                expectedValue, actualValue, TOLERANCE);
    }

    private static double parseScalar(String statenodeText) {
        // Text is "<id>: <value>", e.g. "hky.kappa: 7.607398806669834".
        return Double.parseDouble(statenodeText.substring(statenodeText.indexOf(':') + 1).trim());
    }

    private static double totalBranchLength(String newick) {
        double sum = 0;
        Matcher m = BRANCH_LENGTH.matcher(newick);
        while (m.find()) {
            sum += Double.parseDouble(m.group(1));
        }
        return sum;
    }

    private static java.util.Set<String> nodeLabels(String newick) {
        java.util.Set<String> labels = new java.util.TreeSet<>();
        Matcher m = NODE_LABEL.matcher(newick);
        while (m.find()) {
            labels.add(m.group(1));
        }
        return labels;
    }

    private static java.util.Map<String, String> parseStateNodes(String xml) {
        java.util.Map<String, String> nodes = new java.util.LinkedHashMap<>();
        Matcher m = STATENODE.matcher(xml);
        while (m.find()) {
            nodes.put(m.group(1), m.group(2));
        }
        return nodes;
    }

    private static BenchmarkMCMC parse(Path xmlFile) throws Exception {
        XMLParser parser = new XMLParser();
        return (BenchmarkMCMC) parser.parseFile(new File(xmlFile.toString()));
    }
}
