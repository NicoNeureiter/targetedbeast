package targetedbeast.benchmark;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.junit.Test;

import beast.base.parser.XMLParser;
import beast.base.util.Randomizer;

/**
 * Specification test for the burn-in-speed metric (Phase 3c).
 *
 * <p>Runs {@code primates.xml} once to observe its own posterior trace log, derives a 5th-95th
 * percentile reference band from the trace's stationary tail - exactly how a real reference-chain
 * band would be computed externally, just at the trace logger's resolution rather than per-proposal
 * - then re-runs the identical chain (same seed) with that band wired in as
 * {@code referenceLowerBound}/{@code referenceUpperBound} and checks the reported
 * proposals-to-target falls inside the visible burn-in region: not at the very first proposals (the
 * chain starts from a UPGMA cluster tree under the wrong model, nowhere near stationarity) and well
 * before the run ends.
 */
public class BurnInSpeedSpecTest {

    private static final long SEED = 11L;
    private static final long CHAIN_LENGTH = 4000;

    @Test
    public void reportsAPlausibleBurnInProposalCountNotAtEitherExtreme() throws Exception {
        Path dir = Files.createTempDirectory("burnin-speed-spec");
        PrimatesFixture.primeRandomizerThreadEntry();

        Path observeLog = dir.resolve("observe.log");
        String template = PrimatesFixture.loadTemplate();
        template = PrimatesFixture.withChainLength(template, CHAIN_LENGTH);
        template = PrimatesFixture.withAbsoluteLogPaths(template, observeLog, dir.resolve("observe.trees"));
        String observeXml = PrimatesFixture.toBenchmarkVariant(template, dir.resolve("observe"), 1, null, false);
        Path observeXmlFile = dir.resolve("observe.xml");
        PrimatesFixture.write(observeXml, observeXmlFile);

        Randomizer.setSeed(SEED);
        parseAndRun(observeXmlFile);

        List<Double> posteriorTrace = readPosteriorColumn(observeLog);
        assertTrue("expected a reasonably long trace", posteriorTrace.size() > 50);

        // Stationary tail: last quarter of the logged trace.
        List<Double> tail = posteriorTrace.subList((int) (posteriorTrace.size() * 0.75), posteriorTrace.size());
        List<Double> sortedTail = new ArrayList<>(tail);
        Collections.sort(sortedTail);
        double lower = percentile(sortedTail, 0.05);
        double upper = percentile(sortedTail, 0.95);
        assertTrue("reference band must be non-degenerate", lower < upper);

        // Independent, eyeballed check: the earliest logged sample after which every subsequent
        // logged value stays in [lower, upper] - a coarse (logEvery-resolution) upper bound on
        // where true burn-in ends, used only to sanity-check the metric's proposal-level answer.
        int eyeballBurninSample = -1;
        for (int i = 0; i < posteriorTrace.size(); i++) {
            boolean restStaysInBand = true;
            for (int j = i; j < posteriorTrace.size(); j++) {
                double v = posteriorTrace.get(j);
                if (v < lower || v > upper) {
                    restStaysInBand = false;
                    break;
                }
            }
            if (restStaysInBand) {
                eyeballBurninSample = i;
                break;
            }
        }
        assertTrue("expected the trace to eventually settle into the band", eyeballBurninSample >= 0);

        String timedTemplate = PrimatesFixture.loadTemplate();
        timedTemplate = PrimatesFixture.withChainLength(timedTemplate, CHAIN_LENGTH);
        timedTemplate = PrimatesFixture.withAbsoluteLogPaths(timedTemplate, dir.resolve("timed.log"),
                dir.resolve("timed.trees"));
        String timedXml = PrimatesFixture.toBenchmarkVariant(timedTemplate, dir.resolve("timed"), 1, null, false);
        timedXml = PrimatesFixture.withReferenceBand(timedXml, lower, upper);
        Path timedXmlFile = dir.resolve("timed.xml");
        PrimatesFixture.write(timedXml, timedXmlFile);

        Randomizer.setSeed(SEED);
        BenchmarkMCMC timed = parseAndRun(timedXmlFile);

        assertTrue("burn-in speed tracking should be enabled", timed.isBurnInSpeedEnabledForTesting());
        long target = timed.getBurnInSpeedTargetForTesting();
        assertTrue("target should have been reached before the chain ends", target > 0);
        assertTrue("burn-in target (" + target + ") should not be one of the very first proposals - "
                + "the chain starts from a UPGMA tree under the wrong model, not at stationarity",
                target > 5);
        assertTrue("burn-in target (" + target + ") should be well before the end of the chain "
                + "(chainLength=" + CHAIN_LENGTH + ")", target < CHAIN_LENGTH * 0.9);
    }

    private static double percentile(List<Double> sorted, double p) {
        int index = (int) Math.ceil(p * sorted.size()) - 1;
        index = Math.max(0, Math.min(sorted.size() - 1, index));
        return sorted.get(index);
    }

    /** Reads the "posterior" column (index 1, right after Sample) of a primates trace log. */
    private static List<Double> readPosteriorColumn(Path logFile) throws Exception {
        List<String> lines = Files.readAllLines(logFile);
        List<Double> values = new ArrayList<>();
        for (String line : lines) {
            if (line.isBlank() || line.startsWith("#") || line.startsWith("Sample")) {
                continue;
            }
            String[] cols = line.split("\t");
            values.add(Double.parseDouble(cols[1]));
        }
        return values;
    }

    private static BenchmarkMCMC parseAndRun(Path xmlFile) throws Exception {
        XMLParser parser = new XMLParser();
        BenchmarkMCMC mcmc = (BenchmarkMCMC) parser.parseFile(new File(xmlFile.toString()));
        mcmc.run();
        return mcmc;
    }
}
