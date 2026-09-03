package targetedbeast.benchmark;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import beast.base.util.Randomizer;

/**
 * Builds BenchmarkMCMC/MCMC test XMLs from {@code primates-template.xml} - a copy of
 * {@code examples/primates.xml} (6 taxa, ~92,200 states/sec) with a shortened chain length so
 * tests run in a few seconds instead of the original's 5,000,000-state chain.
 *
 * <p>String substitution, not DOM editing: the template is small and hand-written, and every
 * substitution point is asserted present so a template change breaks loudly instead of silently
 * matching nothing.
 */
final class PrimatesFixture {

    private PrimatesFixture() {}

    static String loadTemplate() throws IOException {
        try (InputStream in = PrimatesFixture.class.getResourceAsStream("primates-template.xml")) {
            if (in == null) {
                throw new IllegalStateException("test resource primates-template.xml not found");
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    static String withChainLength(String xml, long chainLength) {
        String needle = "chainLength=\"5000000\"";
        require(xml, needle);
        return xml.replace(needle, "chainLength=\"" + chainLength + "\"");
    }

    /** Points both loggers at absolute paths instead of relying on cwd-relative $(filebase). */
    static String withAbsoluteLogPaths(String xml, Path logFile, Path treesFile) {
        String logNeedle = "fileName=\"$(filebase).log\"";
        String treesNeedle = "fileName=\"$(filebase).trees\"";
        require(xml, logNeedle);
        require(xml, treesNeedle);
        xml = xml.replace(logNeedle, "fileName=\"" + logFile.toAbsolutePath() + "\"");
        xml = xml.replace(treesNeedle, "fileName=\"" + treesFile.toAbsolutePath() + "\"");
        return xml;
    }

    /**
     * Rewrites the {@code <run spec="MCMC" ...>} element to {@code BenchmarkMCMC}, adds a taxonset
     * and an RNNIMetric, and wires statsOut/jumpInterval/jumpTrace. Every other line - state,
     * operators, priors, loggers - is untouched, which is exactly what the identical-trace test
     * depends on.
     */
    static String toBenchmarkVariant(String xml, Path statsOutPrefix, int jumpInterval, Path jumpTrace) {
        return toBenchmarkVariant(xml, statsOutPrefix, jumpInterval, jumpTrace, true);
    }

    /**
     * @param withRnniMetric if false, no {@code metric} input is added at all - used to test that
     *                       an empty metric list is well-defined (NaN/zero columns, no exception)
     *                       rather than just the common case.
     */
    static String toBenchmarkVariant(String xml, Path statsOutPrefix, int jumpInterval, Path jumpTrace,
            boolean withRnniMetric) {
        String runNeedle = "<run spec=\"MCMC\" id=\"mcmc\" chainLength=";
        require(xml, runNeedle);
        String taxonInput = withRnniMetric
                ? "<input id=\"taxa\" spec=\"beast.base.evolution.alignment.TaxonSet\" alignment=\"@alignment\"/>\n\n    "
                : "";
        xml = xml.replace("<run spec=\"MCMC\" id=\"mcmc\" chainLength=",
                taxonInput + "<run spec=\"targetedbeast.benchmark.BenchmarkMCMC\" id=\"mcmc\""
                        + " statsOut=\"" + statsOutPrefix.toAbsolutePath() + "\""
                        + " jumpInterval=\"" + jumpInterval + "\""
                        + (jumpTrace == null ? "" : " jumpTrace=\"" + jumpTrace.toAbsolutePath() + "\"")
                        + " chainLength=");

        if (withRnniMetric) {
            String stateNeedle = "<state>";
            require(xml, stateNeedle);
            xml = xml.replaceFirst("<state>",
                    "<metric id=\"rnni\" spec=\"beastlabs.evolution.tree.RNNIMetric\" taxonset=\"@taxa\"/>\n"
                    + "        <state>");
        }
        return xml;
    }

    /**
     * Replaces the two standalone narrow/wide Exchange operators with an
     * {@code AdaptableOperatorSampler} that picks between them, so tests can check attribution of
     * nested operators without needing a second XML.
     */
    static String withAdaptableSampler(String xml) {
        String needle = "<operator id=\"narrow\" spec=\"Exchange\" tree=\"@tree\" isNarrow=\"true\" weight=\"1\"/>\n"
                + "        <operator id=\"wide\" spec=\"Exchange\" tree=\"@tree\" isNarrow=\"false\" weight=\"1\"/>";
        require(xml, needle);
        String replacement = "<operator id=\"sampler\" spec=\"targetedbeast.operators.AdaptableOperatorSampler\""
                + " burnin=\"0\" weight=\"2\" tree=\"@tree\">\n"
                + "            <operator id=\"narrow\" spec=\"Exchange\" tree=\"@tree\" isNarrow=\"true\" weight=\"1\"/>\n"
                + "            <operator id=\"wide\" spec=\"Exchange\" tree=\"@tree\" isNarrow=\"false\" weight=\"1\"/>\n"
                + "        </operator>";
        return xml.replace(needle, replacement);
    }

    /** Adds referenceLowerBound/referenceUpperBound to an already-{@code toBenchmarkVariant}'d XML. */
    static String withReferenceBand(String xml, double lower, double upper) {
        String needle = "spec=\"targetedbeast.benchmark.BenchmarkMCMC\" id=\"mcmc\"";
        require(xml, needle);
        return xml.replace(needle, needle
                + " referenceLowerBound=\"" + lower + "\""
                + " referenceUpperBound=\"" + upper + "\"");
    }

    /**
     * Primes {@code Randomizer}'s per-thread generator so that two later, separate
     * {@code Randomizer.setSeed(SAME_SEED)} calls on this thread produce byte-identical streams.
     *
     * <p>{@code Randomizer.random()} lazily creates a per-thread {@code MersenneTwisterFast} on
     * its first call, seeded {@code Randomizer.getSeed() + randomMap.size()} - which, the very
     * first time this happens in a JVM (empty map), is {@code seed - 1}, not {@code seed}.
     * {@code Randomizer.setSeed()} itself only reseeds generators already present in the map, via
     * {@code seed + i}. So calling {@code setSeed(S)} twice in a row, with an actual draw only
     * after the first call, reseeds to two different effective values ({@code S-1} then {@code S})
     * - the second call finds the map non-empty and takes the {@code seed + 0} branch instead.
     * Any test that needs multiple independent runs to see the identical RNG stream from the same
     * nominal seed must call this once, before the first real {@code setSeed}, to populate the map
     * ahead of time.
     */
    static void primeRandomizerThreadEntry() {
        // Not 0/1: Randomizer.setSeed derives the underlying MersenneTwisterFast seed as (this
        // value - 1) plus the current size of its internal thread map, and MersenneTwisterFast
        // rejects an actual seed of exactly 0 ("Non zero random seed required") - a small/edge
        // value here could hit that by coincidence depending on how many other tests already ran
        // in this JVM fork. A large arbitrary constant keeps well clear of that.
        Randomizer.setSeed(987654321L);
        Randomizer.nextDouble();
    }

    static void write(String xml, Path path) throws IOException {
        Files.writeString(path, xml, StandardCharsets.UTF_8);
    }

    private static void require(String xml, String needle) {
        if (!xml.contains(needle)) {
            throw new IllegalStateException("primates-template.xml fixture no longer contains expected text: "
                    + needle);
        }
    }
}
