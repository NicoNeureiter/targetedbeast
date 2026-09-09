package targetedbeast.tools;

import java.io.File;
import java.io.PrintStream;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import beast.base.core.BEASTInterface;
import beast.base.evolution.alignment.Alignment;
import beast.base.inference.Runnable;
import beast.base.parser.XMLParser;
import beast.pkgmgmt.PackageManager;
import targetedbeast.edgeweights.LearnedWeights;
import targetedbeast.util.Alignment2PCA;

/**
 * Standalone CLI to dump PCA embeddings of the LearnedWeights alignment to a TSV file,
 * for use as a fixed basis in the Python surrogate training pipeline.
 */
public class PcaEmbeddingDumpCli {

    public static void main(String[] args) throws Exception {
        Args cli = Args.parse(args);
        if (cli.showHelp) {
            printUsage();
            return;
        }

        if (cli.xmlPath == null || cli.xmlPath.isEmpty()) {
            throw new IllegalArgumentException("Missing --xml <path-to-xml>");
        }
        if (cli.dimension == null) {
            throw new IllegalArgumentException("Missing --dimension <D>");
        }
        if (cli.outPath == null || cli.outPath.isEmpty()) {
            throw new IllegalArgumentException("Missing --out <file.tsv>");
        }

        File xmlFile = new File(cli.xmlPath).getAbsoluteFile();
        if (!xmlFile.exists()) {
            throw new IllegalArgumentException("XML file does not exist: " + xmlFile);
        }

        PackageManager.loadExternalJars();
        BeastCliSupport.registerLocalPackageServices(PcaEmbeddingDumpCli.class, xmlFile);

        // Only the alignment is needed here, but parsing the XML constructs LearnedWeights, whose
        // initAndValidate() would otherwise run the full (irrelevant) training phase - fast only
        // when trainedWeightsFile already holds a cache, slow on a cold one. PATH_BASE_DIR lets
        // relative training-file paths in the XML resolve against the XML's own directory, as
        // LearnedWeightsFitCli does; without it such an XML fails to parse at all.
        String previousSkipLearning = System.getProperty(LearnedWeights.SKIP_LEARNING_SYSTEM_PROPERTY);
        String previousPathBaseDir = System.getProperty(LearnedWeights.PATH_BASE_DIR_SYSTEM_PROPERTY);
        System.setProperty(LearnedWeights.SKIP_LEARNING_SYSTEM_PROPERTY, "true");
        File xmlParent = xmlFile.getParentFile();
        if (xmlParent != null) {
            System.setProperty(LearnedWeights.PATH_BASE_DIR_SYSTEM_PROPERTY, xmlParent.getAbsolutePath());
        }

        try {
            XMLParser parser = new XMLParser(new HashMap<>());
            Runnable runnable = parser.parseFile(xmlFile, false);

            if (!(runnable instanceof BEASTInterface)) {
                throw new IllegalStateException("Parsed runnable is not a BEASTInterface: " + runnable.getClass().getName());
            }

            LearnedWeights learnedWeights = BeastCliSupport.findLearnedWeights((BEASTInterface) runnable, cli.edgeweightsId);
            if (learnedWeights == null) {
                throw new IllegalStateException("Could not find a LearnedWeights object"
                        + (cli.edgeweightsId == null ? "" : " with id='" + cli.edgeweightsId + "'")
                        + " in XML object graph.");
            }

            Alignment alignment = learnedWeights.dataInput.get();
            int dimension = cli.dimension;
            Map<String, double[]> points = Alignment2PCA.getPoints(alignment, dimension, true, true);

            List<String> taxa = alignment.getTaxaNames();
            java.nio.file.Path outPath = Paths.get(cli.outPath);
            try (PrintStream out = new PrintStream(java.nio.file.Files.newOutputStream(outPath))) {
                StringBuilder header = new StringBuilder("taxon");
                for (int d = 0; d < dimension; d++) {
                    header.append("\tpc").append(d);
                }
                out.println(header);

                for (String taxon : taxa) {
                    double[] point = points.get(taxon);
                    if (point == null) {
                        throw new IllegalStateException("Taxon '" + taxon + "' is missing from the PCA embedding map.");
                    }
                    StringBuilder row = new StringBuilder(taxon);
                    for (int d = 0; d < dimension; d++) {
                        row.append('\t').append(String.format(Locale.US, "%.10g", point[d]));
                    }
                    out.println(row);
                }
            }

            System.out.println("PCA embedding written to " + outPath.toAbsolutePath());
        } finally {
            restoreProperty(LearnedWeights.SKIP_LEARNING_SYSTEM_PROPERTY, previousSkipLearning);
            restoreProperty(LearnedWeights.PATH_BASE_DIR_SYSTEM_PROPERTY, previousPathBaseDir);
        }
    }

    private static void restoreProperty(String key, String previousValue) {
        if (previousValue == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previousValue);
        }
    }

    private static void printUsage() {
        System.out.println("Usage: PcaEmbeddingDumpCli --xml <file.xml> --dimension <D> --out <file.tsv> [--edgeweights-id <id>]");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  --xml /path/IECoR_targetedLearned.xml --dimension 30 --out /tmp/pca.tsv");
    }

    private static final class Args {
        String xmlPath;
        Integer dimension;
        String outPath;
        String edgeweightsId = "edgeweights";
        boolean showHelp;

        static Args parse(String[] args) {
            Args parsed = new Args();
            for (int i = 0; i < args.length; i++) {
                String a = args[i];
                switch (a) {
                    case "-h":
                    case "--help":
                        parsed.showHelp = true;
                        break;
                    case "--xml":
                        parsed.xmlPath = nextValue(args, ++i, a);
                        break;
                    case "--dimension":
                        parsed.dimension = Integer.parseInt(nextValue(args, ++i, a));
                        break;
                    case "--out":
                        parsed.outPath = nextValue(args, ++i, a);
                        break;
                    case "--edgeweights-id":
                        parsed.edgeweightsId = nextValue(args, ++i, a);
                        break;
                    default:
                        throw new IllegalArgumentException("Unknown argument: " + a);
                }
            }
            return parsed;
        }

        private static String nextValue(String[] args, int idx, String flag) {
            if (idx >= args.length) {
                throw new IllegalArgumentException("Missing value after " + flag);
            }
            return args[idx];
        }
    }
}
