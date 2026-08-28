package targetedbeast.tools;

import java.io.File;
import java.nio.file.Paths;
import java.util.HashMap;

import beast.base.core.BEASTInterface;
import beast.base.inference.Runnable;
import beast.base.parser.XMLParser;
import beast.pkgmgmt.PackageManager;
import targetedbeast.edgeweights.LearnedWeights;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsFitReporter;

/**
 * Standalone CLI to evaluate LearnedWeights fit directly from an XML configuration,
 * without running full MCMC.
 */
public class LearnedWeightsFitCli {

    public static void main(String[] args) throws Exception {
        Args cli = Args.parse(args);
        if (cli.showHelp) {
            printUsage();
            return;
        }

        if (cli.xmlPath == null || cli.xmlPath.isEmpty()) {
            throw new IllegalArgumentException("Missing --xml <path-to-xml>");
        }

        File xmlFile = new File(cli.xmlPath).getAbsoluteFile();
        if (!xmlFile.exists()) {
            throw new IllegalArgumentException("XML file does not exist: " + xmlFile);
        }

        PackageManager.loadExternalJars();
        BeastCliSupport.registerLocalPackageServices(LearnedWeightsFitCli.class, xmlFile);

        String previousSkipLearning = System.getProperty(LearnedWeights.SKIP_LEARNING_SYSTEM_PROPERTY);
        String previousTrainQuadraticOnly = System.getProperty(LearnedWeights.TRAIN_QUADRATIC_ONLY_SYSTEM_PROPERTY);
        String previousPathBaseDir = System.getProperty(LearnedWeights.PATH_BASE_DIR_SYSTEM_PROPERTY);
        String previousLoadTrainingSamples = System.getProperty(LearnedWeights.LOAD_TRAINING_SAMPLES_SYSTEM_PROPERTY);
        File xmlParent = xmlFile.getParentFile();
        if (xmlParent != null) {
            System.setProperty(LearnedWeights.PATH_BASE_DIR_SYSTEM_PROPERTY, xmlParent.getAbsolutePath());
        }
        System.setProperty(LearnedWeights.LOAD_TRAINING_SAMPLES_SYSTEM_PROPERTY, "true");
        if (cli.trainQuadraticOnly) {
            System.setProperty(LearnedWeights.TRAIN_QUADRATIC_ONLY_SYSTEM_PROPERTY, "true");
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

            LearnedWeightsFitReporter.Report report = LearnedWeightsFitReporter.evaluate(learnedWeights);
            java.nio.file.Path csvOut = Paths.get(cli.outPrefix + ".weights.csv");
            java.nio.file.Path summaryOut = Paths.get(cli.outPrefix + ".summary.tsv");

            LearnedWeightsFitReporter.writeRowsCsv(csvOut, report);
            LearnedWeightsFitReporter.writeSummaryTsv(summaryOut, report);

            System.out.println("LearnedWeights fit report written.");
            System.out.println("  weights CSV:  " + csvOut.toAbsolutePath());
            System.out.println("  summary TSV:  " + summaryOut.toAbsolutePath());
            System.out.println();
            System.out.println(LearnedWeightsFitReporter.formatConsoleSummary(report));
        } finally {
            if (previousSkipLearning == null) {
                System.clearProperty(LearnedWeights.SKIP_LEARNING_SYSTEM_PROPERTY);
            } else {
                System.setProperty(LearnedWeights.SKIP_LEARNING_SYSTEM_PROPERTY, previousSkipLearning);
            }
            if (previousTrainQuadraticOnly == null) {
                System.clearProperty(LearnedWeights.TRAIN_QUADRATIC_ONLY_SYSTEM_PROPERTY);
            } else {
                System.setProperty(LearnedWeights.TRAIN_QUADRATIC_ONLY_SYSTEM_PROPERTY, previousTrainQuadraticOnly);
            }
            if (previousPathBaseDir == null) {
                System.clearProperty(LearnedWeights.PATH_BASE_DIR_SYSTEM_PROPERTY);
            } else {
                System.setProperty(LearnedWeights.PATH_BASE_DIR_SYSTEM_PROPERTY, previousPathBaseDir);
            }
            if (previousLoadTrainingSamples == null) {
                System.clearProperty(LearnedWeights.LOAD_TRAINING_SAMPLES_SYSTEM_PROPERTY);
            } else {
                System.setProperty(LearnedWeights.LOAD_TRAINING_SAMPLES_SYSTEM_PROPERTY, previousLoadTrainingSamples);
            }
        }
    }

    private static void printUsage() {
        System.out.println("Usage: LearnedWeightsFitCli --xml <file.xml> [--out-prefix <prefix>] [--edgeweights-id <id>] [--train-quadratic-only]");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  --xml /path/IECoR_targetedLearned.xml");
        System.out.println("  --xml /path/IECoR_targetedLearned.xml --out-prefix /tmp/iecor-fit");
        System.out.println("  --xml /path/IECoR_targetedLearned.xml --edgeweights-id edgeweights");
        System.out.println("  --xml /path/IECoR_targetedLearned.xml --train-quadratic-only");
    }

    private static final class Args {
        String xmlPath;
        String outPrefix = "learnedweights-fit";
        String edgeweightsId = "edgeweights";
        boolean trainQuadraticOnly;
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
                    case "--out-prefix":
                        parsed.outPrefix = nextValue(args, ++i, a);
                        break;
                    case "--edgeweights-id":
                        parsed.edgeweightsId = nextValue(args, ++i, a);
                        break;
                    case "--train-quadratic-only":
                        parsed.trainQuadraticOnly = true;
                        break;
                    default:
                        // Allow positional shorthand: first arg = xml, second = out prefix
                        if (parsed.xmlPath == null) {
                            parsed.xmlPath = a;
                        } else if ("learnedweights-fit".equals(parsed.outPrefix)) {
                            parsed.outPrefix = a;
                        } else {
                            throw new IllegalArgumentException("Unknown argument: " + a);
                        }
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
