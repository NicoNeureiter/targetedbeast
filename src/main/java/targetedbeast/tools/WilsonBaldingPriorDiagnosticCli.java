package targetedbeast.tools;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;

import beast.base.core.BEASTInterface;
import beast.base.evolution.tree.Tree;
import beast.base.inference.CompoundDistribution;
import beast.base.inference.Distribution;
import beast.base.inference.Runnable;
import beast.base.parser.XMLParser;
import beast.pkgmgmt.PackageManager;
import targetedbeast.edgeweights.EdgeWeights;

public class WilsonBaldingPriorDiagnosticCli {

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
        BeastCliSupport.registerLocalPackageServices(WilsonBaldingPriorDiagnosticCli.class, xmlFile);

        XMLParser parser = new XMLParser(new HashMap<>());
        Runnable runnable = parser.parseFile(xmlFile, false);
        if (!(runnable instanceof BEASTInterface)) {
            throw new IllegalStateException("Parsed runnable is not a BEASTInterface: " + runnable.getClass().getName());
        }
        BEASTInterface root = (BEASTInterface) runnable;

        Tree tree = requireObject(root, Tree.class, cli.treeId, "tree");
        Distribution prior = requireObject(root, Distribution.class, cli.priorId, "prior distribution");
        validateDiagnosticPrior(prior, cli.priorId);
        EdgeWeights likelihoodWeights = requireObject(root, EdgeWeights.class, cli.likelihoodWeightsId, "likelihood edge weights");
        EdgeWeights heuristicWeights = requireObject(root, EdgeWeights.class, cli.heuristicWeightsId, "heuristic edge weights");
        EdgeWeights combinedWeights = requireObject(root, EdgeWeights.class, cli.combinedWeightsId, "combined edge weights");

        WilsonBaldingPriorDiagnostic.Config config = new WilsonBaldingPriorDiagnostic.Config(
                cli.maxHeightRatio,
                cli.useAttachmentIntervalLength,
                cli.seed,
                cli.maxSources);
        WilsonBaldingPriorDiagnostic.Report report = WilsonBaldingPriorDiagnostic.analyze(
                tree,
                prior,
                likelihoodWeights,
                heuristicWeights,
                combinedWeights,
                config);

        Path rowsOut = Paths.get(cli.outPrefix + ".rows.tsv");
        Path summaryOut = Paths.get(cli.outPrefix + ".summary.tsv");
        WilsonBaldingPriorDiagnostic.writeRowsTsv(rowsOut, report);
        WilsonBaldingPriorDiagnostic.writeSummaryTsv(summaryOut, report);

        System.out.println("Wilson-Balding prior diagnostic written.");
        System.out.println("  rows TSV:    " + rowsOut.toAbsolutePath());
        System.out.println("  summary TSV: " + summaryOut.toAbsolutePath());
        System.out.println();
        System.out.println(WilsonBaldingPriorDiagnostic.formatConsoleSummary(report));
    }

    private static <T> T requireObject(BEASTInterface root, Class<T> type, String preferredId, String description) {
        T found = BeastCliSupport.findObject(root, type, preferredId);
        if (found == null) {
            throw new IllegalStateException("Could not find " + description
                    + (preferredId == null ? "" : " with id='" + preferredId + "'")
                    + " in XML object graph.");
        }
        return found;
    }

    static void validateDiagnosticPrior(Distribution prior, String priorId) {
        if (prior instanceof CompoundDistribution) {
            String priorName = priorId == null || priorId.isEmpty() ? prior.getID() : priorId;
            throw new IllegalArgumentException(
                    "Wilson-Balding prior diagnostics require the exact tree prior, not a compound distribution"
                            + (priorName == null || priorName.isEmpty() ? "" : " ('" + priorName + "')")
                            + ". Pass the tree-prior id explicitly, for example '--prior-id CoalescentConstant.t:infB_HA'.");
        }
    }

    private static void printUsage() {
        System.out.println("Usage: WilsonBaldingPriorDiagnosticCli --xml <file.xml> [--out-prefix <prefix>] [--tree-id <id>] [--prior-id <id>] [--likelihood-weights-id <id>] [--heuristic-weights-id <id>] [--combined-weights-id <id>] [--max-height-ratio <value>] [--seed <long>] [--max-sources <n>] [--disable-interval-length]");
    }

    private static final class Args {
        String xmlPath;
        String outPrefix = "wb-prior-diagnostic";
        String treeId = "tree";
        String priorId = "prior";
        String likelihoodWeightsId = "likelihoodEdgeWeights";
        String heuristicWeightsId = "priorHeuristicWeights";
        String combinedWeightsId = "edgeweights";
        double maxHeightRatio = 1.2;
        long seed = 1L;
        int maxSources;
        boolean useAttachmentIntervalLength = true;
        boolean showHelp;

        static Args parse(String[] args) {
            Args parsed = new Args();
            for (int idx = 0; idx < args.length; idx++) {
                String arg = args[idx];
                switch (arg) {
                    case "-h":
                    case "--help":
                        parsed.showHelp = true;
                        break;
                    case "--xml":
                        parsed.xmlPath = nextValue(args, ++idx, arg);
                        break;
                    case "--out-prefix":
                        parsed.outPrefix = nextValue(args, ++idx, arg);
                        break;
                    case "--tree-id":
                        parsed.treeId = nextValue(args, ++idx, arg);
                        break;
                    case "--prior-id":
                        parsed.priorId = nextValue(args, ++idx, arg);
                        break;
                    case "--likelihood-weights-id":
                        parsed.likelihoodWeightsId = nextValue(args, ++idx, arg);
                        break;
                    case "--heuristic-weights-id":
                        parsed.heuristicWeightsId = nextValue(args, ++idx, arg);
                        break;
                    case "--combined-weights-id":
                        parsed.combinedWeightsId = nextValue(args, ++idx, arg);
                        break;
                    case "--max-height-ratio":
                        parsed.maxHeightRatio = Double.parseDouble(nextValue(args, ++idx, arg));
                        break;
                    case "--seed":
                        parsed.seed = Long.parseLong(nextValue(args, ++idx, arg));
                        break;
                    case "--max-sources":
                        parsed.maxSources = Integer.parseInt(nextValue(args, ++idx, arg));
                        break;
                    case "--disable-interval-length":
                        parsed.useAttachmentIntervalLength = false;
                        break;
                    default:
                        if (parsed.xmlPath == null) {
                            parsed.xmlPath = arg;
                        } else {
                            throw new IllegalArgumentException("Unknown argument: " + arg);
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