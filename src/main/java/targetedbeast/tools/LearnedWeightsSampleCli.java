package targetedbeast.tools;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import beast.base.core.BEASTInterface;
import beast.base.evolution.alignment.Alignment;
import beast.base.inference.Runnable;
import beast.base.parser.XMLParser;
import beast.pkgmgmt.PackageManager;
import targetedbeast.edgeweights.LearnedWeights;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsTrainingSampleIO;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsTreeTrainer;

public class LearnedWeightsSampleCli {

    public static void main(String[] args) throws Exception {
        Args cli = Args.parse(args);
        if (cli.showHelp) {
            printUsage();
            return;
        }

        if (cli.xmlPath == null || cli.xmlPath.isEmpty()) {
            throw new IllegalArgumentException("Missing --xml <path-to-xml>");
        }
        if (cli.treeFilePaths == null || cli.treeFilePaths.isEmpty()) {
            throw new IllegalArgumentException("Missing --tree-file <posterior-tree-log>[,<more>...]");
        }
        if (cli.logFilePaths == null || cli.logFilePaths.isEmpty()) {
            throw new IllegalArgumentException("Missing --log-file <posterior-parameter-log>[,<more>...]");
        }
        if (cli.outFilePath == null || cli.outFilePath.isEmpty()) {
            throw new IllegalArgumentException("Missing --out-file <training-samples.tsv>");
        }
        if (cli.treeFilePaths.size() != cli.logFilePaths.size()) {
            throw new IllegalArgumentException(
                "--tree-file and --log-file must list the same number of paths "
                + "(got " + cli.treeFilePaths.size() + " tree file(s) and "
                + cli.logFilePaths.size() + " log file(s))");
        }

        File xmlFile = new File(cli.xmlPath).getAbsoluteFile();
        if (!xmlFile.exists()) {
            throw new IllegalArgumentException("XML file does not exist: " + xmlFile);
        }

        List<File> treeFiles = new ArrayList<>(cli.treeFilePaths.size());
        for (String p : cli.treeFilePaths) {
            File f = new File(p).getAbsoluteFile();
            if (!f.exists()) {
                throw new IllegalArgumentException("Tree file does not exist: " + f);
            }
            treeFiles.add(f);
        }
        List<File> logFiles = new ArrayList<>(cli.logFilePaths.size());
        for (String p : cli.logFilePaths) {
            File f = new File(p).getAbsoluteFile();
            if (!f.exists()) {
                throw new IllegalArgumentException("Log file does not exist: " + f);
            }
            logFiles.add(f);
        }

        PackageManager.loadExternalJars();
        BeastCliSupport.registerLocalPackageServices(LearnedWeightsSampleCli.class, xmlFile);

        String previousSkipLearning = System.getProperty(LearnedWeights.SKIP_LEARNING_SYSTEM_PROPERTY);
        String previousPathBaseDir = System.getProperty(LearnedWeights.PATH_BASE_DIR_SYSTEM_PROPERTY);
        try {
            File xmlParent = xmlFile.getParentFile();
            if (xmlParent != null) {
                System.setProperty(LearnedWeights.PATH_BASE_DIR_SYSTEM_PROPERTY, xmlParent.getAbsolutePath());
            }
            System.setProperty(LearnedWeights.SKIP_LEARNING_SYSTEM_PROPERTY, "true");

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
            LearnedWeightsTreeTrainer trainer = new LearnedWeightsTreeTrainer(
                    learnedWeights.likelihoodInput.get(),
                    alignment,
                    cli.maxSamples > 0 ? cli.maxSamples : learnedWeights.nTrainingTreesInput.get(),
                    learnedWeights.getRootPriorVariance(),
                    learnedWeights.getTargetLogLikelihoodColumn());

                List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = trainer.generateTrainingSamplesFromTreeFiles(
                    treeFiles,
                    logFiles,
                    cli.burninPercentage,
                    cli.thinInterval,
                    cli.maxSamples);

            Path outFile = Paths.get(cli.outFilePath);
            LearnedWeightsTrainingSampleIO.writeSamples(
                    outFile,
                    alignment.getTaxaNames().toArray(new String[0]),
                    samples);

            System.out.println("LearnedWeights posterior training samples written.");
            for (int i = 0; i < treeFiles.size(); i++) {
                System.out.println("  tree file [" + i + "]: " + treeFiles.get(i).getAbsolutePath());
                System.out.println("  log  file [" + i + "]: " + logFiles.get(i).getAbsolutePath());
            }
            System.out.println("  out file:   " + outFile.toAbsolutePath());
            System.out.println("  samples:    " + samples.size());
            System.out.println("  burn-in %:  " + cli.burninPercentage);
            System.out.println("  thin:       " + cli.thinInterval);
            System.out.println("  maxSamples: " + (cli.maxSamples > 0 ? cli.maxSamples : "all"));
        } finally {
            if (previousSkipLearning == null) {
                System.clearProperty(LearnedWeights.SKIP_LEARNING_SYSTEM_PROPERTY);
            } else {
                System.setProperty(LearnedWeights.SKIP_LEARNING_SYSTEM_PROPERTY, previousSkipLearning);
            }
            if (previousPathBaseDir == null) {
                System.clearProperty(LearnedWeights.PATH_BASE_DIR_SYSTEM_PROPERTY);
            } else {
                System.setProperty(LearnedWeights.PATH_BASE_DIR_SYSTEM_PROPERTY, previousPathBaseDir);
            }
        }
    }

    private static void printUsage() {
        System.out.println("Usage: LearnedWeightsSampleCli --xml <file.xml> --tree-file <posterior.trees>[,more...] --log-file <posterior.log>[,more...] --out-file <samples.tsv> [--edgeweights-id <id>] [--burnin-percentage <0-99>] [--thin <n>] [--max-samples <n>]");
        System.out.println();
        System.out.println("Tree/log files may be passed as a comma-separated list, or by repeating the flag.");
        System.out.println("The i-th --log-file must correspond to the i-th --tree-file.");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  --xml /path/IECoR_targetedLearned.xml --tree-file /path/posterior.trees --log-file /path/posterior.log --out-file /tmp/learnedweights-samples.tsv");
        System.out.println("  --xml /path/IECoR_targetedLearned.xml --tree-file chain0.trees,chain1.trees,chain2.trees --log-file chain0.log,chain1.log,chain2.log --out-file /tmp/learnedweights-samples.tsv --burnin-percentage 10 --thin 100 --max-samples 500");
        System.out.println("  --xml /path/IECoR_targetedLearned.xml --tree-file chain0.trees --tree-file chain1.trees --log-file chain0.log --log-file chain1.log --out-file /tmp/learnedweights-samples.tsv");
    }

    private static final class Args {
        String xmlPath;
        List<String> treeFilePaths;
        List<String> logFilePaths;
        String outFilePath;
        String edgeweightsId = "edgeweights";
        int burninPercentage = 10;
        int thinInterval = 100;
        int maxSamples = 500;
        boolean showHelp;

        static Args parse(String[] args) {
            Args parsed = new Args();
            parsed.treeFilePaths = new ArrayList<>();
            parsed.logFilePaths  = new ArrayList<>();
            for (int i = 0; i < args.length; i++) {
                String argument = args[i];
                switch (argument) {
                    case "-h":
                    case "--help":
                        parsed.showHelp = true;
                        break;
                    case "--xml":
                        parsed.xmlPath = nextValue(args, ++i, argument);
                        break;
                    case "--tree-file":
                        appendList(parsed.treeFilePaths, nextValue(args, ++i, argument));
                        break;
                    case "--log-file":
                        appendList(parsed.logFilePaths, nextValue(args, ++i, argument));
                        break;
                    case "--out-file":
                        parsed.outFilePath = nextValue(args, ++i, argument);
                        break;
                    case "--edgeweights-id":
                        parsed.edgeweightsId = nextValue(args, ++i, argument);
                        break;
                    case "--burnin-percentage":
                        parsed.burninPercentage = parseInteger(nextValue(args, ++i, argument), argument, 0, 99);
                        break;
                    case "--thin":
                        parsed.thinInterval = parseInteger(nextValue(args, ++i, argument), argument, 1, Integer.MAX_VALUE);
                        break;
                    case "--max-samples":
                        parsed.maxSamples = parseInteger(nextValue(args, ++i, argument), argument, 0, Integer.MAX_VALUE);
                        break;
                    default:
                        throw new IllegalArgumentException("Unknown argument: " + argument);
                }
            }
            return parsed;
        }

        /**
         * Append a comma-separated path list (or a single path) to {@code dest}.
         * Supports both repeated flags (e.g. {@code --tree-file a --tree-file b})
         * and comma-separated values (e.g. {@code --tree-file a,b}).  Empty entries
         * are skipped so trailing/leading commas are tolerated.
         */
        private static void appendList(List<String> dest, String value) {
            for (String part : value.split(",")) {
                String trimmed = part.trim();
                if (!trimmed.isEmpty()) {
                    dest.add(trimmed);
                }
            }
        }

        private static String nextValue(String[] args, int index, String flag) {
            if (index >= args.length) {
                throw new IllegalArgumentException("Missing value after " + flag);
            }
            return args[index];
        }

        private static int parseInteger(String value, String flag, int minValue, int maxValue) {
            int parsed = Integer.parseInt(value);
            if (parsed < minValue || parsed > maxValue) {
                throw new IllegalArgumentException(flag + " must be in [" + minValue + ", " + maxValue + "], but was " + parsed);
            }
            return parsed;
        }
    }
}