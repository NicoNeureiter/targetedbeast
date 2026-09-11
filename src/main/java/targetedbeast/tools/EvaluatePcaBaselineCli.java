package targetedbeast.tools;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import targetedbeast.edgeweights.learnedweights.LearnedWeightsTreeObjectiveFunction;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsTreeTrainer;
import targetedbeast.util.Alignment2PCA;
import beast.base.evolution.alignment.Alignment;
import beast.base.parser.XMLParser;
import beast.base.core.BEASTInterface;
import beast.base.inference.Runnable;
import beast.pkgmgmt.PackageManager;
import targetedbeast.edgeweights.LearnedWeights;

/**
 * Scores a V6 ProposalExporter TSV with the plain PCA-embedding Brownian
 * likelihood (the same formula the deployed "pca" setting / PCAWeights2 is
 * built on) -- not through an MLP, just PCA embeddings straight into
 * {@link LearnedWeightsTreeObjectiveFunction#computeSurrogateLogLikelihoodDecomposition}
 * with uniform rate 1.0, matching LearnedWeights' own default and what
 * {@code LearnedWeightsGroupedMetrics.evaluate()} already treats as the "pca"
 * comparison arm.
 *
 * <p>V6 is not read by {@link targetedbeast.edgeweights.learnedweights.LearnedWeightsTrainingSampleIO}
 * (V1-V5 only), so this CLI parses the extra V6 columns (operatorType,
 * distanceToCurrent, isCurrent) itself and ignores them for scoring -- they're
 * only carried through to the output CSV for the Python side to consume.
 *
 * <p>Usage:
 * <pre>{@code
 *   EvaluatePcaBaselineCli --xml <xml-with-a-LearnedWeights-or-any-edgeweights-holding-alignment>
 *       --proposals proposals.tsv --dimension 10 --out pca_predictions.csv
 * }</pre>
 */
public class EvaluatePcaBaselineCli {

    public static void main(String[] args) throws Exception {
        String xmlPath = null, proposalsPath = null, outPath = null;
        int dimension = 10;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--xml": xmlPath = args[++i]; break;
                case "--proposals": proposalsPath = args[++i]; break;
                case "--dimension": dimension = Integer.parseInt(args[++i]); break;
                case "--out": outPath = args[++i]; break;
                default: throw new IllegalArgumentException("Unknown argument: " + args[i]);
            }
        }
        if (xmlPath == null || proposalsPath == null || outPath == null) {
            throw new IllegalArgumentException("Usage: --xml <file> --proposals <file> --dimension <D> --out <file>");
        }

        java.io.File xmlFile = new java.io.File(xmlPath).getAbsoluteFile();
        PackageManager.loadExternalJars();
        BeastCliSupport.registerLocalPackageServices(EvaluatePcaBaselineCli.class, xmlFile);
        System.setProperty(LearnedWeights.SKIP_LEARNING_SYSTEM_PROPERTY, "true");
        XMLParser parser = new XMLParser(new java.util.HashMap<>());
        Runnable runnable = parser.parseFile(xmlFile, false);
        LearnedWeights lw = BeastCliSupport.findLearnedWeights((BEASTInterface) runnable, "edgeweights");
        Alignment alignment = lw.dataInput.get();
        List<String> taxa = alignment.getTaxaNames();

        java.util.Map<String, double[]> points = Alignment2PCA.getPoints(alignment, dimension, true, true);
        double[] embeddings = new double[taxa.size() * dimension];
        for (int i = 0; i < taxa.size(); i++) {
            System.arraycopy(points.get(taxa.get(i)), 0, embeddings, i * dimension, dimension);
        }
        double[] brownianRates = new double[dimension];
        Arrays.fill(brownianRates, 1.0);

        List<Row> rows = readV6(Paths.get(proposalsPath));
        System.out.println("Loaded " + rows.size() + " V6 rows, taxonCount=" + rows.get(0).sample.getTaxonCount());

        try (BufferedWriter w = Files.newBufferedWriter(Paths.get(outPath))) {
            w.write("target,pcaPrediction,groupId,operatorType,distanceToCurrent,isCurrent\n");
            for (Row row : rows) {
                double pred = LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihoodDecomposition(
                        row.sample, embeddings, dimension, brownianRates).total;
                w.write(String.format(Locale.US, "%.8g,%.8g,%d,%s,%.8g,%d%n",
                        row.sample.targetLogLikelihood, pred, row.sample.groupId,
                        row.operatorType, row.distanceToCurrent, row.isCurrent ? 1 : 0));
            }
        }
        System.out.println("Wrote " + outPath);
    }

    private static final class Row {
        final LearnedWeightsTreeTrainer.TrainingTreeSample sample;
        final String operatorType;
        final double distanceToCurrent;
        final boolean isCurrent;

        Row(LearnedWeightsTreeTrainer.TrainingTreeSample sample, String operatorType,
                double distanceToCurrent, boolean isCurrent) {
            this.sample = sample;
            this.operatorType = operatorType;
            this.distanceToCurrent = distanceToCurrent;
            this.isCurrent = isCurrent;
        }
    }

    private static List<Row> readV6(Path path) throws Exception {
        List<Row> rows = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            String magic = reader.readLine();
            if (!"# LearnedWeightsTrainingSamplesV6".equals(magic)) {
                throw new IllegalArgumentException("Expected V6 proposals file, got magic: " + magic);
            }
            String taxaLine = reader.readLine();
            String[] taxaTokens = taxaLine.split("\t");
            int n = taxaTokens.length - 1;
            int packedLowerLength = n * (n + 1) / 2;
            int nScalarCols = 13;

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;
                String[] tokens = line.split("\t");
                double targetLogLikelihood = Double.parseDouble(tokens[1]);
                double logDeterminant = Double.parseDouble(tokens[2]);
                int logNormalizerDimension = Integer.parseInt(tokens[3]);
                double rootFreeLogDeterminant = Double.parseDouble(tokens[4]);
                int rootFreeLogNormalizerDimension = Integer.parseInt(tokens[5]);
                long groupId = Long.parseLong(tokens[6]);
                double rootHeight = Double.parseDouble(tokens[7]);
                double totalBranchLength = Double.parseDouble(tokens[8]);
                double totalDiffusionLength = Double.parseDouble(tokens[9]);
                String operatorType = tokens[10];
                double distanceToCurrent = Double.parseDouble(tokens[11]);
                boolean isCurrent = "1".equals(tokens[12]);

                double[][] precision = new double[n][n];
                int idx = nScalarCols;
                for (int row = 0; row < n; row++) {
                    for (int col = 0; col < n; col++) {
                        precision[row][col] = Double.parseDouble(tokens[idx++]);
                    }
                }
                double[] rootFreeCovarianceCholeskyLower = new double[packedLowerLength];
                for (int i = 0; i < packedLowerLength; i++) {
                    rootFreeCovarianceCholeskyLower[i] = Double.parseDouble(tokens[idx++]);
                }

                LearnedWeightsTreeTrainer.TrainingTreeSample sample = new LearnedWeightsTreeTrainer.TrainingTreeSample(
                        targetLogLikelihood, precision, logDeterminant, logNormalizerDimension,
                        rootFreeCovarianceCholeskyLower, rootFreeLogDeterminant, rootFreeLogNormalizerDimension,
                        rootHeight, totalBranchLength, totalDiffusionLength, groupId);
                rows.add(new Row(sample, operatorType, distanceToCurrent, isCurrent));
            }
        }
        return rows;
    }
}
