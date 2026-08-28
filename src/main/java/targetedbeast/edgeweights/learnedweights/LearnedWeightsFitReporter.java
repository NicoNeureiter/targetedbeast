package targetedbeast.edgeweights.learnedweights;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

import targetedbeast.edgeweights.LearnedWeights;

public final class LearnedWeightsFitReporter {

    private LearnedWeightsFitReporter() {
    }

    public static final class Metrics {
        public final int samples;
        public final int pairs;
        public final double meanKl;
        public final double medianKl;
        public final double rmse;
        public final double pearson;
        public final double spearman;
        public final double top1Accuracy;
        public final double rawPearson;
        public final double rawSpearman;
        public final double logDetPearson;
        public final double logDetSpearman;
        public final double quadraticPearson;
        public final double quadraticSpearman;
        public final double targetVsRootHeightPearson;
        public final double targetVsTotalBranchLengthPearson;
        public final double targetVsTotalDiffusionLengthPearson;
        public final double logDetVsRootHeightPearson;
        public final double logDetVsTotalBranchLengthPearson;
        public final double logDetVsTotalDiffusionLengthPearson;

        public Metrics(int samples, int pairs, double meanKl, double medianKl,
                double rmse, double pearson, double spearman, double top1Accuracy,
                double rawPearson, double rawSpearman,
                double logDetPearson, double logDetSpearman,
            double quadraticPearson, double quadraticSpearman,
            double targetVsRootHeightPearson,
            double targetVsTotalBranchLengthPearson,
            double targetVsTotalDiffusionLengthPearson,
            double logDetVsRootHeightPearson,
            double logDetVsTotalBranchLengthPearson,
            double logDetVsTotalDiffusionLengthPearson) {
            this.samples = samples;
            this.pairs = pairs;
            this.meanKl = meanKl;
            this.medianKl = medianKl;
            this.rmse = rmse;
            this.pearson = pearson;
            this.spearman = spearman;
            this.top1Accuracy = top1Accuracy;
            this.rawPearson = rawPearson;
            this.rawSpearman = rawSpearman;
            this.logDetPearson = logDetPearson;
            this.logDetSpearman = logDetSpearman;
            this.quadraticPearson = quadraticPearson;
            this.quadraticSpearman = quadraticSpearman;
            this.targetVsRootHeightPearson = targetVsRootHeightPearson;
            this.targetVsTotalBranchLengthPearson = targetVsTotalBranchLengthPearson;
            this.targetVsTotalDiffusionLengthPearson = targetVsTotalDiffusionLengthPearson;
            this.logDetVsRootHeightPearson = logDetVsRootHeightPearson;
            this.logDetVsTotalBranchLengthPearson = logDetVsTotalBranchLengthPearson;
            this.logDetVsTotalDiffusionLengthPearson = logDetVsTotalDiffusionLengthPearson;
        }
    }

    public static final class Row {
        public final int sampleIndex;
        public final double target;
        public final double predictedLearned;
        public final double predictedPca;
        public final double logDetPart;
        public final double learnedQuadraticPart;
        public final double pcaQuadraticPart;
        public final double rootHeight;
        public final double totalBranchLength;
        public final double totalDiffusionLength;

        public Row(int sampleIndex, double target, double predictedLearned, double predictedPca,
                double logDetPart, double learnedQuadraticPart, double pcaQuadraticPart,
                double rootHeight, double totalBranchLength, double totalDiffusionLength) {
            this.sampleIndex = sampleIndex;
            this.target = target;
            this.predictedLearned = predictedLearned;
            this.predictedPca = predictedPca;
            this.logDetPart = logDetPart;
            this.learnedQuadraticPart = learnedQuadraticPart;
            this.pcaQuadraticPart = pcaQuadraticPart;
            this.rootHeight = rootHeight;
            this.totalBranchLength = totalBranchLength;
            this.totalDiffusionLength = totalDiffusionLength;
        }
    }

    public static final class Report {
        public final Metrics learned;
        public final Metrics pca;
        public final List<Row> rows;

        public Report(Metrics learned, Metrics pca, List<Row> rows) {
            this.learned = learned;
            this.pca = pca;
            this.rows = Collections.unmodifiableList(new ArrayList<>(rows));
        }
    }

    public static Report evaluate(LearnedWeights learnedWeights) {
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = learnedWeights.getTrainingSamples();
        if (samples == null || samples.isEmpty()) {
            throw new IllegalStateException("No LearnedWeights training samples available. "
                    + "Make sure learning was run and produced samples.");
        }

        String[] canonicalTaxa = learnedWeights.dataInput.get().getTaxaNames().toArray(new String[0]);
        int dim = learnedWeights.getEmbeddingDimension();
        double[] brownianTraitRates = learnedWeights.getBrownianTraitRates();
        double[] learnedEmbeddings = learnedWeights.getCurrentLeafEmbeddingsForTaxa(canonicalTaxa);
        double[] pcaEmbeddings = learnedWeights.getPcaLeafEmbeddingsForTaxa(canonicalTaxa);

        List<Row> rows = new ArrayList<>();
        double[] targets = new double[samples.size()];
        double[] learnedPredictions = new double[samples.size()];
        double[] pcaPredictions = new double[samples.size()];
        double[] logDetParts = new double[samples.size()];
        double[] learnedQuadraticParts = new double[samples.size()];
        double[] pcaQuadraticParts = new double[samples.size()];
        double[] rootHeights = new double[samples.size()];
        double[] totalBranchLengths = new double[samples.size()];
        double[] totalDiffusionLengths = new double[samples.size()];

        for (int index = 0; index < samples.size(); index++) {
            LearnedWeightsTreeTrainer.TrainingTreeSample sample = samples.get(index);
            targets[index] = sample.targetLogLikelihood;
            LearnedWeightsTreeObjectiveFunction.SurrogateDecomposition learnedDecomposition =
                LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihoodDecomposition(
                    sample,
                    learnedEmbeddings,
                    dim,
                    brownianTraitRates);
            LearnedWeightsTreeObjectiveFunction.SurrogateDecomposition pcaDecomposition =
                LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihoodDecomposition(
                    sample,
                    pcaEmbeddings,
                    dim,
                    brownianTraitRates);
            learnedPredictions[index] = learnedDecomposition.total;
            pcaPredictions[index] = pcaDecomposition.total;
            logDetParts[index] = learnedDecomposition.logDetPart;
            learnedQuadraticParts[index] = learnedDecomposition.quadraticPart;
            pcaQuadraticParts[index] = pcaDecomposition.quadraticPart;
            rootHeights[index] = sample.rootHeight;
            totalBranchLengths[index] = sample.totalBranchLength;
            totalDiffusionLengths[index] = sample.totalDiffusionLength;
            rows.add(new Row(
                index,
                targets[index],
                learnedPredictions[index],
                pcaPredictions[index],
                logDetParts[index],
                learnedQuadraticParts[index],
                pcaQuadraticParts[index],
                rootHeights[index],
                totalBranchLengths[index],
                totalDiffusionLengths[index]));
        }

        double[] learnedCalibrated = calibrate(learnedPredictions, targets);
        double[] pcaCalibrated = calibrate(pcaPredictions, targets);

        Metrics learnedMetrics = new Metrics(
                samples.size(),
                samples.size(),
                Double.NaN,
                Double.NaN,
                rmse(targets, learnedCalibrated),
                pearson(targets, learnedCalibrated),
                spearman(targets, learnedCalibrated),
            Double.NaN,
            pearson(targets, learnedPredictions),
            spearman(targets, learnedPredictions),
            pearson(targets, logDetParts),
            spearman(targets, logDetParts),
            pearson(targets, learnedQuadraticParts),
            spearman(targets, learnedQuadraticParts),
            pearson(targets, rootHeights),
            pearson(targets, totalBranchLengths),
            pearson(targets, totalDiffusionLengths),
            pearson(logDetParts, rootHeights),
            pearson(logDetParts, totalBranchLengths),
            pearson(logDetParts, totalDiffusionLengths));

        Metrics pcaMetrics = new Metrics(
                samples.size(),
                samples.size(),
                Double.NaN,
                Double.NaN,
                rmse(targets, pcaCalibrated),
                pearson(targets, pcaCalibrated),
                spearman(targets, pcaCalibrated),
            Double.NaN,
            pearson(targets, pcaPredictions),
            spearman(targets, pcaPredictions),
            pearson(targets, logDetParts),
            spearman(targets, logDetParts),
            pearson(targets, pcaQuadraticParts),
            spearman(targets, pcaQuadraticParts),
            pearson(targets, rootHeights),
            pearson(targets, totalBranchLengths),
            pearson(targets, totalDiffusionLengths),
            pearson(logDetParts, rootHeights),
            pearson(logDetParts, totalBranchLengths),
            pearson(logDetParts, totalDiffusionLengths));

        return new Report(learnedMetrics, pcaMetrics, rows);
    }

    public static void writeRowsCsv(Path outputFile, Report report) throws IOException {
        LearnedWeightsReportSupport.ensureParentDirectory(outputFile);

        try (BufferedWriter writer = Files.newBufferedWriter(outputFile)) {
            writer.write("sampleIndex,target,predLearned,predPca,logDetPart,learnedQuadraticPart,pcaQuadraticPart,rootHeight,totalBranchLength,totalDiffusionLength");
            writer.newLine();
            for (Row row : report.rows) {
                writer.write(String.format(Locale.US,
                        "%d,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g",
                        row.sampleIndex,
                        row.target,
                        row.predictedLearned,
                        row.predictedPca,
                        row.logDetPart,
                        row.learnedQuadraticPart,
                        row.pcaQuadraticPart,
                        row.rootHeight,
                        row.totalBranchLength,
                        row.totalDiffusionLength));
                writer.newLine();
            }
        }
    }

    public static void writeSummaryTsv(Path outputFile, Report report) throws IOException {
        LearnedWeightsReportSupport.ensureParentDirectory(outputFile);

        try (BufferedWriter writer = Files.newBufferedWriter(outputFile)) {
            writer.write("model\tsamples\tpairs\tmeanKL\tmedianKL\tRMSE\tpearson\tspearman\ttop1Accuracy\trawPearson\trawSpearman\tlogDetPearson\tlogDetSpearman\tquadraticPearson\tquadraticSpearman\ttargetVsRootHeightPearson\ttargetVsTotalBranchLengthPearson\ttargetVsTotalDiffusionLengthPearson\tlogDetVsRootHeightPearson\tlogDetVsTotalBranchLengthPearson\tlogDetVsTotalDiffusionLengthPearson");
            writer.newLine();
            writeMetricsLine(writer, "learned", report.learned);
            writeMetricsLine(writer, "pca", report.pca);
        }
    }

    public static String formatConsoleSummary(Report report) {
        StringBuilder sb = new StringBuilder();
        sb.append("Fit metrics (lower KL/RMSE is better, higher correlation/top1 is better)\n");
        sb.append(formatMetrics("learned", report.learned)).append('\n');
        sb.append(formatMetrics("pca", report.pca));
        return sb.toString();
    }

    private static void writeMetricsLine(BufferedWriter writer, String label, Metrics m) throws IOException {
        writer.write(String.format(Locale.US,
            "%s\t%d\t%d\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g",
                label,
                m.samples,
                m.pairs,
                m.meanKl,
                m.medianKl,
                m.rmse,
                m.pearson,
                m.spearman,
            m.top1Accuracy,
            m.rawPearson,
            m.rawSpearman,
            m.logDetPearson,
            m.logDetSpearman,
            m.quadraticPearson,
            m.quadraticSpearman,
            m.targetVsRootHeightPearson,
            m.targetVsTotalBranchLengthPearson,
            m.targetVsTotalDiffusionLengthPearson,
            m.logDetVsRootHeightPearson,
            m.logDetVsTotalBranchLengthPearson,
            m.logDetVsTotalDiffusionLengthPearson));
        writer.newLine();
    }

    private static String formatMetrics(String label, Metrics m) {
        return String.format(Locale.US,
            "%s: meanKL=%.6f, medianKL=%.6f, RMSE=%.6f, pearson=%.4f, spearman=%.4f, rawPearson=%.4f, rawSpearman=%.4f, logDetPearson=%.4f, logDetSpearman=%.4f, quadraticPearson=%.4f, quadraticSpearman=%.4f, targetVsRootHeightPearson=%.4f, targetVsTotalBranchLengthPearson=%.4f, targetVsTotalDiffusionLengthPearson=%.4f, logDetVsRootHeightPearson=%.4f, logDetVsTotalBranchLengthPearson=%.4f, logDetVsTotalDiffusionLengthPearson=%.4f, top1=%.4f, samples=%d, pairs=%d",
                label,
                m.meanKl,
                m.medianKl,
                m.rmse,
                m.pearson,
                m.spearman,
            m.rawPearson,
            m.rawSpearman,
            m.logDetPearson,
            m.logDetSpearman,
            m.quadraticPearson,
            m.quadraticSpearman,
            m.targetVsRootHeightPearson,
            m.targetVsTotalBranchLengthPearson,
            m.targetVsTotalDiffusionLengthPearson,
            m.logDetVsRootHeightPearson,
            m.logDetVsTotalBranchLengthPearson,
            m.logDetVsTotalDiffusionLengthPearson,
                m.top1Accuracy,
                m.samples,
                m.pairs);
    }

    private static double[] calibrate(double[] prediction, double[] target) {
        double[] affine = LearnedWeightsTreeObjectiveFunction.fitAffineCalibration(prediction, target);
        double[] calibrated = new double[prediction.length];
        for (int i = 0; i < prediction.length; i++) {
            calibrated[i] = affine[0] * prediction[i] + affine[1];
        }
        return calibrated;
    }

    private static double rmse(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return Math.sqrt(s / a.length);
    }

    private static double pearson(double[] x, double[] y) {
        return LearnedWeightsReportSupport.pearson(x, y);
    }

    private static double spearman(double[] x, double[] y) {
        return LearnedWeightsReportSupport.spearman(x, y);
    }
}
