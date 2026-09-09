package targetedbeast.edgeweights.learnedweights;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

public final class LearnedWeightsTrainingSampleIO {

    private static final String MAGIC_V1 = "# LearnedWeightsTrainingSamplesV1";
    private static final String MAGIC_V2 = "# LearnedWeightsTrainingSamplesV2";
    private static final String MAGIC_V3 = "# LearnedWeightsTrainingSamplesV3";
    private static final String MAGIC_V4 = "# LearnedWeightsTrainingSamplesV4";
    private static final String MAGIC_V5 = "# LearnedWeightsTrainingSamplesV5";

    private LearnedWeightsTrainingSampleIO() {
    }

    public static void writeSamples(
            Path outputFile,
            String[] canonicalTaxa,
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples) throws IOException {
        if (canonicalTaxa == null || canonicalTaxa.length == 0) {
            throw new IllegalArgumentException("canonicalTaxa must not be null or empty");
        }
        if (samples == null) {
            throw new IllegalArgumentException("samples must not be null");
        }
        if (outputFile.getParent() != null) {
            Files.createDirectories(outputFile.getParent());
        }

        boolean writeV4 = true;
        for (LearnedWeightsTreeTrainer.TrainingTreeSample sample : samples) {
            if (!sample.hasRootFreeFactorization()) {
                throw new IllegalArgumentException("Training sample is missing root-free covariance data required by cache output");
            }
            if (sample.rootFreeCovarianceCholeskyLower == null) {
                writeV4 = false;
                break;
            }
        }

        // V5 additionally persists groupId and the tree-size covariates (rootHeight,
        // totalBranchLength, totalDiffusionLength), which V1-V4 silently drop. Only step
        // up to V5 when at least one sample actually carries that information, so a plain
        // V4-shaped set of samples keeps writing the plain V4 file as before.
        boolean writeV5 = writeV4 && samplesCarryV5Data(samples);

        try (BufferedWriter writer = Files.newBufferedWriter(outputFile)) {
            writer.write(writeV5 ? MAGIC_V5 : (writeV4 ? MAGIC_V4 : MAGIC_V3));
            writer.newLine();
            writer.write("taxa");
            for (String taxon : canonicalTaxa) {
                writer.write('\t');
                writer.write(taxon);
            }
            writer.newLine();

            for (LearnedWeightsTreeTrainer.TrainingTreeSample sample : samples) {
                assert sample.getTaxonCount() == canonicalTaxa.length : "Sample taxon count does not match canonicalTaxa length";
                if (writeV5) {
                    writer.write(String.format(Locale.US, "sample\t%.8g\t%.8g\t%d\t%.8g\t%d\t%d\t%.8g\t%.8g\t%.8g",
                            sample.targetLogLikelihood,
                            sample.logDeterminant,
                            sample.logNormalizerDimension,
                            sample.rootFreeLogDeterminant,
                            sample.rootFreeLogNormalizerDimension,
                            sample.groupId,
                            sample.rootHeight,
                            sample.totalBranchLength,
                            sample.totalDiffusionLength));
                } else {
                    writer.write(String.format(Locale.US, "sample\t%.8g\t%.8g\t%d\t%.8g\t%d",
                            sample.targetLogLikelihood,
                        sample.logDeterminant,
                        sample.logNormalizerDimension,
                        sample.rootFreeLogDeterminant,
                        sample.rootFreeLogNormalizerDimension));
                }
                for (int row = 0; row < sample.precisionMatrix.length; row++) {
                    for (int column = 0; column < sample.precisionMatrix[row].length; column++) {
                        writer.write('\t');
                        writer.write(String.format(Locale.US, "%.8g", sample.precisionMatrix[row][column]));
                    }
                }
                if (writeV4) {
                    for (int index = 0; index < sample.rootFreeCovarianceCholeskyLower.length; index++) {
                        writer.write('\t');
                        writer.write(String.format(Locale.US, "%.8g", sample.rootFreeCovarianceCholeskyLower[index]));
                    }
                } else {
                    for (int row = 0; row < sample.rootFreePrecisionMatrix.length; row++) {
                        for (int column = 0; column < sample.rootFreePrecisionMatrix[row].length; column++) {
                            writer.write('\t');
                            writer.write(String.format(Locale.US, "%.8g", sample.rootFreePrecisionMatrix[row][column]));
                        }
                    }
                }
                writer.newLine();
            }
        }
    }

    private static boolean samplesCarryV5Data(List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples) {
        for (LearnedWeightsTreeTrainer.TrainingTreeSample sample : samples) {
            if (sample.groupId != LearnedWeightsTreeTrainer.TrainingTreeSample.NO_GROUP
                    || !Double.isNaN(sample.rootHeight)
                    || !Double.isNaN(sample.totalBranchLength)
                    || !Double.isNaN(sample.totalDiffusionLength)) {
                return true;
            }
        }
        return false;
    }

    public static List<LearnedWeightsTreeTrainer.TrainingTreeSample> readSamples(
            Path inputFile,
            String[] expectedCanonicalTaxa) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(inputFile)) {
            String magic = reader.readLine();
            boolean v1 = MAGIC_V1.equals(magic);
            boolean v2 = MAGIC_V2.equals(magic);
            boolean v3 = MAGIC_V3.equals(magic);
            boolean v4 = MAGIC_V4.equals(magic);
            boolean v5 = MAGIC_V5.equals(magic);
            if (!v1 && !v2 && !v3 && !v4 && !v5) {
                throw new IllegalArgumentException("Unrecognized training sample file format: " + inputFile);
            }

            String taxaLine = reader.readLine();
            if (taxaLine == null) {
                throw new IllegalArgumentException("Missing taxa header in training sample file: " + inputFile);
            }
            String[] taxaTokens = taxaLine.split("\\t");
            if (taxaTokens.length < 2 || !"taxa".equals(taxaTokens[0])) {
                throw new IllegalArgumentException("Invalid taxa header in training sample file: " + inputFile);
            }

            String[] fileTaxa = Arrays.copyOfRange(taxaTokens, 1, taxaTokens.length);
            if (expectedCanonicalTaxa != null && !Arrays.equals(fileTaxa, expectedCanonicalTaxa)) {
                throw new IllegalArgumentException(
                        "Training sample taxa do not match expected taxa order. expected="
                                + Arrays.toString(expectedCanonicalTaxa)
                                + " file=" + Arrays.toString(fileTaxa));
            }

            int taxonCount = fileTaxa.length;
            int packedLowerLength = taxonCount * (taxonCount + 1) / 2;
            int expectedColumns = (v5 ? 10 : (v4 ? 6 : (v3 ? 6 : (v2 ? 4 : 3))))
                    + taxonCount * taxonCount
                    + ((v4 || v5) ? packedLowerLength : (v3 ? taxonCount * taxonCount : 0));
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                String[] tokens = line.split("\\t");
                if (tokens.length != expectedColumns || !"sample".equals(tokens[0])) {
                    throw new IllegalArgumentException("Invalid sample row in training sample file: " + line);
                }

                double targetLogLikelihood = Double.parseDouble(tokens[1]);
                double logDeterminant = Double.parseDouble(tokens[2]);
                int logNormalizerDimension = (v2 || v3 || v4 || v5) ? Integer.parseInt(tokens[3]) : taxonCount;
                double rootFreeLogDeterminant = (v3 || v4 || v5) ? Double.parseDouble(tokens[4]) : Double.NaN;
                int rootFreeLogNormalizerDimension = (v3 || v4 || v5) ? Integer.parseInt(tokens[5]) : 0;
                long groupId = LearnedWeightsTreeTrainer.TrainingTreeSample.NO_GROUP;
                double rootHeight = Double.NaN;
                double totalBranchLength = Double.NaN;
                double totalDiffusionLength = Double.NaN;
                if (v5) {
                    groupId = Long.parseLong(tokens[6]);
                    rootHeight = Double.parseDouble(tokens[7]);
                    totalBranchLength = Double.parseDouble(tokens[8]);
                    totalDiffusionLength = Double.parseDouble(tokens[9]);
                }
                double[][] precision = new double[taxonCount][taxonCount];
                int tokenIndex = v5 ? 10 : ((v3 || v4) ? 6 : (v2 ? 4 : 3));
                for (int row = 0; row < taxonCount; row++) {
                    for (int column = 0; column < taxonCount; column++) {
                        precision[row][column] = Double.parseDouble(tokens[tokenIndex++]);
                    }
                }
                double[][] rootFreePrecision = null;
                double[] rootFreeCovarianceCholeskyLower = null;
                if (v3) {
                    rootFreePrecision = new double[taxonCount][taxonCount];
                    for (int row = 0; row < taxonCount; row++) {
                        for (int column = 0; column < taxonCount; column++) {
                            rootFreePrecision[row][column] = Double.parseDouble(tokens[tokenIndex++]);
                        }
                    }
                } else if (v4 || v5) {
                    rootFreeCovarianceCholeskyLower = new double[packedLowerLength];
                    for (int index = 0; index < packedLowerLength; index++) {
                        rootFreeCovarianceCholeskyLower[index] = Double.parseDouble(tokens[tokenIndex++]);
                    }
                }
                if (v5) {
                    samples.add(new LearnedWeightsTreeTrainer.TrainingTreeSample(
                            targetLogLikelihood,
                            precision,
                            logDeterminant,
                            logNormalizerDimension,
                            rootFreeCovarianceCholeskyLower,
                            rootFreeLogDeterminant,
                            rootFreeLogNormalizerDimension,
                            rootHeight,
                            totalBranchLength,
                            totalDiffusionLength,
                            groupId));
                } else if (v4) {
                    samples.add(new LearnedWeightsTreeTrainer.TrainingTreeSample(
                            targetLogLikelihood,
                            precision,
                            logDeterminant,
                            logNormalizerDimension,
                            rootFreeCovarianceCholeskyLower,
                            rootFreeLogDeterminant,
                            rootFreeLogNormalizerDimension,
                            Double.NaN,
                            Double.NaN,
                            Double.NaN));
                } else {
                    samples.add(new LearnedWeightsTreeTrainer.TrainingTreeSample(
                            targetLogLikelihood,
                            precision,
                            logDeterminant,
                            logNormalizerDimension,
                            rootFreePrecision,
                            rootFreeLogDeterminant,
                            rootFreeLogNormalizerDimension,
                            Double.NaN,
                            Double.NaN,
                            Double.NaN));
                }
            }
            return Collections.unmodifiableList(samples);
        }
    }
}
