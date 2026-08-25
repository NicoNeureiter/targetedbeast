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

        try (BufferedWriter writer = Files.newBufferedWriter(outputFile)) {
            writer.write(writeV4 ? MAGIC_V4 : MAGIC_V3);
            writer.newLine();
            writer.write("taxa");
            for (String taxon : canonicalTaxa) {
                writer.write('\t');
                writer.write(taxon);
            }
            writer.newLine();

            for (LearnedWeightsTreeTrainer.TrainingTreeSample sample : samples) {
                assert sample.getTaxonCount() == canonicalTaxa.length : "Sample taxon count does not match canonicalTaxa length";
                writer.write(String.format(Locale.US, "sample\t%.8g\t%.8g\t%d\t%.8g\t%d",
                        sample.targetLogLikelihood,
                    sample.logDeterminant,
                    sample.logNormalizerDimension,
                    sample.rootFreeLogDeterminant,
                    sample.rootFreeLogNormalizerDimension));
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

    public static List<LearnedWeightsTreeTrainer.TrainingTreeSample> readSamples(
            Path inputFile,
            String[] expectedCanonicalTaxa) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(inputFile)) {
            String magic = reader.readLine();
            boolean v1 = MAGIC_V1.equals(magic);
            boolean v2 = MAGIC_V2.equals(magic);
            boolean v3 = MAGIC_V3.equals(magic);
            boolean v4 = MAGIC_V4.equals(magic);
            if (!v1 && !v2 && !v3 && !v4) {
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
            int expectedColumns = (v4 ? 6 : (v3 ? 6 : (v2 ? 4 : 3)))
                    + taxonCount * taxonCount
                    + (v4 ? packedLowerLength : (v3 ? taxonCount * taxonCount : 0));
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
                int logNormalizerDimension = (v2 || v3 || v4) ? Integer.parseInt(tokens[3]) : taxonCount;
                double rootFreeLogDeterminant = (v3 || v4) ? Double.parseDouble(tokens[4]) : Double.NaN;
                int rootFreeLogNormalizerDimension = (v3 || v4) ? Integer.parseInt(tokens[5]) : 0;
                double[][] precision = new double[taxonCount][taxonCount];
                int tokenIndex = (v3 || v4) ? 6 : (v2 ? 4 : 3);
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
                } else if (v4) {
                    rootFreeCovarianceCholeskyLower = new double[packedLowerLength];
                    for (int index = 0; index < packedLowerLength; index++) {
                        rootFreeCovarianceCholeskyLower[index] = Double.parseDouble(tokens[tokenIndex++]);
                    }
                }
                if (v4) {
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