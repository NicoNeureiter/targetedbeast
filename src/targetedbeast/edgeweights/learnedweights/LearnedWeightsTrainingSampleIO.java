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

        try (BufferedWriter writer = Files.newBufferedWriter(outputFile)) {
            writer.write(MAGIC_V2);
            writer.newLine();
            writer.write("taxa");
            for (String taxon : canonicalTaxa) {
                writer.write('\t');
                writer.write(taxon);
            }
            writer.newLine();

            for (LearnedWeightsTreeTrainer.TrainingTreeSample sample : samples) {
                assert sample.getTaxonCount() == canonicalTaxa.length : "Sample taxon count does not match canonicalTaxa length";
                writer.write(String.format(Locale.US, "sample\t%.17g\t%.17g\t%d",
                        sample.targetLogLikelihood,
                    sample.logDeterminant,
                    sample.logNormalizerDimension));
                for (int row = 0; row < sample.precisionMatrix.length; row++) {
                    for (int column = 0; column < sample.precisionMatrix[row].length; column++) {
                        writer.write('\t');
                        writer.write(String.format(Locale.US, "%.17g", sample.precisionMatrix[row][column]));
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
            if (!v1 && !v2) {
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
            int expectedColumns = (v2 ? 4 : 3) + taxonCount * taxonCount;
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
                int logNormalizerDimension = v2 ? Integer.parseInt(tokens[3]) : taxonCount;
                double[][] precision = new double[taxonCount][taxonCount];
                int tokenIndex = v2 ? 4 : 3;
                for (int row = 0; row < taxonCount; row++) {
                    for (int column = 0; column < taxonCount; column++) {
                        precision[row][column] = Double.parseDouble(tokens[tokenIndex++]);
                    }
                }
                samples.add(new LearnedWeightsTreeTrainer.TrainingTreeSample(
                        targetLogLikelihood,
                        precision,
                        logDeterminant,
                        logNormalizerDimension));
            }
            return Collections.unmodifiableList(samples);
        }
    }
}