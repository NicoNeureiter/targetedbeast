package targetedbeast.edgeweights.learnedweights;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

public final class LearnedWeightsModelIO {

    private static final String MAGIC_V1 = "# LearnedWeightsModelV1";
    private static final String MAGIC_V2 = "# LearnedWeightsModelV2";
    private static final String MAGIC_V3 = "# LearnedWeightsModelV3";
    private static final String MAGIC_V4 = "# LearnedWeightsModelV4";

    private LearnedWeightsModelIO() {
    }

    public static void writeModel(
            Path outputFile,
            String[] canonicalTaxa,
            double[] leafEmbeddings,
            double[] brownianTraitRates,
            double[] rootPriorVariances,
            double[] baseScales,
            double[] quadraticScales,
            double intercept,
            boolean separateBaseAndQuadraticCalibration,
            boolean scaleQuadraticPartOnly,
            boolean trainQuadraticOnly,
            double pcaRegularizationWeight,
            double slopeRegularizationWeight) throws IOException {
        if (outputFile.getParent() != null) {
            Files.createDirectories(outputFile.getParent());
        }
        int dim = brownianTraitRates.length;
        if (rootPriorVariances.length != dim) {
            throw new IllegalArgumentException("rootPriorVariances length mismatch: expected " + dim + " but got " + rootPriorVariances.length);
        }
        if (baseScales.length != dim) {
            throw new IllegalArgumentException("baseScales length mismatch: expected " + dim + " but got " + baseScales.length);
        }
        if (quadraticScales.length != dim) {
            throw new IllegalArgumentException("quadraticScales length mismatch: expected " + dim + " but got " + quadraticScales.length);
        }
        if (leafEmbeddings.length != canonicalTaxa.length * dim) {
            throw new IllegalArgumentException("leafEmbeddings length mismatch: expected "
                    + (canonicalTaxa.length * dim) + " but got " + leafEmbeddings.length);
        }

        try (BufferedWriter writer = Files.newBufferedWriter(outputFile)) {
            writer.write(MAGIC_V4);
            writer.newLine();
            writer.write("taxa");
            for (String taxon : canonicalTaxa) {
                writer.write('\t');
                writer.write(taxon);
            }
            writer.newLine();
            writer.write("dim\t");
            writer.write(Integer.toString(dim));
            writer.newLine();
            writer.write("brownianTraitRates");
            for (double rate : brownianTraitRates) {
                writer.write('\t');
                writer.write(String.format(Locale.US, "%.17g", rate));
            }
            writer.newLine();
            writer.write("rootPriorVariances");
            for (double rootPriorVariance : rootPriorVariances) {
                writer.write('\t');
                writer.write(String.format(Locale.US, "%.17g", rootPriorVariance));
            }
            writer.newLine();
            writer.write("separateBaseAndQuadraticCalibration\t");
            writer.write(Boolean.toString(separateBaseAndQuadraticCalibration));
            writer.newLine();
            writer.write("scaleQuadraticPartOnly\t");
            writer.write(Boolean.toString(scaleQuadraticPartOnly));
            writer.newLine();
            writer.write("trainQuadraticOnly\t");
            writer.write(Boolean.toString(trainQuadraticOnly));
            writer.newLine();
            writer.write(String.format(Locale.US, "pcaRegularizationWeight\t%.17g", pcaRegularizationWeight));
            writer.newLine();
            writer.write(String.format(Locale.US, "slopeRegularizationWeight\t%.17g", slopeRegularizationWeight));
            writer.newLine();
            writer.write("baseScales");
            for (double baseScale : baseScales) {
                writer.write('\t');
                writer.write(String.format(Locale.US, "%.17g", baseScale));
            }
            writer.newLine();
            writer.write("quadraticScales");
            for (double quadraticScale : quadraticScales) {
                writer.write('\t');
                writer.write(String.format(Locale.US, "%.17g", quadraticScale));
            }
            writer.newLine();
            writer.write(String.format(Locale.US, "intercept\t%.17g", intercept));
            writer.newLine();

            for (int taxonIndex = 0; taxonIndex < canonicalTaxa.length; taxonIndex++) {
                writer.write("embedding");
                writer.write('\t');
                writer.write(canonicalTaxa[taxonIndex]);
                for (int axis = 0; axis < dim; axis++) {
                    writer.write('\t');
                    writer.write(String.format(Locale.US, "%.17g", leafEmbeddings[taxonIndex * dim + axis]));
                }
                writer.newLine();
            }
        }
    }

    public static ModelState readModel(Path inputFile, String[] expectedCanonicalTaxa) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(inputFile)) {
            String magic = reader.readLine();
            boolean isV1 = MAGIC_V1.equals(magic);
            boolean isV2 = MAGIC_V2.equals(magic);
            boolean isV3 = MAGIC_V3.equals(magic);
            boolean isV4 = MAGIC_V4.equals(magic);
            if (!isV1 && !isV2 && !isV3 && !isV4) {
                throw new IllegalArgumentException("Unrecognized LearnedWeights model file format: " + inputFile);
            }

            String taxaLine = reader.readLine();
            if (taxaLine == null) {
                throw new IllegalArgumentException("Missing taxa header in LearnedWeights model file: " + inputFile);
            }
            String[] taxaTokens = taxaLine.split("\\t");
            if (taxaTokens.length < 2 || !"taxa".equals(taxaTokens[0])) {
                throw new IllegalArgumentException("Invalid taxa header in LearnedWeights model file: " + inputFile);
            }
            String[] fileTaxa = Arrays.copyOfRange(taxaTokens, 1, taxaTokens.length);
            if (expectedCanonicalTaxa != null && !Arrays.equals(fileTaxa, expectedCanonicalTaxa)) {
                throw new IllegalArgumentException(
                        "LearnedWeights model taxa do not match expected taxa order. expected="
                                + Arrays.toString(expectedCanonicalTaxa)
                                + " file=" + Arrays.toString(fileTaxa));
            }

            int dim = readIntValue(reader.readLine(), "dim", inputFile);
            double[] rates = readVector(reader.readLine(), "brownianTraitRates", dim, inputFile);
            double[] rootPriorVariances = null;
            boolean separateBaseAndQuadraticCalibration = false;
            boolean hasExplicitTrainingObjectiveMetadata = false;
            boolean scaleQuadraticPartOnly = false;
            boolean trainQuadraticOnly = false;
            double pcaRegularizationWeight = Double.NaN;
            double slopeRegularizationWeight = Double.NaN;
            double[] baseScales = null;
            double[] quadraticScales;
            String nextLine = reader.readLine();
            if (isV3 || isV4) {
                rootPriorVariances = readVector(nextLine, "rootPriorVariances", dim, inputFile);
                nextLine = reader.readLine();
            }
            if (isV2 || isV3 || isV4) {
                separateBaseAndQuadraticCalibration = Boolean.parseBoolean(
                        splitLine(nextLine, "separateBaseAndQuadraticCalibration", inputFile, 2)[1]);
                if (isV4) {
                    hasExplicitTrainingObjectiveMetadata = true;
                    scaleQuadraticPartOnly = Boolean.parseBoolean(
                            splitLine(reader.readLine(), "scaleQuadraticPartOnly", inputFile, 2)[1]);
                    trainQuadraticOnly = Boolean.parseBoolean(
                            splitLine(reader.readLine(), "trainQuadraticOnly", inputFile, 2)[1]);
                    pcaRegularizationWeight = readScalar(reader.readLine(), "pcaRegularizationWeight", inputFile);
                    slopeRegularizationWeight = readScalar(reader.readLine(), "slopeRegularizationWeight", inputFile);
                }
                baseScales = readVector(reader.readLine(), "baseScales", dim, inputFile);
                quadraticScales = readVector(reader.readLine(), "quadraticScales", dim, inputFile);
                nextLine = reader.readLine();
            } else {
                quadraticScales = readVector(nextLine, "slopes", dim, inputFile);
            }
            double intercept = readScalar(nextLine, "intercept", inputFile);

            Map<String, double[]> embeddingsByTaxon = new LinkedHashMap<>();
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                String[] tokens = line.split("\\t");
                if (tokens.length != 2 + dim || !"embedding".equals(tokens[0])) {
                    throw new IllegalArgumentException("Invalid embedding row in LearnedWeights model file: " + line);
                }
                String taxon = tokens[1];
                double[] embedding = new double[dim];
                for (int axis = 0; axis < dim; axis++) {
                    embedding[axis] = Double.parseDouble(tokens[2 + axis]);
                }
                embeddingsByTaxon.put(taxon, embedding);
            }

            double[] flattenedEmbeddings = new double[fileTaxa.length * dim];
            for (int taxonIndex = 0; taxonIndex < fileTaxa.length; taxonIndex++) {
                double[] embedding = embeddingsByTaxon.get(fileTaxa[taxonIndex]);
                if (embedding == null) {
                    throw new IllegalArgumentException("Missing embedding row for taxon " + fileTaxa[taxonIndex]);
                }
                System.arraycopy(embedding, 0, flattenedEmbeddings, taxonIndex * dim, dim);
            }

            return new ModelState(
                    flattenedEmbeddings,
                    rates,
                    rootPriorVariances,
                    baseScales,
                    quadraticScales,
                    intercept,
                    isV2 || isV3 || isV4,
                    separateBaseAndQuadraticCalibration,
                    hasExplicitTrainingObjectiveMetadata,
                    scaleQuadraticPartOnly,
                    trainQuadraticOnly,
                    pcaRegularizationWeight,
                    slopeRegularizationWeight);
        }
    }

    private static int readIntValue(String line, String key, Path inputFile) {
        String[] tokens = splitLine(line, key, inputFile, 2);
        return Integer.parseInt(tokens[1]);
    }

    private static double readScalar(String line, String key, Path inputFile) {
        String[] tokens = splitLine(line, key, inputFile, 2);
        return Double.parseDouble(tokens[1]);
    }

    private static double[] readVector(String line, String key, int expectedLength, Path inputFile) {
        String[] tokens = splitLine(line, key, inputFile, expectedLength + 1);
        double[] values = new double[expectedLength];
        for (int i = 0; i < expectedLength; i++) {
            values[i] = Double.parseDouble(tokens[i + 1]);
        }
        return values;
    }

    private static String[] splitLine(String line, String key, Path inputFile, int minTokens) {
        if (line == null) {
            throw new IllegalArgumentException("Missing " + key + " entry in LearnedWeights model file: " + inputFile);
        }
        String[] tokens = line.split("\\t");
        if (tokens.length < minTokens || !key.equals(tokens[0])) {
            throw new IllegalArgumentException("Invalid " + key + " row in LearnedWeights model file: " + inputFile);
        }
        return tokens;
    }

    public record ModelState(
            double[] leafEmbeddings,
            double[] brownianTraitRates,
            double[] rootPriorVariances,
            double[] baseScales,
            double[] quadraticScales,
            double intercept,
            boolean hasExplicitCalibrationMode,
            boolean separateBaseAndQuadraticCalibration,
            boolean hasExplicitTrainingObjectiveMetadata,
            boolean scaleQuadraticPartOnly,
            boolean trainQuadraticOnly,
            double pcaRegularizationWeight,
            double slopeRegularizationWeight) {
    }
}