package targetedbeast.edgeweights.learnedweights;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.Test;

public class LearnedWeightsTrainingSampleIOTest {

    @Test
    public void roundTripPreservesTrainingSamples() throws Exception {
        String[] taxa = new String[] {"A", "B"};
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = List.of(
                new LearnedWeightsTreeTrainer.TrainingTreeSample(
                        -12.5,
                        new double[][] {{1.0, 0.25}, {0.25, 2.0}},
                        0.75),
                new LearnedWeightsTreeTrainer.TrainingTreeSample(
                        -8.0,
                        new double[][] {{3.0, 0.5}, {0.5, 4.0}},
                        1.25));

        Path tempFile = Files.createTempFile("learnedweights-samples", ".tsv");
        try {
            LearnedWeightsTrainingSampleIO.writeSamples(tempFile, taxa, samples);
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> restored = LearnedWeightsTrainingSampleIO.readSamples(tempFile, taxa);

            assertEquals(samples.size(), restored.size());
            for (int i = 0; i < samples.size(); i++) {
                assertEquals(samples.get(i).targetLogLikelihood, restored.get(i).targetLogLikelihood, 0.0);
                assertEquals(samples.get(i).logDeterminant, restored.get(i).logDeterminant, 0.0);
                for (int row = 0; row < taxa.length; row++) {
                    assertArrayEquals(samples.get(i).precisionMatrix[row], restored.get(i).precisionMatrix[row], 0.0);
                }
            }
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }

    @Test
    public void roundTripPreservesGroupIdAndTreeSizeCovariates() throws Exception {
        String[] taxa = new String[] {"A", "B"};
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = List.of(
                new LearnedWeightsTreeTrainer.TrainingTreeSample(
                        -12.5,
                        new double[][] {{1.0, 0.25}, {0.25, 2.0}},
                        0.75,
                        2,
                        new double[] {1.0, 0.5, 1.5},
                        0.6,
                        2,
                        3.5,
                        7.25,
                        9.75,
                        42L),
                new LearnedWeightsTreeTrainer.TrainingTreeSample(
                        -8.0,
                        new double[][] {{3.0, 0.5}, {0.5, 4.0}},
                        1.25,
                        2,
                        new double[] {2.0, 0.1, 3.0},
                        0.9,
                        2,
                        1.1,
                        2.2,
                        3.3,
                        LearnedWeightsTreeTrainer.TrainingTreeSample.NO_GROUP));

        Path tempFile = Files.createTempFile("learnedweights-samples-v5", ".tsv");
        try {
            LearnedWeightsTrainingSampleIO.writeSamples(tempFile, taxa, samples);
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> restored = LearnedWeightsTrainingSampleIO.readSamples(tempFile, taxa);

            assertEquals(samples.size(), restored.size());
            for (int i = 0; i < samples.size(); i++) {
                assertEquals(samples.get(i).groupId, restored.get(i).groupId);
                assertEquals(samples.get(i).rootHeight, restored.get(i).rootHeight, 1e-9);
                assertEquals(samples.get(i).totalBranchLength, restored.get(i).totalBranchLength, 1e-9);
                assertEquals(samples.get(i).totalDiffusionLength, restored.get(i).totalDiffusionLength, 1e-9);
                assertEquals(samples.get(i).targetLogLikelihood, restored.get(i).targetLogLikelihood, 0.0);
                assertEquals(samples.get(i).logDeterminant, restored.get(i).logDeterminant, 0.0);
                for (int row = 0; row < taxa.length; row++) {
                    assertArrayEquals(samples.get(i).precisionMatrix[row], restored.get(i).precisionMatrix[row], 0.0);
                }
            }
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }

    @Test
    public void writeSamplesWithoutGroupOrSizeDataStillWritesV4() throws Exception {
        String[] taxa = new String[] {"A", "B"};
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples = List.of(
                new LearnedWeightsTreeTrainer.TrainingTreeSample(
                        -12.5,
                        new double[][] {{1.0, 0.25}, {0.25, 2.0}},
                        0.75,
                        2,
                        new double[] {1.0, 0.5, 1.5},
                        0.6,
                        2,
                        Double.NaN,
                        Double.NaN,
                        Double.NaN));

        Path tempFile = Files.createTempFile("learnedweights-samples-v4", ".tsv");
        try {
            LearnedWeightsTrainingSampleIO.writeSamples(tempFile, taxa, samples);
            List<String> lines = Files.readAllLines(tempFile);
            assertEquals("# LearnedWeightsTrainingSamplesV4", lines.get(0));

            List<LearnedWeightsTreeTrainer.TrainingTreeSample> restored = LearnedWeightsTrainingSampleIO.readSamples(tempFile, taxa);
            assertEquals(LearnedWeightsTreeTrainer.TrainingTreeSample.NO_GROUP, restored.get(0).groupId);
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void readSamplesRejectsMismatchedTaxaOrder() throws Exception {
        String[] taxa = new String[] {"A", "B"};
        Path tempFile = Files.createTempFile("learnedweights-samples", ".tsv");
        try {
            LearnedWeightsTrainingSampleIO.writeSamples(
                    tempFile,
                    taxa,
                    List.of(new LearnedWeightsTreeTrainer.TrainingTreeSample(
                            -12.5,
                            new double[][] {{1.0, 0.25}, {0.25, 2.0}},
                            0.75)));
            LearnedWeightsTrainingSampleIO.readSamples(tempFile, new String[] {"B", "A"});
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }
}