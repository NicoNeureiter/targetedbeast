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