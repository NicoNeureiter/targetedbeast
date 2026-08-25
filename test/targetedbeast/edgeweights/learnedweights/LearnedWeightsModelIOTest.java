package targetedbeast.edgeweights.learnedweights;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.Test;

public class LearnedWeightsModelIOTest {

    @Test
    public void roundTripPreservesLearnedWeightsState() throws Exception {
        String[] taxa = new String[] {"A", "B"};
        double[] embeddings = new double[] {0.1, 0.2, 0.3, 0.4};
        double[] rates = new double[] {0.5, 1.5};
        double[] rootPriorVariances = new double[] {0.25, 0.75};
        double[] baseScales = new double[] {0.7, 1.3};
        double[] quadraticScales = new double[] {1.1, 0.9};
        double intercept = -0.25;

        Path tempFile = Files.createTempFile("learnedweights-model", ".tsv");
        try {
                LearnedWeightsModelIO.writeModel(
                    tempFile,
                    taxa,
                    embeddings,
                    rates,
                    rootPriorVariances,
                    baseScales,
                    quadraticScales,
                    intercept,
                    true,
                    false,
                    true,
                    0.125,
                    0.375);
            LearnedWeightsModelIO.ModelState restored = LearnedWeightsModelIO.readModel(tempFile, taxa);

            assertArrayEquals(embeddings, restored.leafEmbeddings(), 0.0);
            assertArrayEquals(rates, restored.brownianTraitRates(), 0.0);
            assertArrayEquals(rootPriorVariances, restored.rootPriorVariances(), 0.0);
            assertArrayEquals(baseScales, restored.baseScales(), 0.0);
            assertArrayEquals(quadraticScales, restored.quadraticScales(), 0.0);
            assertEquals(intercept, restored.intercept(), 0.0);
            assertEquals(true, restored.hasExplicitCalibrationMode());
            assertEquals(true, restored.separateBaseAndQuadraticCalibration());
            assertTrue(restored.hasExplicitTrainingObjectiveMetadata());
            assertEquals(false, restored.scaleQuadraticPartOnly());
            assertEquals(true, restored.trainQuadraticOnly());
            assertEquals(0.125, restored.pcaRegularizationWeight(), 0.0);
            assertEquals(0.375, restored.slopeRegularizationWeight(), 0.0);
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }
}