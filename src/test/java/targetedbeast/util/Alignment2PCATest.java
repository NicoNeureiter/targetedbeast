package targetedbeast.util;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.reflect.Method;
import java.util.Map;

import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.datatype.TwoStateCovarion;

public class Alignment2PCATest {

    @Test
    public void distanceBasedEmbeddingUsesClassicalMdsGeometry() throws Exception {
        Alignment2PCA pca = new Alignment2PCA();
        pca.initByName("dimension", 1);

        double[][] distances = new double[][] {
                {0.0, 1.0, 3.0},
                {1.0, 0.0, 2.0},
                {3.0, 2.0, 0.0}
        };

        Method method = Alignment2PCA.class.getDeclaredMethod("getPoints", double[][].class);
        method.setAccessible(true);
        double[][] coordinates = (double[][]) method.invoke(pca, (Object) distances);

        assertEquals(1.0, Math.abs(coordinates[0][0] - coordinates[1][0]), 1e-8);
        assertEquals(2.0, Math.abs(coordinates[1][0] - coordinates[2][0]), 1e-8);
        assertEquals(3.0, Math.abs(coordinates[0][0] - coordinates[2][0]), 1e-8);
    }

    @Test
    public void distanceBasedEmbeddingUsesVisibleCovarionStates() {
        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", new Sequence("A", "0000"),
                "sequence", new Sequence("B", "1111"),
                "sequence", new Sequence("C", "0011"),
            "userDataType", new TwoStateCovarion());

        Map<String, double[]> points = Alignment2PCA.getPoints(alignment, 1, true, true);

        double abDistance = Math.abs(points.get("A")[0] - points.get("B")[0]);
        double acDistance = Math.abs(points.get("A")[0] - points.get("C")[0]);
        double bcDistance = Math.abs(points.get("B")[0] - points.get("C")[0]);

        assertTrue(abDistance > 1e-8);
        assertTrue(acDistance > 1e-8 || bcDistance > 1e-8);
    }

    @Test
    public void compressedEmbeddingHandlesDuplicateSequenceIdsWhenLastTaxonIsNotUnique() {
        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", new Sequence("A", "0000"),
                "sequence", new Sequence("B", "1111"),
                "sequence", new Sequence("C", "0011"),
                "sequence", new Sequence("D", "1111"),
                "userDataType", new TwoStateCovarion());

        Map<String, double[]> points = Alignment2PCA.getPoints(alignment, 2, true, true);

        assertEquals(4, points.size());
        assertEquals(2, points.get("A").length);
        assertArrayEquals(points.get("B"), points.get("D"), 1e-12);
    }
}