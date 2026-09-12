package targetedbeast.edgeweights.learnedweights;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Reads an {@code AxisReweightedFusedV1} JSON weight file produced by
 * {@code surrogate_training/train_axis_fused.py}'s {@code export_fused_weights()}
 * and returns an {@link AxisReweightedFusedSurrogate} ready for inference.
 *
 * <p>Shares {@link EmbeddingMLPSurrogateIO}'s minimal hand-rolled JSON scanner
 * (no external JSON library, flat schema only) rather than duplicating it.
 */
public final class AxisReweightedFusedSurrogateIO {

    static final String FORMAT_TAG = "AxisReweightedFusedV1";

    private AxisReweightedFusedSurrogateIO() {}

    public static AxisReweightedFusedSurrogate read(Path path) throws IOException {
        String text = Files.readString(path);
        EmbeddingMLPSurrogateIO.Scanner sc = new EmbeddingMLPSurrogateIO.Scanner(text);
        sc.expectObject();

        String format = null;
        int dim = -1, nScalars = -1, hidden = -1, nTaxa = -1;
        String[] taxa = null;
        double[] eFlat = null;
        double[] w = null, scalarW = null, scalarMean = null, scalarStd = null;
        double[] contributionMean = null, contributionStd = null;
        double[] mlpW1 = null, mlpB1 = null, mlpW2 = null, mlpB2Arr = null;

        while (sc.hasNextKey()) {
            String key = sc.readKey();
            switch (key) {
                case "format"      -> format     = sc.readString();
                case "dim"         -> dim        = (int) sc.readDouble();
                case "n_scalars"   -> nScalars   = (int) sc.readDouble();
                case "hidden"      -> hidden     = (int) sc.readDouble();
                case "n_taxa"      -> nTaxa      = (int) sc.readDouble();
                case "taxa"        -> taxa       = sc.readStringArray();
                case "E_flat"      -> eFlat      = sc.readDoubleArray();
                case "w"           -> w          = sc.readDoubleArray();
                case "scalar_w"    -> scalarW    = sc.readDoubleArray();
                case "scalar_mean" -> scalarMean = sc.readDoubleArray();
                case "scalar_std"  -> scalarStd  = sc.readDoubleArray();
                case "contribution_mean" -> contributionMean = sc.readDoubleArray();
                case "contribution_std"  -> contributionStd  = sc.readDoubleArray();
                case "mlp_w1"      -> mlpW1      = sc.readDoubleArray();
                case "mlp_b1"      -> mlpB1      = sc.readDoubleArray();
                case "mlp_w2"      -> mlpW2      = sc.readDoubleArray();
                case "mlp_b2"      -> mlpB2Arr   = sc.readDoubleArray();
                default            -> sc.skipValue();
            }
            sc.consumeComma();
        }

        if (!FORMAT_TAG.equals(format))
            throw new IllegalArgumentException("Unexpected format tag: " + format);
        requireNotNull(dim, "dim");
        requireNotNull(nScalars, "n_scalars");
        requireNotNull(hidden, "hidden");
        requireNotNull(nTaxa, "n_taxa");
        requireNotNull(eFlat, "E_flat");
        requireNotNull(w, "w");
        requireNotNull(scalarW, "scalar_w");
        requireNotNull(scalarMean, "scalar_mean");
        requireNotNull(scalarStd, "scalar_std");
        requireNotNull(mlpW1, "mlp_w1");
        requireNotNull(mlpB1, "mlp_b1");
        requireNotNull(mlpW2, "mlp_w2");
        requireNotNull(mlpB2Arr, "mlp_b2");
        if (taxa == null) taxa = defaultTaxa(nTaxa);
        if (mlpB2Arr.length != 1)
            throw new IllegalArgumentException("mlp_b2 expected length 1, got " + mlpB2Arr.length);
        // Absent from older AxisReweightedFusedV1 files (pre-dating contribution z-scoring
        // support): default to mean=0/std=1, a no-op normalisation, so old exports still work.
        if (contributionMean == null) contributionMean = new double[dim];
        if (contributionStd == null) {
            contributionStd = new double[dim];
            java.util.Arrays.fill(contributionStd, 1.0);
        }

        return new AxisReweightedFusedSurrogate(
                nTaxa, dim, nScalars, hidden, taxa,
                eFlat, w, scalarW, scalarMean, scalarStd,
                contributionMean, contributionStd,
                mlpW1, mlpB1, mlpW2, mlpB2Arr[0]);
    }

    private static void requireNotNull(Object val, String field) {
        if (val == null || (val instanceof Integer i && i == -1))
            throw new IllegalArgumentException("Missing required field: " + field);
    }

    private static String[] defaultTaxa(int n) {
        String[] names = new String[n];
        for (int i = 0; i < n; i++) names[i] = "taxon_" + i;
        return names;
    }
}
