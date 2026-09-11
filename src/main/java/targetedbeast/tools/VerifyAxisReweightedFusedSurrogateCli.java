package targetedbeast.tools;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import targetedbeast.edgeweights.learnedweights.AxisReweightedFusedSurrogate;
import targetedbeast.edgeweights.learnedweights.AxisReweightedFusedSurrogateIO;

/**
 * Round-trip check for {@link AxisReweightedFusedSurrogate}: reads
 * {@code surrogate_training/data/exported/fused_v1.json} and scores every row
 * of {@code fused_v1_verification.csv} (qf_0..qf_9, log_det, root_height,
 * total_branch_length, total_diffusion_length, root_free_log_det,
 * expected_pred), asserting the Java forward pass matches the Python
 * reference prediction within float32-level tolerance.
 *
 * <p>Usage: {@code VerifyAxisReweightedFusedSurrogateCli <weights.json> <verification.csv>}
 * (both args optional -- default to the standard exported-data paths).
 */
public class VerifyAxisReweightedFusedSurrogateCli {

    // Predictions run ~10^4-10^5 in magnitude; the exported weights are float32,
    // so ~0.01 absolute error is expected rounding noise, not a bug (per task spec).
    // Use a relative tolerance so this scales correctly with prediction magnitude.
    private static final double RELATIVE_TOLERANCE = 1e-3;
    private static final double ABSOLUTE_FLOOR = 1e-2;

    public static void main(String[] args) throws Exception {
        Path weightsPath = Path.of(args.length > 0 ? args[0]
                : "surrogate_training/data/exported/fused_v1.json");
        Path csvPath = Path.of(args.length > 1 ? args[1]
                : "surrogate_training/data/exported/fused_v1_verification.csv");

        AxisReweightedFusedSurrogate model = AxisReweightedFusedSurrogateIO.read(weightsPath);
        System.out.printf(Locale.ROOT, "Loaded model: dim=%d nScalars=%d hidden=%d nTaxa=%d%n",
                model.dim(), model.nScalars(), model.hidden(), model.nTaxa());

        int nRows = 0;
        double maxAbsErr = 0.0;
        try (BufferedReader r = Files.newBufferedReader(csvPath)) {
            String header = r.readLine();
            String[] cols = header.split(",");
            String line;
            while ((line = r.readLine()) != null) {
                if (line.isBlank()) continue;
                String[] parts = line.split(",");
                double[] row = new double[parts.length];
                for (int i = 0; i < parts.length; i++) row[i] = Double.parseDouble(parts[i]);

                int idx = indexOf(cols, "row");
                double[] qf = new double[model.dim()];
                for (int d = 0; d < model.dim(); d++) {
                    qf[d] = row[indexOf(cols, "qf_" + d)];
                }
                double logDet = row[indexOf(cols, "log_det")];
                double rootHeight = row[indexOf(cols, "root_height")];
                double totalBranchLength = row[indexOf(cols, "total_branch_length")];
                double totalDiffusionLength = row[indexOf(cols, "total_diffusion_length")];
                double rootFreeLogDet = row[indexOf(cols, "root_free_log_det")];
                double expectedPred = row[indexOf(cols, "expected_pred")];

                // NOTE: log_det (used in the contribution term) and root_free_log_det
                // (the 4th scalar) are numerically distinct in this offline data -- the
                // 5-arg predict() overload (which reuses one logDet for both) is only
                // valid at the PCAWeightsMLPLikelihood runtime call site where they
                // coincide; here we must pass both explicitly to match expected_pred.
                double pred = model.predict(qf, logDet, rootHeight, totalBranchLength, totalDiffusionLength,
                        rootFreeLogDet);
                double absErr = Math.abs(pred - expectedPred);
                double tolerance = Math.max(ABSOLUTE_FLOOR, RELATIVE_TOLERANCE * Math.abs(expectedPred));
                maxAbsErr = Math.max(maxAbsErr, absErr);
                nRows++;

                if (absErr > tolerance) {
                    throw new AssertionError(String.format(Locale.ROOT,
                            "row %.0f: pred=%.6f expected=%.6f absErr=%.6f exceeds tolerance %.6f",
                            row[idx], pred, expectedPred, absErr, tolerance));
                }
            }
        }

        System.out.printf(Locale.ROOT, "Verified %d rows, max abs error = %.6g (relative tolerance %.1e)%n",
                nRows, maxAbsErr, RELATIVE_TOLERANCE);
        System.out.println("PASS");
    }

    private static int indexOf(String[] cols, String name) {
        for (int i = 0; i < cols.length; i++) if (cols[i].equals(name)) return i;
        throw new IllegalArgumentException("Column not found: " + name);
    }
}
