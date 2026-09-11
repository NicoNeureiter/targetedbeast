package targetedbeast.edgeweights.learnedweights;

/**
 * Java inference engine for an {@code AxisReweightedFusedV1} model trained
 * in Python ({@code surrogate_training/train_axis_fused.py}).
 *
 * <p>Forward pass for one sample (see {@code export_fused_weights()} docstring):
 * <ol>
 *   <li>{@code contribution[d] = -0.5*qf[d] - 0.5*logDet}  (d = 0..dim-1)
 *   <li>{@code base = sum_d w[d] * contribution[d]}
 *   <li>{@code scalars_norm[s] = (scalars[s] - scalar_mean[s]) / scalar_std[s]}, order:
 *       root_height, total_branch_length, total_diffusion_length, root_free_log_det
 *       (root_free_log_det is the same {@code logDet} passed in above).
 *   <li>{@code linear_extra = sum_s scalar_w[s] * scalars_norm[s]}
 *   <li>{@code mlp_input = concat(contribution, scalars_norm)} (length dim+4, contribution
 *       is RAW here, not z-scored -- only scalars are normalised)
 *   <li>{@code h = ReLU(mlp_input @ mlp_w1^T + mlp_b1)}
 *   <li>{@code mlp_extra = h @ mlp_w2^T + mlp_b2}
 *   <li>{@code pred = base + linear_extra + mlp_extra}
 * </ol>
 */
public final class AxisReweightedFusedSurrogate {

    private final int nTaxa;
    private final int dim;
    private final int nScalars;
    private final int hidden;
    private final String[] taxa;

    private final double[] eFlat;       // row-major, length nTaxa*dim

    private final double[] w;           // length dim
    private final double[] scalarW;     // length nScalars
    private final double[] scalarMean;  // length nScalars
    private final double[] scalarStd;   // length nScalars

    private final double[] mlpW1;       // row-major, length hidden*(dim+nScalars)
    private final double[] mlpB1;       // length hidden
    private final double[] mlpW2;       // length hidden
    private final double   mlpB2;

    AxisReweightedFusedSurrogate(
            int nTaxa, int dim, int nScalars, int hidden, String[] taxa,
            double[] eFlat, double[] w, double[] scalarW, double[] scalarMean, double[] scalarStd,
            double[] mlpW1, double[] mlpB1, double[] mlpW2, double mlpB2) {
        this.nTaxa = nTaxa;
        this.dim = dim;
        this.nScalars = nScalars;
        this.hidden = hidden;
        this.taxa = taxa;
        this.eFlat = eFlat;
        this.w = w;
        this.scalarW = scalarW;
        this.scalarMean = scalarMean;
        this.scalarStd = scalarStd;
        this.mlpW1 = mlpW1;
        this.mlpB1 = mlpB1;
        this.mlpW2 = mlpW2;
        this.mlpB2 = mlpB2;
    }

    /**
     * Predict the surrogate score for one sample.
     *
     * <p>The training data's {@code log_det} (used in the per-axis {@code contribution}
     * term) and {@code root_free_log_det} (the 4th scalar) are numerically distinct
     * columns in general -- see {@code surrogate_training/data/exported/fused_v1_verification.csv}.
     * At the {@link PCAWeightsMLPLikelihood} runtime call site the two coincide (both
     * equal the local {@code logDet = -2*base[0]} value under the single-rate,
     * improper-root model), so callers there may pass the same value for both.
     *
     * @param qf                     raw per-axis quadratic forms, length {@link #dim()}
     * @param logDet                 log-determinant used in the per-axis contribution term
     * @param rootHeight             root height of the proposed tree
     * @param totalBranchLength      total branch length of the proposed tree
     * @param totalDiffusionLength   total diffusion length of the proposed tree
     * @param rootFreeLogDet         4th scalar feature (root_free_log_det); at the
     *                               runtime call site this equals {@code logDet}
     * @return surrogate score, same scale as the Python model's raw output
     */
    public double predict(double[] qf, double logDet,
                           double rootHeight, double totalBranchLength, double totalDiffusionLength,
                           double rootFreeLogDet) {
        double[] contribution = new double[dim];
        double base = 0.0;
        for (int d = 0; d < dim; d++) {
            contribution[d] = -0.5 * qf[d] - 0.5 * logDet;
            base += w[d] * contribution[d];
        }

        double[] scalars = { rootHeight, totalBranchLength, totalDiffusionLength, rootFreeLogDet };
        double[] scalarsNorm = new double[nScalars];
        double linearExtra = 0.0;
        for (int s = 0; s < nScalars; s++) {
            scalarsNorm[s] = (scalars[s] - scalarMean[s]) / scalarStd[s];
            linearExtra += scalarW[s] * scalarsNorm[s];
        }

        int inDim = dim + nScalars;
        double[] mlpInput = new double[inDim];
        System.arraycopy(contribution, 0, mlpInput, 0, dim);
        System.arraycopy(scalarsNorm, 0, mlpInput, dim, nScalars);

        double[] h = new double[hidden];
        for (int i = 0; i < hidden; i++) {
            double v = mlpB1[i];
            int wBase = i * inDim;
            for (int j = 0; j < inDim; j++) {
                v += mlpW1[wBase + j] * mlpInput[j];
            }
            h[i] = Math.max(0.0, v);
        }

        double mlpExtra = mlpB2;
        for (int i = 0; i < hidden; i++) {
            mlpExtra += mlpW2[i] * h[i];
        }

        return base + linearExtra + mlpExtra;
    }

    /**
     * Convenience overload for the {@link PCAWeightsMLPLikelihood} runtime call site,
     * where {@code logDet} and {@code root_free_log_det} are the same value.
     */
    public double predict(double[] qf, double logDet,
                           double rootHeight, double totalBranchLength, double totalDiffusionLength) {
        return predict(qf, logDet, rootHeight, totalBranchLength, totalDiffusionLength, logDet);
    }

    public int nTaxa()   { return nTaxa; }
    public int dim()     { return dim; }
    public int nScalars(){ return nScalars; }
    public int hidden()  { return hidden; }
    public String[] taxa() { return taxa; }
    public double[] eFlat() { return eFlat; }
}
