package targetedbeast.edgeweights.learnedweights;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Java inference engine for an {@code EmbeddingMLPSurrogateV1} model trained
 * in Python ({@code surrogate_training/quadratic_model.py}).
 *
 * <p>Forward pass for one sample:
 * <ol>
 *   <li>Compute per-axis quadratic forms {@code qf[d] = E[:,d] @ P @ E[:,d]}.
 *   <li>Normalise with frozen training-set statistics:
 *       {@code qfNorm[d] = (qf[d] - qfMean[d]) / qfStd[d]}.
 *   <li>Build MLP input {@code [qfNorm | ldNorm | ldNorm/(qfNorm+shift)]}.
 *   <li>Evaluate the MLP (linear + ReLU, linear for the last layer).
 *   <li>Un-normalise the output back to log-likelihood scale.
 * </ol>
 *
 * <p>Load from a weight file written by Python:
 * <pre>{@code
 *   EmbeddingMLPSurrogate model = EmbeddingMLPSurrogateIO.read(Path.of("surrogate_weights.json"));
 *   double logLik = model.predict(precisionMatrix, logDeterminant);
 * }</pre>
 */
public final class EmbeddingMLPSurrogate {

    private final int nTaxa;
    private final int dim;
    private final String[] taxa;

    private final boolean useLogdet;
    private final double  ratioShift;
    private final double  outputBound;  // k > 0: apply k·tanh(raw/k) before denorm; 0 = off

    // Frozen BN stats (length = dim)
    private final double[] qfMean;
    private final double[] qfStd;

    // logDet and target normalisation scalars
    private final double ldMean, ldStd, yMean, yStd;

    // Leaf embeddings: E[taxon * dim + axis]  (row-major, length = nTaxa * dim)
    private final double[] E;

    // MLP layers
    private final int      nLayers;
    private final int[]    layerIn;
    private final int[]    layerOut;
    private final double[][] layerWeight;  // [layer][out * in], row-major
    private final double[][] layerBias;    // [layer][out]

    // ------------------------------------------------------------------
    // Constructor (package-private — use EmbeddingMLPSurrogateIO.read())
    // ------------------------------------------------------------------

    EmbeddingMLPSurrogate(
            int nTaxa, int dim, String[] taxa,
            boolean useLogdet, double ratioShift, double outputBound,
            double[] qfMean, double[] qfStd,
            double ldMean, double ldStd,
            double yMean, double yStd,
            double[] eFlat,
            int nLayers, int[] layerIn, int[] layerOut,
            double[][] layerWeight, double[][] layerBias) {
        this.nTaxa  = nTaxa;
        this.dim    = dim;
        this.taxa   = taxa;
        this.useLogdet   = useLogdet;
        this.ratioShift  = ratioShift;
        this.outputBound = outputBound;
        this.qfMean = qfMean;
        this.qfStd  = qfStd;
        this.ldMean = ldMean;
        this.ldStd  = ldStd;
        this.yMean  = yMean;
        this.yStd   = yStd;
        this.E      = eFlat;
        this.nLayers     = nLayers;
        this.layerIn     = layerIn;
        this.layerOut    = layerOut;
        this.layerWeight = layerWeight;
        this.layerBias   = layerBias;
    }

    // ------------------------------------------------------------------
    // Factory
    // ------------------------------------------------------------------

    public static EmbeddingMLPSurrogate fromFile(Path path) throws IOException {
        return EmbeddingMLPSurrogateIO.read(path);
    }

    // ------------------------------------------------------------------
    // Inference
    // ------------------------------------------------------------------

    /**
     * Predict the surrogate log-likelihood for one sample.
     *
     * @param P              precision matrix, shape [nTaxa][nTaxa], in the
     *                       same taxon order as {@link #taxa()}
     * @param logDeterminant log-determinant of the covariance matrix for this sample
     * @return surrogate log-likelihood in the original (un-normalised) scale
     */
    public double predict(double[][] P, double logDeterminant) {
        // --- Step 1: quadratic forms qf[d] = E[:,d] @ P @ E[:,d] ----------
        double[] qf = new double[dim];
        for (int d = 0; d < dim; d++) {
            double qfd = 0.0;
            for (int i = 0; i < nTaxa; i++) {
                double PEid = 0.0;
                for (int j = 0; j < nTaxa; j++) {
                    PEid += P[i][j] * E[j * dim + d];
                }
                qfd += E[i * dim + d] * PEid;
            }
            qf[d] = qfd;
        }

        // --- Step 2: normalise qf ------------------------------------------
        double[] qfNorm = new double[dim];
        for (int d = 0; d < dim; d++) {
            qfNorm[d] = (qf[d] - qfMean[d]) / qfStd[d];
        }

        // --- Step 3: build MLP input vector ----------------------------------
        double[] x;
        if (useLogdet) {
            double ldNorm = (logDeterminant - ldMean) / ldStd;
            x = new double[2 * dim + 1];
            System.arraycopy(qfNorm, 0, x, 0, dim);
            x[dim] = ldNorm;
            for (int d = 0; d < dim; d++) {
                x[dim + 1 + d] = ldNorm / (qfNorm[d] + ratioShift);
            }
        } else {
            x = Arrays.copyOf(qfNorm, dim);
        }

        // --- Step 4: MLP forward --------------------------------------------
        for (int layer = 0; layer < nLayers; layer++) {
            double[] w   = layerWeight[layer];
            double[] b   = layerBias[layer];
            int      in  = layerIn[layer];
            int      out = layerOut[layer];
            double[] next = new double[out];
            for (int i = 0; i < out; i++) {
                double v = b[i];
                int wBase = i * in;
                for (int j = 0; j < in; j++) {
                    v += w[wBase + j] * x[j];
                }
                // ReLU on all but the last layer
                next[i] = (layer < nLayers - 1) ? Math.max(0.0, v) : v;
            }
            x = next;
        }

        // --- Step 5: bounded activation + un-normalise ----------------------
        double raw = x[0];
        if (outputBound > 0.0) raw = outputBound * Math.tanh(raw / outputBound);
        return raw * yStd + yMean;
    }

    // ------------------------------------------------------------------
    // Inference helpers
    // ------------------------------------------------------------------

    /**
     * Predict using pre-computed raw quadratic form values (avoids recomputing qf
     * when the Cholesky factor is already available).
     *
     * @param qf             raw quadratic forms qf[d] = E[:,d]^T P E[:,d]  (length = dim)
     * @param logDeterminant log-determinant of the covariance matrix
     * @return surrogate log-likelihood in the original (un-normalised) scale
     */
    public double predictFromQfAndLogDet(double[] qf, double logDeterminant) {
        double[] qfNorm = new double[dim];
        for (int d = 0; d < dim; d++) {
            qfNorm[d] = (qf[d] - qfMean[d]) / qfStd[d];
        }

        double[] x;
        if (useLogdet) {
            double ldNorm = (logDeterminant - ldMean) / ldStd;
            x = new double[2 * dim + 1];
            System.arraycopy(qfNorm, 0, x, 0, dim);
            x[dim] = ldNorm;
            for (int d = 0; d < dim; d++) {
                x[dim + 1 + d] = ldNorm / (qfNorm[d] + ratioShift);
            }
        } else {
            x = Arrays.copyOf(qfNorm, dim);
        }

        for (int layer = 0; layer < nLayers; layer++) {
            double[] w   = layerWeight[layer];
            double[] b   = layerBias[layer];
            int      in  = layerIn[layer];
            int      out = layerOut[layer];
            double[] next = new double[out];
            for (int i = 0; i < out; i++) {
                double v = b[i];
                int wBase = i * in;
                for (int j = 0; j < in; j++) {
                    v += w[wBase + j] * x[j];
                }
                next[i] = (layer < nLayers - 1) ? Math.max(0.0, v) : v;
            }
            x = next;
        }

        double raw = x[0];
        if (outputBound > 0.0) raw = outputBound * Math.tanh(raw / outputBound);
        return raw * yStd + yMean;
    }

    /**
     * Compute per-axis quadratic forms from the lower-triangular Cholesky factor
     * of the covariance matrix, using an explicitly supplied embedding matrix.
     *
     * <p>Given L s.t. V = L L^T and an embedding matrix {@code eFlat} (row-major,
     * length {@code n * d}), computes qf[d] = ||L^{-1} E[:,d]||^2 via forward
     * substitution.  This overload is used at Java runtime so that the same
     * embeddings as the {@code runtimeModel} are used for scoring, keeping the
     * MLP's internal {@code E_flat} (from the JSON) unused during MCMC.
     *
     * @param L     lower-triangular Cholesky factor of V, shape [n][n]
     * @param eFlat external embedding matrix, row-major, length n * dim
     * @return qf array of length dim
     */
    public double[] computeQfFromCovarianceCholeskyL(double[][] L, double[] eFlat) {
        double[] qf  = new double[dim];
        double[] xd  = new double[nTaxa];
        for (int d = 0; d < dim; d++) {
            for (int i = 0; i < nTaxa; i++) {
                double sum = eFlat[i * dim + d];
                for (int j = 0; j < i; j++) {
                    sum -= L[i][j] * xd[j];
                }
                xd[i] = sum / L[i][i];
            }
            double qfd = 0.0;
            for (int i = 0; i < nTaxa; i++) {
                qfd += xd[i] * xd[i];
            }
            qf[d] = qfd;
        }
        return qf;
    }

    /**
     * Compute per-axis quadratic forms using the model's own internal embeddings.
     * Prefer {@link #computeQfFromCovarianceCholeskyL(double[][], double[])} when
     * Java-trained embeddings are available.
     */
    public double[] computeQfFromCovarianceCholeskyL(double[][] L) {
        return computeQfFromCovarianceCholeskyL(L, E);
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    public int     nTaxa()      { return nTaxa; }
    public int     dim()        { return dim; }
    public String[] taxa()      { return taxa; }
    public int     nLayers()    { return nLayers; }
    public double  yMean()      { return yMean; }
    public double  yStd()       { return yStd; }

    /** Frozen training-set per-axis qf statistics (length = dim). */
    public double[] qfMean()    { return qfMean; }
    public double[] qfStd()     { return qfStd; }
    public double   ldMean()    { return ldMean; }
    public double   ldStd()     { return ldStd; }

    /** Raw leaf-embedding array, row-major, length {@code nTaxa * dim}. */
    public double[] eFlat()     { return E; }
}
