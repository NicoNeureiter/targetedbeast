package targetedbeast.tools;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import targetedbeast.edgeweights.learnedweights.EmbeddingMLPSurrogate;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsTrainingSampleIO;
import targetedbeast.edgeweights.learnedweights.LearnedWeightsTreeTrainer;

/**
 * Java mirror of {@code surrogate_training/evaluate_pretrained_mlp.py}.
 *
 * <p>Loads a surrogate JSON written by Python and a proposals TSV (V4 schema)
 * written by {@code ProposalExporter}, then scores every row through the
 * Java {@link EmbeddingMLPSurrogate} — using the same code path the runtime
 * uses for {@code getTargetWeights*} via {@code predictFromQfAndLogDet} —
 * and reports Pearson / Spearman / RMSE versus the true target log-likelihood.
 *
 * <p>If the Python evaluator and this CLI report different numbers on the
 * same {@code proposals.tsv} the bug is on the Java side. If they agree the
 * bug is elsewhere in the operator pipeline.
 *
 * <p>Usage:
 * <pre>{@code
 *   java -cp <classpath> targetedbeast.tools.EvaluatePretrainedMLPCli \
 *       --surrogate surrogate_weights.json \
 *       --proposals proposals.tsv \
 *       [--out eval.csv]
 * }</pre>
 */
public class EvaluatePretrainedMLPCli {

    public static void main(String[] args) throws Exception {
        Args cli = Args.parse(args);
        if (cli.showHelp) {
            printUsage();
            return;
        }
        if (cli.surrogatePath == null) {
            throw new IllegalArgumentException("Missing --surrogate <surrogate_weights.json>");
        }
        if (cli.proposalsPath == null) {
            throw new IllegalArgumentException("Missing --proposals <proposals.tsv>");
        }

        Path surrogatePath = Paths.get(cli.surrogatePath).toAbsolutePath();
        Path proposalsPath = Paths.get(cli.proposalsPath).toAbsolutePath();
        if (!Files.exists(surrogatePath)) {
            throw new IllegalArgumentException("Surrogate JSON not found: " + surrogatePath);
        }
        if (!Files.exists(proposalsPath)) {
            throw new IllegalArgumentException("Proposals TSV not found: " + proposalsPath);
        }

        EmbeddingMLPSurrogate surrogate = EmbeddingMLPSurrogate.fromFile(surrogatePath);
        System.out.printf(Locale.US,
            "Loaded surrogate: n_taxa=%d dim=%d n_layers=%d  y_mean=%.4f  y_std=%.4f%n",
            surrogate.nTaxa(), surrogate.dim(), surrogate.nLayers(),
            surrogate.yMean(), surrogate.yStd());

        // Read every row. expectedCanonicalTaxa=null skips the taxa-order check;
        // we trust that the proposals.tsv was emitted by the same alignment.
        List<LearnedWeightsTreeTrainer.TrainingTreeSample> samples =
            LearnedWeightsTrainingSampleIO.readSamples(proposalsPath, null);
        int n = samples.size();
        if (n == 0) {
            System.err.println("No samples in proposals TSV.");
            System.exit(1);
        }
        System.out.printf(Locale.US, "n_samples=%d%n", n);

        if (samples.get(0).precisionMatrix.length != surrogate.nTaxa()) {
            throw new IllegalArgumentException(
                "Taxon count mismatch: surrogate expects " + surrogate.nTaxa()
                + " but proposals TSV has " + samples.get(0).precisionMatrix.length);
        }

        // Score every sample.  Two paths:
        //   - default: qf = E^T P E (matches the Python evaluator exactly)
        //   - --use-cholesky: qf = ||L^{-1} E||^2 via forward substitution
        //                     (matches the runtime pruning path numerically;
        //                      what computeQfFromCovarianceCholeskyL does)
        // Both feed into the same predictFromQfAndLogDet that PCAWeightsMLPLikelihood
        // calls. If the two paths disagree, the stored P and L are inconsistent
        // (a bug in ProposalExporter / buildTrainingSample). If they agree but
        // the operator acceptance is still bad, the bug is downstream of the
        // surrogate (clip / sqrt / Hastings pipeline).
        System.out.printf(Locale.US, "qf path: %s%n",
            cli.useCholesky ? "Cholesky forward-substitution (runtime-style)"
                            : "E^T P E (precision-matrix-style, matches Python)");

        double[] preds   = new double[n];
        double[] targets = new double[n];
        // Also collect both qf paths on the first sample so any disagreement
        // becomes immediately obvious instead of being averaged away by Pearson.
        double[] qfFirstP = null, qfFirstL = null;
        int finite = 0;
        for (int s = 0; s < n; s++) {
            LearnedWeightsTreeTrainer.TrainingTreeSample sample = samples.get(s);
            double[] qf;
            if (cli.useCholesky) {
                double[][] L = unpackLowerTriangular(sample.rootFreeCovarianceCholeskyLower,
                                                    surrogate.nTaxa());
                qf = surrogate.computeQfFromCovarianceCholeskyL(L, surrogate.eFlat());
            } else {
                qf = computeQuadraticForms(sample.precisionMatrix, surrogate.eFlat(),
                                           surrogate.nTaxa(), surrogate.dim());
            }
            if (s == 0) {
                // Compute the OTHER path too so we can print both side-by-side.
                if (cli.useCholesky) {
                    qfFirstL = qf;
                    qfFirstP = computeQuadraticForms(sample.precisionMatrix, surrogate.eFlat(),
                                                    surrogate.nTaxa(), surrogate.dim());
                } else {
                    qfFirstP = qf;
                    if (sample.rootFreeCovarianceCholeskyLower != null) {
                        double[][] L = unpackLowerTriangular(sample.rootFreeCovarianceCholeskyLower,
                                                            surrogate.nTaxa());
                        qfFirstL = surrogate.computeQfFromCovarianceCholeskyL(L, surrogate.eFlat());
                    }
                }
            }
            preds[s]   = surrogate.predictFromQfAndLogDet(qf, sample.logDeterminant);
            targets[s] = sample.targetLogLikelihood;
            if (Double.isFinite(preds[s]) && Double.isFinite(targets[s])) finite++;
        }

        // Diagnostic: qf values from both paths on the first sample.
        if (qfFirstP != null && qfFirstL != null) {
            System.out.println();
            System.out.println("First-sample qf comparison (P-form vs L-form):");
            System.out.printf(Locale.US, "  %-3s  %14s  %14s  %14s%n",
                "d", "qf(E^T P E)", "qf(||L^-1 E||^2)", "abs diff");
            for (int d = 0; d < surrogate.dim(); d++) {
                System.out.printf(Locale.US, "  %-3d  %14.6g  %14.6g  %14.3e%n",
                    d, qfFirstP[d], qfFirstL[d], Math.abs(qfFirstP[d] - qfFirstL[d]));
            }
        }
        if (finite < n) {
            System.err.printf(Locale.US, "WARNING: dropping %d non-finite rows.%n", n - finite);
        }

        // Filter to finite rows for metrics
        double[] p = new double[finite];
        double[] t = new double[finite];
        int k = 0;
        for (int s = 0; s < n; s++) {
            if (Double.isFinite(preds[s]) && Double.isFinite(targets[s])) {
                p[k] = preds[s];
                t[k] = targets[s];
                k++;
            }
        }

        double pearson  = pearsonCorrelation(p, t);
        double spearman = spearmanCorrelation(p, t);
        double rmse     = rootMeanSquaredError(p, t);

        double[] tRange = minMax(t);
        double[] pRange = minMax(p);

        System.out.println();
        System.out.printf(Locale.US, "Evaluated MLP on %d proposed trees (Java)%n", p.length);
        System.out.printf(Locale.US, "  Pearson  (pred vs target) = %+.4f%n", pearson);
        System.out.printf(Locale.US, "  Spearman (pred vs target) = %+.4f%n", spearman);
        System.out.printf(Locale.US, "  RMSE                       = %.4f%n", rmse);
        System.out.printf(Locale.US, "  target  range = [%.3f, %.3f]%n", tRange[0], tRange[1]);
        System.out.printf(Locale.US, "  pred    range = [%.3f, %.3f]%n", pRange[0], pRange[1]);

        if (cli.outPath != null) {
            Path out = Paths.get(cli.outPath).toAbsolutePath();
            if (out.getParent() != null) Files.createDirectories(out.getParent());
            try (BufferedWriter w = Files.newBufferedWriter(out)) {
                w.write("target,prediction,logDet\n");
                for (int s = 0; s < n; s++) {
                    w.write(String.format(Locale.US, "%.8g,%.8g,%.8g%n",
                        targets[s], preds[s], samples.get(s).logDeterminant));
                }
            }
            System.out.println("Per-row CSV written to " + out);
        }
    }

    /**
     * Unpack a row-major-packed lower-triangular matrix back to n×n form.
     * Packing (matches {@code LearnedWeightsTreeTrainer.packLowerTriangular}):
     * for each row, columns 0..row inclusive; entries above the diagonal are 0.
     */
    private static double[][] unpackLowerTriangular(double[] packed, int n) {
        if (packed == null) {
            throw new IllegalArgumentException(
                "rootFreeCovarianceCholeskyLower is null — proposals.tsv must be V4 schema for --use-cholesky.");
        }
        if (packed.length != n * (n + 1) / 2) {
            throw new IllegalArgumentException(
                "Packed Cholesky has length " + packed.length
                + " but expected " + (n * (n + 1) / 2) + " for n=" + n);
        }
        double[][] L = new double[n][n];
        int idx = 0;
        for (int row = 0; row < n; row++) {
            for (int col = 0; col <= row; col++) {
                L[row][col] = packed[idx++];
            }
        }
        return L;
    }

    /**
     * qf[d] = E[:,d]^T P E[:,d].  Matches both the Python evaluator and the
     * Java pruning runtime (modulo numerical equivalence between matrix
     * P-form and Cholesky L-form).  E is row-major: E[i*dim + d].
     */
    private static double[] computeQuadraticForms(double[][] P, double[] E, int nTaxa, int dim) {
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
        return qf;
    }

    // ------------------------------------------------------------------
    // Metrics
    // ------------------------------------------------------------------

    private static double pearsonCorrelation(double[] x, double[] y) {
        int n = x.length;
        double xm = 0, ym = 0;
        for (int i = 0; i < n; i++) { xm += x[i]; ym += y[i]; }
        xm /= n; ym /= n;
        double sxx = 0, syy = 0, sxy = 0;
        for (int i = 0; i < n; i++) {
            double dx = x[i] - xm, dy = y[i] - ym;
            sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
        }
        double denom = Math.sqrt(sxx * syy);
        return denom > 0 ? sxy / denom : Double.NaN;
    }

    private static double spearmanCorrelation(double[] x, double[] y) {
        return pearsonCorrelation(averageRanks(x), averageRanks(y));
    }

    /** Average-rank transform with tie handling. */
    private static double[] averageRanks(double[] v) {
        int n = v.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(v[a], v[b]));
        double[] ranks = new double[n];
        int i = 0;
        while (i < n) {
            int j = i;
            while (j + 1 < n && v[idx[j + 1]] == v[idx[i]]) j++;
            double avg = 0.5 * (i + j) + 1;  // 1-based average rank
            for (int k = i; k <= j; k++) ranks[idx[k]] = avg;
            i = j + 1;
        }
        return ranks;
    }

    private static double rootMeanSquaredError(double[] x, double[] y) {
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            double d = x[i] - y[i];
            s += d * d;
        }
        return Math.sqrt(s / x.length);
    }

    private static double[] minMax(double[] v) {
        double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
        for (double x : v) { if (x < lo) lo = x; if (x > hi) hi = x; }
        return new double[]{lo, hi};
    }

    // ------------------------------------------------------------------
    // CLI plumbing
    // ------------------------------------------------------------------

    private static final class Args {
        String surrogatePath;
        String proposalsPath;
        String outPath;
        boolean useCholesky;
        boolean showHelp;

        static Args parse(String[] argv) {
            Args a = new Args();
            for (int i = 0; i < argv.length; i++) {
                String arg = argv[i];
                switch (arg) {
                    case "--surrogate"    -> a.surrogatePath = argv[++i];
                    case "--proposals"    -> a.proposalsPath = argv[++i];
                    case "--out"          -> a.outPath       = argv[++i];
                    case "--use-cholesky" -> a.useCholesky   = true;
                    case "-h", "--help"   -> a.showHelp = true;
                    default -> throw new IllegalArgumentException("Unknown argument: " + arg);
                }
            }
            return a;
        }
    }

    private static void printUsage() {
        System.out.println("""
            Usage: EvaluatePretrainedMLPCli --surrogate <weights.json> \\
                                             --proposals <proposals.tsv> \\
                                             [--out <per-row.csv>] \\
                                             [--use-cholesky]

            Loads a surrogate JSON written by surrogate_training/train_embeddings.py
            and a proposals TSV (V4 schema) written by ProposalExporter, then scores
            every row through the Java EmbeddingMLPSurrogate using the same
            predictFromQfAndLogDet call the runtime uses in getTargetWeights*.

            Reports Pearson / Spearman / RMSE versus the true target log-likelihood
            for direct comparison with surrogate_training/evaluate_pretrained_mlp.py.

            --use-cholesky  Compute qf via forward-substitution against the stored
                            Cholesky factor L (qf[d] = ||L^-1 E[:,d]||^2) instead of
                            E^T P E.  This matches the runtime pruning path.  If
                            results disagree with the default (P-form) path, then
                            P and L stored in proposals.tsv are inconsistent — a
                            bug in ProposalExporter / buildTrainingSample.
            """);
    }
}
