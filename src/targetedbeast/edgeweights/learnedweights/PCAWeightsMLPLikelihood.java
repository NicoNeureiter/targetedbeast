package targetedbeast.edgeweights.learnedweights;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;

import beast.base.core.Log;
import beast.base.evolution.tree.Node;
import targetedbeast.edgeweights.PCAWeightsBrownianFullLikelihood;

/**
 * MLP-surrogate variant of {@link PCAWeightsBrownianFullLikelihood}.
 *
 * <p>Inherits the full Brownian-motion message-passing machinery from the
 * parent (inside / inside-without / outside caches, per-axis contribution
 * decomposition, normalisation pipeline, edge-weight computation) and only
 * replaces the final per-axis affine summation with a learned MLP forward
 * pass.  The override is a single method.
 *
 * <p>For each proposal, the parent's
 * {@link PCAWeightsBrownianFullLikelihood#computeProposedTreeLogLikelihoodContributionParts}
 * accumulates per-axis {@code (baseContribution, quadraticContribution)} where
 * <pre>
 *   sum over message-combines  of  baseContribution[d]      = −½·logDet(V_axis_d)
 *   sum over message-combines  of  quadraticContribution[d] = −½·qf[d]
 * </pre>
 * Under the training-time setting (single Brownian rate, improper root prior)
 * {@code V_axis_d = V_topology} is shared across axes, so {@code baseContribution[d]}
 * is the same value for every {@code d} and equals {@code −½·logDet}.  Rescaling
 * by {@code −2} gives exactly the {@code (qf, logDet)} pair the MLP was trained on.
 */
public class PCAWeightsMLPLikelihood extends PCAWeightsBrownianFullLikelihood {

    // ------------------------------------------------------------------
    // Diagnostic logger (off by default).
    //
    // Enable by setting the system property
    //   -Dtargetedbeast.surrogate.diagnostic.csv=/tmp/mlp_inputs.csv
    // optionally with
    //   -Dtargetedbeast.surrogate.diagnostic.maxCalls=2000   (default 1000)
    //
    // For each of the first N calls to computeScaledProposedTreeLogLikelihood
    // a CSV row is written containing the raw qf[d], the normalised qf_norm[d]
    // (using the surrogate's frozen qf_mean / qf_std), the raw and normalised
    // logDet, and the MLP prediction.  Compare the distribution of qf_norm and
    // ld_norm against [-3, 3] — anything systematically outside that range is
    // an out-of-distribution input the MLP was not trained to handle.
    // ------------------------------------------------------------------

    private static final String DIAG_CSV_PROP   = "targetedbeast.surrogate.diagnostic.csv";
    private static final String DIAG_MAX_PROP   = "targetedbeast.surrogate.diagnostic.maxCalls";

    private static volatile BufferedWriter diagWriter;
    private static volatile boolean diagInitTried = false;
    private static int diagMaxCalls = -1;
    private static final AtomicInteger diagCalls = new AtomicInteger(0);

    private EmbeddingMLPSurrogate surrogate;
    private ProposalExporter proposalExporter;

    /** Wire the loaded surrogate.  Called from LearnedWeights after JSON load. */
    public void setSurrogate(EmbeddingMLPSurrogate surrogate) {
        this.surrogate = surrogate;
    }

    public EmbeddingMLPSurrogate getSurrogate() {
        return surrogate;
    }

    /** Wire a proposal exporter (system-property gated; may be null). */
    public void setProposalExporter(ProposalExporter exporter) {
        this.proposalExporter = exporter;
    }

    @Override
    protected double computeScaledProposedTreeLogLikelihood(
            Node fromNode, Node toNode, GaussianMessage fromInside,
            GaussianMessage toInsideWithout, double toHeight) {

        assert surrogate != null;
        
        // if (surrogate == null) {
        //     return super.computeScaledProposedTreeLogLikelihood(
        //         fromNode, toNode, fromInside, toInsideWithout, toHeight);
        // }

        int dim = getDim();
        double[] base = new double[dim];
        double[] quad = new double[dim];
        computeProposedTreeLogLikelihoodContributionParts(
            fromNode, toNode, fromInside, toInsideWithout, toHeight, base, quad);

        // logDet = −2·baseContribution[d]  (constant across d under single-rate model)
        // qf[d]  = −2·quadraticContribution[d]
        double logDet = -2.0 * base[0];
        double[] qf = new double[dim];
        for (int d = 0; d < dim; d++) {
            qf[d] = -2.0 * quad[d];
        }
        double pred = surrogate.predictFromQfAndLogDet(qf, logDet);

        if (proposalExporter != null && !proposalExporter.isFull()) {
            proposalExporter.exportProposal(fromNode, toNode, toHeight);
        }

        if (diagWriter != null && diagCalls.get() < diagMaxCalls) {
            // Also score this proposal with the parent's affine path so the
            // analysis script can compare MLP vs affine rankings per batch.
            double affinePred = super.computeScaledProposedTreeLogLikelihood(
                fromNode, toNode, fromInside, toInsideWithout, toHeight);
            maybeLogDiagnostic(fromNode.getNr(), toNode.getNr(), qf, logDet, pred, affinePred);
        } else if (!diagInitTried) {
            // first call: open the file lazily, then log if it really turned on
            initDiagnostic();
            if (diagWriter != null) {
                double affinePred = super.computeScaledProposedTreeLogLikelihood(
                    fromNode, toNode, fromInside, toInsideWithout, toHeight);
                maybeLogDiagnostic(fromNode.getNr(), toNode.getNr(), qf, logDet, pred, affinePred);
            }
        }
        return pred;
    }

    // ------------------------------------------------------------------
    // Diagnostic CSV logger (system-property gated; off by default)
    // ------------------------------------------------------------------



    private void maybeLogDiagnostic(int fromNodeNr, int toNodeNr,
                                    double[] qf, double logDet,
                                    double pred, double affinePred) {
        BufferedWriter w = diagWriter;
        if (w == null) return;
        int callIdx = diagCalls.getAndIncrement();
        if (callIdx >= diagMaxCalls || callIdx < 100000 || callIdx % 10 > 0) return;
        try {
            double[] qfMean = surrogate.qfMean();
            double[] qfStd  = surrogate.qfStd();
            double   ldMean = surrogate.ldMean();
            double   ldStd  = surrogate.ldStd();
            StringBuilder sb = new StringBuilder(256);
            sb.append(callIdx).append(',').append(fromNodeNr).append(',').append(toNodeNr);
            for (double v : qf)               sb.append(',').append(v);
            for (int d = 0; d < qf.length; d++) {
                double z = (qf[d] - qfMean[d]) / qfStd[d];
                sb.append(',').append(z);
            }
            sb.append(',').append(logDet);
            sb.append(',').append((logDet - ldMean) / ldStd);
            sb.append(',').append(pred);
            sb.append(',').append(affinePred);
            sb.append('\n');
            synchronized (PCAWeightsMLPLikelihood.class) {
                w.write(sb.toString());
                if ((callIdx & 0xFF) == 0) w.flush();
                if (callIdx + 10 == diagMaxCalls) {
                    w.flush();
                    Log.info.println("PCAWeightsMLPLikelihood: diagnostic CSV reached " + diagMaxCalls + " rows.");
                }
            }
        } catch (IOException e) {
            Log.warning.println("PCAWeightsMLPLikelihood diagnostic write failed: " + e.getMessage());
        }
    }

    private void initDiagnostic() {
        synchronized (PCAWeightsMLPLikelihood.class) {
            if (diagInitTried) return;
            diagInitTried = true;
            String path = System.getProperty(DIAG_CSV_PROP);
            if (path == null || path.isBlank()) return;
            String maxStr = System.getProperty(DIAG_MAX_PROP);
            diagMaxCalls = (maxStr == null || maxStr.isBlank()) ? 1000 : Integer.parseInt(maxStr.trim());
            try {
                BufferedWriter w = Files.newBufferedWriter(Path.of(path));
                int dim = surrogate.dim();
                StringBuilder hdr = new StringBuilder();
                hdr.append("call,fromNode,toNode");
                for (int d = 0; d < dim; d++) hdr.append(",qf_").append(d);
                for (int d = 0; d < dim; d++) hdr.append(",qf_norm_").append(d);
                hdr.append(",logDet,ld_norm,prediction,affine\n");
                w.write(hdr.toString());
                diagWriter = w;
                Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                    try { w.flush(); w.close(); } catch (IOException ignored) {}
                }));
                Log.info.println("PCAWeightsMLPLikelihood: writing diagnostic CSV to " + path
                    + " (maxCalls=" + diagMaxCalls + ")");
            } catch (IOException e) {
                Log.warning.println("PCAWeightsMLPLikelihood: failed to open diagnostic CSV " + path + ": " + e.getMessage());
            }
        }
    }
}
