package targetedbeast.edgeweights.learnedweights;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import beast.base.core.Log;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
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
    private String currentOperatorTag;

    // ------------------------------------------------------------------
    // Fused axis-reweight+MLP surrogate (AxisReweightedFusedV1), optional
    // second scoring path -- see setFusedSurrogate().
    // ------------------------------------------------------------------
    private AxisReweightedFusedSurrogate fusedSurrogate;
    private LearnedWeightsTreeTrainer fusedTreeTrainer;

    /** Cross-check the incremental scale statistics against a materialized candidate tree. */
    private static final boolean verifyFusedScaleStatistics =
            Boolean.getBoolean("targetedbeast.surrogate.fusedscale.verify");
    private static final AtomicInteger verifiedCandidates = new AtomicInteger(0);

    /** Wire the loaded surrogate.  Called from LearnedWeights after JSON load. */
    public void setSurrogate(EmbeddingMLPSurrogate surrogate) {
        this.surrogate = surrogate;
    }

    /**
     * Wire the fused axis-reweight+MLP surrogate and a {@link LearnedWeightsTreeTrainer}
     * used only for {@link LearnedWeightsTreeTrainer#computeTreeScaleStatistics}
     * (rootHeight/totalBranchLength/totalDiffusionLength of the candidate tree --
     * see {@link #computeScaledProposedTreeLogLikelihood}). When set, this path is
     * used instead of {@link #surrogate} in computeScaledProposedTreeLogLikelihood;
     * when null (default), behavior is unchanged.
     */
    public void setFusedSurrogate(AxisReweightedFusedSurrogate fusedSurrogate, LearnedWeightsTreeTrainer treeTrainer) {
        this.fusedSurrogate = fusedSurrogate;
        this.fusedTreeTrainer = treeTrainer;
    }

    public AxisReweightedFusedSurrogate getFusedSurrogate() {
        return fusedSurrogate;
    }

    /** Set by LearnedWeights.setCurrentOperatorTag, itself called by the operator right
     *  before it invokes getTargetWeights*; read by beginGroup() to tag the ProposalExporter
     *  row with which operator produced the group. */
    public void setCurrentOperatorTag(String tag) {
        this.currentOperatorTag = tag;
    }

    public EmbeddingMLPSurrogate getSurrogate() {
        return surrogate;
    }

    /** Wire a proposal exporter (system-property gated; may be null). */
    public void setProposalExporter(ProposalExporter exporter) {
        this.proposalExporter = exporter;
    }

    @Override
    protected double[] computeProposalLogScores(Node fromNode, GaussianMessage fromInside,
            List<Integer> toNodeNrs, double[] resolvedHeights) {
        // One call here == one competing-candidate set (fromNode against its toNodeNrs
        // batch); mark the group boundary before any of its proposals get exported.
        if (proposalExporter != null) {
            proposalExporter.beginGroup(currentOperatorTag);
        }
        return super.computeProposalLogScores(fromNode, fromInside, toNodeNrs, resolvedHeights);
    }

    @Override
    protected double computeScaledProposedTreeLogLikelihood(
            Node fromNode, Node toNode, GaussianMessage fromInside,
            GaussianMessage toInsideWithout, double toHeight) {

        assert surrogate != null || fusedSurrogate != null;

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

        double pred;
        if (fusedSurrogate != null) {
            pred = computeFusedPrediction(fromNode, toNode, toHeight, qf, logDet);
        } else {
            pred = surrogate.predictFromQfAndLogDet(qf, logDet);
        }

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

    /**
     * Scores one proposal with {@link #fusedSurrogate}. Needs rootHeight/
     * totalBranchLength/totalDiffusionLength of the *proposed* tree, which
     * message-passing does not materialize.
     *
     * <p>All three are sums over branches, and the SPR the candidate represents
     * changes exactly four of them, so they are obtained as a delta on the live
     * tree's own statistics rather than by materializing the candidate -- see
     * {@link #proposedTreeScaleStatistics}. The straightforward version (deep
     * copy the live tree via {@link ProposalExporter#buildCandidateTree}, then
     * rescan it) cost ~40us per scored candidate, which dominated the runtime
     * of every SPR operator; set {@code targetedbeast.surrogate.fusedscale.verify}
     * to run both and cross-check them.
     *
     * <p>{@code qf}/{@code logDet} are reused as-is from the caller (the same
     * root-free message-passing values used by the {@link #surrogate} path)
     * for both the contribution term and the root_free_log_det scalar -- see
     * {@link AxisReweightedFusedSurrogate#predict(double[], double, double, double, double)}.
     */
    private double computeFusedPrediction(Node fromNode, Node toNode, double toHeight,
                                           double[] qf, double logDet) {
        Tree liveTree = (Tree) treeInput.get();
        LearnedWeightsTreeTrainer.TreeScaleStatistics stats =
                proposedTreeScaleStatistics(liveTree, fromNode, toNode, toHeight);
        if (stats == null) {
            // SPR pre-conditions failed / invalid heights -- fall back to the plain
            // surrogate if available, else propagate a very poor score so this
            // candidate is effectively never chosen.
            return surrogate != null ? surrogate.predictFromQfAndLogDet(qf, logDet) : Double.NEGATIVE_INFINITY;
        }
        if (verifyFusedScaleStatistics) {
            verifyAgainstMaterializedCandidate(liveTree, fromNode, toNode, toHeight, stats);
        }
        return fusedSurrogate.predict(qf, logDet, stats.rootHeight(), stats.totalBranchLength(), stats.totalDiffusionLength());
    }

    /**
     * Scale statistics of the tree that {@link ProposalExporter#buildCandidateTree} would
     * build, computed as a delta on {@code liveTree} instead of by building it. Returns
     * null on exactly the inputs for which that method returns null.
     *
     * <p>The move detaches {@code fromNode}'s parent {@code p} (regrafting {@code p}'s
     * other child {@code CiP} onto {@code p}'s parent {@code PiP}) and reinserts it above
     * {@code toNode} at {@code toHeight}. Only the branches above {@code p}, {@code CiP},
     * {@code fromNode} and {@code toNode} change length; the root is never moved, so
     * rootHeight is untouched. Branch rates are per node and node identity survives the
     * move, so each changed branch contributes {@code rate * (newLength - oldLength)} to
     * totalDiffusionLength.
     */
    private LearnedWeightsTreeTrainer.TreeScaleStatistics proposedTreeScaleStatistics(
            Tree liveTree, Node fromNode, Node toNode, double toHeight) {
        // Pre-conditions, mirroring buildCandidateTree exactly.
        Node p = fromNode.getParent();
        if (p == null || p.isRoot()) return null;
        Node PiP = p.getParent();
        if (PiP == null) return null;
        Node jP = toNode.getParent();
        if (jP == null) return null;
        if (jP == p || toNode == p || jP == fromNode) return null;
        Node CiP = otherChild(p, fromNode);
        if (CiP == null) return null;

        // Post-move height validity, mirroring buildCandidateTree's heightsValid checks on
        // the three nodes whose parent/child sets it changes. p's new parent is jP and its
        // new children are fromNode and toNode; PiP gains CiP; jP gains p.
        if (jP.getHeight() <= toHeight) return null;
        if (toHeight <= fromNode.getHeight() || toHeight <= toNode.getHeight()) return null;
        if (PiP.getHeight() <= CiP.getHeight()) return null;

        LearnedWeightsTreeTrainer.TreeScaleStatistics live =
                fusedTreeTrainer.computeTreeScaleStatistics(liveTree);

        double pHeight = p.getHeight();
        double deltaBranchLength = 0.0;
        double deltaDiffusionLength = 0.0;

        Node[] changed = {p, CiP, fromNode, toNode};
        double[] oldLength = {
            PiP.getHeight() - pHeight,
            pHeight - CiP.getHeight(),
            pHeight - fromNode.getHeight(),
            jP.getHeight() - toNode.getHeight(),
        };
        double[] newLength = {
            jP.getHeight() - toHeight,
            PiP.getHeight() - CiP.getHeight(),
            toHeight - fromNode.getHeight(),
            toHeight - toNode.getHeight(),
        };
        for (int i = 0; i < changed.length; i++) {
            double delta = newLength[i] - oldLength[i];
            deltaBranchLength += delta;
            deltaDiffusionLength += fusedTreeTrainer.branchRateFor(changed[i]) * delta;
        }

        return new LearnedWeightsTreeTrainer.TreeScaleStatistics(
                live.rootHeight(),
                live.totalBranchLength() + deltaBranchLength,
                live.totalDiffusionLength() + deltaDiffusionLength);
    }

    private static Node otherChild(Node parent, Node child) {
        if (parent.getChildCount() != 2) return null;
        Node left = parent.getChild(0);
        Node right = parent.getChild(1);
        return left == child ? right : left;
    }

    /** Cross-check {@link #proposedTreeScaleStatistics} against the materialized candidate. */
    private void verifyAgainstMaterializedCandidate(Tree liveTree, Node fromNode, Node toNode,
            double toHeight, LearnedWeightsTreeTrainer.TreeScaleStatistics fast) {
        Tree candidate = ProposalExporter.buildCandidateTree(liveTree, fromNode, toNode, toHeight);
        if (candidate == null) {
            throw new IllegalStateException("fusedscale verify: incremental path accepted a move "
                    + "buildCandidateTree rejects (from=" + fromNode.getNr() + " to=" + toNode.getNr()
                    + " toHeight=" + toHeight + ")");
        }
        LearnedWeightsTreeTrainer.TreeScaleStatistics slow = fusedTreeTrainer.computeTreeScaleStatistics(candidate);
        double tol = 1e-8 * Math.max(1.0, Math.abs(slow.totalBranchLength()));
        if (Math.abs(fast.rootHeight() - slow.rootHeight()) > tol
                || Math.abs(fast.totalBranchLength() - slow.totalBranchLength()) > tol
                || Math.abs(fast.totalDiffusionLength() - slow.totalDiffusionLength()) > tol) {
            throw new IllegalStateException("fusedscale verify: mismatch (from=" + fromNode.getNr()
                    + " to=" + toNode.getNr() + " toHeight=" + toHeight + ")"
                    + " fast=[" + fast.rootHeight() + ", " + fast.totalBranchLength() + ", "
                    + fast.totalDiffusionLength() + "]"
                    + " slow=[" + slow.rootHeight() + ", " + slow.totalBranchLength() + ", "
                    + slow.totalDiffusionLength() + "]");
        }
        int n = verifiedCandidates.incrementAndGet();
        if (Integer.bitCount(n) == 1 && n >= 1024) {
            Log.info.println("PCAWeightsMLPLikelihood: fusedscale verify OK for " + n + " candidates.");
        }
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
            if (surrogate == null) return; // diagnostic CSV only supports the plain EmbeddingMLPSurrogate path
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
