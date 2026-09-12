package targetedbeast.edgeweights.learnedweights;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import beast.base.core.Log;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import targetedbeast.weighttransform.BranchScoreTreeMetric;

/**
 * Exports (proposed-tree precision matrix, true target log-likelihood) pairs
 * in the V6 LearnedWeightsTrainingSamples TSV format.
 *
 * <p>For every SPR proposal that {@link PCAWeightsMLPLikelihood} scores during
 * MCMC, this exporter:
 * <ol>
 *   <li>makes an isolated copy of the live tree,</li>
 *   <li>applies the SPR move (detach {@code fromNode}, regraft above
 *       {@code toNode} at {@code toHeight}) on the copy,</li>
 *   <li>evaluates the true tree likelihood on the copy via the trainer's
 *       independent likelihood evaluator (no live MCMC state is touched),</li>
 *   <li>builds the same (precisionMatrix, logDet, root-free Cholesky, ...)
 *       quantities the training pipeline expects, and</li>
 *   <li>appends a row to the output TSV in the V6 format (V5 plus
 *       {@code operatorType}, {@code distanceToCurrent}, {@code isCurrent}),
 *       which {@code surrogate_training/loader.py} consumes.</li>
 * </ol>
 *
 * <p>Each group additionally gets one extra "virtual candidate" row for the
 * live tree itself ({@code isCurrent=1}, {@code distanceToCurrent=0}), written
 * by {@link #beginGroup}: this is {@code x_from}, needed to evaluate
 * expected-acceptance / expected-ESJD metrics that require both a target and
 * proposal probability for staying put, not just for each candidate move.
 *
 * <p><b>Not read by {@link LearnedWeightsTrainingSampleIO}</b> (still V1-V5
 * only) — {@code EvaluatePretrainedMLPCli}/{@code LearnedWeightsGroupedMetrics}
 * cannot read V6 proposal files. V6 is consumed only by
 * {@code surrogate_training/loader.py}.
 *
 * <p>Activation via system properties:
 * <pre>
 *   -Dtargetedbeast.surrogate.proposalexport.tsv=/path/to/proposals.tsv
 *   -Dtargetedbeast.surrogate.proposalexport.max=5000        (default 1000)
 *   -Dtargetedbeast.surrogate.proposalexport.stride=1        (default 1; keep every N-th proposal)
 * </pre>
 */
public final class ProposalExporter {

    public static final String EXPORT_PATH_PROP   = "targetedbeast.surrogate.proposalexport.tsv";
    public static final String EXPORT_MAX_PROP    = "targetedbeast.surrogate.proposalexport.max";
    public static final String EXPORT_STRIDE_PROP = "targetedbeast.surrogate.proposalexport.stride";

    private final Tree liveTree;
    private final LearnedWeightsTreeTrainer trainer;
    private final String[] canonicalTaxa;
    private final Path outputPath;
    private final int maxRows;
    private final int stride;
    private final BranchScoreTreeMetric distanceMetric = new BranchScoreTreeMetric();
    private final AtomicInteger callCount = new AtomicInteger(0);
    private final AtomicInteger rowCount  = new AtomicInteger(0);
    // Incremented once per competing-candidate set (one fromNode against its toNodeNrs batch);
    // starts at NO_GROUP so a caller that never calls beginGroup() exports ungrouped rows.
    private final AtomicLong groupCounter = new AtomicLong(LearnedWeightsTreeTrainer.TrainingTreeSample.NO_GROUP);
    private volatile String currentOperatorTag = "";
    private BufferedWriter writer;
    private boolean headerWritten = false;

    public ProposalExporter(Tree liveTree, LearnedWeightsTreeTrainer trainer,
            String[] canonicalTaxa, Path outputPath, int maxRows, int stride) {
        this.liveTree = liveTree;
        this.trainer = trainer;
        this.canonicalTaxa = canonicalTaxa;
        this.outputPath = outputPath;
        this.maxRows = maxRows;
        this.stride = Math.max(1, stride);
    }

    /**
     * Read system properties and, if a path is set, construct an exporter.
     * Returns {@code null} when no path is configured.
     */
    public static ProposalExporter createFromSystemProperties(
            Tree liveTree, LearnedWeightsTreeTrainer trainer, String[] canonicalTaxa) {
        String path = System.getProperty(EXPORT_PATH_PROP);
        if (path == null || path.isBlank()) return null;
        int max    = parseInt(System.getProperty(EXPORT_MAX_PROP), 1000);
        int stride = parseInt(System.getProperty(EXPORT_STRIDE_PROP), 1);
        Path out = Path.of(path);
        Log.info.println("ProposalExporter: writing up to " + max + " proposals to "
                + out.toAbsolutePath() + " (stride=" + stride + ")");
        return new ProposalExporter(liveTree, trainer, canonicalTaxa, out, max, stride);
    }

    public boolean isFull() {
        return rowCount.get() >= maxRows;
    }

    /**
     * Mark the start of a new competing-candidate set: call once per
     * {@code getTargetWeightsAtHeights}/{@code getTargetWeightsInteger} invocation,
     * before exporting any of its candidate proposals, so they all share one groupId.
     * Also exports x_from itself as a virtual candidate ({@code isCurrent=1}).
     *
     * @param operatorTag simple class name of the calling operator (may be null/blank if
     *                     the operator hasn't been updated to call setCurrentOperatorTag).
     */
    public void beginGroup(String operatorTag) {
        this.currentOperatorTag = operatorTag == null ? "" : operatorTag;
        groupCounter.incrementAndGet();
        if (isFull()) return;

        Tree copy;
        try {
            copy = new Tree(liveTree.getRoot().copy());
        } catch (RuntimeException ex) {
            return;
        }
        double trueLogPosterior;
        try {
            trueLogPosterior = trainer.computeCurrentTreeLogPosterior(copy);
        } catch (RuntimeException ex) {
            return;
        }
        if (!Double.isFinite(trueLogPosterior)) return;

        LearnedWeightsTreeTrainer.TrainingTreeSample sample;
        try {
            sample = trainer.buildTrainingSample(copy, trueLogPosterior, groupCounter.get());
        } catch (RuntimeException ex) {
            return;
        }
        if (sample.rootFreeCovarianceCholeskyLower == null) return;

        try {
            appendSample(sample, this.currentOperatorTag, 0.0, true);
        } catch (IOException e) {
            Log.warning.println("ProposalExporter: write failed (x_from row): " + e.getMessage());
        }
    }

    /**
     * Capture a single proposal.  Safe to call from inside
     * {@code computeScaledProposedTreeLogLikelihood}: works on an isolated
     * tree copy and uses the trainer's independent likelihood evaluator.
     */
    public void exportProposal(Node fromNode, Node toNode, double toHeight) {
        if (isFull()) return;
        int call = callCount.getAndIncrement();
        if (call % stride != 0) return;

        Tree copy = buildCandidateTree(liveTree, fromNode, toNode, toHeight);
        if (copy == null) return;

        // 3. True target log-posterior (uses trainer's independent evaluator).
        double trueLogPosterior;
        try {
            trueLogPosterior = trainer.computeCurrentTreeLogPosterior(copy);
        } catch (RuntimeException ex) {
            return;
        }
        if (!Double.isFinite(trueLogPosterior)) return;

        // 4. Build training sample (V matrix + logDet + root-free factor + ...).
        LearnedWeightsTreeTrainer.TrainingTreeSample sample;
        try {
            sample = trainer.buildTrainingSample(copy, trueLogPosterior, groupCounter.get());
        } catch (RuntimeException ex) {
            return;
        }
        if (sample.rootFreeCovarianceCholeskyLower == null) return;

        // 5. Distance from the live (pre-move) tree to this candidate, for expected-ESJD.
        double distance;
        try {
            distance = distanceMetric.distance(liveTree, copy);
        } catch (RuntimeException ex) {
            distance = Double.NaN;
        }

        // 6. Write a V6 row.
        try {
            appendSample(sample, this.currentOperatorTag, distance, false);
        } catch (IOException e) {
            Log.warning.println("ProposalExporter: write failed: " + e.getMessage());
        }
    }

    /**
     * Builds an isolated candidate tree for the SPR move (detach {@code fromNode},
     * regraft above {@code toNode} at {@code toHeight}), starting from a deep copy
     * of {@code liveTree}. Returns {@code null} if the move's SPR pre-conditions
     * don't hold, the copy/mutation fails, or the resulting heights are invalid.
     *
     * <p>Shared by {@link #exportProposal} and, for scoring with
     * {@link AxisReweightedFusedSurrogate}, {@link PCAWeightsMLPLikelihood} --
     * both need exactly this "materialize the proposed tree" step.
     */
    public static Tree buildCandidateTree(Tree liveTree, Node fromNode, Node toNode, double toHeight) {
        // Validate SPR pre-conditions (must mirror WeightedWideOperator).
        Node pLive = fromNode.getParent();
        if (pLive == null || pLive.isRoot()) return null;
        Node PiPLive = pLive.getParent();
        if (PiPLive == null) return null;
        Node jPLive = toNode.getParent();
        if (jPLive == null) return null;
        if (jPLive == pLive || toNode == pLive || jPLive == fromNode) return null;

        int fromNr = fromNode.getNr();
        int toNr   = toNode.getNr();
        int pNr    = pLive.getNr();
        int jPNr   = jPLive.getNr();
        int PiPNr  = PiPLive.getNr();

        // 1. Independent deep copy of the live tree.
        Tree copy;
        try {
            copy = new Tree(liveTree.getRoot().copy());
        } catch (RuntimeException ex) {
            return null;
        }
        Node[] copyNodes = copy.getNodesAsArray();
        Node fromCopy = copyNodes[fromNr];
        Node toCopy   = copyNodes[toNr];
        Node pCopy    = copyNodes[pNr];
        Node jPCopy   = copyNodes[jPNr];
        Node PiPCopy  = copyNodes[PiPNr];
        Node CiPCopy  = otherChild(pCopy, fromCopy);
        if (CiPCopy == null) return null;

        // 2. Apply the SPR on the copy.
        try {
            PiPCopy.removeChild(pCopy);
            PiPCopy.addChild(CiPCopy);
            pCopy.removeChild(CiPCopy);
            pCopy.addChild(toCopy);
            jPCopy.removeChild(toCopy);
            jPCopy.addChild(pCopy);
            pCopy.setHeight(toHeight);
        } catch (RuntimeException ex) {
            return null;
        }
        // Heights still need to satisfy parent > child everywhere we touched.
        if (!heightsValid(pCopy) || !heightsValid(jPCopy) || !heightsValid(PiPCopy)) {
            return null;
        }
        return copy;
    }

    private static Node otherChild(Node parent, Node child) {
        if (parent.getChildCount() != 2) return null;
        Node left = parent.getChild(0);
        Node right = parent.getChild(1);
        return left == child ? right : left;
    }

    private static boolean heightsValid(Node n) {
        if (n == null) return true;
        Node p = n.getParent();
        if (p != null && p.getHeight() <= n.getHeight()) return false;
        for (int i = 0; i < n.getChildCount(); i++) {
            Node c = n.getChild(i);
            if (n.getHeight() <= c.getHeight()) return false;
        }
        return true;
    }

    private synchronized void appendSample(LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            String operatorType, double distanceToCurrent, boolean isCurrent) throws IOException {
        if (writer == null) {
            openWriter();
        }
        if (!headerWritten) {
            writeHeader();
            headerWritten = true;
        }
        writeSampleRow(sample, operatorType, distanceToCurrent, isCurrent);
        int n = rowCount.incrementAndGet();
        if ((n & 0xFF) == 0) writer.flush();
        if (n == maxRows) {
            writer.flush();
            Log.info.println("ProposalExporter: reached cap of " + maxRows + " rows.");
        }
    }

    private void openWriter() throws IOException {
        if (outputPath.getParent() != null) {
            Files.createDirectories(outputPath.getParent());
        }
        writer = Files.newBufferedWriter(outputPath);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try { synchronized (this) { if (writer != null) { writer.flush(); writer.close(); } } }
            catch (IOException ignored) {}
        }));
    }

    private void writeHeader() throws IOException {
        writer.write("# LearnedWeightsTrainingSamplesV6");
        writer.newLine();
        writer.write("taxa");
        for (String taxon : canonicalTaxa) {
            writer.write('\t');
            writer.write(taxon);
        }
        writer.newLine();
    }

    private void writeSampleRow(LearnedWeightsTreeTrainer.TrainingTreeSample sample,
            String operatorType, double distanceToCurrent, boolean isCurrent) throws IOException {
        StringBuilder sb = new StringBuilder(8192);
        sb.append("sample");
        sb.append('\t').append(String.format(Locale.US, "%.8g", sample.targetLogLikelihood));
        sb.append('\t').append(String.format(Locale.US, "%.8g", sample.logDeterminant));
        sb.append('\t').append(sample.logNormalizerDimension);
        sb.append('\t').append(String.format(Locale.US, "%.8g", sample.rootFreeLogDeterminant));
        sb.append('\t').append(sample.rootFreeLogNormalizerDimension);
        sb.append('\t').append(sample.groupId);
        sb.append('\t').append(String.format(Locale.US, "%.8g", sample.rootHeight));
        sb.append('\t').append(String.format(Locale.US, "%.8g", sample.totalBranchLength));
        sb.append('\t').append(String.format(Locale.US, "%.8g", sample.totalDiffusionLength));
        sb.append('\t').append(operatorType == null || operatorType.isBlank() ? "unknown" : operatorType);
        sb.append('\t').append(String.format(Locale.US, "%.8g", distanceToCurrent));
        sb.append('\t').append(isCurrent ? 1 : 0);
        for (double[] row : sample.precisionMatrix) {
            for (double v : row) {
                sb.append('\t').append(String.format(Locale.US, "%.8g", v));
            }
        }
        for (double v : sample.rootFreeCovarianceCholeskyLower) {
            sb.append('\t').append(String.format(Locale.US, "%.8g", v));
        }
        sb.append('\n');
        writer.write(sb.toString());
    }

    private static int parseInt(String s, int dflt) {
        if (s == null || s.isBlank()) return dflt;
        try { return Integer.parseInt(s.trim()); }
        catch (NumberFormatException e) { return dflt; }
    }
}
