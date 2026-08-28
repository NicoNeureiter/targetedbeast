package targetedbeast.edgeweights.learnedweights;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;

import beast.base.core.Log;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;

/**
 * Exports (proposed-tree precision matrix, true target log-likelihood) pairs
 * in the V4 LearnedWeightsTrainingSamples TSV format.
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
 *   <li>appends a row to the output TSV in the same V4 format that the Python
 *       loader ({@code surrogate_training/loader.py}) consumes.</li>
 * </ol>
 *
 * <p>Activation via system properties:
 * <pre>
 *   -Dtargetedbeast.surrogate.proposalexport.tsv=/path/to/proposals.tsv
 *   -Dtargetedbeast.surrogate.proposalexport.max=5000        (default 1000)
 *   -Dtargetedbeast.surrogate.proposalexport.stride=1        (default 1; keep every N-th proposal)
 * </pre>
 *
 * <p>The resulting TSV is intentionally identical in schema to a training
 * sample file, so the existing Python pipeline can load it with
 * {@code loader.load_samples(...)} and feed the (V, target) pairs into a
 * pretrained MLP for evaluation.
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
    private final AtomicInteger callCount = new AtomicInteger(0);
    private final AtomicInteger rowCount  = new AtomicInteger(0);
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
     * Capture a single proposal.  Safe to call from inside
     * {@code computeScaledProposedTreeLogLikelihood}: works on an isolated
     * tree copy and uses the trainer's independent likelihood evaluator.
     */
    public void exportProposal(Node fromNode, Node toNode, double toHeight) {
        if (isFull()) return;
        int call = callCount.getAndIncrement();
        if (call % stride != 0) return;

        // Validate SPR pre-conditions (must mirror WeightedWideOperator).
        Node pLive = fromNode.getParent();
        if (pLive == null || pLive.isRoot()) return;
        Node PiPLive = pLive.getParent();
        if (PiPLive == null) return;
        Node jPLive = toNode.getParent();
        if (jPLive == null) return;
        if (jPLive == pLive || toNode == pLive || jPLive == fromNode) return;

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
            return;
        }
        Node[] copyNodes = copy.getNodesAsArray();
        Node fromCopy = copyNodes[fromNr];
        Node toCopy   = copyNodes[toNr];
        Node pCopy    = copyNodes[pNr];
        Node jPCopy   = copyNodes[jPNr];
        Node PiPCopy  = copyNodes[PiPNr];
        Node CiPCopy  = otherChild(pCopy, fromCopy);
        if (CiPCopy == null) return;

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
            return;
        }
        // Heights still need to satisfy parent > child everywhere we touched.
        if (!heightsValid(pCopy) || !heightsValid(jPCopy) || !heightsValid(PiPCopy)) {
            return;
        }

        // 3. True target log-likelihood (uses trainer's independent evaluator).
        double trueLL;
        try {
            trueLL = trainer.computeCurrentTreeLogLikelihood(copy);
        } catch (RuntimeException ex) {
            return;
        }
        if (!Double.isFinite(trueLL)) return;

        // 4. Build training sample (V matrix + logDet + root-free factor + ...).
        LearnedWeightsTreeTrainer.TrainingTreeSample sample;
        try {
            sample = trainer.buildTrainingSample(copy, trueLL);
        } catch (RuntimeException ex) {
            return;
        }
        if (sample.rootFreeCovarianceCholeskyLower == null) return;

        // 5. Write a V4 row.
        try {
            appendSample(sample);
        } catch (IOException e) {
            Log.warning.println("ProposalExporter: write failed: " + e.getMessage());
        }
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

    private synchronized void appendSample(LearnedWeightsTreeTrainer.TrainingTreeSample sample) throws IOException {
        if (writer == null) {
            openWriter();
        }
        if (!headerWritten) {
            writeHeader();
            headerWritten = true;
        }
        writeSampleRow(sample);
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
        writer.write("# LearnedWeightsTrainingSamplesV4");
        writer.newLine();
        writer.write("taxa");
        for (String taxon : canonicalTaxa) {
            writer.write('\t');
            writer.write(taxon);
        }
        writer.newLine();
    }

    private void writeSampleRow(LearnedWeightsTreeTrainer.TrainingTreeSample sample) throws IOException {
        StringBuilder sb = new StringBuilder(8192);
        sb.append("sample");
        sb.append('\t').append(String.format(Locale.US, "%.8g", sample.targetLogLikelihood));
        sb.append('\t').append(String.format(Locale.US, "%.8g", sample.logDeterminant));
        sb.append('\t').append(sample.logNormalizerDimension);
        sb.append('\t').append(String.format(Locale.US, "%.8g", sample.rootFreeLogDeterminant));
        sb.append('\t').append(sample.rootFreeLogNormalizerDimension);
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
