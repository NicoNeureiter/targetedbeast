package targetedbeast.edgeweights.learnedweights;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import targetedbeast.edgeweights.PCAWeightsBrownianFullLikelihood;
import targetedbeast.edgeweights.PCAWeightsBrownianFullLikelihood.GaussianMessage;

/**
 * Proposal-path parity check for the per-axis (base, quad) accumulation.
 *
 * <p>Loads the same fixture format as {@link TreeContributionsParityCheck}
 * (Newick + leaf embeddings + dim + n_taxa), then for every internal node N
 * picks the no-op SPR proposal {@code (fromNode = left(N), toNode = right(N),
 * h = N.height)} (and the symmetric one).  A no-op SPR detaches and re-attaches
 * a leaf at its existing position, so the proposed tree equals the current
 * tree and the per-axis contributions returned by
 * {@link PCAWeightsBrownianFullLikelihood#computeProposedTreeLogLikelihoodContributionParts}
 * must equal those returned by
 * {@link PCAWeightsBrownianFullLikelihood#computeFullTreeContributionParts}.
 *
 * <p>Before this fix the proposal method only accumulated per-axis contributions
 * from the 1–3 message-combines it does explicitly, missing the cached
 * contributions from below {@code fromNode} / {@code toNode} / sibling /
 * outside-grandparent.  With the cached arrays seeded properly, the proposal
 * output should match the whole-tree output to machine precision.
 *
 * <p>Exits with status 0 if every no-op SPR proposal matches the full-tree
 * result within tolerance; status 1 otherwise.
 */
public final class ProposalContributionsParityCheck {

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("Usage: ProposalContributionsParityCheck <fixture.json> [tolerance]");
            System.exit(2);
        }
        Path fixturePath = Path.of(args[0]);
        double tolerance = args.length >= 2 ? Double.parseDouble(args[1]) : 1e-9;

        Fixture fx = Fixture.read(fixturePath);

        List<Sequence> seqs = new ArrayList<>(fx.taxa.size());
        for (String t : fx.taxa) {
            Sequence s = new Sequence();
            s.initByName("taxon", t, "value", "A");
            seqs.add(s);
        }
        Alignment aln = new Alignment();
        aln.initByName("sequence", seqs, "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName(
            "newick",           fx.newick,
            "IsLabelledNewick", true,
            "adjustTipHeights", false,
            "taxa",             aln);

        PCAWeightsBrownianFullLikelihood model = new PCAWeightsBrownianFullLikelihood();
        model.initByName(
            "tree",              tree,
            "data",              aln,
            "dimension",         fx.dim,
            "value",             fx.value,
            "rootPriorVariance", Double.POSITIVE_INFINITY);

        // Force inside recursion + a fresh call to getApproximateTreeLogLikelihood
        // so the cached arrays are populated.
        model.getApproximateTreeLogLikelihood();

        double[] baseFull = new double[fx.dim];
        double[] quadFull = new double[fx.dim];
        model.computeFullTreeContributionParts(baseFull, quadFull);

        System.out.println("# Proposal-path parity test (no-op SPR vs full-tree)");
        System.out.println("# Fixture: " + fixturePath);
        System.out.println("# n_taxa=" + fx.taxa.size() + " dim=" + fx.dim);
        System.out.println("# Tolerance: " + tolerance);
        System.out.println();

        // For each non-root internal node N, try the no-op SPR (left, right, N.height)
        // and the symmetric (right, left, N.height).
        Node root = tree.getRoot();
        Node[] nodes = tree.getNodesAsArray();
        int totalProposals = 0;
        int failures = 0;
        double worstBase = 0.0;
        double worstQuad = 0.0;

        for (Node N : nodes) {
            if (N.isLeaf() || N == root || N.getLeft() == null || N.getRight() == null) continue;
            Node[][] orderings = { { N.getLeft(), N.getRight() }, { N.getRight(), N.getLeft() } };
            for (Node[] pair : orderings) {
                Node fromNode = pair[0];
                Node toNode   = pair[1];
                // Use ensureWithoutNodeMessages + getInsideMessage + getInsideWithoutMessage
                // indirectly via getTargetWeights* would do this; we replicate the
                // setup manually here.
                model.updateByOperatorWithoutNode(fromNode.getNr(), Collections.emptyList());

                GaussianMessage fromInside       = invokeGetInsideMessage(model, fromNode.getNr());
                GaussianMessage toInsideWithout  = invokeGetInsideWithoutMessage(model, toNode.getNr());

                double h = N.getHeight();   // no-op: attachment height equals existing parent's height
                double[] basePrp = new double[fx.dim];
                double[] quadPrp = new double[fx.dim];
                invokeComputeProposedContributionParts(model,
                    fromNode, toNode, fromInside, toInsideWithout, h, basePrp, quadPrp);

                double dB = 0, dQ = 0;
                for (int d = 0; d < fx.dim; d++) {
                    dB = Math.max(dB, Math.abs(basePrp[d] - baseFull[d]));
                    dQ = Math.max(dQ, Math.abs(quadPrp[d] - quadFull[d]));
                }
                totalProposals++;
                worstBase = Math.max(worstBase, dB);
                worstQuad = Math.max(worstQuad, dQ);
                boolean ok = dB <= tolerance && dQ <= tolerance;
                if (!ok) failures++;
                System.out.printf("  N=%d  from=%d  to=%d  |Δ|base=%.3e  |Δ|quad=%.3e  %s%n",
                    N.getNr(), fromNode.getNr(), toNode.getNr(), dB, dQ, ok ? "OK" : "FAIL");
            }
        }

        System.out.println();
        System.out.printf("Proposals tested: %d, failures: %d%n", totalProposals, failures);
        System.out.printf("Worst |Δ|base = %.3e   worst |Δ|quad = %.3e%n", worstBase, worstQuad);
        System.out.println();
        boolean pass = failures == 0;
        System.out.println(pass
            ? "PASS — proposal-path (base, quad) match whole-tree to within " + tolerance
            : "FAIL — at least one proposal diverges from the whole-tree reference");
        System.exit(pass ? 0 : 1);
    }

    // ------------------------------------------------------------------
    // Reflection helpers: getInsideMessage and getInsideWithoutMessage are
    // package-private in PCAWeightsBrownianFullLikelihood.  We invoke them
    // via reflection only inside this test.
    // ------------------------------------------------------------------

    private static GaussianMessage invokeGetInsideMessage(PCAWeightsBrownianFullLikelihood model, int nodeNr) {
        try {
            java.lang.reflect.Method m = PCAWeightsBrownianFullLikelihood.class
                    .getDeclaredMethod("getInsideMessage", int.class);
            m.setAccessible(true);
            return (GaussianMessage) m.invoke(model, nodeNr);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    private static GaussianMessage invokeGetInsideWithoutMessage(PCAWeightsBrownianFullLikelihood model, int nodeNr) {
        try {
            java.lang.reflect.Method m = PCAWeightsBrownianFullLikelihood.class
                    .getDeclaredMethod("getInsideWithoutMessage", int.class);
            m.setAccessible(true);
            return (GaussianMessage) m.invoke(model, nodeNr);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    private static void invokeComputeProposedContributionParts(
            PCAWeightsBrownianFullLikelihood model,
            Node fromNode, Node toNode,
            GaussianMessage fromInside, GaussianMessage toInsideWithout,
            double toHeight, double[] base, double[] quad) {
        try {
            java.lang.reflect.Method m = PCAWeightsBrownianFullLikelihood.class.getDeclaredMethod(
                "computeProposedTreeLogLikelihoodContributionParts",
                Node.class, Node.class, GaussianMessage.class, GaussianMessage.class,
                double.class, double[].class, double[].class);
            m.setAccessible(true);
            m.invoke(model, fromNode, toNode, fromInside, toInsideWithout, toHeight, base, quad);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    // ------------------------------------------------------------------
    // Fixture parser (reuses the JSON schema of TreeContributionsParityCheck)
    // ------------------------------------------------------------------

    private static final class Fixture {
        final String newick;
        final List<String> taxa;
        final int dim;
        final String value;
        Fixture(String newick, List<String> taxa, int dim, String value) {
            this.newick = newick; this.taxa = taxa; this.dim = dim; this.value = value;
        }
        static Fixture read(Path path) throws IOException {
            return new FixtureParser(Files.readString(path)).parse();
        }
    }

    private static final class FixtureParser {
        private final String src; private int p;
        FixtureParser(String src) { this.src = src; this.p = 0; }
        Fixture parse() {
            ws(); expect('{');
            String newick = null, value = null;
            List<String> taxa = null;
            int dim = -1;
            while (true) {
                ws();
                if (peek() == '}') { p++; break; }
                String key = readString(); ws(); expect(':');
                switch (key) {
                    case "newick":  newick = readString();      break;
                    case "value":   value  = readString();      break;
                    case "taxa":    taxa   = readStringArray(); break;
                    case "dim":     dim    = (int) readNumber(); break;
                    default:        skipValue();
                }
                ws();
                if (peek() == ',') { p++; continue; }
                if (peek() == '}') { p++; break; }
                throw new RuntimeException("FixtureParser: expected , or } at offset " + p);
            }
            if (newick == null || taxa == null || value == null || dim < 0) {
                throw new RuntimeException("FixtureParser: incomplete fixture");
            }
            return new Fixture(newick, taxa, dim, value);
        }
        List<String> readStringArray() {
            ws(); expect('['); List<String> out = new ArrayList<>(); ws();
            if (peek() == ']') { p++; return out; }
            while (true) {
                out.add(readString()); ws();
                if (peek() == ',') { p++; ws(); continue; }
                expect(']'); return out;
            }
        }
        double readNumber() {
            ws(); int s = p;
            if (peek() == '-' || peek() == '+') p++;
            while (p < src.length()) {
                char c = src.charAt(p);
                if (Character.isDigit(c) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') p++;
                else break;
            }
            return Double.parseDouble(src.substring(s, p));
        }
        String readString() {
            ws(); expect('"'); StringBuilder sb = new StringBuilder();
            while (p < src.length()) {
                char c = src.charAt(p++);
                if (c == '"') return sb.toString();
                if (c == '\\' && p < src.length()) { char e = src.charAt(p++); sb.append(e == 'n' ? '\n' : e == 't' ? '\t' : e); }
                else sb.append(c);
            }
            throw new RuntimeException("FixtureParser: unterminated string");
        }
        void skipValue() {
            ws(); char c = peek();
            if (c == '"') { readString(); return; }
            if (c == '[') { skipArrayOrObject('[', ']'); return; }
            if (c == '{') { skipArrayOrObject('{', '}'); return; }
            if (c == 't' || c == 'f' || c == 'n') {
                while (p < src.length() && Character.isLetter(src.charAt(p))) p++;
                return;
            }
            readNumber();
        }
        void skipArrayOrObject(char open, char close) {
            expect(open); int depth = 1;
            while (p < src.length() && depth > 0) {
                char c = src.charAt(p++);
                if (c == '"') { p--; readString(); }
                else if (c == open) depth++;
                else if (c == close) depth--;
            }
        }
        void ws() { while (p < src.length() && Character.isWhitespace(src.charAt(p))) p++; }
        char peek() { return p < src.length() ? src.charAt(p) : '\0'; }
        void expect(char c) {
            if (peek() != c) throw new RuntimeException("expected '" + c + "' got '" + peek() + "' at " + p);
            p++;
        }
    }
}
