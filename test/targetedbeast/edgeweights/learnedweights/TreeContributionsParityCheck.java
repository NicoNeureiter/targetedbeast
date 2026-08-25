package targetedbeast.edgeweights.learnedweights;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.TreeParser;
import targetedbeast.edgeweights.PCAWeightsBrownianFullLikelihood;

/**
 * Java-vs-Python parity check for the tree → (qf, logDet) pipeline.
 *
 * <p>Reads a fixture written by {@code surrogate_training/generate_tree_contributions_fixture.py},
 * which carries:
 * <ul>
 *   <li>{@code newick} — a synthetic ultrametric tree</li>
 *   <li>{@code value}  — leaf embeddings in the {@code taxon=e1 e2 …, …} format
 *       that {@link PCAWeightsBrownianFullLikelihood} parses</li>
 *   <li>{@code expected_qf} and {@code expected_logDet} — computed by Python
 *       via {@code LearnedWeightsTreeTrainer.factorizeImproperRootCovariance}'s
 *       projection-trick recipe (the training-time pipeline)</li>
 * </ul>
 *
 * <p>Java parses the Newick, builds a minimal {@link Alignment} with the
 * tree's taxa, instantiates a {@link PCAWeightsBrownianFullLikelihood}
 * configured to match training (improper root prior, unit Brownian rate),
 * runs the inside message-passing with contribution tracking, derives its own
 * {@code logDet = -2·base[0]} and {@code qf[d] = -2·quad[d]}, and compares.
 *
 * <p>If this passes the surrogate's runtime {@code (qf, logDet)} match the
 * training-time pipeline.  If it fails the divergence diagnoses the cause
 * (logDet vs qf, magnitude, etc).
 */
public final class TreeContributionsParityCheck {

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("Usage: TreeContributionsParityCheck <fixture.json> [tolerance]");
            System.exit(2);
        }
        Path fixturePath = Path.of(args[0]);
        double tolerance = args.length >= 2 ? Double.parseDouble(args[1]) : 1e-6;

        Fixture fx = Fixture.read(fixturePath);

        // ---- Build a minimal alignment matching the fixture's taxa ----
        List<Sequence> seqs = new ArrayList<>(fx.taxa.size());
        for (String taxonId : fx.taxa) {
            Sequence s = new Sequence();
            s.initByName("taxon", taxonId, "value", "A");
            seqs.add(s);
        }
        Alignment aln = new Alignment();
        aln.initByName("sequence", seqs, "dataType", "nucleotide");

        // ---- Parse the Newick tree ----
        TreeParser tree = new TreeParser();
        tree.initByName(
            "newick",          fx.newick,
            "IsLabelledNewick", true,
            "adjustTipHeights", false,
            "taxa",             aln
        );

        // ---- Build PCAWeightsBrownianFullLikelihood, matched to training config:
        //      improper root (rootPriorVariance = +Inf), no per-axis rate overrides.
        PCAWeightsBrownianFullLikelihood model = new PCAWeightsBrownianFullLikelihood();
        model.initByName(
            "tree",              tree,
            "data",              aln,
            "dimension",         fx.dim,
            "value",             fx.value,
            "rootPriorVariance", Double.POSITIVE_INFINITY
        );

        // ---- Run inside recursion with contribution tracking ----
        double[] base = new double[fx.dim];
        double[] quad = new double[fx.dim];
        model.computeFullTreeContributionParts(base, quad);

        double logDet = -2.0 * base[0];
        double[] qf = new double[fx.dim];
        for (int d = 0; d < fx.dim; d++) qf[d] = -2.0 * quad[d];

        // ---- Report ----
        System.out.println("# Tree → (qf, logDet) parity test");
        System.out.println("# Fixture: " + fixturePath);
        System.out.println("# n_taxa=" + fx.taxa.size() + " dim=" + fx.dim);
        System.out.println("# rootPriorVariance: Infinity (improper)");
        System.out.println();

        double dLd = Math.abs(logDet - fx.expectedLogDet);
        double dQf = 0.0;
        int worstD = 0;
        for (int d = 0; d < fx.dim; d++) {
            double da = Math.abs(qf[d] - fx.expectedQf[d]);
            if (da > dQf) { dQf = da; worstD = d; }
        }

        System.out.printf("Java   logDet = %.10f%n", logDet);
        System.out.printf("Python logDet = %.10f%n", fx.expectedLogDet);
        System.out.printf("|Δ|logDet     = %.3e%n", dLd);
        System.out.println();
        System.out.println("per-axis qf:");
        System.out.printf("%-5s  %-18s  %-18s  %-12s%n", "d", "java qf", "python qf", "|Δ|");
        for (int d = 0; d < fx.dim; d++) {
            System.out.printf("%-5d  %18.10f  %18.10f  %12.3e%n",
                d, qf[d], fx.expectedQf[d], Math.abs(qf[d] - fx.expectedQf[d]));
        }
        System.out.println();
        System.out.printf("max |Δ|qf = %.3e   worst axis=%d%n", dQf, worstD);

        boolean pass = dLd <= tolerance && dQf <= tolerance;
        System.out.println();
        System.out.println(pass
            ? "PASS — Java Felsenstein (qf, logDet) match Python training recipe within " + tolerance
            : "FAIL — divergence exceeds tolerance " + tolerance);
        System.exit(pass ? 0 : 1);
    }

    // ------------------------------------------------------------------
    // Fixture parsing (minimal JSON)
    // ------------------------------------------------------------------

    private static final class Fixture {
        final String newick;
        final List<String> taxa;
        final int dim;
        final String value;
        final double[] expectedQf;
        final double   expectedLogDet;
        Fixture(String newick, List<String> taxa, int dim, String value,
                double[] expectedQf, double expectedLogDet) {
            this.newick = newick;
            this.taxa = taxa;
            this.dim = dim;
            this.value = value;
            this.expectedQf = expectedQf;
            this.expectedLogDet = expectedLogDet;
        }
        static Fixture read(Path path) throws IOException {
            return FixtureParser.parse(Files.readString(path));
        }
    }

    private static final class FixtureParser {
        private final String src;
        private int p;

        private FixtureParser(String src) { this.src = src; this.p = 0; }

        static Fixture parse(String text) {
            return new FixtureParser(text).parseFixture();
        }

        private Fixture parseFixture() {
            ws(); expect('{');
            String newick = null, value = null;
            List<String> taxa = null;
            int dim = -1;
            double[] qf = null;
            double ld = Double.NaN;
            while (true) {
                ws();
                if (peek() == '}') { p++; break; }
                String key = readString();
                ws(); expect(':');
                switch (key) {
                    case "newick":          newick = readString();      break;
                    case "value":           value  = readString();      break;
                    case "taxa":            taxa   = readStringArray(); break;
                    case "dim":             dim    = (int) readNumber(); break;
                    case "expected_qf":     qf     = readDoubleArray(); break;
                    case "expected_logDet": ld     = readNumber();      break;
                    default:                skipValue();
                }
                ws();
                if (peek() == ',') { p++; continue; }
                if (peek() == '}') { p++; break; }
                throw err("expected , or } at top level");
            }
            if (newick == null || taxa == null || value == null || dim < 0 || qf == null || Double.isNaN(ld)) {
                throw err("incomplete fixture");
            }
            return new Fixture(newick, taxa, dim, value, qf, ld);
        }

        private List<String> readStringArray() {
            ws(); expect('[');
            List<String> out = new ArrayList<>();
            ws();
            if (peek() == ']') { p++; return out; }
            while (true) {
                out.add(readString());
                ws();
                if (peek() == ',') { p++; ws(); continue; }
                expect(']'); return out;
            }
        }

        private double[] readDoubleArray() {
            ws(); expect('[');
            List<Double> list = new ArrayList<>();
            ws();
            if (peek() == ']') { p++; return new double[0]; }
            while (true) {
                list.add(readNumber());
                ws();
                if (peek() == ',') { p++; ws(); continue; }
                expect(']'); break;
            }
            double[] out = new double[list.size()];
            for (int i = 0; i < out.length; i++) out[i] = list.get(i);
            return out;
        }

        private double readNumber() {
            ws();
            int start = p;
            if (peek() == '-' || peek() == '+') p++;
            while (p < src.length()) {
                char c = src.charAt(p);
                if (Character.isDigit(c) || c == '.' || c == 'e' || c == 'E'
                        || c == '+' || c == '-') {
                    p++;
                } else break;
            }
            return Double.parseDouble(src.substring(start, p));
        }

        private String readString() {
            ws(); expect('"');
            StringBuilder sb = new StringBuilder();
            while (p < src.length()) {
                char c = src.charAt(p++);
                if (c == '"') return sb.toString();
                if (c == '\\' && p < src.length()) {
                    char esc = src.charAt(p++);
                    sb.append(esc == 'n' ? '\n' : esc == 't' ? '\t' : esc);
                } else {
                    sb.append(c);
                }
            }
            throw err("unterminated string");
        }

        private void skipValue() {
            ws();
            char c = peek();
            if (c == '"')      { readString();      return; }
            if (c == '[')      { skipArray();       return; }
            if (c == '{')      { skipObject();      return; }
            if (c == 't' || c == 'f' || c == 'n') {
                while (p < src.length() && Character.isLetter(src.charAt(p))) p++;
                return;
            }
            readNumber();
        }

        private void skipArray() {
            expect('['); int depth = 1;
            while (p < src.length() && depth > 0) {
                char c = src.charAt(p++);
                if (c == '"') { p--; readString(); }
                else if (c == '[') depth++;
                else if (c == ']') depth--;
            }
        }

        private void skipObject() {
            expect('{'); int depth = 1;
            while (p < src.length() && depth > 0) {
                char c = src.charAt(p++);
                if (c == '"') { p--; readString(); }
                else if (c == '{') depth++;
                else if (c == '}') depth--;
            }
        }

        private void ws() {
            while (p < src.length() && Character.isWhitespace(src.charAt(p))) p++;
        }
        private char peek() { return p < src.length() ? src.charAt(p) : '\0'; }
        private void expect(char c) {
            if (peek() != c) throw err("expected '" + c + "' got '" + peek() + "'");
            p++;
        }
        private RuntimeException err(String msg) {
            return new RuntimeException("FixtureParser: " + msg + " at offset " + p);
        }
    }
}
