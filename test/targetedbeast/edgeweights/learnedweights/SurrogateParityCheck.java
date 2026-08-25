package targetedbeast.edgeweights.learnedweights;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Java-vs-Python parity check for the MLP surrogate forward pass.
 *
 * <p>Reads two files:
 * <ol>
 *   <li>The surrogate weight JSON used by the runtime.</li>
 *   <li>A fixture JSON written by {@code surrogate_training/generate_parity_fixture.py}
 *       containing {@code (qf, logDet, expected_prediction)} triples.</li>
 * </ol>
 *
 * <p>For each fixture entry the test runs
 * {@link EmbeddingMLPSurrogate#predictFromQfAndLogDet(double[], double)} and
 * compares against the Python-recorded expected prediction.  This validates the
 * JSON loader and the Java MLP forward pass in isolation; the upstream
 * conversion of a proposed tree into {@code (qf, logDet)} is exercised by the
 * parent class's Felsenstein message-passing and is intentionally outside the
 * scope of this test.
 *
 * <p>Exits with status 0 if every absolute prediction diff is within tolerance,
 * otherwise status 1.
 */
public final class SurrogateParityCheck {

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println(
                "Usage: SurrogateParityCheck <surrogate.json> <fixture.json> [tolerance]");
            System.exit(2);
        }
        Path surrogatePath = Path.of(args[0]);
        Path fixturePath   = Path.of(args[1]);
        double tolerance   = args.length >= 3 ? Double.parseDouble(args[2]) : 1e-6;

        EmbeddingMLPSurrogate surrogate = EmbeddingMLPSurrogate.fromFile(surrogatePath);
        Fixture fixture = Fixture.read(fixturePath);

        if (fixture.dim != surrogate.dim()) {
            System.err.println("Fixture dim (" + fixture.dim + ") does not match surrogate dim ("
                + surrogate.dim() + ")");
            System.exit(2);
        }

        System.out.println("# Java-vs-Python parity test for MLP surrogate forward pass");
        System.out.println("# Surrogate:  " + surrogatePath);
        System.out.println("# Fixture:    " + fixturePath);
        System.out.println("# Cases:      " + fixture.cases.size());
        System.out.println("# dim=" + fixture.dim);
        System.out.println("# Tolerance:  " + tolerance);
        System.out.println();
        System.out.printf("%-5s  %-14s  %-14s  %-14s%n",
                          "case", "java pred", "expected pred", "|Δ|");

        double maxAbs = 0.0;
        int badCase = -1;
        for (int c = 0; c < fixture.cases.size(); c++) {
            FixtureCase tc = fixture.cases.get(c);
            double pred = surrogate.predictFromQfAndLogDet(tc.qf, tc.logDet);
            double diff = Math.abs(pred - tc.expectedPrediction);
            if (diff > maxAbs) { maxAbs = diff; badCase = c; }
            System.out.printf("%-5d  %14.6f  %14.6f  %14.3e%n",
                              c, pred, tc.expectedPrediction, diff);
        }

        System.out.println();
        System.out.printf("max |Δ|prediction = %.3e   worst case=%d%n", maxAbs, badCase);
        System.out.println();
        boolean pass = maxAbs <= tolerance;
        System.out.println(pass
            ? "PASS"
            : "FAIL — Java MLP forward pass diverges from Python beyond tolerance " + tolerance);
        System.exit(pass ? 0 : 1);
    }

    // ------------------------------------------------------------------
    // Fixture parsing
    // ------------------------------------------------------------------

    private static final class Fixture {
        final int dim;
        final List<FixtureCase> cases;
        Fixture(int dim, List<FixtureCase> cases) {
            this.dim = dim;
            this.cases = cases;
        }
        static Fixture read(Path path) throws IOException {
            return FixtureParser.parse(Files.readString(path));
        }
    }

    private static final class FixtureCase {
        final double[] qf;
        final double   logDet;
        final double   expectedPrediction;
        FixtureCase(double[] qf, double logDet, double expectedPrediction) {
            this.qf = qf;
            this.logDet = logDet;
            this.expectedPrediction = expectedPrediction;
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
            int dim = -1;
            List<FixtureCase> cases = null;
            while (true) {
                ws();
                if (peek() == '}') { p++; break; }
                String key = readString();
                ws(); expect(':');
                switch (key) {
                    case "dim":         dim   = (int) readNumber(); break;
                    case "test_cases":  cases = readCases();        break;
                    default:            skipValue();
                }
                ws();
                if (peek() == ',') { p++; continue; }
                if (peek() == '}') { p++; break; }
                throw err("expected , or } at top level");
            }
            if (dim < 0 || cases == null) {
                throw err("missing dim / test_cases");
            }
            return new Fixture(dim, cases);
        }

        private List<FixtureCase> readCases() {
            ws(); expect('[');
            List<FixtureCase> out = new ArrayList<>();
            ws();
            if (peek() == ']') { p++; return out; }
            while (true) {
                out.add(readCase());
                ws();
                if (peek() == ',') { p++; continue; }
                expect(']'); return out;
            }
        }

        private FixtureCase readCase() {
            ws(); expect('{');
            double[] qf = null;
            double ld = Double.NaN, pred = Double.NaN;
            while (true) {
                ws();
                if (peek() == '}') { p++; break; }
                String key = readString();
                ws(); expect(':');
                switch (key) {
                    case "qf":                  qf   = readDoubleArray(); break;
                    case "logDet":              ld   = readNumber();      break;
                    case "expected_prediction": pred = readNumber();      break;
                    default:                    skipValue();
                }
                ws();
                if (peek() == ',') { p++; continue; }
                if (peek() == '}') { p++; break; }
                throw err("expected , or } in test case");
            }
            if (qf == null || Double.isNaN(ld) || Double.isNaN(pred)) {
                throw err("incomplete test case");
            }
            return new FixtureCase(qf, ld, pred);
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
                if (c == '\\' && p < src.length()) sb.append(src.charAt(p++));
                else                                sb.append(c);
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
