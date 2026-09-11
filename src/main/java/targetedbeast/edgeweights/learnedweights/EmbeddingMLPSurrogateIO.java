package targetedbeast.edgeweights.learnedweights;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads an {@code EmbeddingMLPSurrogateV1} JSON weight file produced by
 * {@code EmbeddingMLP.export_weights()} in Python and returns an
 * {@link EmbeddingMLPSurrogate} ready for inference.
 *
 * No external JSON library is required; the parser handles the flat schema
 * (scalars, 1-D arrays of doubles, 1-D arrays of strings) written by the
 * Python exporter.
 */
public final class EmbeddingMLPSurrogateIO {

    static final String FORMAT_TAG = "EmbeddingMLPSurrogateV1";

    private EmbeddingMLPSurrogateIO() {}

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    public static EmbeddingMLPSurrogate read(Path path) throws IOException {
        String text = Files.readString(path);
        Scanner sc = new Scanner(text);
        sc.expectObject();

        String format = null;
        int nTaxa = -1, dim = -1, nLayers = -1;
        String[] taxa = null;
        boolean useLogdet = true;
        double ratioShift = 5.0;
        double outputBound = 0.0;
        double[] qfMean = null, qfStd = null;
        double ldMean = 0.0, ldStd = 1.0, yMean = 0.0, yStd = 1.0;
        double[] eFlat = null;

        int maxLayers = 16;
        int[] layerIn  = new int[maxLayers];
        int[] layerOut = new int[maxLayers];
        double[][] layerWeight = new double[maxLayers][];
        double[][] layerBias   = new double[maxLayers][];

        while (sc.hasNextKey()) {
            String key = sc.readKey();
            switch (key) {
                case "format"      -> format     = sc.readString();
                case "n_taxa"      -> nTaxa      = (int) sc.readDouble();
                case "dim"         -> dim        = (int) sc.readDouble();
                case "taxa"        -> taxa       = sc.readStringArray();
                case "use_logdet"    -> useLogdet   = sc.readBoolean();
                case "ratio_shift"   -> ratioShift  = sc.readDouble();
                case "output_bound"  -> outputBound = sc.readDouble();
                case "qf_mean"     -> qfMean     = sc.readDoubleArray();
                case "qf_std"      -> qfStd      = sc.readDoubleArray();
                case "ld_mean"     -> ldMean      = sc.readDouble();
                case "ld_std"      -> ldStd       = sc.readDouble();
                case "y_mean"      -> yMean       = sc.readDouble();
                case "y_std"       -> yStd        = sc.readDouble();
                case "n_layers"    -> nLayers     = (int) sc.readDouble();
                case "E_flat"      -> eFlat       = sc.readDoubleArray();
                default -> {
                    // layer<i>_{in,out,weight,bias}
                    if (key.startsWith("layer")) {
                        int under = key.indexOf('_', 5);
                        if (under > 0) {
                            int idx = Integer.parseInt(key.substring(5, under));
                            String field = key.substring(under + 1);
                            switch (field) {
                                case "in"     -> layerIn[idx]     = (int) sc.readDouble();
                                case "out"    -> layerOut[idx]    = (int) sc.readDouble();
                                case "weight" -> layerWeight[idx] = sc.readDoubleArray();
                                case "bias"   -> layerBias[idx]   = sc.readDoubleArray();
                                default       -> sc.skipValue();
                            }
                            break;
                        }
                    }
                    sc.skipValue();
                }
            }
            sc.consumeComma();
        }

        // Validate
        if (!FORMAT_TAG.equals(format))
            throw new IllegalArgumentException("Unexpected format tag: " + format);
        requireNotNull(nTaxa,    "n_taxa");
        requireNotNull(dim,      "dim");
        requireNotNull(nLayers,  "n_layers");
        requireNotNull(qfMean,   "qf_mean");
        requireNotNull(qfStd,    "qf_std");
        requireNotNull(eFlat,    "E_flat");
        if (taxa == null) taxa = defaultTaxa(nTaxa);

        double[][]  wArrays = new double[nLayers][];
        double[][]  bArrays = new double[nLayers][];
        int[] inDims  = new int[nLayers];
        int[] outDims = new int[nLayers];
        for (int i = 0; i < nLayers; i++) {
            requireNotNull(layerWeight[i], "layer" + i + "_weight");
            requireNotNull(layerBias[i],   "layer" + i + "_bias");
            wArrays[i] = layerWeight[i];
            bArrays[i] = layerBias[i];
            inDims[i]  = layerIn[i];
            outDims[i] = layerOut[i];
        }

        return new EmbeddingMLPSurrogate(
                nTaxa, dim, taxa, useLogdet, ratioShift, outputBound,
                qfMean, qfStd, ldMean, ldStd, yMean, yStd,
                eFlat, nLayers, inDims, outDims, wArrays, bArrays);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static void requireNotNull(Object val, String field) {
        if (val == null || (val instanceof Integer i && i == -1))
            throw new IllegalArgumentException("Missing required field: " + field);
    }

    private static String[] defaultTaxa(int n) {
        String[] names = new String[n];
        for (int i = 0; i < n; i++) names[i] = "taxon_" + i;
        return names;
    }

    // ------------------------------------------------------------------
    // Minimal JSON scanner — handles our flat schema only
    // ------------------------------------------------------------------

    // Package-private (widened from private): shared with AxisReweightedFusedSurrogateIO,
    // which parses the same flat JSON schema style.
    static final class Scanner {
        private final String src;
        private int pos;

        Scanner(String src) { this.src = src; this.pos = 0; }

        void expectObject() {
            skipWs();
            expect('{');
        }

        boolean hasNextKey() {
            skipWs();
            return pos < src.length() && src.charAt(pos) != '}';
        }

        String readKey() {
            skipWs();
            String key = readString();
            skipWs();
            expect(':');
            return key;
        }

        void consumeComma() {
            skipWs();
            if (pos < src.length() && src.charAt(pos) == ',') pos++;
        }

        String readString() {
            skipWs();
            expect('"');
            StringBuilder sb = new StringBuilder();
            while (pos < src.length()) {
                char c = src.charAt(pos++);
                if (c == '"') return sb.toString();
                if (c == '\\' && pos < src.length()) {
                    char esc = src.charAt(pos++);
                    sb.append(esc == 'n' ? '\n' : esc == 't' ? '\t' : esc);
                } else {
                    sb.append(c);
                }
            }
            throw error("unterminated string");
        }

        double readDouble() {
            skipWs();
            int start = pos;
            if (pos < src.length() && src.charAt(pos) == '-') pos++;
            while (pos < src.length()) {
                char c = src.charAt(pos);
                if (Character.isDigit(c) || c == '.' || c == 'e' || c == 'E'
                        || c == '+' || c == '-') pos++;
                else break;
            }
            return Double.parseDouble(src.substring(start, pos));
        }

        boolean readBoolean() {
            skipWs();
            if (src.startsWith("true",  pos)) { pos += 4; return true;  }
            if (src.startsWith("false", pos)) { pos += 5; return false; }
            throw error("expected boolean");
        }

        double[] readDoubleArray() {
            skipWs();
            expect('[');
            List<Double> list = new ArrayList<>();
            skipWs();
            if (pos < src.length() && src.charAt(pos) != ']') {
                list.add(readDouble());
                skipWs();
                while (pos < src.length() && src.charAt(pos) == ',') {
                    pos++;
                    list.add(readDouble());
                    skipWs();
                }
            }
            expect(']');
            double[] arr = new double[list.size()];
            for (int i = 0; i < arr.length; i++) arr[i] = list.get(i);
            return arr;
        }

        String[] readStringArray() {
            skipWs();
            expect('[');
            List<String> list = new ArrayList<>();
            skipWs();
            if (pos < src.length() && src.charAt(pos) != ']') {
                list.add(readString());
                skipWs();
                while (pos < src.length() && src.charAt(pos) == ',') {
                    pos++;
                    list.add(readString());
                    skipWs();
                }
            }
            expect(']');
            return list.toArray(String[]::new);
        }

        void skipValue() {
            skipWs();
            if (pos >= src.length()) return;
            char c = src.charAt(pos);
            if (c == '"')       { readString(); return; }
            if (c == '[')       { skipArray();  return; }
            if (c == '{')       { skipObject(); return; }
            if (c == 't')       { pos += 4;     return; }  // true
            if (c == 'f')       { pos += 5;     return; }  // false
            if (c == 'n')       { pos += 4;     return; }  // null
            readDouble();
        }

        private void skipArray() {
            expect('[');
            int depth = 1;
            while (pos < src.length() && depth > 0) {
                char c = src.charAt(pos++);
                if (c == '[') depth++;
                else if (c == ']') depth--;
                else if (c == '"') { pos--; readString(); }
            }
        }

        private void skipObject() {
            expect('{');
            int depth = 1;
            while (pos < src.length() && depth > 0) {
                char c = src.charAt(pos++);
                if (c == '{') depth++;
                else if (c == '}') depth--;
                else if (c == '"') { pos--; readString(); }
            }
        }

        private void skipWs() {
            while (pos < src.length() && src.charAt(pos) <= ' ') pos++;
        }

        private void expect(char expected) {
            skipWs();
            if (pos >= src.length() || src.charAt(pos) != expected)
                throw error("expected '" + expected + "' but got '"
                        + (pos < src.length() ? src.charAt(pos) : "EOF") + "'");
            pos++;
        }

        private IllegalArgumentException error(String msg) {
            int ctx = Math.min(pos + 40, src.length());
            return new IllegalArgumentException(
                    "JSON parse error at pos " + pos + ": " + msg
                    + " (near: " + src.substring(pos, ctx).replace('\n', ' ') + ")");
        }
    }
}
