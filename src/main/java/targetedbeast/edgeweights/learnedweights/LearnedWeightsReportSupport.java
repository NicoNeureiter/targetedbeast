package targetedbeast.edgeweights.learnedweights;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

final class LearnedWeightsReportSupport {

    private LearnedWeightsReportSupport() {
    }

    static void ensureParentDirectory(Path outputFile) throws IOException {
        if (outputFile.getParent() != null) {
            Files.createDirectories(outputFile.getParent());
        }
    }

    static double mean(double[] values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    static double mean(List<Double> values) {
        if (values.isEmpty()) {
            return Double.NaN;
        }
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.size();
    }

    static double median(List<Double> values) {
        if (values.isEmpty()) {
            return Double.NaN;
        }
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int mid = sorted.size() / 2;
        return sorted.size() % 2 == 0 ? 0.5 * (sorted.get(mid - 1) + sorted.get(mid)) : sorted.get(mid);
    }

    static double pearson(double[] x, double[] y) {
        if (x.length != y.length || x.length == 0) {
            return Double.NaN;
        }

        double meanX = mean(x);
        double meanY = mean(y);
        double sxy = 0.0;
        double sxx = 0.0;
        double syy = 0.0;
        for (int i = 0; i < x.length; i++) {
            double dx = x[i] - meanX;
            double dy = y[i] - meanY;
            sxy += dx * dy;
            sxx += dx * dx;
            syy += dy * dy;
        }

        double denom = Math.sqrt(sxx * syy);
        return denom > 0.0 && Double.isFinite(denom) ? sxy / denom : Double.NaN;
    }

    static double spearman(double[] x, double[] y) {
        return pearson(rank(x), rank(y));
    }

    static double[] rank(double[] values) {
        List<Integer> indices = new ArrayList<>(values.length);
        for (int i = 0; i < values.length; i++) {
            indices.add(i);
        }
        indices.sort(Comparator.comparingDouble(i -> values[i]));

        double[] ranks = new double[values.length];
        int start = 0;
        while (start < indices.size()) {
            int end = start + 1;
            while (end < indices.size()
                    && Double.compare(values[indices.get(start)], values[indices.get(end)]) == 0) {
                end++;
            }
            double averageRank = 0.5 * (start + end - 1);
            for (int i = start; i < end; i++) {
                ranks[indices.get(i)] = averageRank;
            }
            start = end;
        }
        return ranks;
    }

    static String csvEscape(String value) {
        if (value == null) {
            return "";
        }
        if (!value.contains(",") && !value.contains("\"") && !value.contains("\n")) {
            return value;
        }
        return '"' + value.replace("\"", "\"\"") + '"';
    }
}