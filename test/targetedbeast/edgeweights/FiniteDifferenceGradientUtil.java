package targetedbeast.edgeweights;

import org.apache.commons.math3.analysis.MultivariateFunction;

/**
 * Utility methods for finite-difference gradient checks in tests.
 */
public final class FiniteDifferenceGradientUtil {

    private FiniteDifferenceGradientUtil() {
    }

    /**
     * Compute central-difference gradient.
     */
    public static double[] centralDifference(MultivariateFunction f, double[] x, double epsilon) {
        if (f == null) {
            throw new IllegalArgumentException("function must not be null");
        }
        if (x == null || x.length == 0) {
            throw new IllegalArgumentException("x must not be null or empty");
        }
        if (!(epsilon > 0.0)) {
            throw new IllegalArgumentException("epsilon must be > 0");
        }

        double[] grad = new double[x.length];
        double[] xPlus = x.clone();
        double[] xMinus = x.clone();

        for (int i = 0; i < x.length; i++) {
            xPlus[i] += epsilon;
            xMinus[i] -= epsilon;

            double fPlus = f.value(xPlus);
            double fMinus = f.value(xMinus);
            grad[i] = (fPlus - fMinus) / (2.0 * epsilon);

            xPlus[i] = x[i];
            xMinus[i] = x[i];
        }

        return grad;
    }

    public static double dot(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("length mismatch");
        }
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }
}
