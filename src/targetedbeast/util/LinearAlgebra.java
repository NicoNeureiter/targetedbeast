package targetedbeast.util;

public class LinearAlgebra {
    
    public static double sum(double[] xs) {
        double s = 0.0;
        for (double x : xs) {
            s += x;
        }
        return s;
    }

    public static void multiply(double[] x, double[] y, double[] res) {
        for (int i = 0; i < x.length; i++) {
            res[i] = x[i] * y[i];
        }
    }
    
    public static double[] multiply(double[] x, double[] y) {
        double[] res = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            res[i] = x[i] * y[i];
        }
        return res;
    }

    public static double dotProduct(double[] x, double[] y) {
        double s = 0.0;
        for (int i = 0; i < x.length; i++) {
            s += x[i] * y[i];
        }
        return s;
    }
}
