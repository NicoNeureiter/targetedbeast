package targetedbeast.edgeweights.learnedweights;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import targetedbeast.edgeweights.LearnedWeights;

/**
 * Ranking metrics computed within competing-candidate groups (same {@code fromNode},
 * same {@code toNodeNrs} batch, tagged by {@link LearnedWeightsTreeTrainer.TrainingTreeSample#groupId}),
 * as opposed to {@link LearnedWeightsFitReporter}'s metrics which correlate predictions against
 * targets across independent posterior trees.
 *
 * <p>{@code stride}/{@code maxRows} subsampling in {@code ProposalExporter} is not group-aware, so a
 * group read back from a V5 file can have any number of members, including zero or one. Groups with
 * fewer than 2 members carry no ranking information and are excluded; see {@link GroupedMetrics} for
 * how usable/seen counts surface that.
 */
public final class LearnedWeightsGroupedMetrics {

    private LearnedWeightsGroupedMetrics() {
    }

    public static final class GroupedMetrics {
        /** Distinct groupIds observed (excluding {@code NO_GROUP} rows, which aren't a group at all). */
        public final int groupsSeen;
        /** Groups with >= 2 members: the minimum needed for top1/NDCG to be meaningful. */
        public final int groupsUsable;
        /** Groups with >= 3 members: Spearman needs at least 3 items, since a 2-item group only has
         *  two possible orderings and correlation is trivially +-1 regardless of prediction quality. */
        public final int spearmanGroupsUsable;
        public final double top1Accuracy;
        public final double meanSpearman;
        public final double medianSpearman;
        public final double meanNdcg;
        public final double medianNdcg;

        public GroupedMetrics(int groupsSeen, int groupsUsable, int spearmanGroupsUsable,
                double top1Accuracy, double meanSpearman, double medianSpearman,
                double meanNdcg, double medianNdcg) {
            this.groupsSeen = groupsSeen;
            this.groupsUsable = groupsUsable;
            this.spearmanGroupsUsable = spearmanGroupsUsable;
            this.top1Accuracy = top1Accuracy;
            this.meanSpearman = meanSpearman;
            this.medianSpearman = medianSpearman;
            this.meanNdcg = meanNdcg;
            this.medianNdcg = medianNdcg;
        }
    }

    public static final class GroupedReport {
        public final GroupedMetrics learned;
        public final GroupedMetrics pca;

        public GroupedReport(GroupedMetrics learned, GroupedMetrics pca) {
            this.learned = learned;
            this.pca = pca;
        }
    }

    /**
     * Computes grouped metrics for both the learned and PCA embeddings, reusing the same
     * surrogate-decomposition prediction path as {@link LearnedWeightsFitReporter#evaluate}.
     * {@code groupedSamples} must carry real {@code groupId}s, e.g. as loaded from a V5
     * proposal-export file via {@link LearnedWeightsTrainingSampleIO#readSamples}.
     */
    public static GroupedReport evaluate(LearnedWeights learnedWeights,
            List<LearnedWeightsTreeTrainer.TrainingTreeSample> groupedSamples) {
        if (groupedSamples == null || groupedSamples.isEmpty()) {
            throw new IllegalStateException("No grouped training samples available.");
        }

        String[] canonicalTaxa = learnedWeights.dataInput.get().getTaxaNames().toArray(new String[0]);
        int dim = learnedWeights.getEmbeddingDimension();
        double[] brownianTraitRates = learnedWeights.getBrownianTraitRates();
        double[] learnedEmbeddings = learnedWeights.getCurrentLeafEmbeddingsForTaxa(canonicalTaxa);
        double[] pcaEmbeddings = learnedWeights.getPcaLeafEmbeddingsForTaxa(canonicalTaxa);

        int n = groupedSamples.size();
        long[] groupIds = new long[n];
        double[] targets = new double[n];
        double[] learnedPredictions = new double[n];
        double[] pcaPredictions = new double[n];

        for (int index = 0; index < n; index++) {
            LearnedWeightsTreeTrainer.TrainingTreeSample sample = groupedSamples.get(index);
            groupIds[index] = sample.groupId;
            targets[index] = sample.targetLogLikelihood;
            learnedPredictions[index] = LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihoodDecomposition(
                    sample, learnedEmbeddings, dim, brownianTraitRates).total;
            pcaPredictions[index] = LearnedWeightsTreeObjectiveFunction.computeSurrogateLogLikelihoodDecomposition(
                    sample, pcaEmbeddings, dim, brownianTraitRates).total;
        }

        // Ranking metrics (top1/Spearman/NDCG) are invariant to monotone rescaling, so raw
        // (uncalibrated) predictions are used directly - no affine calibration needed.
        GroupedMetrics learnedMetrics = computeGroupedMetrics(groupIds, learnedPredictions, targets);
        GroupedMetrics pcaMetrics = computeGroupedMetrics(groupIds, pcaPredictions, targets);
        return new GroupedReport(learnedMetrics, pcaMetrics);
    }

    /** Pure, independently testable ranking-metrics computation over parallel arrays. */
    public static GroupedMetrics computeGroupedMetrics(long[] groupIds, double[] predicted, double[] trueValues) {
        if (groupIds.length != predicted.length || groupIds.length != trueValues.length) {
            throw new IllegalArgumentException("groupIds, predicted and trueValues must have the same length");
        }

        Map<Long, List<Integer>> groups = new LinkedHashMap<>();
        for (int i = 0; i < groupIds.length; i++) {
            if (groupIds[i] == LearnedWeightsTreeTrainer.TrainingTreeSample.NO_GROUP) {
                continue;
            }
            groups.computeIfAbsent(groupIds[i], k -> new ArrayList<>()).add(i);
        }

        int groupsSeen = groups.size();
        int groupsUsable = 0;
        int spearmanGroupsUsable = 0;
        int top1Hits = 0;
        List<Double> spearmanValues = new ArrayList<>();
        List<Double> ndcgValues = new ArrayList<>();

        for (List<Integer> members : groups.values()) {
            int size = members.size();
            if (size < 2) {
                continue;
            }
            groupsUsable++;

            double[] p = new double[size];
            double[] t = new double[size];
            for (int j = 0; j < size; j++) {
                p[j] = predicted[members.get(j)];
                t[j] = trueValues[members.get(j)];
            }

            if (argmax(p) == argmax(t)) {
                top1Hits++;
            }

            if (size >= 3) {
                spearmanGroupsUsable++;
                spearmanValues.add(LearnedWeightsReportSupport.spearman(p, t));
            }

            ndcgValues.add(ndcg(p, t));
        }

        double top1Accuracy = groupsUsable > 0 ? (double) top1Hits / groupsUsable : Double.NaN;

        return new GroupedMetrics(
                groupsSeen,
                groupsUsable,
                spearmanGroupsUsable,
                top1Accuracy,
                LearnedWeightsReportSupport.mean(spearmanValues),
                LearnedWeightsReportSupport.median(spearmanValues),
                LearnedWeightsReportSupport.mean(ndcgValues),
                LearnedWeightsReportSupport.median(ndcgValues));
    }

    private static int argmax(double[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * NDCG = DCG(predicted order) / DCG(ideal order), unweighted (no truncation - groups are small,
     * the size of one SPR candidate list).
     *
     * <p>Gain definition: {@code gain_i = groupSize - trueRank_i + 1}, where {@code trueRank_i} is
     * the 1-based rank of item i under the true target log-likelihood (rank 1 = highest). This turns
     * the true ranking into a graded relevance scale (best item gets the highest gain, worst gets 1)
     * without exponentiating the raw log-likelihoods, which live on an arbitrary log scale and are
     * not comparable to a relevance scale.
     */
    private static double ndcg(double[] predicted, double[] trueValues) {
        int n = predicted.length;
        int[] predictedRanks = descendingRanks(predicted);
        int[] trueRanks = descendingRanks(trueValues);

        double dcg = 0.0;
        double idcg = 0.0;
        for (int i = 0; i < n; i++) {
            double gain = n - trueRanks[i] + 1;
            dcg += gain / log2(predictedRanks[i] + 1);
            idcg += gain / log2(trueRanks[i] + 1);
        }
        return idcg > 0.0 ? dcg / idcg : Double.NaN;
    }

    private static double log2(double value) {
        return Math.log(value) / Math.log(2.0);
    }

    /** 1-based rank per index, descending by value (rank 1 = largest); ties broken by index. */
    private static int[] descendingRanks(double[] values) {
        Integer[] order = new Integer[values.length];
        for (int i = 0; i < values.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> {
            int cmp = Double.compare(values[b], values[a]);
            return cmp != 0 ? cmp : Integer.compare(a, b);
        });
        int[] ranks = new int[values.length];
        for (int rank = 0; rank < order.length; rank++) {
            ranks[order[rank]] = rank + 1;
        }
        return ranks;
    }
}
