package targetedbeast.tools;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Distribution;
import targetedbeast.edgeweights.EdgeWeights;

public final class WilsonBaldingPriorDiagnostic {

    private WilsonBaldingPriorDiagnostic() {
    }

    public record Config(double maxHeightRatio, boolean useAttachmentIntervalLength, long seed, int maxSources) {

        public Config {
            if (!(maxHeightRatio >= 1.0)) {
                throw new IllegalArgumentException("maxHeightRatio must be >= 1.0");
            }
        }

        public static Config defaults() {
            return new Config(1.2, true, 1L, 0);
        }
    }

    public record Row(
            int sourceNr,
            String sourceId,
            double sourceHeight,
            double sourceLikelihoodWeight,
            double sourceHeuristicWeight,
            double sourceCombinedWeight,
            int candidateNr,
            String candidateId,
            double candidateHeight,
            double attachmentAge,
            double intervalMin,
            double intervalMax,
            double exactPriorDelta,
            double likelihoodTargetWeight,
            double heuristicTargetWeight,
            double combinedTargetWeight,
            double likelihoodSelectionMass,
            double heuristicSelectionMass,
            double combinedSelectionMass,
            int exactRank,
            int likelihoodRank,
            int heuristicRank,
            int combinedRank) {
    }

    public record SourceSummary(
            int sourceNr,
            String sourceId,
            int candidateCount,
            String exactBestCandidateId,
            String likelihoodBestCandidateId,
            String heuristicBestCandidateId,
            String combinedBestCandidateId,
            boolean likelihoodTopMatchesExact,
            boolean heuristicTopMatchesExact,
            boolean combinedTopMatchesExact,
            int likelihoodRankOfExactBest,
            int heuristicRankOfExactBest,
                int combinedRankOfExactBest,
                double likelihoodSpearman,
                double heuristicSpearman,
                double combinedSpearman,
                double likelihoodPearson,
                double heuristicPearson,
                double combinedPearson) {
    }

            public record AggregateSummary(
                int sourceCount,
                int candidateRowCount,
                long likelihoodTopMatchesExact,
                long heuristicTopMatchesExact,
                long combinedTopMatchesExact,
                double averageLikelihoodRankOfExactBest,
                double averageHeuristicRankOfExactBest,
                double averageCombinedRankOfExactBest,
                double averageLikelihoodSpearman,
                double averageHeuristicSpearman,
                double averageCombinedSpearman,
                double averageLikelihoodPearson,
                double averageHeuristicPearson,
                double averageCombinedPearson) {
            }

            public record Report(List<Row> rows, List<SourceSummary> sourceSummaries, AggregateSummary aggregateSummary) {
    }

    public static Report analyze(Tree tree,
            Distribution prior,
            EdgeWeights likelihoodWeights,
            EdgeWeights heuristicWeights,
            EdgeWeights combinedWeights,
            Config config) {
        double baselineLogP = prior.calculateLogP();
        if (!Double.isFinite(baselineLogP)) {
            throw new IllegalStateException("Prior baseline logP is not finite for diagnostic prior '"
                    + prior.getID() + "': " + baselineLogP);
        }
        Random random = new Random(config.seed());

        List<Row> rows = new ArrayList<>();
        List<SourceSummary> sourceSummaries = new ArrayList<>();

        List<Node> sourceNodes = Arrays.stream(tree.getNodesAsArray())
                .filter(node -> !node.isRoot())
                .filter(node -> node.getParent() != null && !node.getParent().isRoot())
                .sorted(Comparator.comparingInt(Node::getNr))
                .collect(Collectors.toList());

        int processedSources = 0;
        for (Node movingNode : sourceNodes) {
            if (config.maxSources() > 0 && processedSources >= config.maxSources()) {
                break;
            }

            List<Row> sourceRows = analyzeSource(tree, prior, baselineLogP, movingNode,
                    likelihoodWeights, heuristicWeights, combinedWeights, config, random);
            if (sourceRows.isEmpty()) {
                continue;
            }

            processedSources++;
            rows.addAll(sourceRows);
            sourceSummaries.add(buildSourceSummary(sourceRows));
        }

        return new Report(rows, sourceSummaries, buildAggregateSummary(rows, sourceSummaries));
    }

    public static void writeRowsTsv(Path output, Report report) throws IOException {
        List<String> lines = new ArrayList<>();
        lines.add(String.join("\t",
                "sourceNr",
                "sourceId",
                "sourceHeight",
                "sourceLikelihoodWeight",
                "sourceHeuristicWeight",
                "sourceCombinedWeight",
                "candidateNr",
                "candidateId",
                "candidateHeight",
                "attachmentAge",
                "intervalMin",
                "intervalMax",
                "exactPriorDelta",
                "likelihoodTargetWeight",
                "heuristicTargetWeight",
                "combinedTargetWeight",
                "likelihoodSelectionMass",
                "heuristicSelectionMass",
                "combinedSelectionMass",
                "exactRank",
                "likelihoodRank",
                "heuristicRank",
                "combinedRank"));

        for (Row row : report.rows()) {
            lines.add(String.join("\t",
                    Integer.toString(row.sourceNr()),
                    row.sourceId(),
                    Double.toString(row.sourceHeight()),
                    Double.toString(row.sourceLikelihoodWeight()),
                    Double.toString(row.sourceHeuristicWeight()),
                    Double.toString(row.sourceCombinedWeight()),
                    Integer.toString(row.candidateNr()),
                    row.candidateId(),
                    Double.toString(row.candidateHeight()),
                    Double.toString(row.attachmentAge()),
                    Double.toString(row.intervalMin()),
                    Double.toString(row.intervalMax()),
                    Double.toString(row.exactPriorDelta()),
                    Double.toString(row.likelihoodTargetWeight()),
                    Double.toString(row.heuristicTargetWeight()),
                    Double.toString(row.combinedTargetWeight()),
                    Double.toString(row.likelihoodSelectionMass()),
                    Double.toString(row.heuristicSelectionMass()),
                    Double.toString(row.combinedSelectionMass()),
                    Integer.toString(row.exactRank()),
                    Integer.toString(row.likelihoodRank()),
                    Integer.toString(row.heuristicRank()),
                    Integer.toString(row.combinedRank())));
        }

        Files.write(output, lines);
    }

    public static void writeSummaryTsv(Path output, Report report) throws IOException {
        List<String> lines = new ArrayList<>();
        AggregateSummary aggregateSummary = report.aggregateSummary();
        lines.add(String.join("\t", "#aggregateMetric", "likelihood", "heuristic", "combined"));
        lines.add(String.join("\t",
            "top1ExactMatches",
            Long.toString(aggregateSummary.likelihoodTopMatchesExact()),
            Long.toString(aggregateSummary.heuristicTopMatchesExact()),
            Long.toString(aggregateSummary.combinedTopMatchesExact())));
        lines.add(String.join("\t",
            "averageRankOfExactBest",
            Double.toString(aggregateSummary.averageLikelihoodRankOfExactBest()),
            Double.toString(aggregateSummary.averageHeuristicRankOfExactBest()),
            Double.toString(aggregateSummary.averageCombinedRankOfExactBest())));
        lines.add(String.join("\t",
            "averageSpearmanAgainstExact",
            Double.toString(aggregateSummary.averageLikelihoodSpearman()),
            Double.toString(aggregateSummary.averageHeuristicSpearman()),
            Double.toString(aggregateSummary.averageCombinedSpearman())));
        lines.add(String.join("\t",
            "averagePearsonAgainstExactDelta",
            Double.toString(aggregateSummary.averageLikelihoodPearson()),
            Double.toString(aggregateSummary.averageHeuristicPearson()),
            Double.toString(aggregateSummary.averageCombinedPearson())));
        lines.add("");
        lines.add(String.join("\t",
                "sourceNr",
                "sourceId",
                "candidateCount",
                "exactBestCandidateId",
                "likelihoodBestCandidateId",
                "heuristicBestCandidateId",
                "combinedBestCandidateId",
                "likelihoodTopMatchesExact",
                "heuristicTopMatchesExact",
                "combinedTopMatchesExact",
                "likelihoodRankOfExactBest",
                "heuristicRankOfExactBest",
                "combinedRankOfExactBest",
                "likelihoodSpearman",
                "heuristicSpearman",
                "combinedSpearman",
                "likelihoodPearson",
                "heuristicPearson",
                "combinedPearson"));

        for (SourceSummary summary : report.sourceSummaries()) {
            lines.add(String.join("\t",
                    Integer.toString(summary.sourceNr()),
                    summary.sourceId(),
                    Integer.toString(summary.candidateCount()),
                    summary.exactBestCandidateId(),
                    summary.likelihoodBestCandidateId(),
                    summary.heuristicBestCandidateId(),
                    summary.combinedBestCandidateId(),
                    Boolean.toString(summary.likelihoodTopMatchesExact()),
                    Boolean.toString(summary.heuristicTopMatchesExact()),
                    Boolean.toString(summary.combinedTopMatchesExact()),
                    Integer.toString(summary.likelihoodRankOfExactBest()),
                    Integer.toString(summary.heuristicRankOfExactBest()),
                    Integer.toString(summary.combinedRankOfExactBest()),
                    Double.toString(summary.likelihoodSpearman()),
                    Double.toString(summary.heuristicSpearman()),
                    Double.toString(summary.combinedSpearman()),
                    Double.toString(summary.likelihoodPearson()),
                    Double.toString(summary.heuristicPearson()),
                    Double.toString(summary.combinedPearson())));
        }

        Files.write(output, lines);
    }

    public static String formatConsoleSummary(Report report) {
        AggregateSummary aggregateSummary = report.aggregateSummary();

        return "Wilson-Balding prior diagnostic\n"
            + "  sources analyzed: " + aggregateSummary.sourceCount() + "\n"
            + "  candidate rows:    " + aggregateSummary.candidateRowCount() + "\n"
                + "  top-1 exact-prior matches\n"
            + "    likelihood:      " + aggregateSummary.likelihoodTopMatchesExact() + "\n"
            + "    heuristic:       " + aggregateSummary.heuristicTopMatchesExact() + "\n"
            + "    combined:        " + aggregateSummary.combinedTopMatchesExact() + "\n"
            + "  average rank of exact best\n"
            + "    likelihood:      " + formatDouble(aggregateSummary.averageLikelihoodRankOfExactBest()) + "\n"
            + "    heuristic:       " + formatDouble(aggregateSummary.averageHeuristicRankOfExactBest()) + "\n"
            + "    combined:        " + formatDouble(aggregateSummary.averageCombinedRankOfExactBest()) + "\n"
            + "  average Spearman vs exact order\n"
            + "    likelihood:      " + formatDouble(aggregateSummary.averageLikelihoodSpearman()) + "\n"
            + "    heuristic:       " + formatDouble(aggregateSummary.averageHeuristicSpearman()) + "\n"
            + "    combined:        " + formatDouble(aggregateSummary.averageCombinedSpearman()) + "\n"
            + "  average Pearson vs exact prior delta\n"
            + "    likelihood:      " + formatDouble(aggregateSummary.averageLikelihoodPearson()) + "\n"
            + "    heuristic:       " + formatDouble(aggregateSummary.averageHeuristicPearson()) + "\n"
            + "    combined:        " + formatDouble(aggregateSummary.averageCombinedPearson());
    }

    private static List<Row> analyzeSource(Tree tree,
            Distribution prior,
            double baselineLogP,
            Node movingNode,
            EdgeWeights likelihoodWeights,
            EdgeWeights heuristicWeights,
            EdgeWeights combinedWeights,
            Config config,
            Random random) {
        Node parent = movingNode.getParent();
        Node sibling = getOtherChild(parent, movingNode);

        List<Node> candidateNodes = getCoExistingLineages(movingNode, tree);
        candidateNodes.remove(movingNode);
        candidateNodes.remove(parent);
        if (candidateNodes.isEmpty()) {
            return List.of();
        }

        List<Integer> ancestors = collectAncestorsAmongCandidates(parent, candidateNodes);
        List<Node> feasibleCandidates = new ArrayList<>();
        List<Double> attachmentAges = new ArrayList<>();
        List<AttachmentInterval> attachmentIntervals = new ArrayList<>();
        for (Node candidate : candidateNodes) {
            AttachmentInterval interval = getAttachmentInterval(movingNode, candidate, config.maxHeightRatio());
            if (interval.range() <= 0.0) {
                continue;
            }
            feasibleCandidates.add(candidate);
            attachmentIntervals.add(interval);
            attachmentAges.add(sampleAttachmentAge(interval, random));
        }

        if (feasibleCandidates.isEmpty()) {
            return List.of();
        }

        ScoredTargets likelihood = scoreTargets(likelihoodWeights, movingNode, ancestors,
                feasibleCandidates, attachmentAges, attachmentIntervals, config.useAttachmentIntervalLength());
        ScoredTargets heuristic = scoreTargets(heuristicWeights, movingNode, ancestors,
                feasibleCandidates, attachmentAges, attachmentIntervals, config.useAttachmentIntervalLength());
        ScoredTargets combined = scoreTargets(combinedWeights, movingNode, ancestors,
                feasibleCandidates, attachmentAges, attachmentIntervals, config.useAttachmentIntervalLength());

        double sourceLikelihoodWeight = likelihoodWeights.getEdgeWeights(movingNode.getNr());
        double sourceHeuristicWeight = heuristicWeights.getEdgeWeights(movingNode.getNr());
        double sourceCombinedWeight = combinedWeights.getEdgeWeights(movingNode.getNr());

        List<RowDraft> rowDrafts = new ArrayList<>();
        for (int idx = 0; idx < feasibleCandidates.size(); idx++) {
            Node candidate = feasibleCandidates.get(idx);
            if (candidate == sibling) {
                continue;
            }

            double exactPriorDelta = computeExactPriorDelta(prior, tree, baselineLogP, movingNode, candidate, attachmentAges.get(idx));
            rowDrafts.add(new RowDraft(
                    movingNode.getNr(),
                    nodeId(movingNode),
                    movingNode.getHeight(),
                    sourceLikelihoodWeight,
                    sourceHeuristicWeight,
                    sourceCombinedWeight,
                    candidate.getNr(),
                    nodeId(candidate),
                    candidate.getHeight(),
                    attachmentAges.get(idx),
                    attachmentIntervals.get(idx).minAge,
                    attachmentIntervals.get(idx).maxAge,
                    exactPriorDelta,
                    likelihood.targetWeights[idx],
                    heuristic.targetWeights[idx],
                    combined.targetWeights[idx],
                    likelihood.selectionMass[idx],
                    heuristic.selectionMass[idx],
                    combined.selectionMass[idx]));
        }

        if (rowDrafts.isEmpty()) {
            return List.of();
        }

        Map<Integer, Integer> exactRanks = computeRanks(rowDrafts, Comparator.comparingDouble(RowDraft::exactPriorDelta).reversed());
        Map<Integer, Integer> likelihoodRanks = computeRanks(rowDrafts, Comparator.comparingDouble(RowDraft::likelihoodSelectionMass).reversed());
        Map<Integer, Integer> heuristicRanks = computeRanks(rowDrafts, Comparator.comparingDouble(RowDraft::heuristicSelectionMass).reversed());
        Map<Integer, Integer> combinedRanks = computeRanks(rowDrafts, Comparator.comparingDouble(RowDraft::combinedSelectionMass).reversed());

        return rowDrafts.stream()
                .map(row -> row.toRow(
                        exactRanks.get(row.candidateNr()),
                        likelihoodRanks.get(row.candidateNr()),
                        heuristicRanks.get(row.candidateNr()),
                        combinedRanks.get(row.candidateNr())))
                .sorted(Comparator.comparingInt(Row::sourceNr).thenComparingInt(Row::exactRank))
                .collect(Collectors.toList());
    }

    private static ScoredTargets scoreTargets(EdgeWeights edgeWeights,
            Node movingNode,
            List<Integer> ancestors,
            List<Node> candidates,
            List<Double> attachmentAges,
            List<AttachmentInterval> attachmentIntervals,
            boolean useAttachmentIntervalLength) {
        edgeWeights.prestore();
        edgeWeights.updateByOperatorWithoutNode(movingNode.getNr(), ancestors);
        double[] targetWeights = edgeWeights.getTargetWeights(movingNode.getNr(), candidates, attachmentAges);
        edgeWeights.reset();

        double[] selectionMass = new double[targetWeights.length];
        for (int idx = 0; idx < targetWeights.length; idx++) {
            double intervalWeight = useAttachmentIntervalLength ? attachmentIntervals.get(idx).range() : 1.0;
            selectionMass[idx] = Math.pow(targetWeights[idx], 2.0) * intervalWeight;
        }
        return new ScoredTargets(targetWeights, selectionMass);
    }

    private static double computeExactPriorDelta(Distribution prior,
            Tree tree,
            double baselineLogP,
            Node movingNode,
            Node candidate,
            double attachmentAge) {
        Node oldParent = movingNode.getParent();
        Node sibling = getOtherChild(oldParent, movingNode);
        Node grandParent = oldParent.getParent();
        Node candidateParent = candidate.getParent();

        double previousHeight = oldParent.getHeight();

        oldParent.setHeight(attachmentAge);
        replace(grandParent, oldParent, sibling);
        replace(oldParent, sibling, candidate);
        replace(candidateParent, candidate, oldParent);
        tree.setEverythingDirty(true);
        double movedLogP = prior.calculateLogP();
        if (!Double.isFinite(movedLogP)) {
            throw new IllegalStateException("Prior produced non-finite logP after diagnostic move for prior '"
                    + prior.getID() + "', movingNode='" + nodeId(movingNode)
                    + "', candidate='" + nodeId(candidate)
                    + "', attachmentAge=" + attachmentAge + ": " + movedLogP);
        }

        replace(candidateParent, oldParent, candidate);
        replace(oldParent, candidate, sibling);
        replace(grandParent, sibling, oldParent);
        oldParent.setHeight(previousHeight);
        tree.setEverythingDirty(true);
        double restoredLogP = prior.calculateLogP();
        if (!Double.isFinite(restoredLogP)) {
            throw new IllegalStateException("Prior produced non-finite restored logP for prior '"
                    + prior.getID() + "', movingNode='" + nodeId(movingNode)
                    + "', candidate='" + nodeId(candidate)
                    + "', attachmentAge=" + attachmentAge + ": " + restoredLogP);
        }
        if (Math.abs(restoredLogP - baselineLogP) > 1e-10) {
            throw new IllegalStateException("Prior baseline was not restored after diagnostic move: restored="
                    + restoredLogP + ", expected=" + baselineLogP);
        }

        double exactPriorDelta = movedLogP - baselineLogP;
        if (!Double.isFinite(exactPriorDelta)) {
            throw new IllegalStateException("Exact prior delta is not finite for prior '"
                    + prior.getID() + "', movingNode='" + nodeId(movingNode)
                    + "', candidate='" + nodeId(candidate)
                    + "', attachmentAge=" + attachmentAge + ": " + exactPriorDelta);
        }
        return exactPriorDelta;
    }

    private static SourceSummary buildSourceSummary(List<Row> sourceRows) {
        Row exactBest = bestRow(sourceRows, Comparator.comparingDouble(Row::exactPriorDelta));
        Row likelihoodBest = bestRow(sourceRows, Comparator.comparingDouble(Row::likelihoodSelectionMass));
        Row heuristicBest = bestRow(sourceRows, Comparator.comparingDouble(Row::heuristicSelectionMass));
        Row combinedBest = bestRow(sourceRows, Comparator.comparingDouble(Row::combinedSelectionMass));
        double likelihoodSpearman = computeSpearman(sourceRows, Row::likelihoodRank);
        double heuristicSpearman = computeSpearman(sourceRows, Row::heuristicRank);
        double combinedSpearman = computeSpearman(sourceRows, Row::combinedRank);
        double likelihoodPearson = computePearson(sourceRows, Row::likelihoodTargetWeight);
        double heuristicPearson = computePearson(sourceRows, Row::heuristicTargetWeight);
        double combinedPearson = computePearson(sourceRows, Row::combinedTargetWeight);

        return new SourceSummary(
                exactBest.sourceNr(),
                exactBest.sourceId(),
                sourceRows.size(),
                exactBest.candidateId(),
                likelihoodBest.candidateId(),
                heuristicBest.candidateId(),
                combinedBest.candidateId(),
                exactBest.candidateNr() == likelihoodBest.candidateNr(),
                exactBest.candidateNr() == heuristicBest.candidateNr(),
                exactBest.candidateNr() == combinedBest.candidateNr(),
                exactBest.likelihoodRank(),
                exactBest.heuristicRank(),
                exactBest.combinedRank(),
                likelihoodSpearman,
                heuristicSpearman,
                combinedSpearman,
                likelihoodPearson,
                heuristicPearson,
                combinedPearson);
    }

    private static AggregateSummary buildAggregateSummary(List<Row> rows, List<SourceSummary> sourceSummaries) {
        long likelihoodMatches = sourceSummaries.stream().filter(SourceSummary::likelihoodTopMatchesExact).count();
        long heuristicMatches = sourceSummaries.stream().filter(SourceSummary::heuristicTopMatchesExact).count();
        long combinedMatches = sourceSummaries.stream().filter(SourceSummary::combinedTopMatchesExact).count();

        return new AggregateSummary(
                sourceSummaries.size(),
                rows.size(),
                likelihoodMatches,
                heuristicMatches,
                combinedMatches,
                averageInt(sourceSummaries, SourceSummary::likelihoodRankOfExactBest),
                averageInt(sourceSummaries, SourceSummary::heuristicRankOfExactBest),
                averageInt(sourceSummaries, SourceSummary::combinedRankOfExactBest),
                averageDouble(sourceSummaries, SourceSummary::likelihoodSpearman),
                averageDouble(sourceSummaries, SourceSummary::heuristicSpearman),
                averageDouble(sourceSummaries, SourceSummary::combinedSpearman),
                averageDouble(sourceSummaries, SourceSummary::likelihoodPearson),
                averageDouble(sourceSummaries, SourceSummary::heuristicPearson),
                averageDouble(sourceSummaries, SourceSummary::combinedPearson));
    }

    private static Row bestRow(List<Row> rows, Comparator<Row> comparator) {
        return rows.stream().max(comparator).orElseThrow();
    }

    private static double computeSpearman(List<Row> sourceRows, java.util.function.ToIntFunction<Row> rankExtractor) {
        int candidateCount = sourceRows.size();
        if (candidateCount < 2) {
            return Double.NaN;
        }

        double sumSquaredDifferences = 0.0;
        for (Row row : sourceRows) {
            double rankDifference = row.exactRank() - rankExtractor.applyAsInt(row);
            sumSquaredDifferences += rankDifference * rankDifference;
        }

        return 1.0 - (6.0 * sumSquaredDifferences)
                / (candidateCount * (candidateCount * candidateCount - 1.0));
    }

    private static double computePearson(List<Row> sourceRows, java.util.function.ToDoubleFunction<Row> valueExtractor) {
        int candidateCount = sourceRows.size();
        if (candidateCount < 2) {
            return Double.NaN;
        }

        double meanExact = sourceRows.stream().mapToDouble(Row::exactPriorDelta).average().orElse(Double.NaN);
        double meanValue = sourceRows.stream().mapToDouble(valueExtractor).average().orElse(Double.NaN);

        double covariance = 0.0;
        double exactVariance = 0.0;
        double valueVariance = 0.0;
        for (Row row : sourceRows) {
            double centeredExact = row.exactPriorDelta() - meanExact;
            double centeredValue = valueExtractor.applyAsDouble(row) - meanValue;
            covariance += centeredExact * centeredValue;
            exactVariance += centeredExact * centeredExact;
            valueVariance += centeredValue * centeredValue;
        }

        if (exactVariance == 0.0 || valueVariance == 0.0) {
            return Double.NaN;
        }
        return covariance / Math.sqrt(exactVariance * valueVariance);
    }

    private static Map<Integer, Integer> computeRanks(List<RowDraft> rows, Comparator<RowDraft> comparator) {
        List<RowDraft> sorted = rows.stream().sorted(comparator).collect(Collectors.toList());
        Map<Integer, Integer> ranks = new HashMap<>();
        for (int idx = 0; idx < sorted.size(); idx++) {
            ranks.put(sorted.get(idx).candidateNr(), idx + 1);
        }
        return ranks;
    }

    private static double averageInt(List<SourceSummary> sourceSummaries, java.util.function.ToIntFunction<SourceSummary> getter) {
        if (sourceSummaries.isEmpty()) {
            return Double.NaN;
        }
        return sourceSummaries.stream().mapToInt(getter).average().orElse(Double.NaN);
    }

    private static double averageDouble(List<SourceSummary> sourceSummaries, java.util.function.ToDoubleFunction<SourceSummary> getter) {
        return sourceSummaries.stream()
                .mapToDouble(getter)
                .filter(Double::isFinite)
                .average()
                .orElse(Double.NaN);
    }

    private static String formatDouble(double value) {
        return Double.isFinite(value) ? String.format(java.util.Locale.US, "%.3f", value) : "NaN";
    }

    private static List<Node> getCoExistingLineages(Node node, Tree tree) {
        return Arrays.stream(tree.getNodesAsArray())
                .filter(candidate -> !candidate.isRoot())
                .filter(candidate -> candidate.getParent().getHeight() > node.getHeight())
                .collect(Collectors.toList());
    }

    private static List<Integer> collectAncestorsAmongCandidates(Node parent, List<Node> candidates) {
        List<Integer> ancestors = new ArrayList<>();
        Node ancestor = parent;
        while (ancestor != null) {
            if (candidates.contains(ancestor)) {
                ancestors.add(ancestor.getNr());
            }
            ancestor = ancestor.getParent();
        }
        return ancestors;
    }

    private static AttachmentInterval getAttachmentInterval(Node movingNode, Node candidate, double maxHeightRatio) {
        double minAge = Math.max(movingNode.getHeight(), candidate.getHeight());
        double maxAge = candidate.getParent().getHeight();
        if (Double.isFinite(maxHeightRatio)) {
            double movingParentHeight = movingNode.getParent().getHeight();
            minAge = Math.max(minAge, movingParentHeight / maxHeightRatio);
            maxAge = Math.min(maxAge, movingParentHeight * maxHeightRatio);
        }
        return new AttachmentInterval(minAge, maxAge);
    }

    private static double sampleAttachmentAge(AttachmentInterval interval, Random random) {
        double range = interval.range();
        if (range <= 0.0) {
            return interval.minAge;
        }
        double epsilon = Math.max(1e-10, 1e-6 * range);
        double lower = interval.minAge + epsilon;
        double upper = interval.maxAge - epsilon;
        if (lower >= upper) {
            return 0.5 * (interval.minAge + interval.maxAge);
        }
        return lower + random.nextDouble() * (upper - lower);
    }

    private static Node getOtherChild(Node parent, Node child) {
        return parent.getLeft() == child ? parent.getRight() : parent.getLeft();
    }

    private static void replace(Node node, Node child, Node replacement) {
        node.removeChild(child);
        node.addChild(replacement);
    }

    private static String nodeId(Node node) {
        return node.getID() != null ? node.getID() : "node-" + node.getNr();
    }

    private record AttachmentInterval(double minAge, double maxAge) {
        double range() {
            return Math.max(0.0, maxAge - minAge);
        }
    }

    private record ScoredTargets(double[] targetWeights, double[] selectionMass) {
    }

    private record RowDraft(
            int sourceNr,
            String sourceId,
            double sourceHeight,
            double sourceLikelihoodWeight,
            double sourceHeuristicWeight,
            double sourceCombinedWeight,
            int candidateNr,
            String candidateId,
            double candidateHeight,
            double attachmentAge,
            double intervalMin,
            double intervalMax,
            double exactPriorDelta,
            double likelihoodTargetWeight,
            double heuristicTargetWeight,
            double combinedTargetWeight,
            double likelihoodSelectionMass,
            double heuristicSelectionMass,
            double combinedSelectionMass) {

        Row toRow(int exactRank, int likelihoodRank, int heuristicRank, int combinedRank) {
            return new Row(
                    sourceNr,
                    sourceId,
                    sourceHeight,
                    sourceLikelihoodWeight,
                    sourceHeuristicWeight,
                    sourceCombinedWeight,
                    candidateNr,
                    candidateId,
                    candidateHeight,
                    attachmentAge,
                    intervalMin,
                    intervalMax,
                    exactPriorDelta,
                    likelihoodTargetWeight,
                    heuristicTargetWeight,
                    combinedTargetWeight,
                    likelihoodSelectionMass,
                    heuristicSelectionMass,
                    combinedSelectionMass,
                    exactRank,
                    likelihoodRank,
                    heuristicRank,
                    combinedRank);
        }
    }
}