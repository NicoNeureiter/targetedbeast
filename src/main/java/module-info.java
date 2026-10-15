open module targetedbeast {
    requires beast.pkgmgmt;
    requires beast.base;
    // targetedbeast.annotator.CladeAnnotator, targetedbeast.alignment.ConsensusAlignment
    requires beast.fx;
    requires beast.labs;                 // RNNIMetric in TreeJumpFunctionTest

    requires java.desktop;               // JOptionPane in targetedbeast.annotator
    requires colt;                       // SVD in targetedbeast.util.Alignment2PCA
    requires commons.math3;              // linear algebra in targetedbeast.edgeweights.PCAWeights

    exports targetedbeast.alignment;
    exports targetedbeast.annotator;
    exports targetedbeast.benchmark;
    exports targetedbeast.coalescent;
    exports targetedbeast.edgeweights;
    exports targetedbeast.edgeweights.learnedweights;
    exports targetedbeast.init;
    exports targetedbeast.likelihood;
    exports targetedbeast.logger;
    exports targetedbeast.operators;
    exports targetedbeast.operatorschedule;
    exports targetedbeast.util;
    exports targetedbeast.weighttransform;

    // targetedbeast.altoperators is deliberately neither exported nor provided:
    // unfinished scratch code, unregistered in version.xml upstream.
    provides beast.base.core.BEASTInterface with
        targetedbeast.alignment.ConsensusAlignment,
        targetedbeast.alignment.RapidAlignment,
        targetedbeast.benchmark.BenchmarkMCMC,
        targetedbeast.coalescent.RapidCoalescent,
        targetedbeast.coalescent.RapidTreeIntervals,
        targetedbeast.edgeweights.CombinedEdgeWeights,
        targetedbeast.edgeweights.ConsensusWeights,
        targetedbeast.edgeweights.ConstantWeights,
        targetedbeast.edgeweights.FelsensteinWeights,
        targetedbeast.edgeweights.FelsensteinWeights2,
        targetedbeast.edgeweights.LearnedWeights,
        targetedbeast.edgeweights.PCAWeights,
        targetedbeast.edgeweights.PCAWeights2,
        targetedbeast.edgeweights.PCAWeightsBrownian,
        targetedbeast.edgeweights.PCAWeightsBrownianFullLikelihood,
        targetedbeast.edgeweights.ParsimonyWeights,
        targetedbeast.edgeweights.ParsimonyWeights2,
        targetedbeast.edgeweights.PartialWeights,
        targetedbeast.edgeweights.ThreadedPartialWeights,
        targetedbeast.edgeweights.SourceTargetWeights,
        targetedbeast.init.ensureNoZeroEdgeLength,
        targetedbeast.likelihood.RapidGenericTreeLikelihood,
        targetedbeast.likelihood.RapidTreeLikelihood,
        targetedbeast.likelihood.SlowTreeLikelihood,
        targetedbeast.logger.CladeLogger,
        targetedbeast.operators.AdaptableOperatorSampler,
        targetedbeast.operators.BactrianIntervalScaleOperator,
        targetedbeast.operators.HeightBasedNodeRandomizer,
        targetedbeast.operators.IntervalScaleOperator,
        targetedbeast.operators.NodeRandomizer2,
        targetedbeast.operators.NodeRandomizer22,
        targetedbeast.operators.RangeSlide,
        targetedbeast.operators.TargetedWilsonBalding,
        targetedbeast.operators.TargetedWilsonBaldingFixedHeight,
        targetedbeast.operators.TargetedWilsonBaldingHeightDelta,
        targetedbeast.operators.WeightBasedNodeRandomizer,
        targetedbeast.operators.WeightedWideOperator,
        targetedbeast.operatorschedule.TargetedOperatorSchedule,
        targetedbeast.util.Alignment2PCA,
        targetedbeast.weighttransform.ExpOffsetTransform,
        targetedbeast.weighttransform.GridTuner,
        targetedbeast.weighttransform.PowerOffsetTransform,
        targetedbeast.weighttransform.RatioGridTuner,
        targetedbeast.weighttransform.SpsaTuner,
        targetedbeast.weighttransform.TreeJumpFunction;
}
