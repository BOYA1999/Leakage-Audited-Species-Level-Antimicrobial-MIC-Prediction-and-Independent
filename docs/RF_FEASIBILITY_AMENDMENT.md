# E01 Random Forest feasibility amendment

Date and decision time: 2026-09-11 01:37 Asia/Shanghai

The original full-run command requested 300 unrestricted-depth trees for each of five seeds and two representations with four CPU workers. It began at 2026-09-10 23:48. During the run, one observed forest-computation interval required approximately 51 minutes at close to four-core saturation and the process working set approached 4.3 GB. At the decision time, the output directory contained no metrics, predictions, summaries, runtime table, or manifest; no Random Forest performance result had been exposed or inspected.

For computational feasibility, the Random Forest baseline is amended before outcome inspection to 100 trees and all available job workers (`n_jobs=-1`). The estimator family, unrestricted depth, square-root feature subsampling, minimum leaf size, random seeds, Morgan and multi-view inputs, species one-hot features, species-balanced weights, scaffold partitions, targets, metrics, and paired analysis remain unchanged. The interrupted partial computation is not an experiment result and will not be reported.

This amendment makes the classical robustness check tractable; it does not constitute estimator-family hyperparameter optimization. Random Forest results remain interpretable only as a fixed task-matched pipeline, not the best attainable performance of that family.
