from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def interval(values: pd.Series) -> tuple[float, float, float]:
    values = values.astype(float)
    mean = float(values.mean())
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CBC estimator interactions and MIC record-quality contrasts.")
    parser.add_argument("--benchmark-by-seed", type=Path, required=True)
    parser.add_argument("--replicate-groups", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = pd.read_csv(args.benchmark_by_seed)
    labels = {
        "LightGBM": ("Morgan + LightGBM", "Multi-view + LightGBM"),
        "Random Forest": ("Morgan + Random Forest", "Multi-view + Random Forest"),
        "XGBoost": ("Morgan + XGBoost", "Multi-view + XGBoost"),
    }
    deltas = []
    for estimator, (morgan, multiview) in labels.items():
        subset = benchmark[benchmark.pipeline.isin([morgan, multiview])].pivot(index="seed", columns="pipeline", values="mae")
        for seed, row in subset.iterrows():
            deltas.append({"estimator": estimator, "seed": int(seed), "multiview_minus_morgan_mae": row[multiview] - row[morgan]})
    delta_frame = pd.DataFrame(deltas).sort_values(["estimator", "seed"])
    interaction_rows = []
    for first, second in (("Random Forest", "LightGBM"), ("Random Forest", "XGBoost"), ("XGBoost", "LightGBM")):
        paired = delta_frame.pivot(index="seed", columns="estimator", values="multiview_minus_morgan_mae")
        values = paired[first] - paired[second]
        mean, lower, upper = interval(values)
        interaction_rows.append({
            "interaction": f"{first} minus {second}",
            "splits": len(values),
            "mean_difference_in_representation_effect": mean,
            "stability_interval_lower": lower,
            "stability_interval_upper": upper,
            "positive_splits": int((values > 0).sum()),
        })
    interaction_frame = pd.DataFrame(interaction_rows)

    groups = pd.read_csv(args.replicate_groups)
    groups = groups[groups.model.isin(["morgan", "multiview"])]
    paired = groups.pivot(index=["seed", "replicate_group"], columns="model", values="mae").reset_index()
    paired["multiview_minus_morgan_mae"] = paired["multiview"] - paired["morgan"]
    quality_rows = []
    for group, subset in paired.groupby("replicate_group", sort=True):
        mean, lower, upper = interval(subset.multiview_minus_morgan_mae)
        quality_rows.append({
            "record_group": group,
            "splits": len(subset),
            "mean_multiview_minus_morgan_mae": mean,
            "stability_interval_lower": lower,
            "stability_interval_upper": upper,
            "multiview_favorable_splits": int((subset.multiview_minus_morgan_mae < 0).sum()),
        })
    quality_frame = pd.DataFrame(quality_rows)

    delta_frame.to_csv(args.output_dir / "estimator_representation_effect_by_seed.csv", index=False)
    interaction_frame.to_csv(args.output_dir / "estimator_representation_interaction.csv", index=False)
    paired.to_csv(args.output_dir / "mic_record_quality_representation_effect_by_seed.csv", index=False)
    quality_frame.to_csv(args.output_dir / "mic_record_quality_representation_effect.csv", index=False)
    manifest = {
        "benchmark_by_seed": {"path": str(args.benchmark_by_seed.resolve()), "sha256": file_hash(args.benchmark_by_seed)},
        "replicate_groups": {"path": str(args.replicate_groups.resolve()), "sha256": file_hash(args.replicate_groups)},
        "interval": "two-sided t interval over five shared scaffold-split values; descriptive stability summary",
        "interaction": "(multi-view minus Morgan MAE for first estimator) minus the same contrast for second estimator",
        "record_quality": "within-seed multi-view minus Morgan LightGBM MAE in pre-existing endpoint record-count/IQR groups",
    }
    (args.output_dir / "cbc_interaction_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
