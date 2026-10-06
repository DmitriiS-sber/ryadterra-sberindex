"""Reproducible retrospective diagnostics added in response to independent validation.

No model is selected, retrained or tuned by this module. Bootstrap intervals
describe the existing sample and are not reliable temporal inference with four origins.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from detect import scores, synthetic
from config import load_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/validation"


def metric(g):
    error = g.prediction - g.actual
    denominator = ((g.actual - g.actual.mean()) ** 2).sum()
    within = (
        (g.actual - g.groupby("territory_id").actual.transform("mean")) ** 2
    ).sum()
    return dict(
        n=len(g),
        territories=g.territory_id.nunique(),
        origins=g.origin.nunique(),
        mae_rub=float(error.abs().mean()),
        rmse_rub=float(np.sqrt((error**2).mean())),
        r2=float(1 - (error**2).sum() / denominator) if denominator else None,
        within_territory_r2=(
            float(1 - (error**2).sum() / within) if within > 1e-9 else None
        ),
        wape_pct=float(100 * error.abs().sum() / g.actual.abs().sum()),
        mean_ape_pct=float(100 * (error.abs() / g.actual).mean()),
    )


def grouped(frame, keys):
    return pd.DataFrame(
        [
            dict(zip(keys, key if isinstance(key, tuple) else (key,)), **metric(g))
            for key, g in frame.groupby(keys, sort=True)
        ]
    )


def forecast_diagnostics(pred, panel):
    grouped(pred, ["model", "horizon"]).to_csv(
        OUT / "forecast_by_horizon.csv", index=False
    )
    grouped(pred, ["model", "origin", "horizon"]).to_csv(
        OUT / "forecast_by_origin_horizon.csv", index=False
    )
    territory = grouped(pred, ["model", "horizon", "territory_id"])
    territory.to_csv(
        OUT / "forecast_by_territory.csv.gz", index=False, compression="gzip"
    )
    denominator = panel.loc[:, "2023-01":"2023-12"].diff(axis=1).abs().mean(axis=1)
    rows = []
    for (model, h), g in territory.groupby(["model", "horizon"]):
        scale = g.territory_id.map(denominator)
        valid = scale.gt(0) & np.isfinite(scale)
        rows.append(
            dict(
                model=model,
                horizon=h,
                n_territories=len(g),
                equal_territory_mae_rub=float(g.mae_rub.mean()),
                mase_lag1_equal_territories=float(
                    (g.mae_rub[valid] / scale[valid]).mean()
                ),
                mase_undefined_count=int((~valid).sum()),
                median_territory_mae=float(g.mae_rub.median()),
                p90_territory_mae=float(g.mae_rub.quantile(0.9)),
                p99_territory_mae=float(g.mae_rub.quantile(0.99)),
            )
        )
    pd.DataFrame(rows).to_csv(OUT / "equal_territory_metrics.csv", index=False)
    selection = pd.read_csv(ROOT / "data/processed/selection.csv").set_index(
        "territory_id"
    )
    q = pred.copy()
    q["baseline_group"] = q.territory_id.map(selection.baseline_group)
    grouped(q, ["model", "horizon", "baseline_group"]).to_csv(
        OUT / "forecast_by_quartile.csv", index=False
    )
    comparisons = []
    for h in [1, 3, 6, 12]:
        reference = "damped_trend" if h == 12 else "last_value"
        wide = territory[territory.horizon.eq(h)].pivot(
            index="territory_id", columns="model", values="mae_rub"
        )
        diff = wide.prophet_auto - wide[reference]
        comparisons.append(
            dict(
                horizon=h,
                reference=reference,
                n=len(diff),
                territories_prophet_worse_pct=float(100 * diff.gt(0).mean()),
                mean_equal_territory_difference=float(diff.mean()),
            )
        )
    pd.DataFrame(comparisons).to_csv(OUT / "territory_comparison.csv", index=False)


def cluster_intervals(pred, iterations=2000, seed=20261004):
    rows = []
    for h in [1, 3, 6, 12]:
        for reference in ["last_value", "prophet_default", "damped_trend"]:
            subset = pred[pred.horizon.eq(h)].copy()
            subset["error"] = (subset.prediction - subset.actual).abs()
            keys = ["territory_id", "origin"]
            left = subset[subset.model.eq("prophet_auto")].set_index(keys).error
            right = subset[subset.model.eq(reference)].set_index(keys).error
            difference = (left - right).unstack("origin")
            values = np.nan_to_num(difference.to_numpy())
            valid = difference.notna().to_numpy()
            n, o = values.shape
            lo = hi = None
            if o > 1:
                rng = np.random.default_rng(seed + h)
                boot = []
                for _ in range(iterations):
                    wt = rng.multinomial(n, np.full(n, 1 / n))
                    wo = rng.multinomial(o, np.full(o, 1 / o))
                    weights = wt[:, None] * wo[None, :]
                    total = (weights * valid).sum()
                    boot.append(float((weights * values).sum() / total))
                lo, hi = np.quantile(boot, [0.025, 0.975])
            rows.append(
                dict(
                    horizon=h,
                    model="prophet_auto",
                    reference=reference,
                    difference_mae_rub=float((left - right).mean()),
                    conditional_lower_95=lo,
                    conditional_upper_95=hi,
                    origins=o,
                    replications=iterations if o > 1 else 0,
                    interpretation=(
                        "descriptive conditional interval; very few time clusters"
                        if o > 1
                        else "temporal uncertainty not estimable from one origin"
                    ),
                )
            )
    pd.DataFrame(rows).to_csv(OUT / "conditional_cluster_intervals.csv", index=False)


def wilson(k, n):
    z = norm.ppf(0.975)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (centre - radius), 100 * (centre + radius)


def detector_stress():
    config = load_config(ROOT / "configs/validation.yaml")
    cfg = config["detectors"]
    v = config["validation"]
    thresholds = json.loads((ROOT / "results/detection/thresholds.json").read_text())
    rows = []
    rng = np.random.default_rng(v["stress_seed"])
    for phi in v["stress_phi"]:
        for kind, magnitude in [
            ("null", 0),
            ("step", 0.5),
            ("step", 1),
            ("step", 2),
            ("seasonal_null", 0),
            ("outlier_null", 0),
        ]:
            for missing in v["missing_fractions"]:
                z = synthetic(
                    rng,
                    v["stress_series"],
                    kind,
                    phi=phi,
                    magnitude=magnitude,
                    missing=missing,
                )
                for method, s in scores(z, 12, cfg).items():
                    for multiplier in v["threshold_multipliers"]:
                        hits = (s > thresholds[method] * multiplier) & np.isfinite(z)
                        hits[:, :12] = False
                        has = hits.any(axis=1)
                        first = np.where(has, hits.argmax(axis=1), 60)
                        success = (first >= 24) & (first < 36)
                        low, high = wilson(int(has.sum()), len(z))
                        rows.append(
                            dict(
                                phi=phi,
                                kind=kind,
                                magnitude_sigma=magnitude,
                                missing_fraction=missing,
                                method=method,
                                threshold_multiplier=multiplier,
                                n=len(z),
                                any_alarm_pct=100 * has.mean(),
                                alarm_wilson_lower_pct=low,
                                alarm_wilson_upper_pct=high,
                                recall_12m_pct=(
                                    100 * success.mean() if kind == "step" else None
                                ),
                                median_delay_months=(
                                    float(np.median(first[success] - 24))
                                    if kind == "step" and success.any()
                                    else None
                                ),
                            )
                        )
    pd.DataFrame(rows).to_csv(OUT / "detector_stress_grid.csv", index=False)


def data_diagnostics(panel):
    missing = panel.isna().stack()
    missing = missing[missing].reset_index()
    missing.columns = ["territory_id", "observation_month", "is_missing"]
    missing["reason"] = "unknown"
    missing.to_csv(OUT / "missing_cells.csv", index=False)
    raw = pd.read_parquet(ROOT / "data/raw/consumption.parquet")
    sha = hashlib.sha256(
        (ROOT / "data/raw/consumption.parquet").read_bytes()
    ).hexdigest()
    temporal = raw.rename(columns={"date": "observation_month"}).copy()
    temporal["published_at"] = pd.NaT
    temporal["retrieved_at"] = "2026-10-03"
    temporal["retrieval_precision"] = "day"
    temporal["vintage_id"] = "downloaded_snapshot_" + sha
    temporal["historical_availability"] = "unknown"
    temporal.to_parquet(ROOT / "data/processed/temporal_registry.parquet", index=False)
    (OUT / "data_availability.json").write_text(
        json.dumps(
            dict(
                rows=len(raw),
                source_sha256=sha,
                known_published_at=0,
                actual_release_calendar="not supplied",
                vintage_history="not supplied",
                retrieved_at="2026-10-03, from original project source manifest",
                territory_dictionary_status="not recovered; source IDs are not confirmed OKTMO",
                target="nominal model estimate of average monthly cashless spending; not total turnover",
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    pred = pd.concat(
        [
            pd.read_parquet(ROOT / "results/main/predictions.parquet"),
            pd.read_parquet(ROOT / "results/default_baseline/predictions.parquet"),
        ],
        ignore_index=True,
    )
    v = load_config(ROOT / "configs/validation.yaml")["validation"]
    forecast_diagnostics(pred, panel)
    cluster_intervals(pred, v["bootstrap_replications"], v["bootstrap_seed"])
    detector_stress()
    data_diagnostics(panel)
    versions = {}
    import importlib.metadata

    for package in ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow"]:
        versions[package] = importlib.metadata.version(package)
    (OUT / "run_manifest.json").write_text(
        json.dumps(
            dict(
                date="2026-10-04",
                input_forecasts_sha256={
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [
                        ROOT / "results/main/predictions.parquet",
                        ROOT / "results/default_baseline/predictions.parquet",
                    ]
                },
                retrospective=True,
                model_parameters_selected=False,
                package_versions=versions,
            ),
            indent=2,
        )
    )
    print("Saved diagnostics to", OUT)


if __name__ == "__main__":
    main()
