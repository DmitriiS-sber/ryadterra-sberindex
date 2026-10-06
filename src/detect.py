"""Sequential change detectors: independent synthetic validation and unlabelled real alerts."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from config import DEFAULT_CONFIG, load_config
from forecast import past_fill, pooled_ridge

ROOT = Path(__file__).resolve().parents[1]


def scores(z: np.ndarray, warmup: int, cfg: dict) -> dict[str, np.ndarray]:
    """Each score at t uses observations through t only. Input: standardised innovations."""
    n, T = z.shape
    out = {m: np.zeros((n, T)) for m in ["cusum", "page_hinkley", "ewma", "shewhart"]}
    cp = np.zeros(n)
    cm = np.zeros(n)
    php = np.zeros(n)
    phm = np.zeros(n)
    lo = np.zeros(n)
    lm = np.zeros(n)
    ew = np.zeros(n)
    running_sum = np.zeros(n)
    reference_count = cfg.get("reference_count", 12)
    observed_count = np.zeros(n)
    # Fixed reference 0 is appropriate for forecast innovations; no future centring.
    for t in range(warmup, T):
        valid = np.isfinite(z[:, t])
        x = np.where(valid, z[:, t], 0.0)
        # Missing observations leave every state and the PH sample count unchanged.
        cp = np.where(valid, np.maximum(0, cp + x - cfg["cusum_drift"]), cp)
        cm = np.where(valid, np.maximum(0, cm - x - cfg["cusum_drift"]), cm)
        out["cusum"][:, t] = np.maximum(cp, cm)
        running_sum += x
        observed_count += valid
        running_mean = running_sum / (reference_count + observed_count)
        php += np.where(valid, x - running_mean - cfg["page_hinkley_drift"], 0)
        phm += np.where(valid, running_mean - x - cfg["page_hinkley_drift"], 0)
        lo = np.where(valid, np.minimum(lo, php), lo)
        lm = np.where(valid, np.minimum(lm, phm), lm)
        out["page_hinkley"][:, t] = np.maximum(php - lo, phm - lm)
        ew = np.where(valid, (1 - cfg["ewma_alpha"]) * ew + cfg["ewma_alpha"] * x, ew)
        out["ewma"][:, t] = np.abs(ew)
        out["shewhart"][:, t] = np.where(valid, np.abs(x), np.nan)
    return out


def synthetic(rng, n, kind, T=60, change=24, phi=0.3, magnitude=2.0, missing=0.0):
    """Generate AR(1) innovations with a known shift or a null disturbance."""
    innovations = rng.normal(size=(n, T))
    z = np.zeros_like(innovations)
    for t in range(T):
        z[:, t] = (phi * z[:, t - 1] if t else 0) + np.sqrt(1 - phi**2) * innovations[
            :, t
        ]
    direction = rng.choice([-1, 1], size=n)
    if kind == "step":
        z[:, change:] += magnitude * direction[:, None]
    if kind == "ramp":
        z[:, change:] += direction[:, None] * np.minimum(
            np.arange(T - change) * 0.15, 2
        )
    if kind == "seasonal_null":
        z += 0.8 * np.sin(2 * np.pi * np.arange(T) / 12)[None, :]
    if kind == "outlier_null":
        times = rng.integers(12, T, size=n)
        z[np.arange(n), times] += direction * 5
    if missing:
        z[rng.random(z.shape) < missing] = np.nan
    return z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/detection")
    parser.add_argument(
        "--monitor-months",
        type=int,
        default=None,
        help="Synthetic calibration length for the real monitoring window",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["detectors"]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    calibration = synthetic(
        np.random.default_rng(cfg["calibration_seed"]),
        cfg["synthetic_null_calibration_series"],
        "null",
    )
    calibr_scores = scores(calibration, cfg["warmup"], cfg)
    thresholds = {
        m: float(
            np.quantile(
                s[:, cfg["warmup"] :].max(axis=1),
                1 - cfg["alarm_false_positive_target"],
                method="higher",
            )
        )
        for m, s in calibr_scores.items()
    }
    rng = np.random.default_rng(cfg["evaluation_seed"])
    rows = []
    examples = []
    for kind in ["null", "step", "ramp", "seasonal_null", "outlier_null"]:
        z = synthetic(rng, cfg["synthetic_test_series_per_kind"], kind)
        for model, s in scores(z, cfg["warmup"], cfg).items():
            alarms = s[:, cfg["warmup"] :] > thresholds[model]
            has = alarms.any(axis=1)
            first = np.where(has, alarms.argmax(axis=1) + cfg["warmup"], 60)
            is_change = kind in ["step", "ramp"]
            ok = (first >= 24) & (first < 36) if is_change else np.zeros(len(z), bool)
            rows.append(
                dict(
                    kind=kind,
                    model=model,
                    n=len(z),
                    threshold=thresholds[model],
                    any_alarm_pct=100 * has.mean(),
                    pre_change_alarm_pct=100 * (first < 24).mean(),
                    recall_12m_pct=100 * ok.mean() if is_change else None,
                    median_delay_months=(
                        float(np.median(first[ok] - 24)) if ok.any() else None
                    ),
                )
            )
            for i in range(len(z)):
                examples.append(
                    dict(
                        kind=kind,
                        model=model,
                        series=i,
                        first_alarm=int(first[i]) if has[i] else None,
                        change_month=24 if is_change else None,
                    )
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "synthetic_metrics.csv", index=False)
    pd.DataFrame(examples).to_csv(out / "synthetic_first_alarms.csv", index=False)
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    # Real alerts use expanding one-month pooled forecasts, with no fitted news effect.
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    values = panel.to_numpy(float)
    dates = pd.to_datetime(panel.columns)
    news = pd.read_csv(ROOT / "data/processed/monthly_news_features.csv").set_index(
        "date"
    )
    news = news.loc[
        panel.columns,
        ["key_rate", "hikes_3m_pp", "forward_tightening", "demand_pressure"],
    ]
    residual = np.full_like(values, np.nan)
    prediction = np.full_like(values, np.nan)
    for target in range(7, len(dates)):
        history = past_fill(values[:, :target])
        p = pooled_ridge(history, dates[:target], news, target - 1, 1, False, config)[
            :, 0
        ]
        prediction[:, target] = p
        residual[:, target] = np.log(values[:, target] / p)
    # Five training residuals Aug-Dec 2023; this short calibration is explicitly a limitation.
    base = residual[:, 7:12]
    centre = np.nanmedian(base, axis=1)
    scale = 1.4826 * np.nanmedian(np.abs(base - centre[:, None]), axis=1)
    floor = float(np.nanmedian(scale))
    scale = np.maximum(scale, max(floor, 0.02))
    z = (residual[:, 12:] - centre[:, None]) / scale[:, None]
    # Recalibrate for the actual monitoring length and the same PH reference count.
    monitor_months = args.monitor_months or config.get("validation", {}).get(
        "real_monitoring_months", 12
    )
    if monitor_months != z.shape[1]:
        raise ValueError(
            "monitor-months must match the length of the real monitoring panel"
        )
    real_calibration = synthetic(
        np.random.default_rng(cfg["calibration_seed"] + 100),
        cfg["synthetic_null_calibration_series"],
        "null",
        T=z.shape[1],
    )
    real_thresholds = {
        m: float(
            np.quantile(
                np.nanmax(s, axis=1),
                1 - cfg["alarm_false_positive_target"],
                method="higher",
            )
        )
        for m, s in scores(real_calibration, 0, cfg).items()
    }
    (out / "real_thresholds.json").write_text(json.dumps(real_thresholds, indent=2))
    real_test_n = config.get("validation", {}).get("real_null_test_series", 500)
    real_null = synthetic(
        np.random.default_rng(cfg["evaluation_seed"] + 100),
        real_test_n,
        "null",
        T=z.shape[1],
    )
    pd.DataFrame(
        [
            dict(
                model=m,
                monitoring_months=z.shape[1],
                n=real_test_n,
                any_alarm_pct=100 * np.mean(np.any(s > real_thresholds[m], axis=1)),
            )
            for m, s in scores(real_null, 0, cfg).items()
        ]
    ).to_csv(out / "real_protocol_null_test.csv", index=False)
    real_scores = scores(z, 0, cfg)
    alerts = []
    for model, s in real_scores.items():
        hit = (s > real_thresholds[model]) & np.isfinite(z)
        for i in range(len(panel)):
            if hit[i].any():
                t = int(np.flatnonzero(hit[i])[0])
                idx = 12 + t
                alerts.append(
                    dict(
                        territory_id=int(panel.index[i]),
                        model=model,
                        first_alarm=panel.columns[idx],
                        signal_observation_month=panel.columns[idx],
                        published_at=None,
                        issued_at=None,
                        economic_onset=None,
                        score=float(s[i, t]),
                        threshold=real_thresholds[model],
                        actual_rub=float(values[i, idx]),
                        forecast_rub=float(prediction[i, idx]),
                        innovation_pct=float(
                            (values[i, idx] / prediction[i, idx] - 1) * 100
                        ),
                    )
                )
    pd.DataFrame(alerts).to_csv(out / "real_alerts.csv", index=False)
    rec = []
    for i, tid in enumerate(panel.index):
        for t in range(12, 24):
            rec.append(
                dict(
                    territory_id=int(tid),
                    date=panel.columns[t],
                    actual=values[i, t],
                    prediction=prediction[i, t],
                    innovation_log=residual[i, t],
                    standardised_innovation=z[i, t - 12],
                    **{m + "_score": s[i, t - 12] for m, s in real_scores.items()},
                )
            )
    pd.DataFrame(rec).to_parquet(out / "real_innovations.parquet", index=False)
    metadata = dict(
        calibration_seed=cfg["calibration_seed"],
        test_seed=cfg["evaluation_seed"],
        calibration_series=len(calibration),
        test_series_each=cfg["synthetic_test_series_per_kind"],
        length_months=60,
        warmup_months=12,
        change_month_index=24,
        noise="AR(1), phi=0.3, marginal sigma=1",
        step_size_sigma=2,
        ramp_sigma_per_month=0.15,
        ramp_cap_sigma=2,
        threshold_target="5% probability of any alarm across 48 monitored null months",
        real_scale_floor_log=floor,
        real_reference="5 residuals Aug-Dec 2023, median and MAD",
        reference_count=cfg.get("reference_count", 12),
        missing_policy="skip state update and PH count; no alarm on missing actual",
        real_monitoring_months=z.shape[1],
        real_threshold_calibration="Independent synthetic null of matching 12-month length, warmup=0; not a measured real FPR",
        real_alarm_interpretation="Unlabelled candidate shifts. No validated real false-positive rate or causal attribution.",
    )
    (out / "detection_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)
    )
    print(metrics.to_string(index=False), flush=True)
    print("REAL_ALERTS", len(alerts), flush=True)


if __name__ == "__main__":
    main()
