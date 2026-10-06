"""Meaningful numerical and temporal checks; exits nonzero on failure."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from forecast import past_fill, pooled_ridge, summarize
from detect import scores

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    checks = []
    raw = pd.read_parquet(ROOT / "data/raw/consumption.parquet")
    p = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    expected = raw[raw.category.eq("Все категории")].pivot(
        index="territory_id", columns="date", values="value"
    )
    eligible = expected.loc[:, :"2023-12"].notna().all(axis=1)
    assert p.index.equals(expected.loc[eligible].sort_index().index)
    checks.append("selection_uses_2023_only")
    a = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, np.nan]])
    np.testing.assert_equal(past_fill(a), [[1.0, 1.0, 3.0], [2.0, 4.0, 4.0]])
    checks.append("past_forward_fill")
    news = pd.read_csv(ROOT / "data/processed/monthly_news_features.csv").set_index(
        "date"
    )
    news = news.loc[
        p.columns, ["key_rate", "hikes_3m_pp", "forward_tightening", "demand_pressure"]
    ]
    history = p.iloc[:10, :12].to_numpy(float)
    dates = pd.to_datetime(p.columns[:12])
    f = pooled_ridge(history, dates, news, 11, 12, True, cfg)
    mutated = news.copy()
    mutated.iloc[12:] = 1e9
    np.testing.assert_allclose(
        f, pooled_ridge(history, dates, mutated, 11, 12, True, cfg), rtol=0, atol=0
    )
    checks.append("forecast_invariant_to_future_news")
    z = np.random.default_rng(42).normal(size=(3, 40))
    before = scores(z, 12, cfg["detectors"])
    z[:, 30:] = 1e9
    after = scores(z, 12, cfg["detectors"])
    for m in before:
        np.testing.assert_equal(before[m][:, :30], after[m][:, :30])
    checks.append("detector_invariant_to_future_observations")
    assert cfg["detectors"]["calibration_seed"] != cfg["detectors"]["evaluation_seed"]
    checks.append("independent_detection_calibration_and_test")
    pred = pd.read_parquet(ROOT / "results/main/predictions.parquet")
    assert not pred.duplicated(["territory_id", "origin", "horizon", "model"]).any()
    assert (
        np.isfinite(pred[["prediction", "actual"]]).all().all()
        and (pred.prediction > 0).all()
    )
    assert (
        pred.groupby(["territory_id", "origin", "horizon"])
        .model.nunique()
        .eq(len(cfg["models"]))
        .all()
    )
    checks.append("finite_positive_common_evaluation_sample")
    for g in pred.groupby(["origin", "horizon"]):
        m = g[1]
        assert m.groupby("model").actual.sum().nunique() == 1
    checks.append("identical_actuals_for_model_comparison")
    selection = pd.read_csv(ROOT / "data/processed/selection.csv")
    metrics = summarize(pred, selection)
    saved = pd.read_csv(ROOT / "results/main/metrics.csv")
    np.testing.assert_allclose(metrics.mae_rub, saved.mae_rub, rtol=1e-12)
    single = pred[pred.model.eq("chronos_bolt_tiny") & pred.horizon.eq(3)]
    manual = sum(
        abs(float(r.prediction) - float(r.actual)) for r in single.itertuples()
    ) / len(single)
    saved_mae = saved[
        saved.model.eq("chronos_bolt_tiny") & saved.scope.eq("horizon_3")
    ].mae_rub.iloc[0]
    assert abs(manual - saved_mae) < 1e-8
    checks.append("saved_metrics_match_independent_mae_sum")
    events = pd.read_csv(ROOT / "data/processed/news_events.csv")
    assert len(events) == 17
    daily = pd.read_csv(ROOT / "data/processed/keyrate_daily.csv", parse_dates=["date"])
    for event in events.itertuples():
        day = pd.Timestamp(event.publication_date)
        assert (
            daily[(daily.date >= day) & (daily.date <= day + pd.Timedelta(days=14))]
            .rate.eq(event.new_rate)
            .any()
        )
    checks.append("announced_rates_confirmed_by_daily_cbr_table")
    for i, row in pd.read_csv(
        ROOT / "data/processed/monthly_news_features.csv"
    ).iterrows():
        if isinstance(row.last_news_date, str):
            assert pd.Timestamp(row.last_news_date) <= pd.Period(row.date).end_time
    checks.append("news_publication_before_feature_origin")
    baseline = pd.read_parquet(ROOT / "results/default_baseline/predictions.parquet")
    reference = pred[pred.model.eq("prophet_auto")].sort_values(
        ["territory_id", "origin", "horizon"]
    )
    baseline = baseline.sort_values(["territory_id", "origin", "horizon"])
    np.testing.assert_equal(
        reference[["territory_id", "horizon", "actual"]].to_numpy(),
        baseline[["territory_id", "horizon", "actual"]].to_numpy(),
    )
    assert np.isfinite(baseline.prediction).all()
    checks.append("exact_default_baseline_has_same_test_pairs")
    latest = pd.read_csv(ROOT / "results/latest_forecast.csv")
    assert len(latest) == len(p) * 4 and latest.data_cutoff.eq("2024-12").all()
    assert latest.target_month.isin(["2025-01", "2025-03", "2025-06", "2025-12"]).all()
    assert np.isfinite(latest.forecast_rub).all() and latest.forecast_rub.gt(0).all()
    checks.append("exported_forecast_cutoff_horizons_and_values")
    warnings = pd.read_parquet(ROOT / "results/future_warning/cases.parquet")
    assert (
        warnings.groupby(["territory_id", "origin", "threshold_pct"])
        .proxy_event.nunique()
        .eq(1)
        .all()
    )
    wmetrics = pd.read_csv(ROOT / "results/future_warning/metrics.csv")
    assert (wmetrics[["tp", "fp", "fn", "tn"]].sum(axis=1) == wmetrics.n).all()
    assert (wmetrics.tp + wmetrics.fn == wmetrics.actual_events).all()
    check = wmetrics[
        wmetrics.model.eq("prophet_auto") & wmetrics.threshold_pct.eq(20)
    ].iloc[0]
    assert abs(check.f1 - 2 * check.tp / (2 * check.tp + check.fp + check.fn)) < 1e-12
    # One proxy label calculated directly from raw panel, separately from benchmark code.
    r = warnings[warnings.threshold_pct.eq(20)].iloc[0]
    i = list(p.columns).index(r.origin)
    ratio = (
        p.loc[r.territory_id].iloc[i + 1 : i + 4].to_numpy()
        / p.loc[r.territory_id].iloc[i - 11 : i - 8].to_numpy()
        - 1
    )
    assert bool(r.proxy_event) == bool(
        (ratio >= 0.2).sum() >= 2 or (ratio <= -0.2).sum() >= 2
    )
    checks.append("future_warning_labels_confusion_counts_and_f1")
    result = {"checks_completed": checks, "count": len(checks), "status": "passed"}
    (ROOT / "results/verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
