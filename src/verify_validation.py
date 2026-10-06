"""Regression checks for corrections requested by the independent validator."""

from pathlib import Path
import json, hashlib, tempfile
import numpy as np
import pandas as pd
from config import load_config
from detect import scores
from asof import asof_panel
from evaluate_events import evaluate
from prospective import validate_keys
import prospective
from validation_diagnostics import wilson

ROOT = Path(__file__).resolve().parents[1]


def rejected(fn):
    try:
        fn()
    except (ValueError, KeyError):
        return
    raise AssertionError("Expected explicit rejection")


def main():
    checks = []
    cfg = load_config(ROOT / "configs/validation.yaml")["detectors"]
    z = np.array([[2.0, np.nan, -1.0, 3.0, np.nan, 2.0]])
    full = scores(z, 0, cfg)
    compact = scores(z[:, np.isfinite(z[0])], 0, cfg)
    for method in full:
        np.testing.assert_allclose(
            full[method][:, np.isfinite(z[0])], compact[method], atol=0, rtol=0
        )
    checks.append("missing_updates_and_PH_counts_skip_exactly")
    for m in ["cusum", "page_hinkley", "ewma"]:
        assert full[m][0, 1] == full[m][0, 0] and full[m][0, 4] == full[m][0, 3]
    checks.append("states_frozen_during_missing_observations")
    changed = z.copy()
    changed[:, 3:] = 999
    for m, s in scores(changed, 0, cfg).items():
        np.testing.assert_equal(s[:, :3], full[m][:, :3])
    checks.append("detector_prefix_invariance_with_missing")
    paths = [
        ROOT / "results/main/predictions.parquet",
        ROOT / "results/default_baseline/predictions.parquet",
    ]
    manifest = json.loads((ROOT / "results/validation/run_manifest.json").read_text())
    original_hashes = json.loads(
        (ROOT / "results/validation/original_forecast_hashes.json").read_text()
    )
    for p in paths:
        assert (
            hashlib.sha256(p.read_bytes()).hexdigest()
            == manifest["input_forecasts_sha256"][str(p.relative_to(ROOT))]
        )
    checks.append("published_forecast_inputs_unchanged")
    pred = pd.read_parquet(paths[0])
    g = pred.query('model=="prophet_auto" and horizon==1').copy()
    g["error"] = (g.actual - g.prediction).abs()
    eq = (
        pd.read_csv(ROOT / "results/validation/equal_territory_metrics.csv")
        .query('model=="prophet_auto" and horizon==1')
        .iloc[0]
    )
    assert (
        abs(g.groupby("territory_id").error.mean().mean() - eq.equal_territory_mae_rub)
        < 1e-9
    )
    assert abs(g.error.mean() - eq.equal_territory_mae_rub) > 1
    checks.append("equal_territory_weighting_distinct_from_test_pair_weighting")
    within = (
        (g.actual - g.groupby("territory_id").actual.transform("mean")) ** 2
    ).sum()
    m = (
        pd.read_csv(ROOT / "results/validation/forecast_by_horizon.csv")
        .query('model=="prophet_auto" and horizon==1')
        .iloc[0]
    )
    assert abs(1 - (g.error**2).sum() / within - m.within_territory_r2) < 1e-12
    checks.append("within_territory_R2_independent_sum")
    cases = pd.read_parquet(ROOT / "results/future_warning/cases.parquet")
    d = pd.read_csv(ROOT / "results/future_warning/direction_metrics.csv")
    for row in d.query('status=="calculated"').itertuples():
        f = cases[
            (cases.model == row.model) & (cases.threshold_pct == row.threshold_pct)
        ]
        correct = (
            f.predicted_warning
            & f.proxy_event
            & f.actual_direction.eq(f.predicted_direction)
        )
        assert correct.sum() == row.tp_correct_direction
        assert (
            f.predicted_warning & ~correct
        ).sum() == row.fp_including_wrong_direction
        assert (f.proxy_event & ~correct).sum() == row.fn_including_wrong_direction
    assert (
        d.query('model=="macro_tightening_flag"')
        .status.eq("direction not predicted; excluded")
        .all()
    )
    checks.append("direction_correct_FP_FN_and_macro_exclusion")
    f = pd.DataFrame(
        [
            dict(
                territory_id=1,
                observation_month="2024-01",
                value=100,
                published_at="2024-02-05T00:00:00Z",
                retrieved_at="2024-02-06T00:00:00Z",
                vintage_id="v1",
            ),
            dict(
                territory_id=1,
                observation_month="2024-01",
                value=999,
                published_at="2024-03-05T00:00:00Z",
                retrieved_at="2024-03-06T00:00:00Z",
                vintage_id="v2",
            ),
        ]
    )
    assert asof_panel(f, "2024-02-15T00:00:00Z").iloc[0, 0] == 100
    assert asof_panel(f, "2024-03-15T00:00:00Z").iloc[0, 0] == 999
    checks.append("asof_excludes_future_revisions")
    rejected(lambda: asof_panel(f.assign(published_at=None), "2024-02-15T00:00:00Z"))
    rejected(lambda: asof_panel(f, "2024-02-15"))
    rejected(
        lambda: asof_panel(
            f.assign(retrieved_at="2024-02-01T00:00:00Z"), "2024-04-15T00:00:00Z"
        )
    )
    checks.append("unknown_timezone_and_impossible_retrieval_rejected")
    temporal = pd.read_parquet(ROOT / "data/processed/temporal_registry.parquet")
    assert temporal.published_at.isna().all()
    rejected(lambda: asof_panel(temporal, "2024-12-31T00:00:00Z"))
    checks.append("historical_unknown_availability_never_fabricated")
    events = pd.DataFrame(
        [
            dict(
                event_id="e1",
                territory_id=1,
                onset_at="2025-06-01T00:00:00Z",
                published_at="2025-07-01T00:00:00Z",
                direction=1,
                source_url="https://example.org/source",
                independent_review=True,
            )
        ]
    )
    warnings = pd.DataFrame(
        [
            dict(
                warning_id="w1",
                territory_id=1,
                issued_at="2025-05-01T00:00:00Z",
                direction=-1,
            )
        ]
    )
    roster = pd.DataFrame([dict(territory_id=1, independent_review=True)])
    args = (
        roster,
        "2025-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        "2026-02-01T00:00:00Z",
    )
    wrong = evaluate(events, warnings, *args)
    assert (wrong["tp"], wrong["fp"], wrong["fn"]) == (0, 1, 1)
    checks.append("wrong_direction_is_not_event_match")
    correct = evaluate(events, warnings.assign(direction=1), *args)
    assert correct["tp"] == 1 and correct["matches"][0]["lead_days"] == 31
    duplicate = pd.concat(
        [warnings.assign(direction=1), warnings.assign(warning_id="w2", direction=1)]
    )
    assert evaluate(events, duplicate, *args)["tp"] == 1
    checks.append("one_to_one_exante_event_matching_and_lead_time")
    rejected(lambda: evaluate(events.iloc[:0], warnings, *args))
    rejected(lambda: evaluate(events.assign(independent_review=False), warnings, *args))
    late = pd.concat(
        [warnings, warnings.assign(warning_id="w3", issued_at="2025-12-01T00:00:00Z")]
    )
    assert evaluate(events, late, *args)["censored_right_warnings"] == 1
    checks.append("unlabelled_accuracy_blocked_and_window_censoring_explicit")
    plan = pd.DataFrame(
        [dict(territory_id=1, origin="2026-10", horizon=3, target_month="2027-01")]
    )
    validate_keys(plan)
    rejected(lambda: validate_keys(plan.assign(target_month="2026-12")))
    checks.append("prospective_horizon_target_consistency")
    low, high = wilson(37, 500)
    assert low < 7.4 < high and low > 0 and high < 100
    checks.append("Wilson_any_alarm_bounds")
    ab = pd.read_csv(ROOT / "results/validation/news_detector_ablation.csv")
    real = pd.read_csv(ROOT / "results/detection/real_alerts.csv")
    for row in ab[~ab.use_news].itertuples():
        assert row.territories_with_candidate_alert == (real.model == row.method).sum()
    checks.append("paired_no_news_arm_matches_real_detector_pipeline")
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        previous = (prospective.LOCK, prospective.PLAN, prospective.SCORE)
        prospective.LOCK = directory / "lock.json"
        prospective.PLAN = directory / "plan.json"
        prospective.SCORE = directory / "score.json"
        try:
            prospective.LOCK.write_text(
                json.dumps(
                    dict(
                        frozen_at="2025-01-01T00:00:00Z",
                        code_sha256=prospective.code_hashes(),
                        model="prophet_auto",
                    )
                )
            )
            keys = dict(
                territory_id=1, origin="2025-03", horizon=1, target_month="2025-04"
            )
            prospective.PLAN.write_text(
                json.dumps(dict(registered_at="2025-02-01T00:00:00Z", keys=[keys]))
            )
            p = directory / "p.csv"
            y = directory / "y.csv"
            output = directory / "out.json"
            forecasts = pd.DataFrame(
                [
                    {
                        **keys,
                        "prediction": 100.0,
                        "model": "prophet_auto",
                        "issued_at": "2025-03-30T00:00:00Z",
                    }
                ]
            )
            forecasts.to_csv(p, index=False)
            actuals = pd.DataFrame(
                [
                    dict(
                        territory_id=1,
                        target_month="2025-04",
                        actual=110.0,
                        published_at="2025-05-10T00:00:00Z",
                        vintage_id="v1",
                    )
                ]
            )
            actuals.to_csv(y, index=False)
            actuals.assign(target_month="2025-06").to_csv(y, index=False)
            rejected(lambda: prospective.evaluate(p, y, output))
            checks.append("prospective_incomplete_registered_targets_rejected")
            actuals.to_csv(y, index=False)
            forecasts.assign(issued_at="2025-04-02T00:00:00Z").to_csv(p, index=False)
            rejected(lambda: prospective.evaluate(p, y, output))
            checks.append("prospective_late_issue_rejected")
            forecasts.to_csv(p, index=False)
            prospective.evaluate(p, y, output)
            assert json.loads(output.read_text())["metrics"][0]["mae_rub"] == 10.0
            rejected(
                lambda: prospective.evaluate(p, y, directory / "another_output.json")
            )
            checks.append("prospective_score_global_single_use_guard")
        finally:
            prospective.LOCK, prospective.PLAN, prospective.SCORE = previous
    result = dict(status="passed", count=len(checks), checks_completed=checks)
    (ROOT / "results/validation/verification.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
