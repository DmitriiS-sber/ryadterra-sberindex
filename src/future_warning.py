"""Exploratory ex-ante warning benchmark using an explicit proxy, not shock ground truth.

At each origin warn if at least two of the next three forecast months exceed
the known same-month-last-year level by 20%, or fall below it by 20%.
The same definition applied to future actuals supplies the retrospective proxy label.
The economic threshold is fixed here without optimisation on test outcomes.
"""

import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    paths = pd.concat(
        [
            pd.read_parquet(ROOT / "results/main/forecast_paths.parquet"),
            pd.read_parquet(ROOT / "results/default_baseline/forecast_paths.parquet"),
        ]
    )
    news = pd.read_csv(ROOT / "data/processed/monthly_news_features.csv").set_index(
        "date"
    )
    rows = []
    for origin_label in sorted(paths.origin.unique()):
        i = list(panel.columns).index(origin_label)
        future_dates = panel.columns[i + 1 : i + 4]
        previous_dates = panel.columns[i - 11 : i - 8]
        future = panel.loc[:, future_dates]
        previous = panel.loc[:, previous_dates]
        valid = np.isfinite(future).all(axis=1) & np.isfinite(previous).all(axis=1)
        actual = future.loc[valid].to_numpy(float)
        baseline = previous.loc[valid].to_numpy(float)
        actual_ratio = actual / baseline - 1
        for model in sorted(paths.model.unique()) + ["macro_tightening_flag"]:
            if model == "macro_tightening_flag":
                r = news.loc[origin_label]
                predicted_ratio = None
                warning = np.full(
                    len(actual), bool(r.forward_tightening or r.hikes_3m_pp >= 2)
                )
            else:
                subset = paths[
                    (paths.origin == origin_label)
                    & (paths.model == model)
                    & paths.target_date.isin(future_dates)
                ]
                forecast = (
                    subset.pivot(
                        index="territory_id", columns="target_date", values="prediction"
                    )
                    .loc[panel.index[valid], future_dates]
                    .to_numpy(float)
                )
                predicted_ratio = forecast / baseline - 1
            for threshold in [0.1, 0.2, 0.3]:
                y = ((actual_ratio >= threshold).sum(axis=1) >= 2) | (
                    (actual_ratio <= -threshold).sum(axis=1) >= 2
                )
                p = (
                    warning
                    if predicted_ratio is None
                    else (
                        ((predicted_ratio >= threshold).sum(axis=1) >= 2)
                        | ((predicted_ratio <= -threshold).sum(axis=1) >= 2)
                    )
                )
                for j, tid in enumerate(panel.index[valid]):
                    true_up = bool((actual_ratio[j] >= threshold).sum() >= 2)
                    true_down = bool((actual_ratio[j] <= -threshold).sum() >= 2)
                    true_direction = int(true_up) - int(true_down)
                    predicted_direction = (
                        0
                        if predicted_ratio is None
                        else int((predicted_ratio[j] >= threshold).sum() >= 2)
                        - int((predicted_ratio[j] <= -threshold).sum() >= 2)
                    )
                    rows.append(
                        dict(
                            territory_id=int(tid),
                            origin=origin_label,
                            model=model,
                            threshold_pct=int(threshold * 100),
                            predicted_warning=bool(p[j]),
                            proxy_event=bool(y[j]),
                            actual_direction=true_direction,
                            predicted_direction=predicted_direction,
                            direction_known=predicted_ratio is not None,
                        )
                    )
    cases = pd.DataFrame(rows)
    out = ROOT / "results/future_warning"
    out.mkdir(exist_ok=True)
    cases.to_parquet(out / "cases.parquet", index=False)
    summary = []
    for (threshold, model), g in cases.groupby(["threshold_pct", "model"]):
        p = g.predicted_warning
        y = g.proxy_event
        tp = int((p & y).sum())
        fp = int((p & ~y).sum())
        fn = int((~p & y).sum())
        tn = int((~p & ~y).sum())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        summary.append(
            dict(
                threshold_pct=threshold,
                model=model,
                n=len(g),
                actual_events=int(y.sum()),
                warnings=int(p.sum()),
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                precision=precision,
                recall=recall,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            )
        )
    pd.DataFrame(summary).to_csv(out / "metrics.csv", index=False)
    directional = []
    for (threshold, model), g in cases.groupby(["threshold_pct", "model"]):
        if not g.direction_known.all():
            directional.append(
                dict(
                    threshold_pct=threshold,
                    model=model,
                    status="direction not predicted; excluded",
                )
            )
            continue
        p = g.predicted_warning
        y = g.proxy_event
        correct = p & y & g.predicted_direction.eq(g.actual_direction)
        wrong = p & y & g.predicted_direction.ne(g.actual_direction)
        tp = int(correct.sum())
        fp = int((p & ~correct).sum())
        fn = int((y & ~correct).sum())
        directional.append(
            dict(
                threshold_pct=threshold,
                model=model,
                status="calculated",
                n=len(g),
                tp_correct_direction=tp,
                fp_including_wrong_direction=fp,
                fn_including_wrong_direction=fn,
                wrong_direction=int(wrong.sum()),
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            )
        )
    pd.DataFrame(directional).to_csv(out / "direction_metrics.csv", index=False)
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "status": "exploratory protocol added after primary forecasting comparison; thresholds not optimised",
                "proxy": "At least two of the next three months have year-over-year deviation >= threshold in the same direction",
                "primary_threshold_pct": 20,
                "direction_rule": "Correct warning direction required; wrong direction counts as FP and FN; macro flag excluded from directional scores",
                "sensitivity_thresholds_pct": [10, 30],
                "limitations": "Proxy can include inflation, baseline effects and measurement changes. Not independently labelled structural shock. Municipality and origin pairs are dependent.",
            },
            indent=2,
        )
    )
    print(
        pd.DataFrame(summary)
        .query("threshold_pct==20")[
            ["model", "n", "actual_events", "warnings", "precision", "recall", "f1"]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
