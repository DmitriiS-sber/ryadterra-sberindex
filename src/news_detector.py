"""Paired residual-detector ablation with and without dated news features.

Candidate-alert counts describe behaviour, not accuracy against economic shocks.
No detector threshold is optimised against observed 2024 alarms.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from config import load_config
from forecast import past_fill, pooled_ridge
from detect import scores

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = load_config(ROOT / "configs/validation.yaml")
    cfg = config["detectors"]
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    values = panel.to_numpy(float)
    dates = pd.to_datetime(panel.columns)
    news = pd.read_csv(ROOT / "data/processed/monthly_news_features.csv").set_index(
        "date"
    )
    features = news.loc[
        panel.columns,
        ["key_rate", "hikes_3m_pp", "forward_tightening", "demand_pressure"],
    ]
    thresholds = json.loads(
        (ROOT / "results/detection/real_thresholds.json").read_text()
    )
    out = ROOT / "results/validation"
    out.mkdir(exist_ok=True)
    rows = []
    summary = []
    paths = []
    for use_news in [False, True]:
        residual = np.full_like(values, np.nan)
        prediction = np.full_like(values, np.nan)
        for t in range(7, len(dates)):
            predicted = pooled_ridge(
                past_fill(values[:, :t]),
                dates[:t],
                features,
                t - 1,
                1,
                use_news,
                config,
            )[:, 0]
            prediction[:, t] = predicted
            residual[:, t] = np.log(values[:, t] / predicted)
        calibration = residual[:, 7:12]
        centre = np.nanmedian(calibration, axis=1)
        scale = 1.4826 * np.nanmedian(np.abs(calibration - centre[:, None]), axis=1)
        floor = max(float(np.nanmedian(scale)), 0.02)
        scale = np.maximum(scale, floor)
        z = (residual[:, 12:] - centre[:, None]) / scale[:, None]
        for method, s in scores(z, 0, cfg).items():
            hits = (s > thresholds[method]) & np.isfinite(z)
            summary.append(
                dict(
                    method=method,
                    use_news=use_news,
                    territories=len(panel),
                    territories_with_candidate_alert=int(hits.any(axis=1).sum()),
                    empirical_shock_precision=None,
                    empirical_shock_recall=None,
                    status="unknown: independent event labels unavailable",
                )
            )
            for i, tid in enumerate(panel.index):
                if not hits[i].any():
                    continue
                t = int(np.flatnonzero(hits[i])[0])
                idx = 12 + t
                rows.append(
                    dict(
                        territory_id=int(tid),
                        method=method,
                        use_news=use_news,
                        signal_observation_month=panel.columns[idx],
                        news_feature_cutoff=panel.columns[idx - 1],
                        news_last_publication=news.loc[
                            panel.columns[idx - 1], "last_news_date"
                        ],
                        actual_rub=float(values[i, idx]),
                        forecast_rub=float(prediction[i, idx]),
                        score=float(s[i, t]),
                        threshold=thresholds[method],
                        published_at=None,
                        economic_onset=None,
                        causal_interpretation="unverified",
                    )
                )
        for t in range(12, len(dates)):
            paths.append(
                pd.DataFrame(
                    dict(
                        territory_id=panel.index,
                        date=panel.columns[t],
                        use_news=use_news,
                        actual=values[:, t],
                        prediction=prediction[:, t],
                        innovation_log=residual[:, t],
                    )
                )
            )
    pd.DataFrame(rows).to_csv(out / "news_detector_alerts.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "news_detector_ablation.csv", index=False)
    pd.concat(paths).to_parquet(out / "news_detector_predictions.parquet", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
