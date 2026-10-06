"""Additional exact Prophet default baseline, apart from uncertainty sampling and fixed seed.

Added after the original comparison to check sensitivity to the choice of baseline.
No parameters are selected against 2024 outcomes.
"""

import json, logging, os, warnings, multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, yaml
from forecast import past_fill, summarize

ROOT = Path(__file__).resolve().parents[1]


def job(args):
    tid, y, dates, length, seed = args
    import prophet
    from prophet import Prophet

    libs = Path(prophet.__file__).resolve().parent.parent / "prophet.libs"
    os.environ["LD_LIBRARY_PATH"] = str(libs)
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").disabled = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(uncertainty_samples=0)
        m.fit(pd.DataFrame({"ds": dates, "y": y}), seed=seed)
    future = pd.DataFrame(
        {
            "ds": pd.date_range(
                dates[-1] + pd.offsets.MonthBegin(1), periods=length, freq="MS"
            )
        }
    )
    return tid, np.maximum(1, m.predict(future).yhat.to_numpy())


def main():
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    dates = pd.to_datetime(panel.columns)
    a = panel.to_numpy(float)
    rows = []
    paths = []
    for origin_label in cfg["origins"]:
        origin = list(panel.columns).index(origin_label)
        hist = past_fill(a[:, : origin + 1])
        length = min(12, 23 - origin)
        jobs = (
            (tid, hist[i], dates[: origin + 1], length, cfg["seed"])
            for i, tid in enumerate(panel.index)
        )
        forecasts = []
        with ProcessPoolExecutor(
            max_workers=cfg["workers"], mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for i, (tid, f) in enumerate(pool.map(job, jobs, chunksize=20)):
                forecasts.append(f)
                if (i + 1) % 1000 == 0:
                    print(origin_label, i + 1, flush=True)
        forecasts = np.asarray(forecasts)
        for h in cfg["horizons"]:
            if origin + h >= 24:
                continue
            valid = np.isfinite(a[:, origin + h])
            rows.append(
                pd.DataFrame(
                    {
                        "territory_id": panel.index[valid],
                        "origin": origin_label,
                        "target_date": panel.columns[origin + h],
                        "horizon": h,
                        "model": "prophet_default",
                        "prediction": forecasts[valid, h - 1],
                        "actual": a[valid, origin + h],
                    }
                )
            )
        paths.append(
            pd.DataFrame(
                {
                    "territory_id": np.repeat(panel.index, length),
                    "origin": origin_label,
                    "model": "prophet_default",
                    "target_date": np.tile(
                        panel.columns[origin + 1 : origin + 1 + length], len(panel)
                    ),
                    "prediction": forecasts.reshape(-1),
                    "actual": a[:, origin + 1 : origin + 1 + length].reshape(-1),
                }
            )
        )
    out = ROOT / "results/default_baseline"
    out.mkdir(exist_ok=True)
    predictions = pd.concat(rows)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    pd.concat(paths).to_parquet(out / "forecast_paths.parquet", index=False)
    combined = pd.concat(
        [pd.read_parquet(ROOT / "results/main/predictions.parquet"), predictions]
    )
    summary = summarize(combined, pd.read_csv(ROOT / "data/processed/selection.csv"))
    summary.to_csv(out / "metrics_comparison.csv", index=False)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "purpose": "Sensitivity to the exact default Prophet baseline",
                "added_after_original_comparison": True,
                "parameters": "Prophet defaults except uncertainty_samples=0; fixed optimiser seed",
                "n": len(predictions),
                "failures": 0,
            },
            indent=2,
        )
    )
    print(
        summary[summary.model.eq("prophet_default")][
            ["scope", "n", "mae_rub"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
