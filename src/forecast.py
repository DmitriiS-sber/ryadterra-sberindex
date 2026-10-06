"""Fixed-parameter historical forecasts; every model sees only its origin's history."""

from __future__ import annotations
import argparse, hashlib, json, logging, os, time, warnings, multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from config import DEFAULT_CONFIG, foundation_revision, load_config
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def past_fill(x: np.ndarray) -> np.ndarray:
    """Forward-fill a supplied history, never using a later observation."""
    a = np.asarray(x, dtype=float).copy()
    for i in range(1, a.shape[1]):
        missing = ~np.isfinite(a[:, i])
        a[missing, i] = a[missing, i - 1]
    if not np.isfinite(a).all():
        raise ValueError("Unfillable past history")
    return a


def basic_forecasts(y: np.ndarray, length: int, config: dict) -> dict[str, np.ndarray]:
    """Return level, annual seasonal and damped log-trend forecast paths."""
    last = np.repeat(y[-1], length)
    seasonal = []
    seq = list(y)
    for _ in range(length):
        value = seq[-12] if len(seq) >= 12 else seq[-1]
        seasonal.append(value)
        seq.append(value)
    cfg = config["damped_trend"]
    recent = np.log(y[-cfg["recent_months"] :])
    slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
    slope = np.clip(
        slope, -cfg["maximum_monthly_growth"], cfg["maximum_monthly_growth"]
    )
    damping = cfg["damping"]
    steps = np.arange(1, length + 1)
    damped = y[-1] * np.exp(slope * damping * (1 - damping**steps) / (1 - damping))
    return {
        "last_value": last,
        "seasonal_naive": np.asarray(seasonal),
        "damped_trend": damped,
    }


def prophet_job(args):
    """Fit the selected Prophet variants in an isolated worker process."""
    territory_id, y, dates, length, config = args
    import prophet
    from prophet import Prophet

    libs = Path(prophet.__file__).resolve().parent.parent / "prophet.libs"
    if libs.exists():
        os.environ["LD_LIBRARY_PATH"] = str(libs) + (
            ":" + os.environ["LD_LIBRARY_PATH"]
            if os.environ.get("LD_LIBRARY_PATH")
            else ""
        )
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").disabled = True
    frame = pd.DataFrame({"ds": dates, "y": y})
    future = pd.DataFrame(
        {
            "ds": pd.date_range(
                pd.Timestamp(dates[-1]) + pd.offsets.MonthBegin(1),
                periods=length,
                freq="MS",
            )
        }
    )
    result = {}
    errors = []
    p = config["prophet"]
    for name, season, prior in [
        ("prophet_auto", "auto", 10.0),
        ("prophet_monthly", p["yearly_fourier_order"], p["seasonality_prior_scale"]),
    ]:
        if name not in config["models"]:
            continue
        try:
            m = Prophet(
                yearly_seasonality=season,
                weekly_seasonality=False,
                daily_seasonality=False,
                n_changepoints=p["n_changepoints"],
                changepoint_prior_scale=p["changepoint_prior_scale"],
                seasonality_prior_scale=prior,
                uncertainty_samples=0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(frame, seed=config["seed"])
            result[name] = np.maximum(m.predict(future).yhat.to_numpy(), 1.0)
        except Exception as exc:
            errors.append(
                {
                    "territory_id": int(territory_id),
                    "model": name,
                    "error": str(exc)[:600],
                }
            )
    return territory_id, result, errors


def ridge_features(
    seq: np.ndarray, target_month: int, news_vector: np.ndarray | None
) -> np.ndarray:
    """Features formed before observing the target. Input is log-normalised history."""
    recent = seq[:, -6:]
    columns = [
        recent[:, -1],
        recent[:, -2],
        recent[:, -3],
        recent[:, -6],
        recent.mean(axis=1),
        recent.std(axis=1),
        recent[:, -1] - recent[:, -3],
        np.full(len(seq), np.sin(2 * np.pi * target_month / 12)),
        np.full(len(seq), np.cos(2 * np.pi * target_month / 12)),
        np.full(len(seq), np.sin(4 * np.pi * target_month / 12)),
        np.full(len(seq), np.cos(4 * np.pi * target_month / 12)),
    ]
    if news_vector is not None:
        columns += [np.full(len(seq), float(value)) for value in news_vector]
    return np.column_stack(columns)


def pooled_ridge(
    history: np.ndarray,
    dates: pd.DatetimeIndex,
    news: pd.DataFrame,
    origin: int,
    length: int,
    use_news: bool,
    config: dict,
) -> np.ndarray:
    """Fit pooled log-growth regression and freeze news at the forecast origin."""
    scales = np.exp(np.log(history[:, :6]).mean(axis=1))
    logy = np.log(history / scales[:, None])
    X = []
    Y = []
    for target in range(config["ridge"]["minimum_lags"], len(dates)):
        # At the origin of a one-month prediction, only the previous month's news is known.
        v = news.iloc[target - 1].to_numpy(float) if use_news else None
        X.append(ridge_features(logy[:, :target], dates[target].month, v))
        Y.append(logy[:, target] - logy[:, target - 1])
    model = make_pipeline(StandardScaler(), Ridge(alpha=config["ridge"]["alpha"]))
    model.fit(np.concatenate(X), np.concatenate(Y))
    seq = logy.copy()
    result = []
    frozen = news.iloc[origin].to_numpy(float) if use_news else None
    limit = config["ridge"]["maximum_recursive_monthly_change"]
    for step in range(1, length + 1):
        date = dates[-1] + pd.DateOffset(months=step)
        pred = model.predict(ridge_features(seq, date.month, frozen))
        next_value = seq[:, -1] + np.clip(pred, -limit, limit)
        result.append(np.exp(next_value) * scales)
        seq = np.column_stack([seq, next_value])
    return np.asarray(result).T


def chronos_predict(
    history: np.ndarray, length: int, config: dict, pipeline
) -> np.ndarray:
    """Batch Chronos inference and return the quantile selected in YAML."""
    import torch

    out = []
    batch_size = config["chronos"]["batch_size"]
    for start in range(0, len(history), batch_size):
        tensor = torch.tensor(history[start : start + batch_size], dtype=torch.float32)
        with torch.inference_mode():
            q = pipeline.predict(tensor, prediction_length=length)
        # Pipeline reports quantiles in its model config; locate the declared median.
        index = list(pipeline.quantiles).index(config["chronos"]["quantile"])
        out.append(q[:, index, :].detach().cpu().numpy())
    return np.maximum(np.concatenate(out), 1.0)


def summarize(pred: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired forecast errors by horizon, origin and spending quartile."""
    rows = []
    groups = [("all", pred)]
    for h, g in pred.groupby("horizon"):
        groups.append((f"horizon_{h}", g))
    for date, g in pred.groupby("origin"):
        groups.append((f"origin_{date}", g))
    s = selection.set_index("territory_id").baseline_group
    pred = pred.copy()
    pred["baseline_group"] = pred.territory_id.map(s)
    for q, g in pred.groupby("baseline_group"):
        groups.append((q, g))
    for scope, g in groups:
        for model, m in g.groupby("model"):
            error = m.prediction - m.actual
            denom = float(((m.actual - m.actual.mean()) ** 2).sum())
            rows.append(
                dict(
                    scope=scope,
                    model=model,
                    n=len(m),
                    municipalities=m.territory_id.nunique(),
                    origins=m.origin.nunique(),
                    mae_rub=float(error.abs().mean()),
                    median_ape_pct=float((error.abs() / m.actual).median() * 100),
                    mean_ape_pct=float((error.abs() / m.actual).mean() * 100),
                    rmse_rub=float(np.sqrt((error**2).mean())),
                    r2=float(1 - (error**2).sum() / denom) if denom else None,
                )
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-chronos", action="store_true")
    parser.add_argument("--availability-lag", type=int, default=0)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML experiment configuration",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Separate output folder for a new experiment"
    )
    parser.add_argument(
        "--workers", type=int, help="Override the process count in YAML"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.workers is not None:
        if args.workers < 1:
            parser.error("--workers must be positive")
        config["workers"] = args.workers
    if args.limit < 0 or args.availability_lag < 0:
        parser.error("--limit and --availability-lag must be nonnegative")
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    if args.limit:
        panel = panel.iloc[: args.limit]
    ids = panel.index.to_numpy()
    values = panel.to_numpy(float)
    all_dates = pd.to_datetime(panel.columns)
    news = pd.read_csv(ROOT / "data/processed/monthly_news_features.csv").set_index(
        "date"
    )
    news = news.loc[
        panel.columns,
        ["key_rate", "hikes_3m_pp", "forward_tightening", "demand_pressure"],
    ]
    output_dir = (
        ROOT
        / "results"
        / (f"lag_{args.availability_lag}" if args.availability_lag else "main")
    )
    if args.limit:
        output_dir = ROOT / "results" / "smoke"
    if args.output_dir is not None:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(config["models"])
    pipeline = None
    if args.skip_chronos and "chronos_bolt_tiny" in model_names:
        model_names.remove("chronos_bolt_tiny")
    if not model_names:
        parser.error("No models remain after --skip-chronos")
    if "chronos_bolt_tiny" in model_names:
        import torch
        from chronos import ChronosBoltPipeline

        torch.set_num_threads(4)
        os.environ.setdefault("HF_HOME", str(ROOT.parent / "hf_cache"))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        pin = foundation_revision(config["chronos"]["model_id"])
        pipeline = ChronosBoltPipeline.from_pretrained(
            config["chronos"]["model_id"],
            device_map=config["chronos"]["device"],
            dtype=torch.float32,
            revision=pin,
        )
    errors = []
    frames = []
    paths = []
    runtimes = []
    start = time.time()
    for origin_label in config["origins"]:
        origin = int(np.where(panel.columns == origin_label)[0][0])
        train_end = origin - args.availability_lag
        minimum = (
            max(7, config["ridge"]["minimum_lags"] + 1)
            if any(m.startswith("pooled_ridge") for m in model_names)
            else 2
        )
        if train_end + 1 < minimum:
            raise ValueError(
                f"Insufficient training history at {origin_label} with the requested publication lag"
            )
        history = past_fill(values[:, : train_end + 1])
        dates = all_dates[: train_end + 1]
        length = min(max(config["horizons"]), len(all_dates) - 1 - origin)
        model_length = length + args.availability_lag
        if length < 1:
            raise ValueError(
                f"Origin {origin_label} has no subsequent test observation"
            )
        print(
            "ORIGIN",
            origin_label,
            "HISTORY",
            history.shape,
            "FUTURE",
            length,
            flush=True,
        )
        origin_start = time.time()
        arrays = {
            name: np.full((len(ids), model_length), np.nan) for name in model_names
        }
        for i, y in enumerate(history):
            for name, forecast in basic_forecasts(y, model_length, config).items():
                if name in arrays:
                    arrays[name][i] = forecast
        for name, use_news in [("pooled_ridge", False), ("pooled_ridge_news", True)]:
            if name not in arrays:
                continue
            # News is available at the decision origin even when spending has a publication lag.
            arrays[name] = pooled_ridge(
                history, dates, news, origin, model_length, use_news, config
            )
        jobs = (
            (tid, history[i], dates, model_length, config) for i, tid in enumerate(ids)
        )
        if any(name.startswith("prophet_") for name in model_names):
            with ProcessPoolExecutor(
                max_workers=config["workers"],
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                for i, (tid, out, err) in enumerate(
                    pool.map(prophet_job, jobs, chunksize=20)
                ):
                    for name, forecast in out.items():
                        arrays[name][i] = forecast
                    errors += [dict(origin=origin_label, **e) for e in err]
                    if (i + 1) % 500 == 0:
                        print("PROPHET", origin_label, i + 1, "/", len(ids), flush=True)
        if pipeline is not None:
            arrays["chronos_bolt_tiny"] = chronos_predict(
                history, model_length, config, pipeline
            )
        arrays = {k: v[:, args.availability_lag :] for k, v in arrays.items()}
        # Restrict every model to exactly the same evaluable municipality/target pairs.
        for horizon in config["horizons"]:
            if origin + horizon >= len(all_dates):
                continue
            actual = values[:, origin + horizon]
            valid = np.isfinite(actual)
            for arr in arrays.values():
                valid &= np.isfinite(arr[:, horizon - 1])
            for model, arr in arrays.items():
                frames.append(
                    pd.DataFrame(
                        {
                            "territory_id": ids[valid],
                            "origin": origin_label,
                            "target_date": panel.columns[origin + horizon],
                            "horizon": horizon,
                            "model": model,
                            "prediction": arr[valid, horizon - 1],
                            "actual": actual[valid],
                        }
                    )
                )
        for model, arr in arrays.items():
            paths.append(
                pd.DataFrame(
                    {
                        "territory_id": np.repeat(ids, length),
                        "origin": origin_label,
                        "model": model,
                        "target_date": np.tile(
                            panel.columns[origin + 1 : origin + 1 + length], len(ids)
                        ),
                        "prediction": arr.reshape(-1),
                        "actual": values[:, origin + 1 : origin + 1 + length].reshape(
                            -1
                        ),
                    }
                )
            )
        runtimes.append(dict(origin=origin_label, seconds=time.time() - origin_start))
        # Checkpoint computed origins: resumed work never loses completed numerical output.
        pd.concat(frames, ignore_index=True).to_parquet(
            output_dir / "predictions.parquet", index=False
        )
        pd.concat(paths, ignore_index=True).to_parquet(
            output_dir / "forecast_paths.parquet", index=False
        )
        print(
            "DONE",
            origin_label,
            "SECONDS",
            round(time.time() - origin_start, 1),
            flush=True,
        )
    pred = pd.concat(frames, ignore_index=True)
    selection = pd.read_csv(ROOT / "data/processed/selection.csv")
    summary = summarize(pred, selection)
    summary.to_csv(output_dir / "metrics.csv", index=False)
    pred.to_csv(output_dir / "predictions.csv.gz", index=False, compression="gzip")
    (output_dir / "model_errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2)
    )
    versions = {}
    import importlib.metadata

    for name in [
        "numpy",
        "pandas",
        "scikit-learn",
        "prophet",
        "cmdstanpy",
        "torch",
        "chronos-forecasting",
        "transformers",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    model_revision = None
    if pipeline is not None:
        model_revision = getattr(pipeline.model.config, "_commit_hash", None) or pin
    run = dict(
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        effective_config=config,
        protocol_sha256=hashlib.sha256(
            (ROOT / "report/protocol.md").read_bytes()
        ).hexdigest(),
        municipalities=len(ids),
        models=model_names,
        availability_lag_months=args.availability_lag,
        total_seconds=time.time() - start,
        origin_runtime=runtimes,
        model_failures=len(errors),
        package_versions=versions,
        foundation_model_revision=model_revision,
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2)
    )
    print(
        summary[summary.scope.str.startswith("horizon")][
            ["scope", "model", "n", "mae_rub", "median_ape_pct", "r2"]
        ].to_string(index=False),
        flush=True,
    )
    if errors:
        raise RuntimeError(
            f"{len(errors)} model failures: inspect {output_dir}/model_errors.json"
        )


if __name__ == "__main__":
    main()
