"""Export forecasts from a chosen data cutoff; Chronos quantiles are uncalibrated."""

import argparse, json, os, hashlib
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from config import DEFAULT_CONFIG, foundation_revision, load_config
from forecast import past_fill, prophet_job, basic_forecasts

ROOT = Path(__file__).resolve().parents[1]


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--input", type=Path, default=ROOT / "data/processed/panel.parquet"
    )
    cli.add_argument("--origin", default="2024-12")
    cli.add_argument(
        "--output", type=Path, default=ROOT / "results/latest_forecast.csv"
    )
    cli.add_argument(
        "--model",
        choices=["prophet_auto", "damped_trend", "chronos_bolt_tiny"],
        default="prophet_auto",
    )
    cli.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    a = cli.parse_args()
    cfg = load_config(a.config)
    cfg["models"] = [a.model]
    supplied = pd.read_parquet(a.input)
    periods = pd.PeriodIndex(supplied.columns, freq="M")
    if (
        not periods.is_monotonic_increasing
        or periods.has_duplicates
        or len(periods) < 12
    ):
        raise ValueError(
            "Input must contain at least 12 unique chronological monthly columns"
        )
    if not np.all(np.diff(periods.asi8) == 1):
        raise ValueError(
            "Monthly columns must be consecutive; represent missing values as NaN"
        )
    if (supplied.to_numpy(float) <= 0).any():
        raise ValueError("Observed spending values must be positive")
    if a.origin not in supplied.columns:
        raise ValueError("Origin must be an observed month in the supplied panel")
    panel = supplied.loc[:, : a.origin]
    if len(panel.columns) < 12:
        raise ValueError("At least 12 historical months are required at the cutoff")
    history = past_fill(panel.to_numpy(float))
    assert (
        panel.columns[-1] == a.origin
    ), "Origin must be an observed month in the supplied panel"
    os.environ.setdefault("HF_HOME", str(ROOT.parent / "hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if a.model == "chronos_bolt_tiny":
        import torch
        from chronos import ChronosBoltPipeline

        torch.set_num_threads(4)
    revision = foundation_revision(cfg["chronos"]["model_id"])
    kwargs = {"revision": revision} if revision else {}
    pipe = (
        ChronosBoltPipeline.from_pretrained(
            cfg["chronos"]["model_id"],
            device_map=cfg["chronos"]["device"],
            dtype=torch.float32,
            **kwargs,
        )
        if a.model == "chronos_bolt_tiny"
        else None
    )
    forecast = None
    if a.model == "prophet_auto":
        dates = pd.to_datetime(panel.columns)
        jobs = ((tid, history[i], dates, 12, cfg) for i, tid in enumerate(panel.index))
        arrays = []
        with ProcessPoolExecutor(
            max_workers=cfg["workers"], mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for tid, result, errors in pool.map(prophet_job, jobs, chunksize=20):
                if errors:
                    raise RuntimeError(errors)
                arrays.append(result["prophet_auto"])
        forecast = np.asarray(arrays)
    elif a.model == "damped_trend":
        forecast = np.asarray(
            [basic_forecasts(y, 12, cfg)["damped_trend"] for y in history]
        )
    rows = []
    issued_at = datetime.now(timezone.utc).isoformat()
    input_sha256 = hashlib.sha256(a.input.read_bytes()).hexdigest()
    batch = cfg["chronos"]["batch_size"]
    for start in range(0, len(panel), batch):
        if pipe is not None:
            with torch.inference_mode():
                q = (
                    pipe.predict(
                        torch.tensor(
                            history[start : start + batch], dtype=torch.float32
                        ),
                        prediction_length=12,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
        for i, tid in enumerate(panel.index[start : start + batch]):
            for h in cfg["horizons"]:
                rows.append(
                    dict(
                        territory_id=int(tid),
                        data_cutoff=a.origin,
                        origin=a.origin,
                        issued_at=issued_at,
                        input_sha256=input_sha256,
                        horizon=h,
                        target_month=str(pd.Period(a.origin) + h),
                        model=a.model,
                        q10_rub=(
                            max(1, float(q[i, list(pipe.quantiles).index(0.1), h - 1]))
                            if pipe is not None
                            else None
                        ),
                        forecast_rub=(
                            max(
                                1,
                                float(
                                    q[
                                        i,
                                        list(pipe.quantiles).index(
                                            cfg["chronos"]["quantile"]
                                        ),
                                        h - 1,
                                    ]
                                ),
                            )
                            if pipe is not None
                            else float(forecast[start + i, h - 1])
                        ),
                        q90_rub=(
                            max(1, float(q[i, list(pipe.quantiles).index(0.9), h - 1]))
                            if pipe is not None
                            else None
                        ),
                        status="unverified beyond dataset; any model quantiles uncalibrated",
                    )
                )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(rows)
    output["prediction"] = output["forecast_rub"]
    output.to_csv(a.output, index=False)
    print("SAVED", str(a.output), "ROWS", len(rows))


if __name__ == "__main__":
    main()
