"""Freeze code, register future test keys, and score complete released targets once.

Local timestamps/hashes are audit aids. Genuine ex-ante issuance requires an
external immutable forecast log; this module cannot prove authenticity by itself.
"""

from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/prospective_lock.json"
PLAN = ROOT / "configs/prospective_plan.json"
SCORE = ROOT / "configs/prospective_scored.json"
KEY = ["territory_id", "origin", "horizon", "target_month"]


def timestamp(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        raise ValueError("Explicit timezone required")
    return t.tz_convert("UTC")


def code_hashes():
    files = list((ROOT / "src").glob("*.py")) + [
        ROOT / "configs/experiment.yaml",
        ROOT / "configs/validation.yaml",
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }


def validate_keys(p):
    if p.empty or not set(KEY) <= set(p) or p[KEY].isna().any().any():
        raise ValueError("Complete future test keys required")
    if p.duplicated(KEY).any():
        raise ValueError("Duplicate test keys")
    expected = [
        str(pd.Period(o, freq="M") + int(h)) for o, h in zip(p.origin, p.horizon)
    ]
    if list(p.target_month) != expected:
        raise ValueError("Target month does not match origin plus horizon")
    if not p.horizon.isin([1, 3, 6, 12]).all():
        raise ValueError("Unexpected horizon")


def freeze():
    if LOCK.exists():
        raise ValueError("Freeze exists; do not overwrite")
    payload = dict(
        frozen_at=datetime.now(timezone.utc).isoformat(),
        status="code frozen; independent test awaits pre-registered plan and genuinely new data",
        model="prophet_auto",
        seen_history_end="2024-12",
        horizons=[1, 3, 6, 12],
        primary_metric="MAE equally weighted test pairs",
        secondary_metric="mean of per-territory MAE",
        code_sha256=code_hashes(),
        selection_note="Model chosen retrospectively on 2024; only targets after freeze and plan registration qualify",
        provenance_requirement="External immutable forecast log and authentic release/vintage history required",
        event_matching=dict(
            warning_horizon_months=3,
            direction_required=True,
            one_to_one=True,
            onset_and_publication_separate=True,
            independent_labels=True,
            explicit_boundary_censoring=True,
        ),
    )
    LOCK.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print("Frozen", LOCK)


def register(path):
    if PLAN.exists():
        raise ValueError("Future test plan already registered")
    lock = json.loads(LOCK.read_text())
    if lock["code_sha256"] != code_hashes():
        raise ValueError("Code changed after freeze")
    p = pd.read_csv(path)
    validate_keys(p)
    now = pd.Timestamp.now(tz="UTC")
    if not (
        pd.to_datetime(p.target_month + "-01")
        .dt.tz_localize("Europe/Moscow")
        .dt.tz_convert("UTC")
        > now
    ).all():
        raise ValueError("All targets must be future months at registration")
    PLAN.write_text(
        json.dumps(
            dict(
                registered_at=now.isoformat(),
                keys=p[KEY].to_dict("records"),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                status="Fixed keys; no pruning based on outcomes",
            ),
            indent=2,
        )
    )
    print("Registered", PLAN)


def evaluate(predictions, actuals, output):
    lock = json.loads(LOCK.read_text())
    plan = json.loads(PLAN.read_text())
    frozen = timestamp(lock["frozen_at"])
    registered = timestamp(plan["registered_at"])
    if lock["code_sha256"] != code_hashes():
        raise ValueError("Code/config changed after freeze")
    if output.exists() or SCORE.exists():
        raise ValueError("Test is single-use; score already exists")
    p = pd.read_csv(predictions)
    y = pd.read_csv(actuals)
    validate_keys(p)
    if not {"prediction", "model", "issued_at"} <= set(p) or not {
        "territory_id",
        "target_month",
        "actual",
        "published_at",
        "vintage_id",
    } <= set(y):
        raise ValueError("Missing provenance")
    if (
        p[["prediction", "model", "issued_at"]].isna().any().any()
        or y[["actual", "published_at", "vintage_id"]].isna().any().any()
    ):
        raise ValueError("Unknown provenance")
    if y.vintage_id.astype(str).str.strip().eq("").any():
        raise ValueError("Empty vintage IDs")
    wanted = pd.DataFrame(plan["keys"])
    if set(map(tuple, p[KEY].to_numpy())) != set(map(tuple, wanted[KEY].to_numpy())):
        raise ValueError("Forecast keys differ from registered test population")
    issued = p.issued_at.map(timestamp)
    published = y.published_at.map(timestamp)
    target = (
        pd.to_datetime(p.target_month + "-01")
        .dt.tz_localize("Europe/Moscow")
        .dt.tz_convert("UTC")
    )
    if not ((issued >= registered) & (issued >= frozen) & (issued < target)).all():
        raise ValueError(
            "Forecast issuance must follow registration and precede target month"
        )
    if not published.gt(frozen).all() or not p.model.eq(lock["model"]).all():
        raise ValueError("Non-prospective actual or different model")
    if not published.le(pd.Timestamp.now(tz="UTC")).all():
        raise ValueError("Actual publication is still in the future")
    if (
        not np.isfinite(p.prediction).all()
        or not np.isfinite(y.actual).all()
        or not (p.prediction.gt(0).all() and y.actual.gt(0).all())
    ):
        raise ValueError("Values must be positive and finite")
    if y.duplicated(["territory_id", "target_month"]).any():
        raise ValueError("Duplicate actual keys")
    m = p.merge(
        y, on=["territory_id", "target_month"], validate="many_to_one", how="left"
    )
    if m.actual.isna().any():
        raise ValueError(
            "Some registered targets are unreleased: no partial one-time scoring"
        )
    if not (m.published_at.map(timestamp) > m.issued_at.map(timestamp)).all():
        raise ValueError("Actual released before forecast")
    rows = []
    for h, g in m.groupby("horizon"):
        e = (g.prediction - g.actual).abs()
        rows.append(
            dict(
                horizon=int(h),
                n=len(g),
                mae_rub=float(e.mean()),
                equal_territory_mae_rub=float(
                    g.assign(error=e).groupby("territory_id").error.mean().mean()
                ),
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            dict(
                status="scored once; ex-ante authenticity requires external evidence",
                frozen_at=lock["frozen_at"],
                registered_at=plan["registered_at"],
                metrics=rows,
                forecast_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
                actuals_sha256=hashlib.sha256(actuals.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
    )
    SCORE.write_text(
        json.dumps(
            dict(
                output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                scored_at=datetime.now(timezone.utc).isoformat(),
            ),
            indent=2,
        )
    )
    print(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["freeze", "register", "evaluate"])
    for name in ["plan", "predictions", "actuals", "output"]:
        p.add_argument("--" + name, type=Path)
    a = p.parse_args()
    if a.action == "freeze":
        freeze()
    elif a.action == "register":
        if not a.plan:
            p.error("register requires --plan")
        register(a.plan)
    else:
        if not all([a.predictions, a.actuals, a.output]):
            p.error("evaluate requires all input/output paths")
        evaluate(a.predictions, a.actuals, a.output)


if __name__ == "__main__":
    main()
