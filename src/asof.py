"""Build a monthly panel using only snapshots actually available by a cutoff."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def asof_panel(frame: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Reject unknown availability; choose the latest available vintage per key.

    Publication and retrieval timestamps must be timezone-aware. An observation
    month alone is never treated as a publication timestamp. published_at must
    refer to this revision, not the first publication of the observation.
    Retrieval controls eligibility, never the revision ordering.
    """
    if "category" in frame:
        frame = frame[frame.category.eq("Все категории")]
    required = {
        "territory_id",
        "observation_month",
        "value",
        "published_at",
        "retrieved_at",
        "vintage_id",
    }
    if not required <= set(frame):
        raise ValueError(f"Missing columns: {required-set(frame)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError(
            "As-of evaluation requires known timestamps and vintage IDs for every input row"
        )
    if (
        not frame.observation_month.astype(str)
        .str.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])")
        .all()
    ):
        raise ValueError("Observation month must use YYYY-MM")
    if (
        frame.vintage_id.astype(str).str.strip().eq("").any()
        or not np.isfinite(frame.value).all()
    ):
        raise ValueError("Vintage IDs must be nonempty and values finite")
    limit = pd.Timestamp(cutoff)
    if limit.tzinfo is None:
        raise ValueError("Cutoff must include an explicit timezone")
    work = frame.copy()
    for column in ["published_at", "retrieved_at"]:
        if (
            not work[column]
            .astype(str)
            .str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True)
            .all()
        ):
            raise ValueError(f"{column} must include timezone offsets")
        work[column] = pd.to_datetime(work[column], utc=True)
    if (work.published_at > work.retrieved_at).any():
        raise ValueError(
            "A snapshot cannot be retrieved before its declared publication"
        )
    if work.duplicated(["territory_id", "observation_month", "vintage_id"]).any():
        raise ValueError("Duplicate observation within a vintage")
    available = work[(work.published_at <= limit) & (work.retrieved_at <= limit)]
    available = available[
        available.observation_month.le(
            limit.tz_convert("Europe/Moscow").strftime("%Y-%m")
        )
    ]
    if available.empty:
        raise ValueError("No observed vintage was available at this cutoff")
    # published_at describes the publication of this particular revision.
    # Reject ambiguous revisions rather than guess their order.
    version_keys = ["territory_id", "observation_month", "published_at"]
    if available.groupby(version_keys).vintage_id.nunique().gt(1).any():
        raise ValueError("Ambiguous revision order: explicit version timestamp required")
    chosen = available.sort_values(["published_at", "retrieved_at"], kind="stable").drop_duplicates(
        ["territory_id", "observation_month"], keep="last"
    )
    if (chosen.value <= 0).any():
        raise ValueError("Observed spending must be positive")
    return chosen.pivot(
        index="territory_id", columns="observation_month", values="value"
    ).sort_index(axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--cutoff", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    f = (
        pd.read_parquet(a.input)
        if a.input.suffix == ".parquet"
        else pd.read_csv(a.input)
    )
    panel = asof_panel(f, a.cutoff)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(a.output)
    print("Saved", a.output)


if __name__ == "__main__":
    main()
