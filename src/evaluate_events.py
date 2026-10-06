"""One-to-one, direction-aware scoring on a declared surveillance window.

The independently reviewed territory roster and complete event log are external
inputs. Code checks their schema but cannot establish completeness or independence.
Boundary events/warnings are censored explicitly rather than scored as failures.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def timestamp(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        raise ValueError("An explicit timezone is required")
    return t.tz_convert("UTC")


def evaluate(
    events, warnings, population, window_start, window_end, adjudicated_at, months=3
):
    re = {
        "event_id",
        "territory_id",
        "onset_at",
        "published_at",
        "direction",
        "source_url",
        "independent_review",
    }
    rw = {"warning_id", "territory_id", "issued_at", "direction"}
    if not re <= set(events) or not rw <= set(warnings):
        raise ValueError("Incomplete event or warning schema")
    if events.empty:
        raise ValueError("No independent labels: real shock accuracy remains unknown")
    if population.empty or not {"territory_id", "independent_review"} <= set(
        population
    ):
        raise ValueError(
            "A fixed independently reviewed surveillance roster is required"
        )
    for f, cols in [
        (events, re),
        (warnings, rw),
        (population, {"territory_id", "independent_review"}),
    ]:
        if f[list(cols)].isna().any().any():
            raise ValueError("Unknown provenance prohibits scoring")
    if (
        not events.independent_review.astype(str).str.lower().isin(["true", "1"]).all()
        or not population.independent_review.astype(str)
        .str.lower()
        .isin(["true", "1"])
        .all()
    ):
        raise ValueError("Independent review is required")
    if (
        population.territory_id.duplicated().any()
        or events.event_id.duplicated().any()
        or warnings.warning_id.duplicated().any()
    ):
        raise ValueError("Duplicate IDs")
    if (
        not events.direction.isin([-1, 1]).all()
        or not warnings.direction.isin([-1, 1]).all()
    ):
        raise ValueError("Directions must be -1 or +1")
    if (
        not events.territory_id.isin(population.territory_id).all()
        or not warnings.territory_id.isin(population.territory_id).all()
    ):
        raise ValueError("Event/warning outside the registered territory population")
    start, end, review = map(timestamp, [window_start, window_end, adjudicated_at])
    if not start < end <= review or months < 1:
        raise ValueError("Invalid observation or adjudication window")
    events = events.copy()
    warnings = warnings.copy()
    for col in ["onset_at", "published_at"]:
        events[col] = events[col].map(timestamp)
    warnings["issued_at"] = warnings.issued_at.map(timestamp)
    if (
        not events.onset_at.between(start, end, inclusive="left").all()
        or not warnings.issued_at.between(start, end, inclusive="left").all()
    ):
        raise ValueError("Inputs must be limited to the declared observation window")
    if not events.published_at.le(review).all():
        raise ValueError("Label not available by adjudication date")
    left = events.onset_at.lt(start + pd.DateOffset(months=months))
    right = warnings.issued_at.ge(end - pd.DateOffset(months=months))
    eligible = events.loc[~left]
    scored = warnings.loc[~right]
    if eligible.empty:
        raise ValueError("No events with a complete preceding warning window")
    used = set()
    matches = []
    for event in eligible.sort_values(["onset_at", "event_id"]).itertuples():
        candidates = scored[
            (scored.territory_id == event.territory_id)
            & (scored.direction == event.direction)
            & (scored.issued_at >= event.onset_at - pd.DateOffset(months=months))
            & (scored.issued_at < event.onset_at)
            & ~scored.warning_id.isin(used)
        ]
        if candidates.empty:
            continue
        warning = candidates.sort_values(["issued_at", "warning_id"]).iloc[0]
        used.add(warning.warning_id)
        matches.append(
            dict(
                event_id=event.event_id,
                warning_id=warning.warning_id,
                lead_days=(event.onset_at - warning.issued_at).total_seconds() / 86400,
            )
        )
    tp = len(matches)
    fp = len(scored) - tp
    fn = len(eligible) - tp
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn),
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        matches=matches,
        censored_left_events=int(left.sum()),
        censored_right_warnings=int(right.sum()),
        window_start=start.isoformat(),
        window_end=end.isoformat(),
        adjudicated_at=review.isoformat(),
        territory_count=len(population),
        matching_rule="Earliest unused matching-direction warning before onset, sorted events",
        limitation="Completeness and independence of labels must be established externally; counts are conditional on declared inputs",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["events", "warnings", "population", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    for name in ["window-start", "window-end", "adjudicated-at"]:
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    result = evaluate(
        pd.read_csv(a.events),
        pd.read_csv(a.warnings),
        pd.read_csv(a.population),
        a.window_start,
        a.window_end,
        a.adjudicated_at,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
