"""Fixed dictionary warning about future policy tightening, not a causal expense shock label."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    events = pd.read_csv(
        ROOT / "data/processed/news_events.csv", parse_dates=["publication_date"]
    )
    end = pd.Timestamp("2024-12-31")
    rows = []
    for r in events.itertuples():
        until = r.publication_date + pd.Timedelta(days=90)
        censored = until > end
        future = events[
            (events.publication_date > r.publication_date)
            & (events.publication_date <= until)
            & (events.rate_change_pp > 0)
        ]
        rows.append(
            dict(
                publication_date=str(r.publication_date.date()),
                source_url=r.source_url,
                forward_warning=r.forward_tightening,
                current_hike_warning=int(r.rate_change_pp > 0),
                target_hike_within_90d=None if censored else int(len(future) > 0),
                censored=censored,
                next_hike_date=(
                    str(future.publication_date.min().date()) if len(future) else None
                ),
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/news_warning_cases.csv", index=False)
    valid = df[~df.censored]
    metrics = []
    for model in ["forward_warning", "current_hike_warning"]:
        y = valid.target_hike_within_90d.astype(bool).to_numpy()
        p = valid[model].astype(bool).to_numpy()
        tp = int((y & p).sum())
        fp = int((~y & p).sum())
        fn = int((y & ~p).sum())
        tn = int((~y & ~p).sum())
        metrics.append(
            dict(
                model=model,
                n=len(valid),
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                accuracy=float((y == p).mean()),
            )
        )
    pd.DataFrame(metrics).to_csv(ROOT / "results/news_warning_metrics.csv", index=False)
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
