"""Recheck primary-source metadata without redistributing full news texts."""

from __future__ import annotations
import hashlib, json, re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def fetch(row):
    result = dict(
        source_url=row["source_url"],
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    try:
        r = requests.get(row["source_url"], timeout=12)
        r.raise_for_status()
        result["sha256"] = hashlib.sha256(r.content).hexdigest()
        if row.get("kind") == "news":
            soup = BeautifulSoup(r.content, "html.parser")
            body = soup.select_one(".document") or soup.select_one("main") or soup
            text = body.get_text(" ", strip=True).replace("\xa0", " ").lower()
            tightening = int(
                any(
                    phrase in text
                    for phrase in [
                        "допускает возможность дальнейшего повышения",
                        "будет оценивать целесообразность дальнейшего повышения",
                        "будет оценивать целесообразность повышения",
                        "будет оценивать целесообразность дополнительного повышения",
                        "не исключает возможности повышения",
                    ]
                )
            )
            demand = int(
                any(
                    phrase in text
                    for phrase in [
                        "спроса превышает возможности",
                        "спроса продолжает значительно опережать",
                        "спроса продолжает опережать",
                        "расширение внутреннего спроса",
                        "внутренний спрос продолжает",
                    ]
                )
            )
            result.update(
                forward_tightening=tightening,
                demand_pressure=demand,
                feature_flags_match=bool(
                    tightening == row["forward_tightening"]
                    and demand == row["demand_pressure"]
                ),
                html_hash_matches_original=result["sha256"] == row["source_sha256"],
                status="current primary text checked; historical version availability unknown",
            )
        elif row.get("kind") == "archive":
            result.update(
                original_hash_matches=result["sha256"] == row["expected_sha256"],
                status="downloaded and hashed",
            )
        else:
            result["status"] = (
                "page retrieved; geographic ID mapping still requires a validated dictionary file"
            )
    except Exception as e:
        result.update(status="not verified", error=str(e)[:250])
    return result


def main():
    news = pd.read_csv(ROOT / "data/processed/news_events.csv").to_dict("records")
    for row in news:
        row["kind"] = "news"
    rows = news + [
        dict(
            source_url="https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip",
            kind="archive",
            expected_sha256="a9f932ff4096a7df797d1547987937f34d3995ac445b4748177114488d12b010",
        ),
        dict(
            source_url="https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities",
            kind="dictionary",
        ),
    ]
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(fetch, rows))
    p = ROOT / "results/validation/source_recheck.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print({"checked": len(results), "retrieved": sum("sha256" in r for r in results)})


if __name__ == "__main__":
    main()
