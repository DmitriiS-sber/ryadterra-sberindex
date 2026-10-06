"""Download primary sources, validate the municipal panel and extract dated news features."""

from __future__ import annotations
import argparse, hashlib, json, re, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED, RESULTS = (ROOT / x for x in ["data/raw", "data/processed", "results"])
ARCHIVE_URL = "https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip"
ARCHIVE_SHA256 = "a9f932ff4096a7df797d1547987937f34d3995ac445b4748177114488d12b010"
NEWS_FILES = [
    "10022023_133000key.htm",
    "17032023_133000Key.htm",
    "28042023_133000Key.htm",
    "09062023_133024Key.htm",
    "21072023_133000Key.htm",
    "15082023_103000Key.htm",
    "15092023_133000key.htm",
    "27102023_133000key.htm",
    "15122023_133000key.htm",
    "16022024_133000key.htm",
    "22032024_133000key.htm",
    "26042024_133000key.htm",
    "07062024_133000Key.htm",
    "26072024_133000Key.htm",
    "13092024_133000Key.htm",
    "25102024_133000Key.htm",
    "20122024_133000key.htm",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, path: Path) -> dict:
    if path.exists() and path.stat().st_size:
        return {
            "url": url,
            "local_file": str(path.relative_to(ROOT)),
            "status": "cached",
            "sha256": sha256(path),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(2):
        try:
            r = requests.get(
                url, timeout=30, headers={"User-Agent": "SberIndexResearch/1.0"}
            )
            r.raise_for_status()
            path.write_bytes(r.content)
            return {
                "url": url,
                "local_file": str(path.relative_to(ROOT)),
                "status": "downloaded",
                "sha256": sha256(path),
            }
        except requests.RequestException as exc:
            last_error = str(exc)
    raise RuntimeError(f"Primary-source download failed: {url}: {last_error}")


def extract_news(file_name: str) -> dict:
    url = f"https://www.cbr.ru/press/pr/?file={file_name}"
    path = RAW / "news" / file_name.replace(".htm", ".html")
    source = download(url, path)
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    main = soup.select_one(".document") or soup.select_one("main") or soup
    title = soup.select_one("h1").get_text(" ", strip=True)
    if "ключевую ставку" not in title.lower():
        raise ValueError(f"Unexpected source title for {file_name}: {title}")
    text = main.get_text(" ", strip=True).replace("\xa0", " ")
    # Publication date/time encoded in the primary-source URL, verified against its page date.
    published = pd.to_datetime(file_name[:8], format="%d%m%Y")
    date_marker = published.strftime("%d.%m.%Y")
    if date_marker not in soup.get_text(" ", strip=True):
        raise ValueError(
            f"Publication date not present on the source page: {file_name}"
        )
    rate_match = re.search(r"(?:до|уровне)\s+(\d+[,.]\d+)\s*%", title)
    delta_match = re.search(r"на\s+(\d+)\s*б", title)
    if not rate_match:
        raise ValueError(f"Cannot parse rate: {title}")
    delta = (float(delta_match[1]) / 100) if delta_match else 0.0
    if "снизить" in title.lower():
        delta *= -1
    lower = text.lower()
    forward_tightening = int(
        any(
            x in lower
            for x in [
                "допускает возможность дальнейшего повышения",
                "будет оценивать целесообразность дальнейшего повышения",
                "будет оценивать целесообразность повышения",
                "будет оценивать целесообразность дополнительного повышения",
                "не исключает возможности повышения",
            ]
        )
    )
    demand_pressure = int(
        any(
            x in lower
            for x in [
                "спроса превышает возможности",
                "спроса продолжает значительно опережать",
                "спроса продолжает опережать",
                "расширение внутреннего спроса",
                "внутренний спрос продолжает",
            ]
        )
    )
    return dict(
        publication_date=published.strftime("%Y-%m-%d"),
        publication_time=file_name[9:15],
        new_rate=float(rate_match[1].replace(",", ".")),
        rate_change_pp=delta,
        forward_tightening=forward_tightening,
        demand_pressure=demand_pressure,
        source_url=url,
        source_sha256=source["sha256"],
        text_characters=len(text),
        title=title,
    )


def main():
    for p in [RAW, PROCESSED, RESULTS]:
        p.mkdir(parents=True, exist_ok=True)
    archive = RAW / "hackathon_http.zip"
    if not archive.exists():
        try:
            download(ARCHIVE_URL, archive)
        except RuntimeError:
            download(ARCHIVE_URL.replace("https:", "http:"), archive)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError(
            "Archive version differs from the pinned experiment; inspect before changing its hash."
        )
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith("consumption.parquet"):
                (RAW / "consumption.parquet").write_bytes(z.read(name))
            if name.endswith("лицензия.pdf"):
                (RAW / "data_license.pdf").write_bytes(z.read(name))
    df = pd.read_parquet(RAW / "consumption.parquet")
    expected = {"date", "territory_id", "category", "value"}
    if set(df.columns) != expected:
        raise ValueError(f"Unexpected columns: {df.columns.tolist()}")
    if df.duplicated(["date", "territory_id", "category"]).any():
        raise ValueError("Duplicate panel keys")
    if df.value.isna().any() or (df.value <= 0).any():
        raise ValueError("Missing/nonpositive stored values")
    panel = (
        df[df.category.eq("Все категории")]
        .pivot(index="territory_id", columns="date", values="value")
        .sort_index()
    )
    train = panel.loc[:, "2023-01":"2023-12"]
    selected = panel.loc[train.notna().all(axis=1)].copy()
    selected.to_parquet(PROCESSED / "panel.parquet")
    selected.to_csv(PROCESSED / "panel.csv", index=True)
    selection = pd.DataFrame(
        {
            "territory_id": panel.index,
            "eligible_2023": train.notna().all(axis=1).values,
            "mean_2023_rub": train.mean(axis=1).values,
        }
    )
    selection["baseline_group"] = pd.qcut(
        selection["mean_2023_rub"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)
    selection.to_csv(PROCESSED / "selection.csv", index=False)
    audit = dict(
        raw_rows=len(df),
        municipalities=df.territory_id.nunique(),
        months=df.date.nunique(),
        date_min=df.date.min(),
        date_max=df.date.max(),
        categories=sorted(df.category.unique().tolist()),
        selected_municipalities=len(selected),
        complete_all_24=int(panel.notna().all(axis=1).sum()),
        excluded_for_incomplete_2023=int(len(panel) - len(selected)),
        selected_missing_2024=int(selected.loc[:, "2024-01":].isna().sum().sum()),
        duplicates=0,
        units="RUB: estimate of average monthly cashless spending of residents",
        observed_column="value",
        documented_column="consumption",
        archive_url=ARCHIVE_URL,
        actual_download_url=ARCHIVE_URL.replace("https:", "http:"),
        archive_sha256=sha256(archive),
        consumption_sha256=sha256(RAW / "consumption.parquet"),
        download_date="2026-10-03",
        license="CC BY-SA 4.0",
    )
    (RESULTS / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2)
    )
    print("AUDIT", json.dumps(audit, ensure_ascii=False), flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        news = list(pool.map(extract_news, NEWS_FILES))
    news = pd.DataFrame(news).sort_values("publication_date")
    news.to_csv(PROCESSED / "news_events.csv", index=False)
    # Official daily key rate, independently sourced from the CBR table.
    keyrate_url = "https://www.cbr.ru/hd_base/KeyRate/?UniDbQuery.Posted=True&UniDbQuery.From=01.01.2023&UniDbQuery.To=31.12.2024"
    download(keyrate_url, RAW / "cbr_keyrate.html")
    soup = BeautifulSoup((RAW / "cbr_keyrate.html").read_bytes(), "html.parser")
    rows = []
    for tr in soup.select("table tr"):
        cells = [x.get_text(" ", strip=True) for x in tr.select("td")]
        if len(cells) == 2 and re.match(r"\d{2}\.\d{2}\.\d{4}", cells[0]):
            rows.append(
                [
                    pd.to_datetime(cells[0], format="%d.%m.%Y"),
                    float(cells[1].replace(",", ".")),
                ]
            )
    daily = pd.DataFrame(rows, columns=["date", "rate"]).sort_values("date")
    if daily.empty:
        raise ValueError("CBR daily table is empty")
    daily.to_csv(PROCESSED / "keyrate_daily.csv", index=False)
    # All monthly features are computed by end-of-month publication availability.
    monthly = []
    news["date"] = pd.to_datetime(news.publication_date)
    for period in pd.period_range("2023-01", "2024-12", freq="M"):
        end = period.end_time
        known = news[news.date <= end]
        recent = known[known.date >= (period - 2).start_time]
        last = known.iloc[-1] if len(known) else None
        rate = float(daily[daily.date <= end].iloc[-1].rate)
        monthly.append(
            dict(
                date=str(period),
                key_rate=rate,
                hikes_3m_pp=float(recent.rate_change_pp.clip(lower=0).sum()),
                forward_tightening=(
                    int(last.forward_tightening) if last is not None else 0
                ),
                demand_pressure=int(last.demand_pressure) if last is not None else 0,
                last_news_date=str(last.publication_date) if last is not None else "",
            )
        )
    pd.DataFrame(monthly).to_csv(PROCESSED / "monthly_news_features.csv", index=False)
    manifest = {
        "retrieved_at": "2026-10-03",
        "archive_sha256": ARCHIVE_SHA256,
        "news_sources": news[
            ["publication_date", "source_url", "source_sha256"]
        ].to_dict("records"),
        "keyrate_url": keyrate_url,
        "keyrate_sha256": sha256(RAW / "cbr_keyrate.html"),
    }
    (RESULTS / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print("NEWS", len(news), "DAILY_RATES", len(daily), flush=True)


if __name__ == "__main__":
    main()
