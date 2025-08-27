"""
collect_news.py — minimal, working Week 1 news collector using Google News RSS.

Run:
    python src/collect_news.py
Output:
    data/raw/news/news_YYYYMMDD_HHMMSS.csv
"""
from urllib.parse import quote_plus
import feedparser
import pandas as pd
import os
from datetime import datetime, timezone

def fetch_google_news(query: str, region: str = "IN", lang: str = "en", max_items: int = 50):
    """
    Fetch headlines from Google News RSS for a given query.
    - query: search terms, e.g., "market trends"
    - region/lang: regionalization for India English by default
    - max_items: limit the number of items
    Returns a list of dict rows.
    """
    q = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={q}&hl={lang}-{region}&gl={region}&ceid={region}:{lang}"
    feed = feedparser.parse(url)

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for entry in feed.entries[:max_items]:
        rows.append({
            "query": query,
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
            "fetched_at_utc": fetched_at
        })
    return rows

def save_rows_to_csv(rows, out_dir="data/raw/news"):
    """
    Save list of dicts to a timestamped CSV.
    """
    if not rows:
        print("No rows to save.")
        return None
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"news_{ts}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path

def main():
    # You can edit these search queries for your domain later
    queries = [
        "competitor",
        "emerging market trends",
        "AI market",
        "company earnings",
        "merger acquisition"
    ]

    all_rows = []
    for q in queries:
        rows = fetch_google_news(q, max_items=40)
        all_rows.extend(rows)
        print(f"Fetched {len(rows)} items for '{q}'")

    out_path = save_rows_to_csv(all_rows)
    if out_path:
        print(f"Saved: {out_path}")
    else:
        print("Nothing saved. Try a different query or check your internet connection.")

if __name__ == "__main__":
    main()
