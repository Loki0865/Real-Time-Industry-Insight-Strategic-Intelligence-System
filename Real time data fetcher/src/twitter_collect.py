"""
twitter_collect.py — optional Twitter collector for later (requires a Bearer Token)
Setup:
  1) Get a developer account and generate a Bearer Token.
  2) Put it in a `.env` file as: TWITTER_BEARER_TOKEN=YOUR_TOKEN
Run:
    python src/twitter_collect.py
Output:
    data/raw/twitter/tweets_YYYYMMDD_HHMMSS.csv
"""
import os
from datetime import datetime
import pandas as pd

from tweepy import Client
from config import get_env

def fetch_tweets(query: str, max_results: int = 30):
    token = get_env("TWITTER_BEARER_TOKEN")
    if not token:
        print("No TWITTER_BEARER_TOKEN found in environment. Skipping.")
        return []

    client = Client(bearer_token=token)
    resp = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=["created_at","lang","public_metrics"])
    rows = []
    if resp.data:
        for t in resp.data:
            rows.append({
                "query": query,
                "id": str(t.id),
                "text": t.text,
                "created_at": getattr(t, "created_at", ""),
                "lang": getattr(t, "lang", ""),
            })
    return rows

def save_rows_to_csv(rows, out_dir="data/raw/twitter"):
    if not rows:
        print("No tweets to save.")
        return None
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"tweets_{ts}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    return out_path

def main():
    queries = [
        "AI market trends lang:en -is:retweet",
        "mergers acquisitions lang:en -is:retweet",
    ]
    all_rows = []
    for q in queries:
        items = fetch_tweets(q, max_results=30)
        all_rows.extend(items)
        print(f"Fetched {len(items)} tweets for '{q}'")
    path = save_rows_to_csv(all_rows)
    if path:
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
