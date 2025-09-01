import os
import requests
import csv
from datetime import datetime, timezone
from config import get_env

def collect_tweets_to_csv(query="python", max_results=10, csv_file="tweets.csv"):
    token = get_env("TWITTER_BEARER_TOKEN")
    if not token:
        print("❌ No TWITTER_BEARER_TOKEN found. Skipping.")
        return

    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results={max_results}&tweet.fields=created_at"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        tweets = data.get("data", [])

        if not tweets:
            print("⚠️ No tweets found.")
            return

        # Save to CSV
        with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if file.tell() == 0:  # write header if file is empty
                writer.writerow(["query", "title", "link", "published", "fetched_at_utc"])

            fetched_time = datetime.now(timezone.utc).isoformat()

            for tweet in tweets:
                tweet_id = tweet["id"]
                text = tweet["text"]
                created_at = tweet.get("created_at", "")
                link = f"https://twitter.com/i/web/status/{tweet_id}"

                writer.writerow([query, text, link, created_at, fetched_time])

        print(f"✅ {len(tweets)} tweets saved to {csv_file} successfully!")

    elif response.status_code == 429:
        print("⚠️ Rate limit hit. Try again later.")
    else:
        print("❌ Error:", response.status_code, response.text)


if __name__ == "__main__":
    collect_tweets_to_csv()
