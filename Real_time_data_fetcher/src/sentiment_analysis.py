import os
import re
import string
import requests
import pandas as pd
import feedparser
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm
from urllib.parse import quote
import json
import time
import requests as http
from textblob import TextBlob  # Added for fallback

# --------------------------
# Load API Keys
# --------------------------
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# Configure Gemini only if key is present - Updated to gemini-2.5-flash
gemini_model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        print(f"⚠️ Gemini configuration failed, falling back to neutral sentiment: {e}")

# Quota alert guard to avoid spamming
QUOTA_ALERT_SENT = False

def _write_quota_status(exceeded: bool, retry_seconds: int | None = None, detail: str | None = None):
    try:
        os.makedirs("data/processed", exist_ok=True)
        status_path = "data/processed/gemini_quota_status.json"
        payload = {
            "timestamp": int(time.time()),
            "exceeded": bool(exceeded),
            "retry_seconds": int(retry_seconds) if retry_seconds is not None else None,
            "detail": detail or ""
        }
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as e:
        print(f"⚠️ Could not write quota status: {e}")

def _notify_quota_exceeded(detail: str, retry_seconds: int | None = None):
    global QUOTA_ALERT_SENT
    if QUOTA_ALERT_SENT:
        return
    QUOTA_ALERT_SENT = True
    # Console
    retry_msg = f" Retry in ~{retry_seconds}s." if retry_seconds else ""
    print(f"🚫 Gemini quota exceeded.{retry_msg}")
    # Slack
    if SLACK_WEBHOOK_URL:
        try:
            http.post(SLACK_WEBHOOK_URL, json={
                "text": f"[Gemini] Quota exceeded.{retry_msg}\n{detail[:500]}"
            })
        except Exception as e:
            print(f"⚠️ Slack notify failed: {e}")
    # File status
    _write_quota_status(True, retry_seconds, detail)

# --------------------------
# Cleaning Function
# --------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"  
        "\U0001F300-\U0001F5FF"  
        "\U0001F680-\U0001F6FF"  
        "\U0001F1E0-\U0001F1FF"  
        "\U00002700-\U000027BF"  
        "\U0001F900-\U0001F9FF"  
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)
    allowed_chars = string.ascii_letters + string.digits + " .,;:!?'-"
    text = "".join([c for c in text if c in allowed_chars])
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]

# --------------------------
# Fetch News from NewsAPI
# --------------------------
def fetch_news(query="AI", language="en", page_size=10):
    if not NEWS_API_KEY:
        return pd.DataFrame(columns=["title", "description", "content", "publishedAt", "url", "source", "topic"])
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": language,
            "pageSize": page_size,
            "apiKey": NEWS_API_KEY
        }
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        data = []
        for art in articles:
            data.append({
                "title": art.get("title"),
                "description": art.get("description"),
                "content": art.get("content"),
                "publishedAt": art.get("publishedAt"),
                "url": art.get("url"),
                "source": art.get("source", {}).get("name", ""),
                "topic": query
            })
        return pd.DataFrame(data)
    except Exception as e:
        print(f"⚠️ NewsAPI fetch failed for query '{query}': {e}")
        return pd.DataFrame(columns=["title", "description", "content", "publishedAt", "url", "source", "topic"])

# --------------------------
# Fetch Google News RSS
# --------------------------
def fetch_google_news(query="AI", language="en", max_articles=10):
    query_encoded = quote(query)
    rss_url = f"https://news.google.com/rss/search?q={query_encoded}&hl={language}-US&gl=US&ceid=US:{language.upper()}&when:1d"
    feed = feedparser.parse(rss_url)
    articles = []
    for entry in feed.entries[:max_articles]:
        articles.append({
            "title": entry.get("title"),
            "description": entry.get("summary"),
            "content": "",
            "publishedAt": entry.get("published", ""),
            "url": entry.get("link"),
            "source": entry.get("source", {}).get("title", ""),
            "topic": query
        })
    return pd.DataFrame(articles)

# --------------------------
# Batched Sentiment with Gemini + TextBlob Fallback
# --------------------------
def classify_sentiments_batch(texts, batch_size=10):
    results = []
    if gemini_model is None:
        print("⚠️ Gemini not configured; using TextBlob fallback")
        for text in texts:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            label = "Positive" if polarity > 0.05 else "Negative" if polarity < -0.05 else "Neutral"
            results.append((label, float(polarity)))
        return results

    for i in tqdm(range(0, len(texts), batch_size), desc="Classifying Sentiment (batched)"):
        batch = texts[i:i+batch_size]
        joined = "\n".join([f"{j+1}. {t}" for j, t in enumerate(batch)])
        prompt = f"""
        Analyze the sentiment for the following {len(batch)} texts.

        For each text, return ONLY a JSON object in a list with:
        - "label": one of ["Positive", "Negative", "Neutral"]
        - "score": number between -1.0 and 1.0

        Example output:
        [
          {{"label": "Positive", "score": 0.8}},
          {{"label": "Negative", "score": -0.6}}
        ]

        Texts:
        {joined}
        """

        try:
            response = gemini_model.generate_content(prompt)
            raw = response.text.strip()
            match = re.search(r'\[.*\]', raw, re.S)
            if not match:
                raise ValueError("No JSON array found")
            parsed = json.loads(match.group())
            for item in parsed:
                label = item.get("label", "Neutral").capitalize()
                score = float(item.get("score", 0.0))
                if label not in ["Positive", "Negative", "Neutral"]:
                    label = "Neutral"
                if score < -1.0 or score > 1.0:
                    score = 0.0
                results.append((label, score))
        except Exception as e:
            err_text = str(e)
            retry_seconds = None
            m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err_text)
            if m:
                retry_seconds = int(m.group(1))
            if ("quota" in err_text.lower()) or ("429" in err_text) or ("404" in err_text):
                _notify_quota_exceeded(err_text, retry_seconds)
            print(f"⚠️ Batch sentiment failed: {e}, using TextBlob fallback for this batch")
            for text in batch:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                label = "Positive" if polarity > 0.05 else "Negative" if polarity < -0.05 else "Neutral"
                results.append((label, float(polarity)))

    return results

# --------------------------
# Main Pipeline
# --------------------------
def run_pipeline(topics=None, aggregate=True):
    if topics is None:
        topics = ["Artificial Intelligence", "Technology", "Trending", "Market"]

    all_dfs = []
    for topic in topics:
        print(f"Fetching news for topic: {topic}")
        df_newsapi = fetch_news(topic)
        df_google = fetch_google_news(topic)
        frames = [d for d in [df_newsapi, df_google] if not d.empty]
        if len(frames) == 0:
            continue
        df = pd.concat(frames, ignore_index=True)
        df["topic"] = topic
        all_dfs.append(df)

    if not all_dfs:
        print("⚠️ No articles found for any topic.")
        return None

    df_all = pd.concat(all_dfs, ignore_index=True)
    if df_all.empty:
        print("⚠️ No articles found for any topic.")
        return None

    df_all.drop_duplicates(subset=["title", "url"], inplace=True)
    df_all["date"] = pd.to_datetime(df_all["publishedAt"], errors="coerce").dt.tz_localize(None).dt.date
    
    today = pd.Timestamp.now().date()
    seven_days_ago = today - pd.Timedelta(days=7)
    df_all = df_all[df_all["date"] >= seven_days_ago]
    
    print(f"📅 Date range: {df_all['date'].min()} to {df_all['date'].max()}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("data/raw", exist_ok=True)
    raw_path = f"data/raw/news_multi_{timestamp}.csv"
    df_all.to_csv(raw_path, index=False)
    print(f"✅ Raw data saved to {raw_path}")

    df_all["clean_text"] = (
        df_all["title"].fillna("") + " " + df_all["description"].fillna("")
    ).apply(clean_text)

    sentiments = classify_sentiments_batch(df_all["clean_text"].tolist(), batch_size=10)
    labels, scores = zip(*sentiments)
    df_all["sentiment_gemini"] = labels
    df_all["sentiment_score"] = scores

    os.makedirs("data/processed", exist_ok=True)
    articles_path = f"data/processed/articles_with_sentiment_{timestamp}.csv"
    df_all.to_csv(articles_path, index=False)
    print(f"✅ Articles with sentiment saved to {articles_path}")

    if aggregate:
        trend_df = (
            df_all.groupby(["topic", "date"])
            .agg(sentiment_score=("sentiment_score", "mean"),
                 articles_count=("sentiment_score", "count"))
            .reset_index()
            .rename(columns={"topic": "keyword"})
        )
        final_path = f"data/processed/news_with_sentiment_multi_{timestamp}.csv"
        trend_df.to_csv(final_path, index=False)
        print(f"✅ Final dataset saved to {final_path}")
        return trend_df

    return df_all