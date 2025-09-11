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

# --------------------------
# Load API Keys
# --------------------------
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

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
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params)
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

# --------------------------
# Fetch Google News RSS
# --------------------------
def fetch_google_news(query="AI", language="en", max_articles=5):
    query_encoded = quote(query)
    rss_url = f"https://news.google.com/rss/search?q={query_encoded}&hl={language}-US&gl=US&ceid=US:{language.upper()}"
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
# Batched Sentiment with Gemini
# --------------------------
def classify_sentiments_batch(texts, batch_size=10):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Classifying Sentiment (batched)"):
        batch = texts[i:i+batch_size]

        # Prepare prompt
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

            # Extract JSON
            match = re.search(r'\[.*\]', raw, re.S)
            if not match:
                results.extend([("Neutral", 0.0)] * len(batch))
                continue

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
            print(f"⚠️ Batch sentiment failed: {e}")
            results.extend([("Neutral", 0.0)] * len(batch))

    return results

# --------------------------
# Main Pipeline
# --------------------------
def run_pipeline(topics=None):
    if topics is None:
        topics = ["Artificial Intelligence", "Technology", "Trending", "Market"]

    all_dfs = []

    for topic in topics:
        print(f"Fetching news for topic: {topic}")
        df_newsapi = fetch_news(topic)
        df_google = fetch_google_news(topic)
        df = pd.concat([df_newsapi, df_google], ignore_index=True)
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)

    if df_all.empty:
        print("⚠️ No articles found for any topic.")
        return

    # Deduplicate
    df_all.drop_duplicates(subset=["title", "url"], inplace=True)

    # Save raw
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = f"data/raw/news_multi_{timestamp}.csv"
    os.makedirs("data/raw", exist_ok=True)
    df_all.to_csv(raw_path, index=False)
    print(f"✅ Raw data saved to {raw_path}")

    # Clean text
    df_all["clean_text"] = df_all["title"].fillna("") + " " + df_all["description"].fillna("")
    df_all["clean_text"] = df_all["clean_text"].apply(clean_text)

    # Batch classify sentiment
    sentiments = classify_sentiments_batch(df_all["clean_text"].tolist(), batch_size=10)
    labels, scores = zip(*sentiments)
    df_all["sentiment_gemini"] = labels
    df_all["sentiment_score"] = scores

    # Save processed
    os.makedirs("data/processed", exist_ok=True)
    final_path = f"data/processed/news_with_sentiment_multi_{timestamp}.csv"
    df_all.to_csv(final_path, index=False)
    print(f"✅ Final dataset with sentiment saved to {final_path}")

# --------------------------
# Run Script
# --------------------------
if __name__ == "__main__":
    topics_to_fetch = ["Artificial Intelligence", "Technology", "Trending", "Market", "AI Trends"]
    run_pipeline(topics_to_fetch)
