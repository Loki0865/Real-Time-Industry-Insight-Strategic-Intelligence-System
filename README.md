# Real-Time Industry Insight — Strategic Intelligence System

A Streamlit-powered dashboard that continuously ingests industry news, performs sentiment analysis, forecasts sentiment trends, and raises actionable alerts. Built entirely in Python.

## Key Features

- Real-time news ingestion by keyword
- Robust text cleaning and batch sentiment classification
- Daily sentiment aggregation per keyword
- Time-series forecasting using Prophet
- Smart alerting (e.g., significant sentiment drops, surge detection via z-score)
- Interactive visualizations with Plotly (charts, subplots)
- Streamlit UI with wide layout, tabs, metric cards, and alert styling
- Caching to keep the app fast and API-efficient

## Tech Stack

- Python (100%)
- Streamlit (UI)
- pandas, NumPy (data handling)
- Prophet (time-series forecasting)
- Plotly (interactive charts)
- SciPy (z-score computations)
- requests (HTTP/API calls)

## Project Structure

This repository contains the Streamlit app and supporting modules:

```
Real-Time-Industry-Insight-Strategic-Intelligence-System/
├─ Real_time_data_fetcher/
│  └─ src/
│     ├─ app.py                       # Streamlit entry point
│     ├─ sentiment_analysis.py        # fetch_news, classify_sentiments_batch, clean_text
│     ├─ forecasting.py               # forecast_sentiment
│     └─ alerts.py                    # check_alerts, safe_zscore
└─ (other project files)
```

Notes:
- The main Streamlit app is at `Real_time_data_fetcher/src/app.py`.
- The app imports:
  - `from sentiment_analysis import fetch_news, classify_sentiments_batch, clean_text`
  - `from forecasting import forecast_sentiment`
  - `from alerts import check_alerts, safe_zscore`

## How It Works

1. Configuration and UI
   - Streamlit is configured for a wide layout and a dashboard experience with custom CSS (metric cards, alert boxes).
   - Tabs and headers help organize insights.

2. Data Ingestion and Preprocessing
   - News articles are fetched per keyword.
   - Text is cleaned: titles/descriptions combined, standardized, and prepared for modeling.

3. Sentiment Analysis
   - Batch classification returns a label and a confidence score per article.
   - Results are aggregated by keyword and date (daily mean sentiment).

4. Forecasting
   - Uses Prophet to forecast future sentiment trajectories for each keyword.
   - Supports visual trend analysis and forward-looking insights.

5. Alerting
   - Alerts are generated when thresholds are crossed (e.g., notable negative drops, high z-score surges).
   - Thresholds (from `app.py`):
     - `sentiment_drop`: -0.3
     - `surge_zscore`: 1.5

6. Caching
   - Data fetch and processing are cached (`st.cache_data(ttl=300)`) to reduce latency and API overhead.

## Getting Started

### Prerequisites

- Python 3.9–3.11 recommended
- System build tools required by Prophet (platform-dependent)

### Installation

- If the project contains a `requirements.txt`, run:
  ```
  pip install -r requirements.txt
  ```
- If not, install core dependencies manually:
  ```
  pip install streamlit pandas numpy prophet plotly scipy requests
  ```

### Configuration

- News/API credentials:
  - The `fetch_news` function likely requires a news/search API key.
  - Supply secrets either via environment variables or Streamlit secrets.

Using Streamlit secrets (recommended for deployment):
Create `.streamlit/secrets.toml`:
```toml
# Example keys (adapt to your provider and code)
NEWS_API_KEY = "your_api_key_here"
```

Alternatively, export environment variables as supported by your `fetch_news` implementation:
```
export NEWS_API_KEY="your_api_key_here"
```

### Run Locally

From the repository root:
```
streamlit run Real_time_data_fetcher/src/app.py
```

Then open the local URL that Streamlit prints (usually http://localhost:8501).

## Using the App

- Enter/select the keywords you want to monitor.
- The app will:
  - Fetch the latest articles per keyword (default pagination: pages=2, articles_per_page=20).
  - Clean text, classify sentiments in batch, and compute daily means.
  - Forecast sentiment via Prophet.
  - Display insights and raise alerts when thresholds are exceeded.

## Deployment (Streamlit)

This project is deployed using Streamlit. To deploy on Streamlit Community Cloud:

1. Push your code to a public GitHub repository (this repo).
2. In Streamlit Community Cloud, create a new app:
   - Repository: `Loki0865/Real-Time-Industry-Insight-Strategic-Intelligence-System`
   - Main file path: `Real_time_data_fetcher/src/app.py`
3. Set required secrets in the app settings:
   - Add your news/API provider keys in “Secrets” (same keys as in `.streamlit/secrets.toml`).
4. Deploy. Streamlit will install dependencies and run the app.

Tips:
- If you don’t have a `requirements.txt`, add one to ensure consistent environment setup in the cloud.

## Customization

- Thresholds: Adjust `THRESHOLDS` in `app.py` to tune sensitivity (e.g., `sentiment_drop`, `surge_zscore`).
- Forecast horizon: Modify forecasting parameters in `forecasting.py`.
- Visualization: Update Plotly figures/subplots in `app.py` for custom charts.
- Data sources: Extend or swap `fetch_news` to use different providers or add domain filtering.

## Troubleshooting

- Prophet installation issues:
  - Ensure compatible Python version and that system compilers are available.
- Empty charts or no results:
  - Check API keys/secrets and keyword selection.
  - Confirm rate limits or provider availability.
- Slow performance:
  - Reduce pages/articles per page.
  - Increase cache TTL, or add intermediate caching layers.

## Roadmap Ideas

- Multi-provider news ingestion with fallback/merge
- Entity-level sentiment and topic clustering
- Anomaly detection beyond z-score (e.g., seasonal-HBOS, STL residuals)
- Alert notifications (email/Slack/Webhooks)
- Export dashboards and data snapshots

## License

Add your preferred license (e.g., MIT, Apache-2.0) at the root of the repository as `LICENSE`.

## Acknowledgments

- Streamlit for rapid app development
- Prophet for robust time-series forecasting
- Plotly for interactive visualizations
- Open-source ML/NLP ecosystem
