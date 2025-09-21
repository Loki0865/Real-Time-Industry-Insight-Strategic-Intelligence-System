# 📰 News Sentiment Analysis System

A simple, automated news sentiment analysis system that fetches news articles, analyzes their sentiment using AI, and generates forecasts.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file with your API keys:
```
GOOGLE_API_KEY=your_gemini_api_key
NEWS_API_KEY=your_newsapi_key
SLACK_WEBHOOK_URL=your_slack_webhook_url
```

### 3. Run Daily Analysis
```bash
python run_daily.py
```

## 📊 What It Does

1. **Fetches News**: Collects latest articles from Google News RSS and NewsAPI
2. **Analyzes Sentiment**: Uses Google Gemini AI to classify sentiment (-1.0 to 1.0)
3. **Aggregates Data**: Combines with historical data for trend analysis
4. **Generates Forecasts**: Creates 7-day predictions using Prophet
5. **Sends Alerts**: Notifies about significant sentiment changes

## 📁 Project Structure

```
Real time data fetcher/
├── run_daily.py              # Main daily runner
├── src/
│   ├── main.py              # Core analysis pipeline
│   ├── sentiment_analysis.py # News fetching & sentiment analysis
│   ├── forecasting.py       # Prophet forecasting
│   └── alerts.py            # Alert system
├── data/
│   ├── raw/                 # Raw news articles
│   └── processed/           # Processed data + history
├── logs/                    # System logs
└── requirements.txt         # Python dependencies
```

## 🔧 Configuration

### Topics Analyzed
- Artificial Intelligence
- Technology  
- Trending
- Market

### Alert Thresholds
- **Sentiment Drop**: Below -0.3
- **Volume Surge**: Z-score above 1.5

### Forecast Period
- **7 days ahead** using Facebook Prophet

## 📈 Output Files

- `data/processed/news_history.csv` - Historical trend data
- `data/processed/articles_with_sentiment_*.csv` - Individual articles with sentiment
- `logs/` - System logs and error tracking

## 🎯 Daily Usage

Simply run once per day:
```bash
python run_daily.py
```

The system will:
- ✅ Fetch fresh news data
- ✅ Analyze sentiment using AI
- ✅ Update historical trends
- ✅ Generate 7-day forecasts
- ✅ Check for alerts
- ✅ Save all data and logs

## 🔍 Monitoring

Check the console output for:
- Data fetching status
- Sentiment analysis progress
- Forecast generation
- Alert notifications
- Error messages

## 🛠️ Troubleshooting

### Common Issues:
1. **No API Key**: Check `.env` file
2. **No Internet**: Verify connectivity
3. **No Data**: Check news sources availability
4. **Forecast Errors**: Ensure enough historical data

### Manual Testing:
```bash
# Test individual components
python src/main.py
```

## 📊 Features

- **Multi-source News**: Google News RSS + NewsAPI
- **AI Sentiment Analysis**: Google Gemini 1.5 Flash
- **Time Series Forecasting**: Facebook Prophet
- **Real-time Alerts**: Sentiment drops and volume surges
- **Historical Tracking**: Maintains data across runs
- **Comprehensive Logging**: Full activity tracking

## 🎉 Success!

Your news sentiment analysis system is ready! Run `python run_daily.py` daily to get:
- Fresh news sentiment analysis
- 7-day trend forecasts
- Real-time alerts for significant changes
- Complete historical data tracking

**Happy analyzing!** 📈