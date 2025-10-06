import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt  

from sentiment_analysis import run_pipeline
from forecasting import forecast_sentiment
from alerts import check_alerts 

load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

THRESHOLDS = {
    'sentiment_drop': -0.3,
    'surge_zscore': 1.5
}


def load_historical_data():
    """Load existing historical data if available"""
    history_file = "data/processed/news_history.csv"
    if os.path.exists(history_file):
        try:
            historical_df = pd.read_csv(history_file)
            historical_df['date'] = pd.to_datetime(historical_df['date'])
            print(f"📚 Loaded {len(historical_df)} historical data points")
            return historical_df
        except Exception as e:
            print(f"⚠️ Could not load historical data: {e}")
    return pd.DataFrame()


def save_updated_history(combined_df):
    """Save the combined historical + new data"""
    history_file = "data/processed/news_history.csv"
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    combined_df.to_csv(history_file, index=False)
    print(f"💾 Updated history file with {len(combined_df)} total data points")


def process_new_data(new_df):
    """Process and normalize new data"""
    # Normalize date column
    possible_date_cols = ['date', 'publishedAt', 'time', 'published_date']
    found_date_col = next((col for col in possible_date_cols if col in new_df.columns), None)
    if not found_date_col:
        raise KeyError("No date-like column found in new dataframe!")
    if found_date_col != 'date':
        new_df = new_df.rename(columns={found_date_col: 'date'})

    new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce').dt.tz_localize(None)
    new_df = new_df.dropna(subset=['date'])

    # Normalize topic column
    possible_topic_cols = ['topic', 'keyword', 'query', 'Topic', 'search_term']
    found_topic_col = next((col for col in possible_topic_cols if col in new_df.columns), None)
    if not found_topic_col:
        raise KeyError("No topic-like column found in new dataframe!")
    if found_topic_col != 'topic':
        new_df = new_df.rename(columns={found_topic_col: 'topic'})

    # Aggregate by topic + day
    new_trend = (
        new_df.groupby(['topic', pd.Grouper(key='date', freq='D')])
          .agg(
              avg_sentiment=('sentiment_score', 'mean'),
              articles_count=('sentiment_score', 'count')
          )
          .reset_index()
    )

    # Rename for consistency
    new_trend = new_trend.rename(columns={
        'topic': 'keyword',
        'avg_sentiment': 'sentiment_score'
    })

    return new_trend


def combine_data(historical_df, new_trend):
    """Combine historical and new data"""
    if not historical_df.empty:
        # Ensure column names match
        if 'content' in historical_df.columns:
            historical_df = historical_df.rename(columns={'content': 'keyword'})
        
        # Combine and remove duplicates (keep latest)
        combined_df = pd.concat([historical_df, new_trend], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['keyword', 'date'], keep='last')
        combined_df = combined_df.sort_values(['keyword', 'date'])
        
        print(f"🔄 Combined {len(historical_df)} historical + {len(new_trend)} new = {len(combined_df)} total data points")
        
        # Save updated history
        save_updated_history(combined_df)
        return combined_df
    else:
        save_updated_history(new_trend)
        return new_trend


def plot_trend(keyword, data):
    """Plot sentiment trend for a keyword"""
    plt.figure(figsize=(10, 5))
    
    plt.plot(data['date'], data['sentiment_score'], 'o-', 
            linewidth=2, markersize=6, 
            color='#2E86AB', markerfacecolor='#A23B72', 
            markeredgecolor='#2E86AB')
    
    # Reference lines
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.axhline(y=0.3, color='green', linestyle=':', linewidth=1, alpha=0.7)
    plt.axhline(y=-0.3, color='red', linestyle=':', linewidth=1, alpha=0.7)
    
    plt.title(f'Sentiment Trend: {keyword}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Sentiment Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Statistics
    avg_sentiment = data['sentiment_score'].mean()
    latest_sentiment = data['sentiment_score'].iloc[-1]
    stats_text = f'Average: {avg_sentiment:.2f}\nLatest: {latest_sentiment:.2f}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_forecast(keyword, trend_data, forecast_data):
    """Plot forecast for a keyword"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    historical_data = trend_data[trend_data['keyword'] == keyword].copy()
    historical_data = historical_data.sort_values('date')
    
    forecast_start_date = historical_data['date'].max()
    future_forecast = forecast_data[forecast_data['ds'] > forecast_start_date]
    
    # Plot data
    ax.plot(historical_data['date'], historical_data['sentiment_score'], 
           'o-', label='Historical Data', 
           linewidth=2, markersize=5, 
           color='#2E86AB', alpha=0.8)
    
    ax.plot(future_forecast['ds'], future_forecast['yhat'], 
           'o-', label='7-Day Forecast', 
           linewidth=2, markersize=5, 
           color='#E74C3C', alpha=0.8)
    
    # Confidence interval
    ax.fill_between(future_forecast['ds'], future_forecast['yhat_lower'], future_forecast['yhat_upper'], 
                   alpha=0.15, color='#E74C3C')
    
    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.6)
    ax.axhline(y=0.3, color='green', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=-0.3, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=forecast_start_date, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    
    ax.set_title(f'Sentiment Forecast: {keyword}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sentiment Score', fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Statistics
    avg_historical = historical_data['sentiment_score'].mean()
    latest_sentiment = historical_data['sentiment_score'].iloc[-1]
    forecast_avg = future_forecast['yhat'].mean()
    forecast_trend = "↗ Improving" if future_forecast['yhat'].iloc[-1] > future_forecast['yhat'].iloc[0] else "↘ Declining"
    
    stats_text = f'Historical Avg: {avg_historical:.2f}\nLatest: {latest_sentiment:.2f}\nForecast Avg: {forecast_avg:.2f}\nTrend: {forecast_trend}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    print("🚀 Starting full pipeline...")

    try:
        # Load data
        historical_df = load_historical_data()
        topics = ["Artificial Intelligence", "Technology", "Trending", "Market"]
        new_df = run_pipeline(topics)

        if new_df is None or new_df.empty:
            if not historical_df.empty:
                print("📊 Using only historical data for analysis...")
                trend = historical_df
            else:
                print("❌ No data available for analysis.")
                return
        else:
            # Process and combine data
            new_trend = process_new_data(new_df)
            trend = combine_data(historical_df, new_trend)
            
            # Ensure all dates are Timestamps
            trend['date'] = pd.to_datetime(trend['date'])
            trend = trend.sort_values(['keyword', 'date'])

        print("📊 Trend data sample:")
        print(trend.head())

        # Plot trends
        for kw in trend['keyword'].unique():
            sub = trend[trend['keyword'] == kw]
            plot_trend(kw, sub)

        # Generate forecasts and plots
        for kw in trend['keyword'].unique():
            model, forecast = forecast_sentiment(trend, kw)
            if model is not None and forecast is not None:
                print(f"📈 Forecast ready for {kw}")
                plot_forecast(kw, trend, forecast)
            else:
                print(f"⚠️ Not enough data to forecast for {kw}")

        # Check alerts
        alerts = check_alerts(trend, THRESHOLDS, SLACK_WEBHOOK_URL)
        if alerts:
            print("⚡ Alerts triggered:", alerts)
        else:
            print("✅ No alerts.")

    except Exception as e:
        print(f"❌ Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()