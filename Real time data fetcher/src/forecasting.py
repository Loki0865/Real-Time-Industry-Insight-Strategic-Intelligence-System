from prophet import Prophet
import pandas as pd
import numpy as np

def forecast_sentiment(trend_df, keyword, days=7):
    sub = trend_df[trend_df['keyword'] == keyword][['date', 'sentiment_score']].rename(
        columns={'date': 'ds', 'sentiment_score': 'y'}
    )
    if len(sub) < 5:
        print(f"⚠️ Insufficient data for {keyword}: only {len(sub)} points (minimum 5 required)")
        return None, None
    
    # Ensure dates are properly formatted for Prophet
    sub['ds'] = pd.to_datetime(sub['ds'])
    sub = sub.sort_values('ds')
    
    # Simple data info
    print(f"📈 Forecasting {keyword}: {len(sub)} data points from {sub['ds'].min().date()} to {sub['ds'].max().date()}")
    
    # Configure Prophet with better parameters for sentiment data
    m = Prophet(
        yearly_seasonality=False,  # Disable yearly seasonality for short-term sentiment
        weekly_seasonality=True,    # Enable weekly patterns
        daily_seasonality=False,   # Disable daily patterns
        changepoint_prior_scale=0.05,  # More conservative trend changes
        seasonality_prior_scale=10.0,  # Allow more seasonal variation
        interval_width=0.8  # 80% confidence interval
    )
    
    # Add custom seasonality for weekly patterns
    m.add_seasonality(name='weekly', period=7, fourier_order=3)
    
    m.fit(sub)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    
    # Simple forecast info
    forecast_end = forecast['ds'].max().date()
    print(f"📈 Forecast for {keyword} extends to: {forecast_end}")
    
    return m, forecast
