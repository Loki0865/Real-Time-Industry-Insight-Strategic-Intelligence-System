from prophet import Prophet
import pandas as pd
import numpy as np
from datetime import datetime, date

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
    
    # Configure Prophet
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.8
    )
    
    m.add_seasonality(name='weekly', period=7, fourier_order=3)
    
    m.fit(sub)
    
    # Get today and last historical date
    today = date.today()
    last_hist = sub['ds'].max().date()
    
    # Calculate gap days to fill to today + extend 'days' ahead
    gap_days = (today - last_hist).days
    total_periods = gap_days + days if gap_days > 0 else days
    
    future = m.make_future_dataframe(periods=total_periods)
    
    # Filter to start from today
    future = future[future['ds'] >= pd.to_datetime(today)]
    
    forecast = m.predict(future)
    
    # Simple forecast info
    forecast_end = forecast['ds'].max().date()
    print(f"📈 Forecast for {keyword} starts {today} and extends to: {forecast_end}")
    
    return m, forecast