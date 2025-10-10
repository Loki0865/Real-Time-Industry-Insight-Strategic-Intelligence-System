# alerts.py
import pandas as pd
import numpy as np
from scipy.stats import zscore
import requests

# ──────────────────────────────
# Safe z-score helper
# ──────────────────────────────
def safe_zscore(series: pd.Series) -> np.ndarray:
    """
    Return z-scores but avoid NaNs or warnings when variance ~0.
    """
    s = pd.Series(series)
    if len(s) < 2 or np.isclose(s.std(ddof=0), 0):
        return np.zeros_like(s, dtype=float)
    return zscore(s, nan_policy='omit')

# ──────────────────────────────
# Main alert checker
# ──────────────────────────────
def check_alerts(trend_df: pd.DataFrame, thresholds: dict, slack_webhook: str = None):
    """
    trend_df must have columns:
    ['keyword', 'date', 'sentiment_score', 'articles_count']

    thresholds example:
    {
        'sentiment_drop': -0.3,   # avg sentiment below this triggers alert
        'surge_zscore': 1.5       # z-score of article count above this triggers alert
    }
    """
    alerts = []

    for kw in trend_df['keyword'].unique():
        sub = trend_df[trend_df['keyword'] == kw].copy()

        # add z-score for article counts safely
        sub['count_z'] = safe_zscore(sub['articles_count'])

        # check for volume surge
        surge_rows = sub[sub['count_z'] > thresholds['surge_zscore']]
        for _, row in surge_rows.iterrows():
            msg = f"🚨 ALERT for {kw} on {row['date']}: Article surge (z={row['count_z']:.2f})"
            alerts.append(msg)
            if slack_webhook:
                try:
                    requests.post(slack_webhook, json={"text": msg})
                except Exception as e:
                    print(f"⚠️ Slack send failed: {e}")

        # check for sentiment drop
        drop_rows = sub[sub['sentiment_score'] < thresholds['sentiment_drop']]
        for _, row in drop_rows.iterrows():
            msg = f"⚠️ ALERT for {kw} on {row['date']}: Sentiment drop ({row['sentiment_score']:.2f})"
            alerts.append(msg)
            if slack_webhook:
                try:
                    requests.post(slack_webhook, json={"text": msg})
                except Exception as e:
                    print(f"⚠️ Slack send failed: {e}")

    return alerts
