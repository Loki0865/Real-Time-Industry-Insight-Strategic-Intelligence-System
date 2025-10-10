import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import os
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats  # For z-score

# Custom CSS for full-screen (red alert theme)
st.markdown("""
    <style>
    .main-header { font-size: 3rem; color: #2E86C1; text-align: center; margin-bottom: 1rem; }
    .tab-header { font-size: 2rem; color: #34495E; text-align: center; margin-bottom: 1rem; }
    .metric-card { background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 0.5rem; }
    .alert-box { background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 10px; padding: 1rem; margin: 1rem 0; color: #721c24; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { height: 3rem; font-size: 1.2rem; }
    </style>
""", unsafe_allow_html=True)

# Imports
from sentiment_analysis import fetch_news, classify_sentiments_batch, clean_text
from forecasting import forecast_sentiment
from alerts import check_alerts, safe_zscore

# Config
st.set_page_config(page_title="Strategic Intelligence Dashboard", layout="wide", page_icon="🧠", initial_sidebar_state="collapsed")

# Thresholds
THRESHOLDS = {'sentiment_drop': -0.3, 'surge_zscore': 1.5}

# Session state
if 'trend_df' not in st.session_state:
    st.session_state.trend_df = pd.DataFrame()
if 'raw_articles' not in st.session_state:
    st.session_state.raw_articles = pd.DataFrame()
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = {}
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Load/Fetch data
@st.cache_data(ttl=300)
def load_or_fetch_data(keywords, pages=2, articles_per_page=20):
    all_data = []
    for keyword in keywords:
        for page in range(1, pages + 1):
            df = fetch_news(query=keyword, page_size=articles_per_page)
            if not df.empty:
                df['keyword'] = keyword
                df['page'] = page
                all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['clean_text'] = (combined_df['title'].fillna('') + ' ' + combined_df['description'].fillna('')).apply(clean_text)
        sentiments = classify_sentiments_batch(combined_df['clean_text'].tolist())
        labels, scores = zip(*sentiments)
        combined_df['sentiment_label'] = labels
        combined_df['sentiment_score'] = scores
        
        combined_df['date'] = pd.to_datetime(combined_df['publishedAt'], errors='coerce').dt.date
        trend_df = combined_df.groupby(['keyword', 'date']).agg({
            'sentiment_score': 'mean',
            'title': 'count'
        }).reset_index().rename(columns={'title': 'articles_count'})
        
        return combined_df, trend_df
    return pd.DataFrame(), pd.DataFrame()

# Title
st.markdown('<h1 class="main-header">🧠 Strategic Intelligence Dashboard</h1>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Live News & Insights", "📈 Forecast Dashboard", "🚨 Alerts & Benchmarks"])

with tab1:
    st.markdown('<h2 class="tab-header">Fetch Latest News & Sentiment</h2>', unsafe_allow_html=True)
    
    # API Keys input section
    col_key1, col_key2 = st.columns(2)
    with col_key1:
        api_key = st.text_input("🔑 Enter your NewsAPI Key", type="password", help="Get free key from newsapi.org")
        if api_key:
            os.environ["NEWS_API_KEY"] = api_key  # Override env
            st.session_state.api_key = api_key
    with col_key2:
        gemini_key = st.text_input("🔑 Enter your Gemini API Key", type="password", help="Get key from Google AI Studio")
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key  # Override env
            st.session_state.gemini_key = gemini_key
    
    keywords_input = st.text_area("📝 Enter keywords (comma-separated)", 
                                  value="Artificial Intelligence,Machine Learning,Deep Learning,Neural Networks,NLP,Generative AI,Computer Vision,Ethics,Chatbots,Robotics",
                                  height=100)
    keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
    
    col3, col4 = st.columns(2)
    articles_per_page = col3.slider("Articles per page", 10, 50, 20)
    pages_to_fetch = col4.slider("Pages to fetch", 1, 5, 2)
    
    col_btn = st.columns([3, 1, 3])
    with col_btn[1]:
        if st.button("🚀 Fetch News", type="primary", use_container_width=True):
            if 'api_key' in st.session_state and 'gemini_key' in st.session_state:
                with st.spinner("Fetching and analyzing news..."):
                    raw, trend = load_or_fetch_data(keywords, pages_to_fetch, articles_per_page)
                    st.session_state.raw_articles = raw
                    st.session_state.trend_df = trend
                    st.success(f"✅ Fetched {len(raw)} articles across {len(trend)} trends!")
                    
                    # Download trend CSV
                    if not trend.empty:
                        csv_trend = trend.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Trend CSV", data=csv_trend, file_name='trend_data.csv', mime='text/csv', use_container_width=True)
                    
                    # Download raw articles CSV (prominent)
                    if not raw.empty:
                        csv_raw = raw.to_csv(index=False).encode('utf-8')
                        st.download_button("📄 Download Raw Fetched Articles CSV", data=csv_raw, file_name='raw_articles.csv', mime='text/csv', use_container_width=True)
            else:
                st.error("⚠️ Please enter both your NewsAPI and Gemini API keys.")
    
    if not st.session_state.raw_articles.empty:
        # Metrics (NaN fix)
        col_a, col_b, col_c, col_d = st.columns(4)
        total_articles = len(st.session_state.raw_articles)
        avg_sentiment = st.session_state.raw_articles['sentiment_score'].mean()
        if pd.isna(avg_sentiment):
            avg_sentiment = 0.0
        top_keyword = st.session_state.trend_df.loc[st.session_state.trend_df['articles_count'].idxmax(), 'keyword'] if not st.session_state.trend_df.empty else "N/A"
        unique_sources = st.session_state.raw_articles['source'].nunique()
        
        with col_a:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Articles", total_articles)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_c:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Hottest Topic", top_keyword)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_d:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Unique Sources", unique_sources)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Sentiment pie
        sentiment_counts = st.session_state.raw_articles['sentiment_label'].value_counts()
        fig_pie = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Raw fetched data as table
        st.subheader("📄 Raw Fetched Articles Data")
        st.dataframe(st.session_state.raw_articles, use_container_width=True, height=400)
        
        # Trend viz
        filtered_trend = st.session_state.trend_df[st.session_state.trend_df['keyword'].isin(keywords[:6])]
        if not filtered_trend.empty:
            fig = px.line(filtered_trend, x='date', y='sentiment_score', color='keyword', 
                          title="Sentiment Trends Over Time", markers=True, height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Volume bar
        fig_bar = px.bar(filtered_trend, x='date', y='articles_count', color='keyword', 
                         title="Articles Volume Over Time", height=500)
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.markdown('<h2 class="tab-header">Forecast Dashboard from CSV</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📁 Upload CSV (Columns: keyword, date, sentiment_score, articles_count)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'date' not in df.columns or 'sentiment_score' not in df.columns:
            st.error("❌ CSV must have 'date' and 'sentiment_score' columns.")
            st.stop()
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df = df.dropna(subset=['date'])
        st.session_state.trend_df = df
        
        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        time_range = col_f1.slider("Time Range (Days)", 7, 365, 30)
        forecast_days = col_f2.slider("Forecast Horizon (Days)", 3, 14, 7)
        select_keywords = col_f3.multiselect("Filter Keywords", df['keyword'].unique(), default=df['keyword'].unique()[:3])
        
        filtered_df = df[(df['keyword'].isin(select_keywords)) & 
                         (df['date'] >= (datetime.now().date() - timedelta(days=time_range)))]
        
        if filtered_df.empty:
            st.warning("⚠️ No data after filters. Adjust sliders.")
            st.stop()
        
        # Metrics
        col_d, col_e, col_f = st.columns(3)
        with col_d:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Data Points", len(filtered_df))
            st.markdown('</div>', unsafe_allow_html=True)
        with col_e:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Forecast Horizon", f"{forecast_days} Days")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_f:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_forecast = np.mean([v['yhat'].mean() for v in st.session_state.forecasts.values() if not v.empty]) if st.session_state.forecasts else filtered_df['sentiment_score'].mean()
            st.metric("Avg Forecast Sentiment", f"{avg_forecast:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔮 Generate Forecasts", type="primary", use_container_width=True):
            with st.spinner("Forecasting trends with Prophet..."):
                st.session_state.forecasts = {}
                for kw in filtered_df['keyword'].unique():
                    sub_df = filtered_df[filtered_df['keyword'] == kw][['date', 'sentiment_score']].rename(columns={'date': 'ds', 'sentiment_score': 'y'})
                    if len(sub_df) >= 5:
                        m, forecast = forecast_sentiment(filtered_df, kw, days=forecast_days)
                        if m is not None:
                            st.session_state.forecasts[kw] = forecast
                            st.success(f"✅ Forecast for {kw}")
                    else:
                        st.warning(f"⚠️ Skipping {kw}: Need ≥5 points.")
        
        if st.session_state.forecasts:
            selected_kw = st.selectbox("Select Keyword", list(st.session_state.forecasts.keys()))
            forecast_df = st.session_state.forecasts[selected_kw]
            
            hist = filtered_df[filtered_df['keyword'] == selected_kw]
            future = forecast_df[forecast_df['ds'] > hist['date'].max()]
            
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            fig.add_trace(go.Scatter(x=hist['date'], y=hist['sentiment_score'], mode='lines+markers', name='Historical', line=dict(color='blue', width=3)), secondary_y=False)
            fig.add_trace(go.Scatter(x=future['ds'], y=future['yhat'], mode='lines+markers', name='Forecast', line=dict(color='red', width=3)), secondary_y=False)
            fig.add_trace(go.Scatter(x=future['ds'], y=future['yhat_upper'], mode='lines', line=dict(color='red', dash='dash'), showlegend=False), secondary_y=False)
            fig.add_trace(go.Scatter(x=future['ds'], y=future['yhat_lower'], mode='lines', line=dict(color='red', dash='dash'), fill='tonexty', fillcolor='rgba(255,0,0,0.2)', showlegend=False), secondary_y=False)
            fig.update_layout(title=f"Sentiment Forecast: {selected_kw}", xaxis_title="Date", yaxis_title="Sentiment Score", height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast table
            future_table = future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            future_table.columns = ['Date', 'Predicted', 'Lower CI', 'Upper CI']
            st.subheader("Forecast Table")
            st.dataframe(future_table, use_container_width=True)
        
        # Volume scatter
        fig_scatter = px.scatter(filtered_df, x='date', y='sentiment_score', size='articles_count', color='keyword', 
                                 title="Sentiment vs Volume Scatter", hover_data=['articles_count'], height=600)
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.markdown('<h2 class="tab-header">Alerts & Benchmarks</h2>', unsafe_allow_html=True)
    
    if not st.session_state.trend_df.empty:
        # Slack input
        slack_webhook = st.text_input("🔗 Slack Webhook URL (optional for alerts)", type="password", help="Paste your Slack incoming webhook URL")
        if slack_webhook:
            os.environ["SLACK_WEBHOOK_URL"] = slack_webhook
        
        if st.button("🔔 Check Alerts", type="primary", use_container_width=True):
            st.session_state.alerts = check_alerts(st.session_state.trend_df, THRESHOLDS, os.getenv("SLACK_WEBHOOK_URL"))
            if st.session_state.alerts:
                st.error(f"🚨 {len(st.session_state.alerts)} alerts sent to Slack! Check your channel.")
            else:
                st.success("✅ No alerts sent.")
        
        # Filters
        col_t1, col_t2 = st.columns(2)
        time_range_alert = col_t1.slider("Alert Time Range (Days)", 7, 365, 30)
        z_threshold = col_t2.slider("Anomaly Z-Threshold", 1.0, 3.0, 2.0)
        
        filtered_trend = st.session_state.trend_df[st.session_state.trend_df['date'] >= (datetime.now().date() - timedelta(days=time_range_alert))]
        
        # Alert df
        alert_df = filtered_trend.copy()
        alert_df['count_z'] = alert_df.groupby('keyword')['articles_count'].transform(safe_zscore)
        alert_df['alert_sentiment'] = alert_df['sentiment_score'] < THRESHOLDS['sentiment_drop']
        alert_df['alert_surge'] = alert_df['count_z'] > THRESHOLDS['surge_zscore']
        
        # Anomalies
        def detect_anomalies(df, z_threshold):
            anomalies = []
            for kw in df['keyword'].unique():
                sub = df[df['keyword'] == kw]
                z_scores = stats.zscore(sub['sentiment_score'])
                anomaly_mask = np.abs(z_scores) > z_threshold
                anomalies.extend(sub.loc[anomaly_mask, ['keyword', 'date', 'sentiment_score']].values.tolist())
            return pd.DataFrame(anomalies, columns=['keyword', 'date', 'sentiment_score']) if anomalies else pd.DataFrame()
        
        anomalies_df = detect_anomalies(alert_df, z_threshold)
        
        col_f, col_g, col_h = st.columns(3)
        with col_f:
            st.subheader("Alert History")
            st.dataframe(alert_df[['keyword', 'date', 'sentiment_score', 'articles_count', 'alert_sentiment', 'alert_surge']], use_container_width=True)
        with col_g:
            st.subheader("Anomalies Table")
            if not anomalies_df.empty:
                st.dataframe(anomalies_df, use_container_width=True)
            else:
                st.info("No anomalies.")
        with col_h:
            alert_types = pd.Series(['Sentiment Drop' if row['alert_sentiment'] else 'Surge' if row['alert_surge'] else 'None' for _, row in alert_df.iterrows()])
            fig_pie = px.pie(values=alert_types.value_counts(), names=alert_types.value_counts().index, title="Alert Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Trend line
        if not filtered_trend.empty:
            fig_line = px.line(filtered_trend, x='date', y='sentiment_score', color='keyword', 
                               title="Trend Progressions", facet_col='keyword', facet_col_wrap=3, height=600)
            fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_line, use_container_width=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; font-size: 1rem;">*Dashboard powered by Streamlit & Prophet | Real-Time AI News Analytics*</p>', unsafe_allow_html=True)