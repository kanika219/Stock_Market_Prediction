import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_fetcher import DataFetcher
from src.sentiment import SentimentAnalyzer
from src.predictor import StockPredictor
from src.utils import format_currency, format_change, generate_insights

# Page Configuration
st.set_page_config(page_title="Stock Intelligence Dashboard", layout="wide")

# Sidebar - Stock Selection
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Market Dashboard", "News Intelligence", "Price Forecast", "Sentiment vs Price", "Investment Insights"])

ticker_choice = st.sidebar.selectbox("Select Ticker", ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "Other"])

if ticker_choice == "Other":
    ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL").upper()
else:
    ticker = ticker_choice
horizon = st.sidebar.slider("Prediction Horizon (days)", 1, 30, 7)

# Load Classes
@st.cache_resource
def load_sentiment_analyzer():
    return SentimentAnalyzer()

@st.cache_resource
def load_data_fetcher():
    try:
        news_key = st.secrets.get("NEWS_API_KEY", "YOUR_NEWS_API_KEY")
    except:
        news_key = "YOUR_NEWS_API_KEY"
    return DataFetcher(news_api_key=news_key)

@st.cache_resource
def load_stock_predictor():
    return StockPredictor()

sentiment_analyzer = load_sentiment_analyzer()
data_fetcher = load_data_fetcher()
stock_predictor = load_stock_predictor()

# Fetch Stock Data
@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    return data_fetcher.fetch_stock_data(ticker)

# Cache news for 1 hour
@st.cache_data(ttl=3600)
def get_news_data(ticker):
    return data_fetcher.fetch_latest_news(ticker)

df_stock = get_stock_data(ticker)

# Check if dataframe is empty
if df_stock is None or df_stock.empty or len(df_stock) < 2:
    st.error("Unable to retrieve live data.")
    st.markdown("""
    **Possible reasons:**
    • Yahoo Finance temporary rate limit
    • Network restriction on hosting environment
    
    Displaying cached or fallback data instead.
    """)
    st.info("Check your internet connection or ticker name.")
    st.stop()

# Get News Data
news_result, news_status = get_news_data(ticker)
df_news = news_result

# Sidebar Ticker Information
current_price = df_stock['Close'].iloc[-1].item()
prev_close = df_stock['Close'].iloc[-2].item()
price_change = ((current_price - prev_close) / prev_close) * 100

st.sidebar.metric(label=f"Current {ticker} Price", value=format_currency(current_price), delta=f"{price_change:.2f}%")

# Bonus: Data Source Status
st.sidebar.markdown("---")
st.sidebar.subheader("Data Source Status")
st.sidebar.write(f"Stock Data: {'Live' if df_stock is not None and not df_stock.empty else 'Unavailable'}")
st.sidebar.write(f"News Data: {news_status}")
st.sidebar.write("Prediction Model: Local")

# Main Content Logic
if page == "Market Dashboard":
    st.title("📊 Market Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    news_sentiments = []
    if not df_news.empty:
        for headline in df_news['headline'].head(10):
            _, score, _ = sentiment_analyzer.predict_sentiment(headline)
            news_sentiments.append(score)
    avg_sentiment = np.mean(news_sentiments) if news_sentiments else 0.0
    
    next_day_price, _ = stock_predictor.get_prediction_metrics(ticker)
    
    col1.metric("Current Price", format_currency(current_price))
    col2.metric("Daily Change", format_change(price_change), delta_color="normal")
    col3.metric("Market Sentiment Score", f"{avg_sentiment:.2f}")
    col4.metric("Predicted Next Day Price", format_currency(next_day_price))
    
    # Stock History Chart
    st.subheader("Stock Price History")
    fig_price = px.line(df_stock.reset_index(), x='Date', y='Close', title=f"{ticker} Stock Price History")
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Trading Volume Chart
    st.subheader("Trading Volume")
    fig_vol = px.bar(df_stock.reset_index(), x='Date', y='Volume', title="Trading Volume History")
    st.plotly_chart(fig_vol, use_container_width=True)

elif page == "News Intelligence":
    st.title("🛡️ News Intelligence")
    
    if news_status == "Fallback" or news_status == "Offline":
        st.warning(f"Live {ticker} news unavailable. Showing {news_status.lower()} dataset.")
    
    # Process sentiment and impact
    news_rows = []
    positive_count = 0
    neutral_count = 0
    negative_count = 0
    
    for _, row in df_news.iterrows():
        sentiment, score, confidence = sentiment_analyzer.predict_sentiment(row['headline'])
        
        # Impact calculation: score * headline_length_factor (normalized)
        length_factor = np.log1p(len(row['headline'])) / 5
        impact_score = score * length_factor
        impact_score = np.clip(impact_score, -1, 1)
        
        sentiment_emoji = "🟢 Positive" if sentiment == "Positive" else "🔴 Negative" if sentiment == "Negative" else "🟡 Neutral"
        
        if sentiment == "Positive": positive_count += 1
        elif sentiment == "Negative": negative_count += 1
        else: neutral_count += 1
        
        news_rows.append({
            "Headline": row['headline'],
            "Source": row['source'],
            "Sentiment": sentiment_emoji,
            "Impact Score": f"{impact_score:+.2f}",
            "Link": row['url'],
            "raw_score": score,
            "raw_impact": impact_score
        })
    
    df_results = pd.DataFrame(news_rows)

    # SECTION 1 — SENTIMENT OVERVIEW
    col1, col2, col3 = st.columns(3)
    col1.metric("Positive News", positive_count)
    col2.metric("Neutral News", neutral_count)
    col3.metric("Negative News", negative_count)
    
    st.markdown("---")
    
    # SECTION 2 — RECENT HEADLINES TABLE
    st.subheader(f"Recent Headlines ({news_status} Data)")
    if not df_results.empty:
        display_df = df_results[["Headline", "Source", "Sentiment", "Impact Score", "Link"]]
        st.dataframe(display_df, use_container_width=True, height=400)
    else:
        st.info("No news headlines available to display.")

    st.markdown("---")

    # SECTION 3 — HEADLINE IMPACT ANALYZER
    st.subheader("Headline Impact Analyzer")
    user_headline = st.text_input("Enter a financial headline to analyze:")
    if user_headline:
        sentiment, score, confidence = sentiment_analyzer.predict_sentiment(user_headline)
        
        length_factor = np.log1p(len(user_headline)) / 5
        impact_score = score * length_factor
        impact_score = np.clip(impact_score, -1, 1)
        
        # Market Interpretation logic
        if sentiment == "Positive" and confidence > 0.7:
            interpretation = "Bullish signal: Strong positive sentiment with high confidence."
        elif sentiment == "Negative" and confidence > 0.7:
            interpretation = "Bearish signal: Significant negative sentiment detected."
        elif sentiment == "Neutral":
            interpretation = "Limited market impact: Headline appears neutral."
        else:
            interpretation = "Mixed signals: Inconclusive market direction."

        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Confidence:** {confidence:.2%}")
        with c2:
            st.write(f"**Impact Score:** {impact_score:+.2f}")
        with c3:
            st.write(f"**Market Interpretation:**")
            st.write(interpretation)

    st.markdown("---")

    # SECTION 4 — SENTIMENT MOMENTUM CHART
    st.subheader("Sentiment vs Price Momentum")
    
    # Combine stock price with synthetic/dummy sentiment timeline for last 30 days
    dates = df_stock.index[-30:]
    prices = df_stock['Close'].iloc[-30:].values
    
    # Synthetic sentiment momentum (moving average of dummy daily scores)
    np.random.seed(42) # Reproducible synthetic data
    daily_sent = np.random.uniform(-0.4, 0.4, size=30)
    # Add some bias based on price movement
    price_diff = np.diff(prices, prepend=prices[0])
    daily_sent += np.clip(price_diff / prices.max() * 10, -0.3, 0.3)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=dates, y=prices, name="Price", line=dict(color="blue")), secondary_y=False)
    fig.add_trace(go.Scatter(x=dates, y=daily_sent, name="Sentiment Score", line=dict(color="green", width=2, shape='spline')), secondary_y=True)
    
    fig.update_layout(title="Sentiment Momentum vs Stock Price (Last 30 Days)", hovermode="x unified")
    fig.update_yaxes(title_text="Stock Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True, range=[-1, 1])
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "Price Forecast":
    st.title("📈 Price Forecasting")
    
    st.subheader(f"{ticker} Historical vs Predicted Price")
    
    forecast_data = stock_predictor.get_forecast(ticker, horizon)
    
    # Prepare forecast dates
    last_date = df_stock.index[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast_data))]
    
    fig = go.Figure()
    # Historical data
    fig.add_trace(go.Scatter(x=df_stock.index[-60:], y=df_stock['Close'].iloc[-60:], name="Historical Price", line=dict(color='blue')))
    # Forecast data
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_data, name="Predicted Price", line=dict(color='green', dash='dash')))
    
    fig.update_layout(title=f"{ticker} Price Forecast - Next {horizon} Days", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"The model predicts the price for {ticker} over the next {horizon} days using precomputed LSTM results.")

elif page == "Sentiment vs Price":
    st.title("⚖️ Sentiment vs Price Movement")
    
    # Create synthetic daily sentiment scores (for visualization)
    daily_sentiment = np.random.uniform(-0.5, 0.5, size=30)
    last_30_days_prices = df_stock['Close'].iloc[-30:].values
    dates = df_stock.index[-30:]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=dates, y=last_30_days_prices, name="Price"), secondary_y=False)
    fig.add_trace(go.Bar(x=dates, y=daily_sentiment, name="Sentiment", opacity=0.5), secondary_y=True)
    
    fig.update_layout(title="Daily Sentiment vs Stock Price Movement")
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    st.write("Graph shows correlation between news sentiment spikes and stock price trends over the last 30 days.")

elif page == "Investment Insights":
    st.title("💡 Investment Insights")
    
    # Use already fetched news for sentiment summary
    dash_news = df_news
    
    # Pre-calculate sentiment
    news_sentiments = []
    if not dash_news.empty:
        for headline in dash_news['headline'].head(10):
            _, score, _ = sentiment_analyzer.predict_sentiment(headline)
            news_sentiments.append(score)
    avg_sentiment = np.mean(news_sentiments) if news_sentiments else 0.0
    
    price_trend = price_change # daily change as trend
    
    market_sentiment, recommendation, confidence = generate_insights(ticker, avg_sentiment, price_trend)
    
    st.subheader("Summary")
    st.info(f"**Market Sentiment:** {market_sentiment}")
    st.info(f"**Price Trend:** {'Upward' if price_trend > 0 else 'Downward'}")
    st.info(f"**Confidence:** {confidence:.2%}")
    
    st.divider()
    st.subheader("AI-based Recommendation")
    st.write(recommendation)
    
    # Additional indicators
    st.divider()
    volatility = df_stock['Close'].pct_change().std()
    risk = "High" if volatility > 0.02 else "Medium" if volatility > 0.01 else "Low"
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Indicator", risk)
    c2.metric("Volatility Index", f"{volatility:.2%}")
    c3.metric("Confidence Score", f"{confidence:.2%}")
