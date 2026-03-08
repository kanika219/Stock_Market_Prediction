# Stock Intelligence Dashboard

A real-time financial analytics dashboard that combines stock price trends, news sentiment, and short-term price forecasting.

## 🚀 Overview
This dashboard analyzes financial news sentiment and stock price trends to provide investment insights and price predictions for top tech stocks (AAPL, TSLA, NVDA, MSFT, AMZN).

### 🛠 Business Problem
Investors need a consolidated view of market data and sentiment to make informed decisions. This dashboard provides a lightweight, real-time interface for sentiment-aware stock analysis without the overhead of heavy ML libraries in production.

## 🏗 Model Architecture
- **Sentiment Analysis:** TF-IDF Vectorization followed by a Logistic Regression classifier (trained on financial news data).
- **Time Series Forecasting:** LSTM (Long Short-Term Memory) neural network trained locally with stock price, volume, and sentiment features.
- **Inference:** The Streamlit app uses precomputed LSTM predictions to ensure high performance and zero dependency on heavy ML frameworks like TensorFlow on Streamlit Cloud.

## 📊 Dataset & APIs
- **Stock Data:** Live and historical data fetched via `yfinance`.
- **Financial News:** Latest headlines retrieved from [NewsAPI](https://newsapi.org/).
- **Sentiment Labels:** Pre-trained on a custom financial news dataset.

## ⚙️ Local Training Instructions
To retrain the models locally (requires `tensorflow`):

1. **Install Local Requirements:**
   ```bash
   pip install tensorflow yfinance pandas scikit-learn joblib nltk
   ```

2. **Train Sentiment Model:**
   ```bash
   python training/train_sentiment_model.py
   ```

3. **Train LSTM Model:**
   ```bash
   python training/train_lstm_model.py
   ```
   *This will generate `models/lstm_model_<TICKER>.h5` and update `data/predicted_prices.csv`.*

## 🌐 Streamlit Cloud Deployment Guide
1. Push the repository to GitHub.
2. Link your repository to [Streamlit Cloud](https://share.streamlit.io/).
3. Set your NewsAPI key as a secret in the Streamlit Cloud dashboard:
   - Go to App Settings -> Secrets.
   - Add: `NEWS_API_KEY = "your_key_here"`
4. Deploy! (The app will automatically install dependencies from `requirements.txt`).

---
*Built with Python, Streamlit, and Plotly.*
