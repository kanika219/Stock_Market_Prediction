import yfinance as yf
import pandas as pd
import sys

ticker_name = "AAPL"
try:
    ticker = yf.Ticker(ticker_name)
    news = ticker.news
    print(f"Ticker: {ticker_name}")
    print(f"News type: {type(news)}")
    if isinstance(news, list):
        print(f"News length: {len(news)}")
        if len(news) > 0:
            print(f"First item type: {type(news[0])}")
            print(f"First item keys: {news[0].keys() if isinstance(news[0], dict) else 'N/A'}")
            # print(f"First item: {news[0]}")
    elif isinstance(news, pd.DataFrame):
        print(f"News columns: {news.columns.tolist()}")
        print(f"News shape: {news.shape}")
    else:
        print(f"Unexpected news type: {news}")
except Exception as e:
    print(f"Error: {e}")
