import yfinance as yf
import pandas as pd
import sys

ticker_name = "AAPL"
try:
    ticker = yf.Ticker(ticker_name)
    news = ticker.news
    print(f"Ticker: {ticker_name}")
    print(f"News length: {len(news)}")
    if len(news) > 0:
        item = news[0]
        print(f"Keys: {item.keys()}")
        if 'content' in item:
            print(f"Content keys: {item['content'].keys()}")
except Exception as e:
    print(f"Error: {e}")
