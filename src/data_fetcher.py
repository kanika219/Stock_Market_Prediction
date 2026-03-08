import yfinance as yf
import requests
import pandas as pd
import os
import feedparser
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, news_api_key=None):
        self.news_api_key = news_api_key
        self.cache_path = "data/cached_news.csv"

    def fetch_stock_data(self, ticker, period="1y", interval="1d"):
        """Fetch historical stock price data."""
        data = yf.download(ticker, period=period, interval=interval)
        # Robust flattening of MultiIndex columns
        if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
            data.columns = data.columns.get_level_values(0)
        return data

    def fetch_latest_news(self, ticker):
        """Multi-source news pipeline with yfinance (Primary), Google News RSS (Fallback)."""
        news_items = []
        is_fallback = False

        # Source 1: yfinance News
        try:
            yticker = yf.Ticker(ticker)
            yf_news = yticker.news
            if yf_news:
                for item in yf_news[:10]:
                    content = item.get('content', {})
                    if content:
                        news_items.append({
                            'headline': content.get('title'),
                            'source': content.get('provider', {}).get('displayName', 'yfinance'),
                            'date': content.get('pubDate'),
                            'url': content.get('canonicalUrl', {}).get('url', '')
                        })
                    else:
                        # Backup for older schema just in case
                        news_items.append({
                            'headline': item.get('title'),
                            'source': item.get('publisher'),
                            'date': datetime.fromtimestamp(item.get('providerPublishTime')).strftime('%Y-%m-%dT%H:%M:%SZ') if item.get('providerPublishTime') else '',
                            'url': item.get('link')
                        })
        except Exception as e:
            print(f"yfinance news error: {e}")

        # Source 2: Google News RSS (Ticker Specific)
        if not news_items:
            try:
                rss_url = f"https://news.google.com/rss/search?q={ticker}+stock"
                feed = feedparser.parse(rss_url)
                if feed.entries:
                    for entry in feed.entries[:10]:
                        news_items.append({
                            'headline': entry.title,
                            'source': entry.get('source', {}).get('title', 'Google News'),
                            'date': entry.published,
                            'url': entry.link
                        })
            except Exception as e:
                print(f"Google News RSS error: {e}")

        # Source 3: Google News RSS (General Market - Final Fallback)
        if not news_items:
            is_fallback = True
            try:
                rss_url = "https://news.google.com/rss/search?q=stock+market"
                feed = feedparser.parse(rss_url)
                if feed.entries:
                    for entry in feed.entries[:10]:
                        news_items.append({
                            'headline': entry.title,
                            'source': entry.get('source', {}).get('title', 'Google News'),
                            'date': entry.published,
                            'url': entry.link
                        })
            except Exception as e:
                print(f"General news fallback error: {e}")

        df = pd.DataFrame(news_items)
        if not df.empty:
            self.cache_news(df)
            
        return df, is_fallback

    def cache_news(self, df):
        """Save news to CSV."""
        if not os.path.exists("data"):
            os.makedirs("data")
            
        if os.path.exists(self.cache_path):
            try:
                existing_df = pd.read_csv(self.cache_path)
                df = pd.concat([df, existing_df]).drop_duplicates(subset=['headline', 'url']).head(100)
            except:
                pass
        df.to_csv(self.cache_path, index=False)
