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
        """Fetch historical stock price data with deployment-safe parameters."""
        try:
            # Use threads=False and progress=False for Streamlit Cloud reliability
            data = yf.download(
                ticker, 
                period=period, 
                interval=interval, 
                progress=False, 
                threads=False
            )
            
            if data is None or data.empty:
                return pd.DataFrame()

            # Robust flattening of MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Ensure index is datetime
            data.index = pd.to_datetime(data.index)
            return data
        except Exception:
            return pd.DataFrame()

    def fetch_latest_news(self, ticker_symbol):
        """Multi-source news pipeline with yfinance (Primary), Google News RSS (Fallback)."""
        news_items = []
        
        # Source 1: yfinance News
        try:
            yticker = yf.Ticker(ticker_symbol)
            yf_news = yticker.news
            if yf_news:
                for item in yf_news[:10]:
                    # Some versions of yfinance have 'content' nesting, others don't
                    content = item.get('content', {})
                    if content:
                        news_items.append({
                            'headline': content.get('title'),
                            'source': content.get('provider', {}).get('displayName', 'yfinance'),
                            'date': content.get('pubDate'),
                            'url': content.get('canonicalUrl', {}).get('url', '')
                        })
                    else:
                        news_items.append({
                            'headline': item.get('title'),
                            'source': item.get('publisher', 'yfinance'),
                            'date': datetime.fromtimestamp(item.get('providerPublishTime')).strftime('%Y-%m-%d') if item.get('providerPublishTime') else '',
                            'url': item.get('link')
                        })
        except Exception:
            pass

        # Source 2: Google News RSS (Fallback)
        if not news_items:
            try:
                rss_url = f"https://news.google.com/rss/search?q={ticker_symbol}+stock"
                feed = feedparser.parse(rss_url)
                if feed.entries:
                    for entry in feed.entries[:10]:
                        news_items.append({
                            'headline': entry.title,
                            'source': entry.get('source', {}).get('title', 'Google News'),
                            'date': getattr(entry, 'published', ''),
                            'url': entry.link
                        })
            except Exception:
                pass

        return pd.DataFrame(news_items)

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
