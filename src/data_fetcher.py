import yfinance as yf
import requests
import pandas as pd
import os
import feedparser
import json
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, news_api_key=None):
        self.news_api_key = news_api_key
        self.cache_path = "data/cached_news.csv"
        self.sample_news_path = "data/sample_news.json"

    def fetch_stock_data(self, ticker):
        """Fetch historical stock price data with deployment-safe parameters."""
        try:
            data = yf.download(
                ticker,
                period="1y",
                interval="1d",
                progress=False,
                threads=False
            )

            if data is None or data.empty:
                raise ValueError("No stock data")

            # Robust flattening of MultiIndex columns if they exist
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Ensure index is datetime
            data.index = pd.to_datetime(data.index)
            return data

        except Exception:
            return None

    def fetch_yfinance_news(self, ticker_symbol):
        """Primary source: yfinance news"""
        try:
            ticker = yf.Ticker(ticker_symbol)
            news = ticker.news
            headlines = []
            if news:
                for item in news[:10]:
                    # Handle new yfinance news structure where data is inside 'content'
                    content = item.get('content')
                    if not isinstance(content, dict):
                        content = item

                    # Extract headline (try multiple common fields)
                    headline = content.get('title') or content.get('headline') or item.get('title') or item.get('headline') or ""
                    
                    # Extract source
                    source = content.get('provider', {}).get('displayName') or content.get('publisher') or item.get('publisher') or 'yfinance'
                    
                    # Extract date
                    date_val = content.get('pubDate') or content.get('providerPublishTime') or item.get('providerPublishTime')
                    if isinstance(date_val, int):
                        date_str = datetime.fromtimestamp(date_val).strftime('%Y-%m-%d')
                    elif isinstance(date_val, str):
                        try:
                            date_str = date_val.split('T')[0]
                        except:
                            date_str = date_val
                    else:
                        date_str = ''

                    # Extract URL
                    url = ""
                    click_through = content.get('clickThroughUrl')
                    if isinstance(click_through, dict):
                        url = click_through.get('url')
                    
                    if not url:
                        url = content.get('link') or item.get('link') or content.get('url') or item.get('url') or ""

                    headlines.append({
                        "headline": headline,
                        "source": source,
                        "date": date_str,
                        "url": url
                    })
            return headlines
        except Exception:
            return []

    def fetch_google_news(self, ticker_symbol):
        """Secondary source: Google News RSS"""
        try:
            url = f"https://news.google.com/rss/search?q={ticker_symbol}+stock"
            feed = feedparser.parse(url)
            headlines = []
            if feed.entries:
                for entry in feed.entries[:10]:
                    headlines.append({
                        "headline": getattr(entry, 'title', ''),
                        "source": entry.source.title if hasattr(entry, 'source') else "Google News",
                        "date": getattr(entry, 'published', ''),
                        "url": getattr(entry, 'link', '')
                    })
            return headlines
        except Exception:
            return []

    def load_sample_news(self):
        """Offline fallback dataset"""
        try:
            if os.path.exists(self.sample_news_path):
                with open(self.sample_news_path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def fetch_latest_news(self, ticker_symbol):
        """Unified news function with fallback system."""
        # Try yfinance news first
        news = self.fetch_yfinance_news(ticker_symbol)
        if news and len(news) > 0:
            # Check if headlines are actually present
            valid_headlines = [n for n in news if n.get('headline') and n.get('headline').strip()]
            if valid_headlines:
                return pd.DataFrame(news), "Live"

        # Try Google News fallback
        news = self.fetch_google_news(ticker_symbol)
        if news and len(news) > 0:
            return pd.DataFrame(news), "Fallback"

        # Try offline sample data
        news = self.load_sample_news()
        if news:
            # Filter for ticker if possible, otherwise just return sample
            return pd.DataFrame(news), "Offline"

        return pd.DataFrame(), "None"

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
