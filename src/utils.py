import pandas as pd
import numpy as np

def format_currency(value):
    """Format currency value."""
    return f"${value:,.2f}"

def format_change(change):
    """Format price change with sign and color."""
    sign = "+" if change > 0 else ""
    return f"{sign}{change:.2f}%"

def get_risk_indicator(volatility):
    """Return risk indicator based on volatility."""
    if volatility < 0.015: return "Low"
    elif volatility < 0.03: return "Medium"
    else: return "High"

def generate_insights(ticker, sentiment_score, price_trend):
    """Generate investment insights based on sentiment and trend."""
    if sentiment_score > 0.2 and price_trend > 0:
        return "Bullish", "Consider holding or buying. Strong sentiment and upward trend.", 0.85
    elif sentiment_score < -0.2 and price_trend < 0:
        return "Bearish", "Strong downward pressure. Consider selling or reducing position.", 0.72
    else:
        return "Neutral", "Mixed signals. Wait for clearer indicators or hold current position.", 0.65
