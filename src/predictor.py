import pandas as pd
import joblib
import os
import numpy as np

class StockPredictor:
    def __init__(self, data_path="data/predicted_prices.csv"):
        self.data_path = data_path
        self.predictions = None
        self.load_predictions()

    def load_predictions(self):
        """Load precomputed predictions from CSV."""
        if os.path.exists(self.data_path):
            self.predictions = pd.read_csv(self.data_path)
        else:
            print(f"Precomputed predictions not found at {self.data_path}. Please run training first.")
            # Fallback dummy data if not trained
            tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]
            self.predictions = pd.DataFrame({t: np.random.uniform(100, 500, size=30) for t in tickers})

    def get_forecast(self, ticker, horizon=7):
        """Get precomputed forecast for a specific ticker and horizon."""
        if self.predictions is not None and ticker in self.predictions.columns:
            return self.predictions[ticker].head(horizon).tolist()
        return [0.0] * horizon

    def get_prediction_metrics(self, ticker):
        """Return metrics like next day price and confidence."""
        forecast = self.get_forecast(ticker, horizon=1)
        next_day = forecast[0] if forecast else 0.0
        confidence = np.random.uniform(0.6, 0.9) # Placeholder confidence
        return next_day, confidence
