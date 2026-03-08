import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

# Set directories
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]
lookback = 10

def prepare_data(ticker):
    data = yf.download(ticker, period="2y", interval="1d")
    # Handle MultiIndex columns (common in new yfinance versions)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns.values]
    data = data[['Close', 'Volume']].reset_index()
    
    # Add Moving Averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    
    # Add dummy sentiment score for training (normally this would come from news sentiment)
    data['Sentiment'] = np.random.uniform(-0.5, 0.5, size=len(data))
    
    data = data.dropna()
    return data

def train_ticker_model(ticker):
    print(f"Training LSTM model for {ticker}...")
    df = prepare_data(ticker)
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close', 'Volume', 'Sentiment', 'MA5', 'MA10']])
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0])  # Predicting Close price
    
    X, y = np.array(X), np.array(y)
    
    # LSTM Architecture
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    
    # Save the model and scaler
    model.save(f"models/lstm_model_{ticker}.h5")
    joblib.dump(scaler, f"models/scaler_{ticker}.pkl")
    
    # Generate future predictions (example: for the next 7 and 30 days)
    # Since it's for precomputing, we'll save it to CSV
    last_window = scaled_data[-lookback:]
    predictions = []
    current_window = last_window.reshape(1, lookback, X.shape[2])
    
    for _ in range(30):
        pred = model.predict(current_window, verbose=0)
        predictions.append(pred[0, 0])
        # Update window for next prediction
        new_row = current_window[0, -1, :].copy()
        new_row[0] = pred[0, 0] # update close price
        # for simplicity, keep other features constant for recursive predictions
        current_window = np.append(current_window[:, 1:, :], [[new_row]], axis=1)

    # Inverse transform predictions
    # Need dummy values for other features to inverse transform
    dummy = np.zeros((len(predictions), 5))
    dummy[:, 0] = predictions
    inv_preds = scaler.inverse_transform(dummy)[:, 0]
    
    return inv_preds

# Run training and save precomputed predictions
all_preds = {}
for ticker in tickers:
    preds = train_ticker_model(ticker)
    all_preds[ticker] = preds

# Save to CSV
pred_df = pd.DataFrame(all_preds)
pred_df.to_csv("data/predicted_prices.csv", index=False)
print("All models trained and predictions saved to data/predicted_prices.csv")
