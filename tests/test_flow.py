import sys
import os

# Add local directory to path
sys.path.append(os.getcwd())

from src.data_loader import DataLoader
from src.train import train_model, predict
from src.strategy import TradingStrategy
import pandas as pd

def test_pipeline():
    print("1. Testing Data Loading...")
    # Fetch small amount of data
    loader = DataLoader("AAPL", "2023-01-01", "2023-06-01")
    loader.fetch_stock_data()
    X, y, scaler = loader.preprocess_data(sequence_length=10)
    print(f"   Data loaded. X shape: {X.shape}, y shape: {y.shape}")
    
    print("2. Testing Model Training...")
    model = train_model(X, y, input_size=1, hidden_size=10, num_layers=1, num_epochs=2)
    print("   Model trained.")
    
    print("3. Testing Prediction...")
    preds = predict(model, X, scaler)
    print(f"   Predictions shape: {preds.shape}")
    
    print("4. Testing Strategy...")
    # Mock aligned data
    actual = y.numpy().flatten() # This is scaled, but for strategy logic test it's fine just to run it
    # We need to inverse transform 'actual' to make it realistic for strategy which checks percentage change
    actual_inv = scaler.inverse_transform(y.numpy().reshape(-1, 1)).flatten()
    
    strategy = TradingStrategy()
    signals, portfolio = strategy.simple_strategy(actual_inv, preds.flatten())
    print(f"   Strategy run. Final Portfolio Value: {portfolio[-1]}")
    
    print("5. Testing Sentiment...")
    s = loader.fetch_news_sentiment()
    print(f"   Sentiment score: {s}")
    
    print("SUCCESS: Pipeline Verified.")

if __name__ == "__main__":
    test_pipeline()
