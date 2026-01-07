import sys
import os

# Add local directory to path
sys.path.append(os.getcwd())

from src.data_loader import DataLoader  # noqa: E402
from src.train import train_model, predict  # noqa: E402
from src.strategy import TradingStrategy  # noqa: E402


def test_pipeline():
    print("1. Testing Data Loading...")
    # Fetch small amount of data
    loader = DataLoader("AAPL", "2023-01-01", "2023-06-01")
    loader.fetch_stock_data()
    X_train, y_train, X_test, y_test, target_scaler, input_size, offset = loader.preprocess_data(sequence_length=10)
    X, y, scaler = X_train, y_train, target_scaler
    print(f"   Data loaded. X shape: {X.shape}, y shape: {y.shape}")

    print("2. Testing Model Training...")
    model = train_model(X, y, input_size=input_size, hidden_size=10, num_layers=1, num_epochs=2)
    print("   Model trained.")

    print("3. Testing Prediction...")
    preds = predict(model, X, scaler)
    print(f"   Predictions shape: {preds.shape}")

    print("4. Testing Strategy...")
    # Mock aligned data
    # We need to inverse transform to make it realistic for strategy which checks percentage change
    actual_inv = scaler.inverse_transform(y.numpy().reshape(-1, 1)).flatten()

    strategy = TradingStrategy()
    signals, portfolio = strategy.simple_strategy(actual_inv, preds.flatten())
    print(f"   Strategy run. Final Portfolio Value: {portfolio[-1]}")

    print("4. Testing Strategy...")
    # Mock aligned data
    # We need to inverse transform to make it realistic for strategy which checks percentage change
    actual_inv = scaler.inverse_transform(y.numpy().reshape(-1, 1)).flatten()

    strategy = TradingStrategy()
    signals, portfolio = strategy.simple_strategy(actual_inv, preds.flatten())
    print(f"   Strategy run. Final Portfolio Value: {portfolio[-1]}")

    print("5. Testing Sentiment...")
    avg_sentiment, details = loader.fetch_news_sentiment()
    print(f"   Sentiment score: {avg_sentiment}")
    print(f"   Details count: {len(details)}")

    print("SUCCESS: Pipeline Verified.")


if __name__ == "__main__":
    test_pipeline()
