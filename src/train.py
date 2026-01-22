import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
import numpy as np
from src.model import StockPredictorLSTM


def train_model(X_train, y_train, input_size, hidden_size=50, num_layers=2, 
                num_epochs=50, dropout=0.2, batch_size=32, seed=42):
    """
    Train the LSTM model with minibatch gradient descent.
    
    Args:
        X_train: Training features tensor
        y_train: Training targets tensor
        input_size: Number of input features
        hidden_size: LSTM hidden units
        num_layers: Number of LSTM layers
        num_epochs: Training epochs
        dropout: Dropout rate
        batch_size: Minibatch size (default 32)
        seed: Random seed for reproducibility
        
    Returns:
        Trained model
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = StockPredictorLSTM(input_size, hidden_size, num_layers, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(X_train, y_train)
    loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / num_batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model


def predict(model, X_test, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze().cpu().numpy()

    predictions = predictions.reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    return predictions


def walk_forward_validate(loader, sequence_length=60, n_splits=5, hidden_size=32, 
                          num_layers=1, num_epochs=20, dropout=0.0, verbose=True):
    """
    Run walk-forward validation across multiple time periods.
    
    Args:
        loader: DataLoader instance with data already fetched
        sequence_length: Lookback window for sequences
        n_splits: Number of validation folds
        hidden_size: LSTM hidden units
        num_layers: Number of LSTM layers
        num_epochs: Training epochs per fold
        dropout: Dropout rate
        verbose: Print progress
        
    Returns:
        Dictionary with aggregated metrics and per-fold results
    """
    from src.data_loader import calculate_metrics
    
    all_metrics = []
    fold_results = []
    
    for X_train, y_train, X_test, y_test, scaler, fold_info in loader.walk_forward_splits(
        sequence_length=sequence_length, n_splits=n_splits
    ):
        if verbose:
            print(f"\n--- Fold {fold_info['fold']} ---")
            print(f"Train: {fold_info['train_period']} ({fold_info['train_size']} samples)")
            print(f"Test:  {fold_info['test_period']} ({fold_info['test_size']} samples)")
        
        input_size = X_train.shape[2]
        
        model = train_model(
            X_train, y_train, 
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_epochs=num_epochs,
            dropout=dropout,
            seed=42 + fold_info['fold']
        )
        
        preds = predict(model, X_test, scaler)
        y_true = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
        
        metrics = calculate_metrics(y_true, preds)
        
        if verbose:
            print(f"MSE: {metrics['mse']:.6f}")
            print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        
        all_metrics.append(metrics)
        fold_results.append({
            "fold_info": fold_info,
            "metrics": metrics
        })
    
    if all_metrics:
        avg_metrics = {
            "avg_mse": np.mean([m["mse"] for m in all_metrics]),
            "std_mse": np.std([m["mse"] for m in all_metrics]),
            "avg_directional_accuracy": np.mean([m["directional_accuracy"] for m in all_metrics]),
            "std_directional_accuracy": np.std([m["directional_accuracy"] for m in all_metrics]),
            "avg_sharpe": np.mean([m["sharpe_ratio"] for m in all_metrics]),
            "std_sharpe": np.std([m["sharpe_ratio"] for m in all_metrics]),
        }
        
        if verbose:
            print(f"\n=== Walk-Forward Summary ({len(all_metrics)} folds) ===")
            print(f"MSE: {avg_metrics['avg_mse']:.6f} ± {avg_metrics['std_mse']:.6f}")
            print(f"Directional Accuracy: {avg_metrics['avg_directional_accuracy']:.2%} ± {avg_metrics['std_directional_accuracy']:.2%}")
            print(f"Sharpe Ratio: {avg_metrics['avg_sharpe']:.2f} ± {avg_metrics['std_sharpe']:.2f}")
    else:
        avg_metrics = {}
    
    return {
        "summary": avg_metrics,
        "folds": fold_results
    }

