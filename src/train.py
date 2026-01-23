import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
import numpy as np
from src.model import StockPredictorLSTM


def train_model(X_train, y_train, input_size, hidden_size=50, num_layers=2, 
                num_epochs=50, dropout=0.2, batch_size=32, seed=42, 
                val_data=None, verbose=True):
    """
    Train the LSTM model with advanced techniques: Early Stopping, LR Scheduling, and Regularization.
    
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
        bidirectional: Whether to use Bidirectional LSTM
        val_data: Tuple of (X_val, y_val) for validation
        verbose: Whether to print progress
        
    Returns:
        Trained model
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = StockPredictorLSTM(input_size, hidden_size, num_layers, dropout)
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    dataset = TensorDataset(X_train, y_train)
    loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)
    patience = 10
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            
            if torch.isnan(loss):
                print(f"nan loss detected at epoch {epoch+1}")
                return model # Abort training
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        
        # Validation Phase
        val_loss = avg_train_loss # Default to train loss if no val data
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                X_val, y_val = val_data
                val_preds = model(X_val)
                val_loss = criterion(val_preds.squeeze(), y_val).item()
            
            # Step the scheduler based on validation loss
            scheduler.step(val_loss)
            
            # Early Stopping Logic 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

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

