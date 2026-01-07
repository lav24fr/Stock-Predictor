import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.model import StockPredictorLSTM


def train_model(X_train, y_train, input_size, hidden_size=50, num_layers=2, num_epochs=50, dropout=0.2, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = StockPredictorLSTM(input_size, hidden_size, num_layers, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model


def predict(model, X_test, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze().cpu().numpy()

    predictions = predictions.reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    return predictions
