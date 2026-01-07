import torch
import torch.nn as nn
import torch.optim as optim
from src.model import StockPredictorLSTM
import numpy as np


def train_model(
    X_train,
    y_train,
    input_size=1,
    hidden_size=50,
    num_layers=2,
    num_epochs=50,
    learning_rate=0.001,
    dropout=0.2,
    device="cpu",
    seed=42,
):
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = StockPredictorLSTM(input_size, hidden_size, num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        outputs = model(X_train.to(device))
        optimizer.zero_grad()
        loss = criterion(outputs, y_train.view(-1, 1).to(device))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model


def predict(model, X_test, scaler, device="cpu"):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.to(device))
        predictions = predictions.cpu().numpy()
        predictions = scaler.inverse_transform(predictions)
    return predictions
