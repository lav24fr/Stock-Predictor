import torch
import torch.nn as nn


class StockPredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2):
        super(StockPredictorLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # NOTE: Bidirectional LSTMs are NOT suitable for time series forecasting.
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        lstm_output_dim = hidden_size
        
        self.fc_1 = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # We take the output of the LAST time step in the sequence
        last_time_step = lstm_out[:, -1, :]
        
        predictions = self.fc_1(last_time_step)
        
        return predictions
