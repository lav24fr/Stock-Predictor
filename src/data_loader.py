import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
import torch

class DataLoader:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None

    def fetch_stock_data(self):
        """Fetches historical stock data from yfinance."""
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        # Flatten MultiIndex columns if necessary (yfinance update)
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)
            
        return self.data

    def fetch_news_sentiment(self):
        """
        Fetches news from yfinance and calculates average sentiment.
        Note: Historical news is hard to get for free. This fetches *recent* news
        and applies a dummy sentiment for historical illustration if actual data isn't available.
        """
        ticker_obj = yf.Ticker(self.ticker)
        news = ticker_obj.news
        
        sentiments = []
        for article in news:
            title = article.get('title', '')
            blob = TextBlob(title)
            sentiments.append(blob.sentiment.polarity)
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        return avg_sentiment

    def preprocess_data(self, sequence_length=60):
        """
        Preprocesses data for LSTM model.
        Args:
            sequence_length: Number of time steps to look back.
        Returns:
            X, y, scaler: Preprocessed features, targets, and the scaler object.
        """
        if self.data is None:
            self.fetch_stock_data()
            
        # Use 'Close' price for prediction
        # Ensure we have data
        if self.data.empty:
            raise ValueError("No data fetched.")

        dataset = self.data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(dataset)

        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        
        # Reshape X for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), self.scaler

    def get_raw_data(self):
        return self.data
