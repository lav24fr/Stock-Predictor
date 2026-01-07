import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class DataLoader:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def fetch_stock_data(self):
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date, auto_adjust=True)
        if data.empty:
            raise ValueError(f"No data found for {self.ticker}")
        self.data = data
        return self.data

    def fetch_news_sentiment(self, limit=10):
        ticker_obj = yf.Ticker(self.ticker)
        news = ticker_obj.news[:limit] if ticker_obj.news else []

        analyzer = SentimentIntensityAnalyzer()
        weighted_scores = []
        weights = []
        results = []

        for article in news:
            title = article.get("title", "")
            score = analyzer.polarity_scores(title)["compound"]

            is_relevant = any(
                keyword in title.lower()
                for keyword in [self.ticker.lower(), "stock", "market", "earning", "revenue"]
            )

            weight = 1.0 if is_relevant else 0.2

            results.append({"Title": title, "Score": score, "Relevant": "Yes" if is_relevant else "No"})

            weighted_scores.append(score * weight)
            weights.append(weight)

        if weights:
            avg_sentiment = np.sum(weighted_scores) / np.sum(weights)
        else:
            avg_sentiment = 0

        return avg_sentiment, results

    def preprocess_data(self, sequence_length=60, split_ratio=0.8):
        if self.data is None:
            self.fetch_stock_data()

        if self.data.isnull().values.any():
            self.data = self.data.dropna()

        df = self.data.copy()

        df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Vol_Change"] = np.log((df["Volume"] + 1) / (df["Volume"].shift(1) + 1))

        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        exp12 = df["Close"].ewm(span=12, adjust=False).mean()
        exp26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp12 - exp26
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        df = df.dropna()

        features = ["Log_Ret", "Vol_Change", "RSI", "MACD", "Signal_Line"]
        dataset = df[features].values

        offset = len(self.data) - len(df)
        
        train_size = int(len(dataset) * split_ratio)

        train_data = dataset[:train_size]
        test_data = dataset[train_size:]

        self.scaler.fit(train_data)
        self.target_scaler.fit(train_data[:, 0].reshape(-1, 1))

        scaled_train = self.scaler.transform(train_data)
        scaled_test = self.scaler.transform(test_data)

        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(seq_len, len(data)):
                X.append(data[i - seq_len : i])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(scaled_train, sequence_length)

        if len(scaled_train) >= sequence_length:
            test_context = scaled_train[-sequence_length:]
            scaled_test_extended = np.concatenate((test_context, scaled_test), axis=0)
        else:
            scaled_test_extended = scaled_test

        X_test, y_test = create_sequences(scaled_test_extended, sequence_length)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        input_size = X_train.shape[2]

        return X_train, y_train, X_test, y_test, self.target_scaler, input_size, offset
