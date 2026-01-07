import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")


class DataLoader:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.sia = SentimentIntensityAnalyzer()

    def fetch_stock_data(self):
        """Fetches historical stock data from yfinance."""
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date, auto_adjust=True)

        # Flatten MultiIndex columns if necessary (yfinance update)
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)

        return self.data

    def fetch_news_sentiment(self):
        """
        Fetches news from yfinance and calculates weighted average sentiment using VADER.
        Prioritizes news that explicitly mentions the ticker or company name.
        Returns:
            avg_sentiment (float): Weighted Mean compound score.
            details (list): List of dicts with 'title', 'score', 'relevant'.
        """
        ticker_obj = yf.Ticker(self.ticker)
        news = ticker_obj.news

        # Try to get short name for better matching (e.g. "Apple" from "Apple Inc.")
        # We wrap in try/except because .info can sometimes be flaky/slow
        company_name = self.ticker
        try:
            info = ticker_obj.info
            if info and "shortName" in info:
                # Simplify name: "Apple Inc." -> "Apple"
                company_name = info["shortName"].split(" ")[0]
        except Exception:
            pass  # Fallback to ticker

        results = []
        weighted_scores = []
        weights = []

        try:
            news = ticker_obj.news
        except Exception as e:
            print(f"Error fetching news for {self.ticker}: {e}")
            return 0.0, []

        for article in news:
            # Handle nested structure if present (common in newer yfinance versions)
            if "content" in article and isinstance(article["content"], dict):
                title = article["content"].get("title", "")
            else:
                title = article.get("title", "")

            score = self.sia.polarity_scores(title)["compound"]

            # Relevance Check
            is_relevant = False
            if self.ticker.lower() in title.lower() or company_name.lower() in title.lower():
                is_relevant = True

            # Weighting
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
        """
        Preprocesses data using Log Returns for Stationarity.
        """
        if self.data is None:
            self.fetch_stock_data()

        if self.data.isnull().values.any():
            self.data = self.data.dropna()

        # --- Feature Engineering ---
        df = self.data.copy()

        # 1. Log Returns (Target & Feature)
        # Log(P_t / P_{t-1})
        df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))

        # 2. Volume Change (Log)
        # Add small epsilon to avoid log(0)
        df["Vol_Change"] = np.log((df["Volume"] + 1) / (df["Volume"].shift(1) + 1))

        # 3. RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # 4. MACD
        exp12 = df["Close"].ewm(span=12, adjust=False).mean()
        exp26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp12 - exp26
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # Drop NaNs created by shift/rolling
        df = df.dropna()

        # Select Features
        # Note: We use Log_Ret instead of Price
        features = ["Log_Ret", "Vol_Change", "RSI", "MACD", "Signal_Line"]

        dataset = df[features].values

        # --- Split THEN Scale ---
        train_size = int(len(dataset) * split_ratio)
        train_data = dataset[:train_size]
        test_data = dataset[train_size:]

        # Fit scaler ONLY on training data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit
        self.scaler.fit(train_data)
        self.target_scaler.fit(train_data[:, 0].reshape(-1, 1))  # Log_Ret is index 0

        # Transform
        scaled_train = self.scaler.transform(train_data)
        scaled_test = self.scaler.transform(test_data)

        # Create Sequences
        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(seq_len, len(data)):
                X.append(data[i - seq_len : i])
                y.append(data[i, 0])  # Predicting 'Log_Ret'
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(scaled_train, sequence_length)
        X_test, y_test = create_sequences(scaled_test, sequence_length)

        # Convert to Tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # Calculate offset (rows dropped)
        # Original length: len(self.data) (before local copy & dropna)
        # We need to account for self.data.dropna() at line 103 if it happened.
        # But clearer: valid_start_index in self.data corresponding to df.index[0]

        # Or simply:
        # dropped_rows = len(original_len) - len(df)
        # But we need to know exactly WHERE the cut happened (usually start).
        # Since we use rolling/shifts, it cuts from the start.

        offset = len(self.data) - len(df)  # Assuming self.data was the reference

        return X_train, y_train, X_test, y_test, self.target_scaler, len(features), offset

    def get_raw_data(self):
        return self.data
