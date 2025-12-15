import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
import torch
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

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
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
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
            if info and 'shortName' in info:
                # Simplify name: "Apple Inc." -> "Apple"
                company_name = info['shortName'].split(' ')[0]
        except Exception:
            pass # Fallback to ticker
            
        results = []
        weighted_scores = []
        weights = []
        
        for article in news:
            # Handle nested structure if present (common in newer yfinance versions)
            if 'content' in article and isinstance(article['content'], dict):
                title = article['content'].get('title', '')
            else:
                title = article.get('title', '')
                
            score = self.sia.polarity_scores(title)['compound']
            
            # Relevance Check
            is_relevant = False
            if self.ticker.lower() in title.lower() or company_name.lower() in title.lower():
                is_relevant = True
                
            # Weighting
            weight = 1.0 if is_relevant else 0.2
            
            results.append({
                'Title': title,
                'Score': score,
                'Relevant': "Yes" if is_relevant else "No"
            })
            
            weighted_scores.append(score * weight)
            weights.append(weight)
        
        if weights:
            avg_sentiment = np.sum(weighted_scores) / np.sum(weights)
        else:
            avg_sentiment = 0
            
        return avg_sentiment, results

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
