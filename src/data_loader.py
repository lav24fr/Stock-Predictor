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

    def _add_indicators(self, df, warmup_df=None):
        """
        Compute RSI, MACD, and Signal Line indicators.
        
        Args:
            df: DataFrame to compute indicators on
            warmup_df: Optional DataFrame to use as warm-up data (for test set).
                       This ensures rolling/ewm computations use proper history.
        
        Returns:
            DataFrame with indicators added
        """
        # Create a copy of df to avoid modifying original
        result_df = df.copy()
        
        if warmup_df is not None:
            # Use only the Close column from warmup for indicator calculations
            warmup_close = warmup_df["Close"].copy()
            combined_close = pd.concat([warmup_close, df["Close"]], axis=0)
            warmup_len = len(warmup_close)
        else:
            combined_close = df["Close"].copy()
            warmup_len = 0
        
        # RSI calculation
        delta = combined_close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD calculation
        exp12 = combined_close.ewm(span=12, adjust=False).mean()
        exp26 = combined_close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        
        # Extract only the portion corresponding to df (remove warmup)
        if warmup_len > 0:
            result_df["RSI"] = rsi.iloc[warmup_len:].values
            result_df["MACD"] = macd.iloc[warmup_len:].values
            result_df["Signal_Line"] = signal_line.iloc[warmup_len:].values
        else:
            result_df["RSI"] = rsi.values
            result_df["MACD"] = macd.values
            result_df["Signal_Line"] = signal_line.values
            
        return result_df

    def preprocess_data(self, sequence_length=60, split_ratio=0.8, include_sentiment=False):
        """
        Preprocess stock data for LSTM training.
        
        Args:
            sequence_length: Lookback window for sequences
            split_ratio: Train/test split ratio
            include_sentiment: If True, adds current sentiment as a feature.
                              Historical data uses neutral sentiment (0) since
                              we don't have historical news data.
        
        Returns:
            X_train, y_train, X_test, y_test, scaler, input_size, offset
        """
        if self.data is None:
            self.fetch_stock_data()

        if self.data.isnull().values.any():
            self.data = self.data.dropna()

        df = self.data.copy()

        df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Vol_Change"] = np.log((df["Volume"] + 1) / (df["Volume"].shift(1) + 1))

        df = df.dropna()

        offset = len(self.data) - len(df)
        
        train_size = int(len(df) * split_ratio)
        
        MIN_SAMPLES_PER_PARAM = 10
        estimated_params = 50 * 50 * 4
        min_recommended = estimated_params * MIN_SAMPLES_PER_PARAM // 100
        
        if train_size < min_recommended:
            import warnings
            warnings.warn(
                f"Training set has only {train_size} samples. Consider using at least {min_recommended}.",
                UserWarning
            )
        
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        train_df = self._add_indicators(train_df)
        train_df = train_df.dropna()

        test_df = self._add_indicators(test_df, warmup_df=train_df)

        features = ["Log_Ret", "Vol_Change", "RSI", "MACD", "Signal_Line"]
        
        if include_sentiment:
            current_sentiment, _ = self.fetch_news_sentiment()
            
            train_df["Sentiment"] = 0.0
            
            news_lookback_days = min(5, len(test_df))
            test_df["Sentiment"] = 0.0
            if news_lookback_days > 0:
                test_df.iloc[-news_lookback_days:, test_df.columns.get_loc("Sentiment")] = current_sentiment
            
            features.append("Sentiment")
        
        train_data = train_df[features].values
        test_data = test_df[features].values

        offset += (train_size - len(train_df))

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

    def walk_forward_splits(self, sequence_length=60, n_splits=5, test_size=0.1):
        """
        Generator that yields train/test splits for walk-forward validation.
        
        Uses an expanding window approach:
        - Fold 1: Train on [0:60%], Test on [60%:70%]
        - Fold 2: Train on [0:70%], Test on [70%:80%]
        - etc.
        
        Args:
            sequence_length: Lookback window for LSTM sequences
            n_splits: Number of validation folds
            test_size: Fraction of data for each test fold (default 10%)
            
        Yields:
            Tuple of (X_train, y_train, X_test, y_test, scaler, fold_info)
        """
        if self.data is None:
            self.fetch_stock_data()

        if self.data.isnull().values.any():
            self.data = self.data.dropna()

        df = self.data.copy()
        df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Vol_Change"] = np.log((df["Volume"] + 1) / (df["Volume"].shift(1) + 1))
        df = df.dropna()

        n_samples = len(df)
        test_samples = int(n_samples * test_size)
        
        min_train_size = max(sequence_length * 2, int(n_samples * 0.5))
        
        for fold in range(n_splits):
            test_end = n_samples - (n_splits - fold - 1) * test_samples
            test_start = test_end - test_samples
            train_end = test_start
            
            if train_end < min_train_size:
                continue
                
            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            train_df = self._add_indicators(train_df)
            train_df = train_df.dropna()
            
            if len(train_df) < sequence_length + 10:
                continue
                
            test_df = self._add_indicators(test_df, warmup_df=train_df)
            
            features = ["Log_Ret", "Vol_Change", "RSI", "MACD", "Signal_Line"]
            train_data = train_df[features].values
            test_data = test_df[features].values
            
            fold_scaler = StandardScaler()
            fold_target_scaler = StandardScaler()
            
            fold_scaler.fit(train_data)
            fold_target_scaler.fit(train_data[:, 0].reshape(-1, 1))
            
            scaled_train = fold_scaler.transform(train_data)
            scaled_test = fold_scaler.transform(test_data)
            
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
            
            if len(X_train) == 0 or len(X_test) == 0:
                continue
            
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
            
            fold_info = {
                "fold": fold + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_period": f"{train_df.index[0].date()} to {train_df.index[-1].date()}",
                "test_period": f"{test_df.index[0].date()} to {test_df.index[-1].date()}"
            }
            
            yield X_train, y_train, X_test, y_test, fold_target_scaler, fold_info


def calculate_metrics(y_true, y_pred, strategy_returns=None, risk_free_rate=0.02):
    """
    Calculate evaluation metrics for stock prediction.
    
    Args:
        y_true: Actual log returns
        y_pred: Predicted log returns  
        strategy_returns: Optional array of actual trading returns from following predictions.
                         If provided, Sharpe is calculated on these. Otherwise, uses
                         returns from a simple direction-following strategy.
        risk_free_rate: Annual risk-free rate (default 2%)
        
    Returns:
        Dictionary with MSE, directional accuracy, and Sharpe ratio
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        return {
            "mse": 0.0,
            "directional_accuracy": 0.0,
            "sharpe_ratio": 0.0
        }
    
    mse = np.mean((y_true - y_pred) ** 2)
    
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    dir_accuracy = np.mean(direction_true == direction_pred)
    
    # Sharpe Ratio: Calculate on ACTUAL returns achieved by following predictions
    # If strategy_returns not provided, simulate a simple direction-following strategy
    if strategy_returns is not None:
        returns_for_sharpe = np.array(strategy_returns).flatten()
    else:
        # Simple strategy: go long if predict up, short if predict down
        # Actual return = sign(prediction) * actual_return
        returns_for_sharpe = np.sign(y_pred) * y_true
    
    if len(returns_for_sharpe) > 1 and np.std(returns_for_sharpe) > 0:
        daily_rf = risk_free_rate / 252
        excess_returns = returns_for_sharpe - daily_rf
        sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(returns_for_sharpe)
    else:
        sharpe = 0.0
    
    return {
        "mse": float(mse),
        "directional_accuracy": float(dir_accuracy),
        "sharpe_ratio": float(sharpe)
    }

