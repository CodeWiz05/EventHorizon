import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import RobustScaler
import logging
from src.config import TICKER_TO_ASSET_CLASS_MAP, ASSET_CLASS_CONTEXT_MAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketDataEngineer:
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.asset_class = TICKER_TO_ASSET_CLASS_MAP.get(ticker, "EQUITY")
        self.context = ASSET_CLASS_CONTEXT_MAP.get(self.asset_class, {"behavior": "pro-cyclical"})
        self.feature_names = []
        
    def fetch_data(self) -> pd.DataFrame:
        logging.info(f"Fetching data for {self.ticker} ({self.asset_class}) from {self.start_date} to {self.end_date}")
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df

    def _engineer_macro_fear_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Original features: Highly predictive for Gold and Bonds"""
        df['Rolling_Mean_21d'] = df['Log_Return'].rolling(window=21).mean()
        
        def get_slope(y):
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]
        df['Rolling_Slope_63d'] = df['Close'].rolling(window=63).apply(get_slope, raw=True)
        
        df['Vol_21d'] = df['Log_Return'].rolling(window=21).std()
        df['Vol_63d'] = df['Log_Return'].rolling(window=63).std()
        df['Vol_126d'] = df['Log_Return'].rolling(window=126).std()
        df['Vol_Struct_21_126'] = df['Vol_21d'] / df['Vol_126d'].replace(0, np.nan)
        
        rolling_peak_21d = df['Close'].rolling(window=21).max()
        df['Max_DD_21d'] = (df['Close'] / rolling_peak_21d) - 1
        df['Sign_Persistence_21d'] = (df['Log_Return'] > 0).rolling(window=21).mean()

        self.feature_names = [
            'Rolling_Mean_21d', 'Rolling_Slope_63d', 'Vol_21d', 'Vol_63d', 
            'Vol_Struct_21_126', 'Max_DD_21d', 'Sign_Persistence_21d'
        ]
        return df

    def _engineer_equity_distribution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fast-Twitch distribution features designed to capture V-shaped recoveries 
        and institutional distribution without lagging structural trends.
        """
        # 1. EWMA Volatility (Replaces Vol_21d)
        # Reacts immediately to spikes, but decays the "memory" of a crash rapidly
        df['Vol_EWMA_10d'] = df['Log_Return'].ewm(span=10, adjust=False).std()
        
        # 2. Volatility Rate of Change
        # Allows HMM to differentiate between "Crash" (High + Rising Vol) 
        # and "Recovery" (High + Falling Vol)
        df['Vol_Change'] = df['Vol_EWMA_10d'].diff()
        
        # 3. Short-Term Momentum
        # Catches the exact inflection point of a V-shape recovery
        df['Momentum_5d'] = df['Log_Return'].rolling(window=5).mean()
        
        # 4. Structural Anchors (Retained)
        df['Skew_21d'] = df['Log_Return'].rolling(window=21).skew()
        df['Trend_Distance_63d'] = df['Close'] / df['Close'].rolling(window=63).mean() - 1

        self.feature_names = [
            'Vol_EWMA_10d', 'Vol_Change', 'Momentum_5d', 
            'Skew_21d', 'Trend_Distance_63d'
        ]
        return df

    def engineer_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        if self.context['behavior'] == 'pro-cyclical':
            return self._engineer_equity_distribution_features(df)
        else:
            return self._engineer_macro_fear_features(df)

    def run_pipeline(self) -> pd.DataFrame:
        raw_df = self.fetch_data()
        engineered_df = self.engineer_structural_features(raw_df)
        
        logging.info(f"Cleaning data and extracting {len(self.feature_names)} features...")
        engineered_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        engineered_df.dropna(inplace=True)
            
        return engineered_df

class FeatureFusion:
    def __init__(self, feature_cols: list):
        self.feature_cols = feature_cols
        self.scaler = RobustScaler()
        self.is_fit = False
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raw_features = df[self.feature_cols].values
        scaled_features = self.scaler.fit_transform(raw_features)
        self.is_fit = True
        scaled_df = pd.DataFrame(data=scaled_features, index=df.index, columns=self.feature_cols)
        return pd.concat([df[['Close']], scaled_df], axis=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raw_features = df[self.feature_cols].values
        scaled_features = self.scaler.transform(raw_features)
        scaled_df = pd.DataFrame(data=scaled_features, index=df.index, columns=self.feature_cols)
        return pd.concat([df[['Close']], scaled_df], axis=1)