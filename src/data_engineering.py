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

    def _engineer_equity_distribution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Vol_EWMA_10d'] = df['Log_Return'].ewm(span=10, adjust=False).std()
        df['Vol_Change'] = df['Vol_EWMA_10d'].diff()
        df['Momentum_5d'] = df['Log_Return'].rolling(window=5).mean()
        df['Skew_21d'] = df['Log_Return'].rolling(window=21).skew()
        df['Trend_Distance_63d'] = df['Close'] / df['Close'].rolling(window=63).mean() - 1

        self.feature_names = [
            'Vol_EWMA_10d', 'Vol_Change', 'Momentum_5d', 
            'Skew_21d', 'Trend_Distance_63d'
        ]
        return df

    def _engineer_fx_features(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI_14d'] = 100 - (100 / (1 + rs))

        roll_mean = df['Close'].rolling(window=21).mean()
        roll_std = df['Close'].rolling(window=21).std()
        df['BB_Width_21d'] = (roll_std * 2) / roll_mean

        df['Dist_MA_63d'] = (df['Close'] / df['Close'].rolling(window=63).mean()) - 1.0

        df['Autocorr_5d'] = df['Log_Return'].rolling(window=21).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(pd.Series(x).dropna()) > 1 else np.nan
        )

        self.feature_names = ['RSI_14d', 'BB_Width_21d', 'Dist_MA_63d', 'Autocorr_5d']
        return df

    def _engineer_safe_haven_features(self, df: pd.DataFrame) -> pd.DataFrame:
        def get_slope(y):
            y_series = pd.Series(y).dropna()
            x = np.arange(len(y_series))
            if len(x) < 2: return np.nan
            return np.polyfit(x, y_series, 1)[0]
            
        df['Macro_Slope_126d'] = df['Close'].rolling(window=126).apply(get_slope, raw=True)

        df['Vol_21d'] = df['Log_Return'].rolling(window=21).std()
        df['Vol_252d'] = df['Log_Return'].rolling(window=252).std()
        df['Vol_Ratio'] = df['Vol_21d'] / df['Vol_252d'].replace(0, np.nan)

        negative_returns = df['Log_Return'].copy()
        negative_returns[negative_returns > 0] = 0
        df['Downside_Dev_21d'] = negative_returns.rolling(window=21).std()

        df['Max_DD_63d'] = (df['Close'] / df['Close'].rolling(window=63).max()) - 1.0

        self.feature_names = ['Macro_Slope_126d', 'Vol_Ratio', 'Downside_Dev_21d', 'Max_DD_63d']
        return df

    def engineer_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        behavior = self.context.get('behavior', 'pro-cyclical')
        
        if behavior == 'pro-cyclical':
            return self._engineer_equity_distribution_features(df)
        elif behavior == 'mean-reverting':
            return self._engineer_fx_features(df)
        else:
            return self._engineer_safe_haven_features(df)

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