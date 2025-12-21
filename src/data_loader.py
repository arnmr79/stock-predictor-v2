import yfinance as yf

import pandas as pd

def fetch_data(ticker):
    """
    Fetches historical data for a given ticker using yfinance.
    """
    data = yf.download(ticker, period="1y")
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    return data

def add_indicators(df):
    """
    Adds technical indicators.
    Tries to use pandas-ta, falls back to manual calculation if missing.
    """
    try:
        import pandas_ta as ta
        # Ensure pandas-ta strategy or direct calls
        if not hasattr(df, 'ta'):
             df.ta = ta.AnalysisIndicators(df)
        df['RSI'] = df.ta.rsi(length=14)
        df['SMA_20'] = df.ta.sma(length=20)
    except ImportError:
        print("pandas-ta not found, using manual calculation.")
        # Manual SMA
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Manual RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
    return df
