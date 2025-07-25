import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad)

def calculate_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return sma, upper, lower

def load_and_align_data(symbol_files, base_path):
    all_data = []
    for symbol, file in symbol_files.items():
        filepath = os.path.join(base_path, file)
        df = pd.read_csv(filepath, sep=';', header=None)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Add indicators
        df["RSI"] = calculate_rsi(df["close"])
        df["%K"], df["%D"] = calculate_stochastic(df["high"], df["low"], df["close"])
        df["CCI"] = calculate_cci(df["high"], df["low"], df["close"])
        df["BB_MID"], df["BB_UPPER"], df["BB_LOWER"] = calculate_bollinger_bands(df["close"])

        df["symbol"] = symbol
        all_data.append(df)

    return pd.concat(all_data)