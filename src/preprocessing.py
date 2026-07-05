"""Cleaning and feature-engineering helpers for the asset price data."""
import numpy as np
import pandas as pd


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a business-day DatetimeIndex, correct dtypes, and fill gaps.

    YFinance data has no missing rows on trading days, but reindexing to a
    full business-day calendar surfaces holidays/gaps which are then
    forward-filled (a standard approach for prices: the last known price
    holds until the next trade).
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    full_range = pd.bdate_range(df.index.min(), df.index.max())
    df = df.reindex(full_range)
    df.index.name = "Date"

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.ffill().bfill()
    return df


def add_return_features(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    """Add daily return, log return, and rolling volatility columns."""
    df = df.copy()
    df["daily_return"] = df[price_col].pct_change()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    df["rolling_mean_30"] = df[price_col].rolling(window=30).mean()
    df["rolling_std_30"] = df["daily_return"].rolling(window=30).std()
    return df


def detect_outliers(returns: pd.Series, n_std: float = 3.0) -> pd.Series:
    """Flag daily returns more than `n_std` standard deviations from the mean."""
    mean, std = returns.mean(), returns.std()
    lower, upper = mean - n_std * std, mean + n_std * std
    return (returns < lower) | (returns > upper)
