"""Fetch and cache historical price data from YFinance."""
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2026-06-30"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def fetch_asset_data(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    cache_dir: Path = RAW_DIR,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download (or load cached) OHLCV data for each ticker.

    Returns a dict mapping ticker -> DataFrame indexed by Date with
    columns Open, High, Low, Close, Adj Close, Volume.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for ticker in tickers:
        cache_path = cache_dir / f"{ticker}.csv"
        if cache_path.exists() and not force_refresh:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            df = yf.download(
                ticker, start=start, end=end, auto_adjust=False, progress=False
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index.name = "Date"
            df.to_csv(cache_path)
        data[ticker] = df
    return data


def combine_close_prices(data: dict[str, pd.DataFrame], field: str = "Adj Close") -> pd.DataFrame:
    """Combine a single field (e.g. Adj Close) from each asset into one wide DataFrame."""
    combined = pd.DataFrame({ticker: df[field] for ticker, df in data.items()})
    combined = combined.sort_index()
    return combined


if __name__ == "__main__":
    raw = fetch_asset_data()
    for ticker, df in raw.items():
        print(ticker, df.shape, df.index.min(), df.index.max())
