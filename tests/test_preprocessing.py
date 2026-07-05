import numpy as np
import pandas as pd
import pytest

from src.preprocessing import add_return_features, clean_prices, detect_outliers
from src.risk_metrics import adf_test, historical_var, sharpe_ratio


@pytest.fixture
def sample_prices():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    prices = pd.DataFrame(
        {"Adj Close": [100, 101, np.nan, 103, 104, 106, 105, np.nan, 108, 110]},
        index=dates,
    )
    return prices


def test_clean_prices_fills_missing_and_sorts(sample_prices):
    cleaned = clean_prices(sample_prices)
    assert cleaned["Adj Close"].isna().sum() == 0
    assert cleaned.index.is_monotonic_increasing


def test_add_return_features_creates_expected_columns(sample_prices):
    cleaned = clean_prices(sample_prices)
    featured = add_return_features(cleaned)
    for col in ["daily_return", "log_return", "rolling_mean_30", "rolling_std_30"]:
        assert col in featured.columns


def test_detect_outliers_flags_extreme_values():
    returns = pd.Series([0.01, 0.02, -0.01, 0.5, 0.015, -0.02])
    flags = detect_outliers(returns, n_std=1.5)
    assert flags.iloc[3]  # the 0.5 return should be flagged


def test_adf_test_returns_expected_keys():
    rng = np.random.default_rng(42)
    stationary_series = pd.Series(rng.normal(size=200))
    result = adf_test(stationary_series, name="synthetic")
    assert "p_value" in result
    assert bool(result["is_stationary"]) is True


def test_historical_var_and_sharpe_ratio_run():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0005, 0.02, size=500))
    var_95 = historical_var(returns, confidence=0.95)
    sr = sharpe_ratio(returns)
    assert var_95 > 0
    assert isinstance(sr, float)
