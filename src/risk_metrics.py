"""Risk and statistical-test helpers: ADF stationarity test, VaR, Sharpe ratio."""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

TRADING_DAYS_PER_YEAR = 252


def adf_test(series: pd.Series, name: str = "") -> dict:
    """Run the Augmented Dickey-Fuller test and return key results."""
    series = series.dropna()
    result = adfuller(series, autolag="AIC")
    return {
        "name": name,
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (empirical) Value at Risk at the given confidence level.

    Returned as a positive number representing the loss threshold, e.g.
    0.03 means a 3% daily loss at the given confidence level.
    """
    returns = returns.dropna()
    return -np.percentile(returns, (1 - confidence) * 100)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio from daily returns."""
    returns = returns.dropna()
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    return returns.mean() * TRADING_DAYS_PER_YEAR


def annualized_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """Maximum drawdown of a cumulative-return series (e.g. (1+r).cumprod())."""
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1
    return drawdown.min()
