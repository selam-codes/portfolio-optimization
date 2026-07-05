# Portfolio Optimization — Time Series Forecasting for GMF Investments

10 Academy KAIM — Week 9 Challenge: *Time Series Forecasting for Portfolio Management Optimization*

## Business Context

GMF Investments is a financial advisory firm that uses data-driven, time-series forecasting
to inform portfolio management decisions. This project builds forecasting models for
**Tesla (TSLA)**, and uses them alongside historical data for **BND** (bonds) and **SPY**
(S&P 500) to construct and backtest an optimized portfolio using Modern Portfolio Theory.

## Assets

| Asset | Ticker | Role |
|---|---|---|
| Tesla | TSLA | High-growth, high-risk |
| Vanguard Total Bond Market ETF | BND | Low-risk, stability |
| S&P 500 ETF | SPY | Moderate-risk, diversified market exposure |

Data period: **2015-01-01 to 2026-06-30**, sourced from Yahoo Finance via `yfinance`.

## Project Status

- [x] **Task 1** — Data extraction, cleaning, EDA, stationarity testing (ADF), risk metrics (VaR, Sharpe)
- [x] **Task 2 (in progress)** — Chronological train/test split + ARIMA model (`pmdarima.auto_arima`)
- [ ] Task 2 (remaining) — LSTM model + model comparison
- [ ] Task 3 — Future forecast with confidence intervals, trend/risk analysis
- [ ] Task 4 — Efficient Frontier & optimal portfolio (PyPortfolioOpt)
- [ ] Task 5 — Strategy backtest vs. 60/40 SPY/BND benchmark

## Project Structure

```
portfolio-optimization/
├── .vscode/settings.json
├── .github/workflows/unittests.yml
├── requirements.txt
├── data/
│   ├── raw/           # cached yfinance downloads (gitignored)
│   └── processed/      # cleaned/feature-engineered CSVs, metric summaries
├── notebooks/
│   ├── 1.0-data-preprocessing-eda.ipynb
│   └── 2.0-arima-modeling.ipynb
├── reports/figures/    # exported charts used in the interim/final report
├── src/                 # reusable data loading, preprocessing, risk-metric functions
├── tests/               # pytest unit tests for src/
└── scripts/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Fetch/refresh data and regenerate processed CSVs
jupyter nbconvert --to notebook --execute --inplace notebooks/1.0-data-preprocessing-eda.ipynb

# ARIMA model
jupyter nbconvert --to notebook --execute --inplace notebooks/2.0-arima-modeling.ipynb

# Unit tests
pytest tests/ -v
```

## Key Findings So Far (Task 1 & initial Task 2)

- Price levels for TSLA, BND, and SPY are **non-stationary** (ADF test, p > 0.05); daily
  returns for all three are **stationary** (p < 0.05) — confirming **d = 1** is the correct
  differencing order for ARIMA on the price series.
- TSLA has the highest annualized return (~44%) but also by far the highest annualized
  volatility (~56%) and 1-day 95% VaR (~5.1%), versus SPY (~17% return / ~17% vol) and BND
  (~2% return / ~5% vol).
- `auto_arima` selected an ARIMA(0,1,0) model for TSLA on the training data (2015–2024);
  forecast test-period metrics: MAE ≈ 54.2, RMSE ≈ 70.2, MAPE ≈ 17.1% (test period:
  2025-01 to 2026-06). Confidence intervals widen rapidly over the forecast horizon,
  consistent with the Efficient Market Hypothesis.

## Team

Kerod, Mahbubah, Feven (10 Academy KAIM Week 9 facilitators)
