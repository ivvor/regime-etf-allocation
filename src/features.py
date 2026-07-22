"""Regime features. All use only backward-looking data.

Sign convention before z-scoring: trend positive=bullish, vol positive=
high vol (bearish), drawdown negative=in drawdown (bearish), defensive
strength positive=defensives beating equity (bearish for equity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_trend(prices: pd.DataFrame, ticker: str, lookbacks=(63, 126, 252)) -> pd.Series:
    s = prices[ticker]
    moms = pd.concat([s / s.shift(lb) - 1 for lb in lookbacks], axis=1)
    return moms.mean(axis=1).rename("trend")


def compute_realized_vol(prices: pd.DataFrame, ticker: str, window: int = 20) -> pd.Series:
    r = prices[ticker].pct_change()
    return (r.rolling(window, min_periods=window).std() * np.sqrt(252)).rename(f"vol_{window}d")


def compute_drawdown(prices: pd.DataFrame, ticker: str) -> pd.Series:
    s = prices[ticker]
    return (s / s.cummax() - 1).rename("drawdown")


def compute_defensive_strength(
    prices: pd.DataFrame, equity_ticker: str, bond_ticker: str, gold_ticker: str,
    lookback: int = 63,
) -> pd.Series:
    def _mom(ticker):
        s = prices[ticker]
        return s / s.shift(lookback) - 1
    eq, bond, gold = _mom(equity_ticker), _mom(bond_ticker), _mom(gold_ticker)
    return ((bond + gold) / 2 - eq).rename("defensive_strength")


def rolling_zscore(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """Rolling z-score. Never uses future data (unlike full-sample standardization)."""
    mu = series.rolling(window, min_periods=min_periods).mean()
    sigma = series.rolling(window, min_periods=min_periods).std()
    z = (series - mu) / sigma
    return z.replace([np.inf, -np.inf], np.nan).rename(f"z_{series.name}")


def compute_all_features(
    prices: pd.DataFrame,
    core_ticker: str = "0050.TW",
    bond_ticker: str = "00679B.TWO",
    gold_ticker: str = "00635U.TW",
    zscore_window: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame({
        "trend":              compute_trend(prices, core_ticker),
        "vol_20d":            compute_realized_vol(prices, core_ticker, 20),
        "drawdown":           compute_drawdown(prices, core_ticker),
        "defensive_strength": compute_defensive_strength(prices, core_ticker, bond_ticker, gold_ticker),
    })
    z = pd.DataFrame({
        f"z_{col}": rolling_zscore(raw[col], window=zscore_window)
        for col in raw.columns
    })
    return raw, z
