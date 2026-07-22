"""Baseline allocation strategies.

Each function returns target weights indexed by signal date (last trading
day of month), columns = tickers, values = weight (row sums <= 1, remainder
is cash). Every function uses only data available on or before the signal date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_monthly_signal_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    return prices.groupby(prices.index.to_period("M")).tail(1).index


def buy_and_hold(signal_dates, tickers, hold_ticker) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=signal_dates, columns=tickers)
    w[hold_ticker] = 1.0
    return w


def equal_weight(signal_dates, tickers) -> pd.DataFrame:
    return pd.DataFrame(1.0 / len(tickers), index=signal_dates, columns=tickers)


def static_allocation(signal_dates, tickers, weight_map: dict[str, float]) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=signal_dates, columns=tickers)
    for t, val in weight_map.items():
        if t in w.columns:
            w[t] = val
    return w


def momentum_rotation(
    prices: pd.DataFrame, signal_dates, lookback_days: int = 126, top_n: int = 2,
) -> pd.DataFrame:
    """Hold top_n ETFs by trailing return, equal-weighted."""
    tickers = prices.columns.tolist()
    rows = []
    for sd in signal_dates:
        hist = prices.loc[:sd]
        if len(hist) < lookback_days:
            row = {t: 1.0 / len(tickers) for t in tickers}
        else:
            mom = hist.iloc[-1] / hist.iloc[-lookback_days] - 1
            top = mom.nlargest(top_n).index.tolist()
            row = {t: (1.0 / top_n if t in top else 0.0) for t in tickers}
        rows.append(row)
    return pd.DataFrame(rows, index=signal_dates)


def inverse_volatility(
    prices: pd.DataFrame, signal_dates, lookback_days: int = 60,
) -> pd.DataFrame:
    tickers = prices.columns.tolist()
    daily_rets = prices.pct_change()
    rows = []
    for sd in signal_dates:
        hist = daily_rets.loc[:sd]
        if len(hist) < lookback_days:
            row = {t: 1.0 / len(tickers) for t in tickers}
        else:
            vol = hist.iloc[-lookback_days:].std().replace(0, np.nan)
            inv_vol = (1.0 / vol).fillna(0)
            total = inv_vol.sum()
            row = (inv_vol / total).to_dict() if total > 0 else {t: 1.0 / len(tickers) for t in tickers}
        rows.append(row)
    return pd.DataFrame(rows, index=signal_dates)
