"""Performance metrics for backtest evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import BacktestResult


def compute_metrics(result: BacktestResult, risk_free_annual: float = 0.015) -> dict:
    r = result.daily_returns
    rf_daily = risk_free_annual / 252
    n_years = len(r) / 252

    total_return = result.cumulative.iloc[-1] - 1
    ann_ret = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = r.std() * np.sqrt(252)

    excess = r.mean() - rf_daily
    sharpe = excess / r.std() * np.sqrt(252) if r.std() > 0 else np.nan

    cum = result.cumulative
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()

    underwater = dd < 0
    if underwater.any():
        groups = (~underwater).cumsum()
        max_dd_duration = int(underwater.groupby(groups).sum().max())
    else:
        max_dd_duration = 0

    downside = r[r < rf_daily] - rf_daily
    downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(252) if len(downside) > 0 else np.nan
    sortino = (ann_ret - risk_free_annual) / downside_dev if downside_dev and downside_dev > 0 else np.nan

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    win_rate = (monthly > 0).mean()

    annual_turnover = result.turnover.sum() / n_years if n_years > 0 else 0

    return {
        "Annual Return":     round(ann_ret, 4),
        "Annual Volatility": round(ann_vol, 4),
        "Sharpe":            round(sharpe, 3),
        "Sortino":           round(sortino, 3),
        "Max Drawdown":      round(max_dd, 4),
        "Max DD Duration":   max_dd_duration,
        "Calmar":            round(calmar, 3),
        "Monthly Win Rate":  round(win_rate, 3),
        "Worst Month":       round(monthly.min(), 4),
        "Best Month":        round(monthly.max(), 4),
        "Annual Turnover":   round(annual_turnover, 3),
        "Total Cost":        round(result.costs.sum(), 4),
    }


def compare_strategies(results: list[BacktestResult], risk_free_annual: float = 0.015) -> pd.DataFrame:
    rows = {res.name: compute_metrics(res, risk_free_annual) for res in results}
    return pd.DataFrame(rows).T
