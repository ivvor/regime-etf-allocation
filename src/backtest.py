"""Monthly-rebalanced backtest engine.

Signal at month-end close; execution at next trading day's close. Weights
drift daily between rebalances. Turnover measured from drifted weights to
new target weights, with cost deducted on rebalance day. Cash earns a
configurable daily rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    cumulative:    pd.Series
    weights:       pd.DataFrame
    turnover:      pd.Series
    costs:         pd.Series
    name:          str = ""


def map_signal_to_execution(
    signal_dates: pd.DatetimeIndex,
    trading_days: pd.DatetimeIndex,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Map each signal date to the next trading day (execution date).

    Used by both the backtest engine and the no-look-ahead tests, so a
    signal generated on date X can never be executed at date X's own close.
    """
    pairs = []
    for sd in signal_dates:
        future = trading_days[trading_days > sd]
        if len(future) > 0:
            pairs.append((sd, future[0]))
    return pairs


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_bps: float = 20.0,
    cash_rate_annual: float = 0.015,
    name: str = "",
) -> BacktestResult:
    etf_cols = prices.columns.tolist()
    daily_rets = prices.pct_change()
    cash_daily = (1 + cash_rate_annual) ** (1 / 252) - 1

    pairs = map_signal_to_execution(target_weights.index, prices.index)
    exec_map = {exec_date: sd for sd, exec_date in pairs}
    if not exec_map:
        raise ValueError("No valid signal->execution date pairs found.")

    trading_days = prices.index
    first_exec = min(exec_map.keys())

    active_w = None
    active_cash = 0.0
    records = []
    turn_records = {}
    cost_records = {}

    for day in trading_days:
        if day < first_exec:
            continue

        if day in exec_map:
            sig_date = exec_map[day]
            new_w = target_weights.loc[sig_date].reindex(etf_cols, fill_value=0.0)
            new_cash = max(0.0, 1.0 - new_w.sum())

            turn = ((new_w - active_w).abs().sum() + abs(new_cash - active_cash)
                    if active_w is not None else 0.0)
            cost = cost_bps / 10000 * turn
            turn_records[day] = turn
            cost_records[day] = cost

            active_w = new_w.copy()
            active_cash = new_cash

        if active_w is None:
            continue
        if day not in daily_rets.index:
            continue
        day_r = daily_rets.loc[day]
        if day_r.isna().any():
            continue

        port_ret = (active_w * day_r).sum() + active_cash * cash_daily
        if day in cost_records:
            port_ret -= cost_records[day]
        records.append({"date": day, "return": port_ret})

        for t in etf_cols:
            active_w[t] *= 1 + day_r[t]
        active_cash *= 1 + cash_daily
        total = active_w.sum() + active_cash
        if total > 0:
            active_w /= total
            active_cash /= total

    ret_series = pd.DataFrame(records).set_index("date")["return"]
    cum = (1 + ret_series).cumprod()

    return BacktestResult(
        daily_returns=ret_series,
        cumulative=cum,
        weights=target_weights,
        turnover=pd.Series(turn_records, name="turnover"),
        costs=pd.Series(cost_records, name="cost"),
        name=name,
    )
