"""Monthly-rebalanced backtest engine.

Signal at month-end close; execution at next trading day's close. This means:
  - On the execution day, the OLD weights still earn that day's return (the
    trade happens at that day's close, so the new position hasn't existed
    for any part of the day yet).
  - The NEW weights start earning returns from the day AFTER execution.
  - Transaction cost is deducted from the execution day's return (the cost
    of trading at that close).

Turnover is measured only on the risk-asset weights (not cash), since moving
into/out of cash is not itself a trade that incurs cost -- only buying/selling
the ETFs does. Cash is the residual (1 - sum of ETF weights), so its change is
already implied by the ETF weight changes and must not be added separately.

The very first position is entered from 100% cash, and that initial trade
is charged the same cost_bps as any other rebalance (no free initial entry).
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
    """Map each signal date to the next trading day (execution date)."""
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

    # Start from 100% cash before the first execution -- the first rebalance
    # is a real trade (cash -> target weights) and is charged like any other.
    active_w = pd.Series(0.0, index=etf_cols)
    active_cash = 1.0

    records = []
    turn_records = {}
    cost_records = {}

    for day in trading_days:
        if day < first_exec:
            continue
        if day not in daily_rets.index:
            continue
        day_r = daily_rets.loc[day]
        if day_r.isna().any():
            continue

        # Step 1: today's return is earned on the weights that were ALREADY
        # in place entering today (i.e., before any rebalance that happens
        # at today's close).
        raw_ret = (active_w * day_r).sum() + active_cash * cash_daily

        cost = 0.0
        if day in exec_map:
            sig_date = exec_map[day]
            new_w = target_weights.loc[sig_date].reindex(etf_cols, fill_value=0.0)
            new_cash = max(0.0, 1.0 - new_w.sum())

            # Turnover on risk-asset weights only -- cash is the residual,
            # its change is implied and must not be counted again.
            turn = (new_w - active_w).abs().sum()
            cost = cost_bps / 10000 * turn
            turn_records[day] = turn
            cost_records[day] = cost

        net_ret = raw_ret - cost
        records.append({"date": day, "return": net_ret})

        if day in exec_map:
            # The trade executes at today's close: the new weights become
            # active starting tomorrow, with no drift applied yet today.
            active_w = new_w.copy()
            active_cash = new_cash
        else:
            # No rebalance today: drift yesterday's weights forward by
            # today's realized return.
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
