"""Data quality checks for the ETF price panel."""

from __future__ import annotations

import numpy as np
import pandas as pd


def audit_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker date range and missing-value summary, before dropna."""
    return pd.DataFrame({
        "first_valid_date": prices.apply(lambda s: s.first_valid_index()),
        "last_valid_date":  prices.apply(lambda s: s.last_valid_index()),
        "n_observations":   prices.notna().sum(),
        "n_missing":        prices.isna().sum(),
        "missing_pct":      (prices.isna().mean() * 100).round(2),
    })


def check_date_continuity(prices: pd.DataFrame, max_gap_days: int = 10) -> pd.Series:
    """Flag gaps between consecutive trading dates larger than max_gap_days.

    Normal weekend/holiday gaps are 1-4 calendar days. Anything bigger
    usually means a partial download failure got silently dropped by dropna.
    """
    gaps = prices.index.to_series().diff().dt.days.dropna()
    suspicious = gaps[gaps > max_gap_days]
    if not suspicious.empty:
        print("WARNING: gaps larger than expected:")
        for date, gap in suspicious.items():
            print(f"  {int(gap)} days ending {date.date()}")
    else:
        print(f"Date continuity OK, no gaps > {max_gap_days} days.")
    return suspicious


def return_audit(prices: pd.DataFrame) -> pd.DataFrame:
    """Basic per-ticker return statistics (mean, std, min/max, annualized)."""
    rets = prices.pct_change().dropna()
    stats = rets.agg(["mean", "std", "min", "max"]).T
    stats["annualized_return"] = (stats["mean"] * 252).round(4)
    stats["annualized_vol"]    = (stats["std"] * np.sqrt(252)).round(4)
    stats["min_date"] = rets.idxmin()
    stats["max_date"] = rets.idxmax()
    return stats


def flag_large_moves(prices: pd.DataFrame, threshold: float = 0.08) -> pd.DataFrame:
    """Flag single-day returns exceeding threshold (default 8%)."""
    rets = prices.pct_change().dropna()
    flagged = rets.where(rets.abs() > threshold).stack().dropna()
    if flagged.empty:
        print(f"No single-day moves exceeding ±{threshold:.0%}.")
        return pd.DataFrame(columns=["return"])
    result = flagged.rename("return").to_frame().sort_values("return")
    print(f"Found {len(result)} moves exceeding ±{threshold:.0%}:")
    print(result.to_string())
    return result


def verify_dividend_adjustment(
    adj_close: pd.Series,
    unadj_close: pd.Series,
    dividends: pd.Series,
    ticker: str,
    n_checks: int = 5,
) -> pd.DataFrame:
    """Cross-check Adj Close against reported dividends on ex-dividend dates.

    On the ex-date, the adjusted series should satisfy:
        adj_ratio = (close_on_ex + dividend) / close_before
    Rearranged for the dividend:
        implied_div = close_before * (adj_ratio - price_return)
    Flags discrepancy > 50 bps between implied and reported dividend.
    """
    div_dates = dividends[dividends > 0].index
    if len(div_dates) == 0:
        print(f"  {ticker}: no dividends recorded.")
        return pd.DataFrame()

    check_dates = div_dates[-n_checks:]
    rows = []

    for ex_date in check_dates:
        loc = adj_close.index.get_loc(ex_date)
        if loc == 0:
            continue
        prev_date = adj_close.index[loc - 1]

        div_amount   = dividends[ex_date]
        close_before = unadj_close[prev_date]
        adj_before   = adj_close[prev_date]
        adj_on_ex    = adj_close[ex_date]
        close_on_ex  = unadj_close[ex_date]

        price_return = close_on_ex / close_before
        adj_ratio    = adj_on_ex / adj_before
        implied_div  = close_before * (adj_ratio - price_return)
        discrepancy_bps = abs(implied_div - div_amount) / close_before * 10000

        rows.append({
            "ex_date":          ex_date.date(),
            "dividend":         round(div_amount, 4),
            "close_before":     round(close_before, 2),
            "implied_dividend": round(implied_div, 4),
            "discrepancy_bps":  round(discrepancy_bps, 1),
            "status":           "OK" if discrepancy_bps < 50 else "CHECK",
        })

    result = pd.DataFrame(rows)
    if len(result) == 0:
        print(f"  {ticker}: could not verify (dividend dates at edge of sample).")
        return result

    n_check = (result["status"] != "OK").sum()
    if n_check == 0:
        print(f"  {ticker}: {len(result)}/{len(result)} dividend adjustments OK.")
    else:
        print(f"  {ticker}: {n_check}/{len(result)} discrepancies > 50 bps, "
              f"cross-check with FinMind/TEJ.")
    return result
