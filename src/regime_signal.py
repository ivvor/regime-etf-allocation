"""Continuous regime score and weight allocation.

S_t = beta_trend*z_trend - beta_vol*z_vol + beta_dd*z_drawdown - beta_def*z_defensive
Higher S_t -> more risk-on -> higher equity weight via sigmoid mapping:
    w_equity = w_min + (w_max - w_min) * sigmoid(S_t)
Remaining weight splits across bond, gold, and cash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_COEFFICIENTS = {
    "z_trend":              +0.35,
    "z_vol_20d":            -0.25,
    "z_drawdown":           +0.20,
    "z_defensive_strength": -0.20,
}

EQUAL_COEFFICIENTS = {
    "z_trend":              +0.25,
    "z_vol_20d":            -0.25,
    "z_drawdown":           +0.25,
    "z_defensive_strength": -0.25,
}


def compute_regime_score(z_features: pd.DataFrame, coefficients: dict[str, float] | None = None) -> pd.Series:
    if coefficients is None:
        coefficients = DEFAULT_COEFFICIENTS
    score = pd.Series(0.0, index=z_features.index)
    for col, beta in coefficients.items():
        if col not in z_features.columns:
            raise KeyError(f"Feature '{col}' not found. Available: {list(z_features.columns)}")
        score += beta * z_features[col]
    return score.rename("regime_score")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_regime_weights(
    prices: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    regime_score: pd.Series,
    core_ticker: str = "0050.TW",
    defensive_ticker: str = "00713.TW",
    bond_ticker: str = "00679B.TWO",
    gold_ticker: str = "00635U.TW",
    w_min: float = 0.20,
    w_max: float = 0.90,
    core_def_lookback: int = 63,
    non_equity_split: dict[str, float] | None = None,
    smoothing_rho: float = 0.0,
) -> pd.DataFrame:
    """Build monthly target weights from regime score.

    non_equity_split default: {"bond": 0.45, "gold": 0.25, "cash": 0.30}.
    smoothing_rho: 0 = no smoothing, blends with previous month's weights otherwise.
    """
    if non_equity_split is None:
        non_equity_split = {"bond": 0.45, "gold": 0.25, "cash": 0.30}

    tickers = [core_ticker, defensive_ticker, bond_ticker, gold_ticker]
    rows = []
    prev_w = None

    for sd in signal_dates:
        score_at_sd = regime_score.loc[:sd].dropna()
        if len(score_at_sd) == 0:
            rows.append({t: 0.25 for t in tickers})
            continue

        s = score_at_sd.iloc[-1]
        eq_w = w_min + (w_max - w_min) * sigmoid(s)

        hist = prices.loc[:sd]
        if len(hist) >= core_def_lookback:
            core_mom = hist[core_ticker].iloc[-1] / hist[core_ticker].iloc[-core_def_lookback] - 1
            def_mom = hist[defensive_ticker].iloc[-1] / hist[defensive_ticker].iloc[-core_def_lookback] - 1
            q = sigmoid((core_mom - def_mom) * 10)
        else:
            q = 0.5

        core_w = eq_w * q
        def_w = eq_w * (1 - q)
        non_eq = 1.0 - eq_w
        bond_w = non_eq * non_equity_split["bond"]
        gold_w = non_eq * non_equity_split["gold"]

        row = {core_ticker: core_w, defensive_ticker: def_w, bond_ticker: bond_w, gold_ticker: gold_w}

        if smoothing_rho > 0 and prev_w is not None:
            row = {t: smoothing_rho * prev_w[t] + (1 - smoothing_rho) * row[t] for t in tickers}

        rows.append(row)
        prev_w = row.copy()

    return pd.DataFrame(rows, index=signal_dates)
