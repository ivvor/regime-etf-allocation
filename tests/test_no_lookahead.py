"""
No-look-ahead-bias tests.

Run with: pytest tests/test_no_lookahead.py -v

These tests verify that truncating the data at some date T does not
change any computed value at or before T. If it does, future data is
leaking into past calculations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_config
from src.features import compute_all_features, rolling_zscore
from src.regime_signal import (
    compute_regime_score, build_regime_weights, DEFAULT_COEFFICIENTS,
)
from src.allocation import get_monthly_signal_dates
from src.backtest import map_signal_to_execution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prices():
    path = ROOT / "data" / "processed" / "adj_close.parquet"
    if not path.exists():
        pytest.skip("Run 01_data_audit.ipynb first to generate processed data.")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "config" / "settings.json")


# ---------------------------------------------------------------------------
# 1. Features do not use future data
# ---------------------------------------------------------------------------

def test_features_no_lookahead(prices):
    """Truncating prices at date T should not change feature values at T."""
    raw_full, z_full = compute_all_features(prices)

    # Truncate at the 60% mark
    cut_idx = int(len(prices) * 0.6)
    cut_date = prices.index[cut_idx]
    raw_trunc, z_trunc = compute_all_features(prices.loc[:cut_date])

    for col in raw_trunc.columns:
        v_full = raw_full.loc[cut_date, col]
        v_trunc = raw_trunc.loc[cut_date, col]
        if np.isnan(v_full) and np.isnan(v_trunc):
            continue
        assert abs(v_full - v_trunc) < 1e-10, (
            f"Feature '{col}' at {cut_date.date()}: "
            f"full={v_full:.8f}, trunc={v_trunc:.8f}"
        )

    for col in z_trunc.columns:
        v_full = z_full.loc[cut_date, col]
        v_trunc = z_trunc.loc[cut_date, col]
        if np.isnan(v_full) and np.isnan(v_trunc):
            continue
        assert abs(v_full - v_trunc) < 1e-10, (
            f"Z-score '{col}' at {cut_date.date()}: "
            f"full={v_full:.8f}, trunc={v_trunc:.8f}"
        )


# ---------------------------------------------------------------------------
# 2. Rolling z-score uses rolling window, not full sample
# ---------------------------------------------------------------------------

def test_rolling_zscore_no_future():
    """Rolling z-score at date T must be identical whether or not
    data after T exists."""
    np.random.seed(42)
    s = pd.Series(np.random.randn(500), name="test")

    z_full = rolling_zscore(s, window=100, min_periods=30)

    mid = 300
    z_trunc = rolling_zscore(s.iloc[:mid], window=100, min_periods=30)

    last = z_trunc.index[-1]
    assert abs(z_full.iloc[last] - z_trunc.iloc[-1]) < 1e-10, (
        f"Z-score differs: full={z_full.iloc[last]}, trunc={z_trunc.iloc[-1]}"
    )


# ---------------------------------------------------------------------------
# 3. Execution date is strictly AFTER signal date
# ---------------------------------------------------------------------------

def test_execution_after_signal(prices):
    """Every execution date must be strictly after its signal date.
    This prevents same-close look-ahead bias."""
    signal_dates = get_monthly_signal_dates(prices)[12:]
    pairs = map_signal_to_execution(signal_dates, prices.index)

    for sig, exec_date in pairs:
        assert exec_date > sig, (
            f"Execution date {exec_date.date()} is not after "
            f"signal date {sig.date()}"
        )


# ---------------------------------------------------------------------------
# 4. Weights at signal date T are unchanged by future data
# ---------------------------------------------------------------------------

def test_weights_no_lookahead(prices):
    """Weights computed with full data vs. truncated data should match
    at a shared signal date (with smoothing=0)."""
    _, z_full = compute_all_features(prices)
    score_full = compute_regime_score(z_full, DEFAULT_COEFFICIENTS)
    signal_dates = get_monthly_signal_dates(prices)[12:]

    # Pick a signal date at the 50% mark
    test_idx = len(signal_dates) // 2
    test_sd = signal_dates[test_idx]

    # Full data
    w_full = build_regime_weights(
        prices, signal_dates, score_full, smoothing_rho=0.0,
    )

    # Truncated data (end 5 trading days after test_sd)
    loc = prices.index.get_loc(test_sd)
    trunc_end = prices.index[min(loc + 5, len(prices) - 1)]
    p_trunc = prices.loc[:trunc_end]
    sd_trunc = signal_dates[signal_dates <= test_sd]
    _, z_trunc = compute_all_features(p_trunc)
    score_trunc = compute_regime_score(z_trunc, DEFAULT_COEFFICIENTS)
    w_trunc = build_regime_weights(
        p_trunc, sd_trunc, score_trunc, smoothing_rho=0.0,
    )

    for col in w_full.columns:
        v_full = w_full.loc[test_sd, col]
        v_trunc = w_trunc.loc[test_sd, col]
        assert abs(v_full - v_trunc) < 1e-10, (
            f"Weight '{col}' at {test_sd.date()}: "
            f"full={v_full:.6f}, trunc={v_trunc:.6f}"
        )


# ---------------------------------------------------------------------------
# 5. Regime score at date T unchanged by future data
# ---------------------------------------------------------------------------

def test_regime_score_no_lookahead(prices):
    """Regime score at T must be identical with or without future data."""
    _, z_full = compute_all_features(prices)
    score_full = compute_regime_score(z_full, DEFAULT_COEFFICIENTS)

    cut_idx = int(len(prices) * 0.6)
    cut_date = prices.index[cut_idx]
    _, z_trunc = compute_all_features(prices.loc[:cut_date])
    score_trunc = compute_regime_score(z_trunc, DEFAULT_COEFFICIENTS)

    v_full = score_full.loc[cut_date]
    v_trunc = score_trunc.loc[cut_date]

    if not (np.isnan(v_full) and np.isnan(v_trunc)):
        assert abs(v_full - v_trunc) < 1e-10, (
            f"Score at {cut_date.date()}: full={v_full}, trunc={v_trunc}"
        )


# ---------------------------------------------------------------------------
# 6. Weight row sums never exceed 1.0
# ---------------------------------------------------------------------------

def test_weight_sums(prices):
    """Target weights must sum to <= 1.0 (remainder is cash)."""
    _, z = compute_all_features(prices)
    score = compute_regime_score(z, DEFAULT_COEFFICIENTS)
    signal_dates = get_monthly_signal_dates(prices)[12:]
    w = build_regime_weights(prices, signal_dates, score, smoothing_rho=0.5)

    sums = w.sum(axis=1)
    assert (sums <= 1.0 + 1e-9).all(), (
        f"Weight sums exceed 1.0: max={sums.max():.6f}"
    )
    assert (w >= -1e-9).all().all(), "Negative weights found"
