"""Download and validate Taiwan-listed ETF price data from Yahoo Finance.

00679B trades on the Taipei Exchange (TPEx / 上櫃), so it needs the .TWO
suffix, not .TW. Getting this wrong doesn't raise an error -- it silently
returns an all-NaN column, which only surfaces later as a confusing
"no common sample" failure. download_etf_data() checks for this directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "settings.json"
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def download_etf_data(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV + dividends for each ticker.

    Returns dict: ticker -> DataFrame with Open/High/Low/Close/Adj Close/
    Volume/Dividends.
    """
    raw = yf.download(
        tickers=tickers, start=start, end=end,
        auto_adjust=False, actions=True, progress=False,
        group_by="ticker", threads=True,
    )

    if raw.empty:
        raise RuntimeError("yfinance returned no data. Check connection and tickers.")

    result = {}
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                df = raw[ticker].copy()
            except KeyError:
                raise RuntimeError(
                    f"Ticker '{ticker}' not found. Available: "
                    f"{sorted(set(raw.columns.get_level_values(0)))}"
                )
        else:
            df = raw.copy()

        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        if df[price_col].isna().all():
            raise RuntimeError(
                f"No price data for '{ticker}'. Check the exchange suffix: "
                f".TW = TWSE main board, .TWO = Taipei Exchange (TPEx)."
            )
        result[ticker] = df

    return result


def build_adj_close_panel(ticker_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({
        t: df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        for t, df in ticker_data.items()
    })


def build_dividend_panel(ticker_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({
        t: df["Dividends"] if "Dividends" in df.columns else pd.Series(0.0, index=df.index)
        for t, df in ticker_data.items()
    })


def build_close_panel(ticker_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({t: df["Close"] for t, df in ticker_data.items()})


def get_common_sample(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any NaN to get the common trading-day panel."""
    common = prices.dropna(how="any").copy()
    if common.empty:
        raise RuntimeError("No common sample after dropping NaN rows.")
    return common
