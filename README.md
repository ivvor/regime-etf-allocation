# Regime-Aware Tactical Allocation across Taiwan-Listed ETFs

Uses trend, volatility, drawdown, and cross-asset momentum to build a continuous
market regime score. The score is mapped to equity allocation weights via sigmoid,
with the remainder split across bonds, gold, and cash. Monthly rebalanced, with
transaction costs.

## Result

Regime strategy vs five baselines (Buy & Hold, Equal Weight, Static 50/20/20/10,
Momentum Top-2, Inverse Vol), after transaction costs:

![Cumulative return](figures/regime_vs_baselines_cumulative.png)

The strategy's main advantage is drawdown control, not return improvement.
Absolute return is lower than Buy & Hold in every sub-period tested.

![Drawdown comparison](figures/regime_drawdown.png)

## Ablation

Removing each feature one at a time shows trend is the primary driver of the
drawdown/Calmar improvement; the other three features contribute marginally.

![Ablation study](figures/ablation.png)

## Regime score vs price

![Regime score](figures/regime_score.png)

## How to run

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pip install pytest
    jupyter lab

Run the notebooks in order: 01_data_audit -> 02_baselines -> 03_regime_signal -> 04_robustness.

Data is downloaded from Yahoo Finance on the first run and cached in data/processed/.

## ETF universe

| Ticker | Name | Role | Exchange |
|---|---|---|---|
| 0050.TW | 元大台灣50 | Core equity | TWSE |
| 00713.TW | 元大高息低波 | Defensive equity | TWSE |
| 00679B.TWO | 元大美債20年 | Long-term US Treasury | TPEx |
| 00635U.TW | 元大S&P黃金 | Gold futures | TWSE |

00679B is listed on the Taipei Exchange (TPEx), so it uses .TWO in Yahoo Finance, not .TW.
00679B is an unhedged USD bond ETF; 00635U has ~1.5-2% annual futures roll cost.

## Structure

    src/
      data_loader.py        Download + panel construction
      data_validation.py    Quality checks, dividend verification
      features.py           Regime signal features
      regime_signal.py      Continuous score + weight mapping
      allocation.py         Baseline strategies
      backtest.py           Monthly-rebalanced backtest engine
      metrics.py             Sharpe, Sortino, Calmar, drawdown, etc.

    notebooks/
      01_data_audit          Data download, validation, dividend cross-check
      02_baselines           Five baselines (B&H, equal weight, static, momentum, inv-vol)
      03_regime_signal       Regime strategy backtest vs baselines
      04_robustness          Ablation, sensitivity, cost scenarios, sub-period analysis

    tests/
      test_no_lookahead.py   Verify no future data leaks into features/scores/weights

## Tests

    pytest tests/test_no_lookahead.py -v
