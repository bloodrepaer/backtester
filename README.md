# Backtester

A rigorous backtesting engine for Indian retail traders. Write your strategy in Python — the engine handles everything else: lookahead prevention, slippage, commission, vol-targeted sizing, walk-forward validation, and parameter optimisation.

## Why this exists

Every free Indian backtesting tool has at least one of these problems:
- **Lookahead bias** — using tomorrow's data to make today's decision. Sharpe looks great. It's a lie.
- **No walk-forward validation** — parameters tuned on the full dataset are overfit. They fail in live trading.
- **Wrong transaction costs** — ignoring slippage, STT, exchange fees. Returns are inflated.

This engine fixes all three structurally — the user physically cannot introduce lookahead bias because `shift(1)` is applied by the engine, not the user.

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/yourname/backtester
cd backtester
pip install -r requirements.txt

# 2. Run the web UI
streamlit run app.py

# 3. Or use the CLI
python cli.py --strategy my_strategy.py --mode 2
```

Opens at `http://localhost:8501`

---

## Writing a strategy

Create a `.py` file with a `strategy` function:

```python
# my_strategy.py
import pandas as pd

def strategy(df: pd.DataFrame) -> pd.Series:
    """
    df has columns: open, high, low, close, volume
    Return a Series of -1 (short), 0 (flat), 1 (long) with index = df.index
    """
    fast = df["close"].rolling(20).mean()
    slow = df["close"].rolling(50).mean()
    return (fast > slow).astype(int)
```

**Do not** call `.shift()` — the engine does this.  
**Do not** handle slippage or position sizing — the engine does this.

For parameter optimisation (mode 3), also define:

```python
def strategy_factory(params: dict):
    fast = params["fast"]
    slow = params["slow"]
    def strat(df):
        return (df["close"].rolling(fast).mean() >
                df["close"].rolling(slow).mean()).astype(int)
    return strat

PARAM_GRID = {
    "fast": [20, 50, 100],
    "slow": [100, 150, 200],
}
```

---

## Three modes

| Mode | Command | What it does |
|------|---------|-------------|
| 1 | `--mode 1` | Full backtest on all data. Benchmark vs buy-and-hold + risk-free. |
| 2 | `--mode 2` | Walk-forward validation. Tests if strategy generalises across time periods. |
| 3 | `--mode 3` | Optimised walk-forward. Grid searches params on train, evaluates on unseen test. |

---

## Supported data formats

Upload any of these — format is auto-detected:

| Source | Format |
|--------|--------|
| NSE Bhavcopy | `nseindia.com` daily CSV |
| Zerodha Kite | Timezone-aware datetime index |
| Upstox | Standard OHLCV + OI column |
| AngelOne / Shoonya | Unix timestamp index |
| Yahoo Finance | `yfinance` download with Adj Close |
| TradingView | ISO8601 time column export |
| Generic | Any CSV with open/high/low/close/volume + date index |

Intraday data (1-min, 5-min, 15-min) is automatically resampled to daily OHLCV.

---

## CLI reference

```bash
python cli.py [OPTIONS]

  --data        PATH     OHLCV CSV file (default: simulated data)
  --strategy    PATH     Strategy .py file
  --mode        INT      1=backtest, 2=walk-forward, 3=optimised WF
  --name        STR      Strategy display name
  --windows     INT      Walk-forward windows (default: 5)
  --train-pct   FLOAT    Train fraction per window (default: 0.70)
  --optimise-on STR      sharpe | calmar | cagr (default: sharpe)
  --slippage    FLOAT    Slippage in bps (default: 8)
  --commission  FLOAT    Commission in bps (default: 2)
  --vol-target  FLOAT    Vol target annualised (default: 0.20)
  --max-lev     FLOAT    Max leverage (default: 3.0)
  --no-vol-size          Disable vol-targeted sizing
  --out         PATH     Output directory for chart
  --no-plot              Skip chart generation
```

---

## Project structure

```
backtester/
├── app.py           ← Streamlit web UI
├── cli.py           ← Terminal interface
├── engine.py        ← Core runner (shift, costs, sizing, metrics)
├── ingestion.py     ← Smart data loader (7 formats, auto-resample)
├── walk_forward.py  ← Rolling OOS validation + consistency verdict
├── optimiser.py     ← Grid search + IS→OOS decay check
├── metrics.py       ← Sharpe, Calmar, Sortino, drawdown, CAGR
├── strategy.py      ← Contract + validator
├── returns.py       ← Log returns, equity curve
├── data_loader.py   ← Base loader + GBM simulator
└── requirements.txt
```

---

## Metrics explained

| Metric | What it measures | Good threshold |
|--------|-----------------|----------------|
| CAGR | Compound annual growth rate | > benchmark |
| Sharpe | Return per unit of total volatility | > 1.0 |
| Sortino | Return per unit of downside volatility | > 1.5 |
| Calmar | Annual return / max drawdown | > 1.0 |
| Max drawdown | Worst peak-to-trough decline | < -20% |
| IS→OOS decay | How much Sharpe drops from train to test | < 0.3 = robust |

---

## Security note

Strategy code is executed via `exec()` — safe for local use. For a hosted product, replace `execute_strategy_code()` in `app.py` with a sandboxed subprocess or RestrictedPython.

---

## Built with

- `pandas` — data manipulation
- `numpy` — vectorised math  
- `matplotlib` — charts
- `streamlit` — web UI
