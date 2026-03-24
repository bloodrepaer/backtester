import pandas as pd
import numpy as np
from data_loader import load_data


# ── STEP 1: COMPUTE DAILY RETURNS ────────────────────────────────────────────

def compute_returns(df):
    """
    WHY log returns and not simple returns?

    Simple return:  r = (P_t - P_{t-1}) / P_{t-1}
    Log return:     r = log(P_t / P_{t-1})

    Log returns are ADDITIVE across time:
        log(P_3/P_1) = log(P_3/P_2) + log(P_2/P_1)
    Simple returns are NOT additive — you can't just sum them.

    For backtesting, log returns let us do:
        total_return = exp(sum of daily log returns) - 1
    which is mathematically exact.

    .pct_change() gives simple returns. np.log(close/close.shift(1)) gives log returns.
    We'll use log returns.
    """
    close = df["close"]

    # shift(1) means "yesterday's close"
    # So log(today / yesterday) = today's log return
    # shift(1) shifts the series FORWARD by 1 — i.e. at row t, shift(1) gives row t-1
    log_ret = np.log(close / close.shift(1))

    # First row is NaN because there's no "yesterday" — drop it
    log_ret = log_ret.dropna()

    return log_ret


# ── STEP 2: BUILD AN SMA CROSSOVER SIGNAL ────────────────────────────────────

def sma_crossover_signal(df, fast=50, slow=200):
    """
    SMA crossover: the simplest trend-following strategy.

    Logic:
    - fast SMA > slow SMA → market trending UP → go LONG (+1)
    - fast SMA < slow SMA → market trending DOWN → stay FLAT (0)

    WHY .shift(1) on the signal?
    This is the MOST IMPORTANT line in any backtest.

    If the signal at time t uses close[t], and we also trade at close[t],
    that's LOOKAHEAD BIAS — we're using information we didn't have
    when the trade was placed.

    In reality: signal fires at close[t] → trade executes at open[t+1]
    OR conservatively: signal fires at close[t] → trade executes at close[t+1]

    We use the conservative version: shift the signal by 1 day.
    This means: "the signal I computed yesterday tells me what to hold today."

    WITHOUT shift(1): Sharpe looks 30-50% higher but it's a lie.
    WITH shift(1):    Sharpe is lower but it's real.
    """
    close = df["close"]

    fast_ma = close.rolling(fast).mean()   # rolling window mean
    slow_ma = close.rolling(slow).mean()

    # Raw signal: 1 if fast > slow, else 0
    raw_signal = (fast_ma > slow_ma).astype(int)

    # SHIFT BY 1 — this is Domain 3 (backtest correctness) in one line
    signal = raw_signal.shift(1)

    return signal, fast_ma, slow_ma


# ── STEP 3: COMPUTE STRATEGY RETURNS ─────────────────────────────────────────

def strategy_returns(log_ret, signal):
    """
    Strategy return on day t = signal[t] * market_return[t]

    If signal = 1 (long): you earn the market return
    If signal = 0 (flat):  you earn nothing
    If signal = -1 (short): you earn the negative of market return

    We align signal and log_ret on the same index first.
    pandas does this automatically with .mul() but being explicit is safer.
    """
    # Align both series — drops NaNs from rolling window warmup period
    aligned = pd.DataFrame({"ret": log_ret, "signal": signal}).dropna()

    strat_ret = aligned["ret"] * aligned["signal"]
    market_ret = aligned["ret"]

    return strat_ret, market_ret, aligned


# ── STEP 4: COMPUTE EQUITY CURVE ─────────────────────────────────────────────

def equity_curve(log_returns, starting_capital=100):
    """
    Converts a series of daily log returns into a running portfolio value.

    cumsum() of log returns = total log return up to that day
    exp() converts back to price space

    Starting at 100 means the curve shows "how much Rs.100 became".
    """
    return starting_capital * np.exp(log_returns.cumsum())


# ── RUN EVERYTHING ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    log_ret = compute_returns(df)
    signal, fast_ma, slow_ma = sma_crossover_signal(df, fast=50, slow=200)
    strat_ret, market_ret, aligned = strategy_returns(log_ret, signal)

    strat_curve  = equity_curve(strat_ret)
    market_curve = equity_curve(market_ret)

    print(f"Date range after warmup: {strat_ret.index[0].date()} to {strat_ret.index[-1].date()}")
    print(f"Strategy final value : Rs. {strat_curve.iloc[-1]:.2f} (started at Rs. 100)")
    print(f"Market   final value : Rs. {market_curve.iloc[-1]:.2f} (started at Rs. 100)")
    print()
    print("Daily strategy returns (first 5):")
    print(strat_ret.head())
