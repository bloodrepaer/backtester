import pandas as pd
import numpy as np
from data_loader import load_data
from returns import compute_returns, sma_crossover_signal, strategy_returns, equity_curve


# ── SHARPE RATIO ──────────────────────────────────────────────────────────────

def sharpe_ratio(log_returns, risk_free_rate=0.065, periods_per_year=252):
    """
    Sharpe = (mean return - risk free rate) / std of returns
             annualised

    WHY 252: trading days in a year (not 365 — markets are closed on weekends)
    WHY risk_free_rate=0.065: India 10Y Gsec yield ~6.5% — use local rf rate

    The annualisation:
    - mean daily return * 252 = annualised mean
    - daily std * sqrt(252)   = annualised std  (by the sqrt-time rule)
    - Sharpe = annualised excess return / annualised std

    Sqrt-time rule: variance scales linearly with time, so std scales with sqrt(time).
    This comes from the assumption that daily returns are i.i.d. (independent).
    """
    rf_daily = risk_free_rate / periods_per_year

    excess_returns = log_returns - rf_daily
    mean_excess = excess_returns.mean() * periods_per_year
    std_ann      = log_returns.std() * np.sqrt(periods_per_year)

    if std_ann == 0:
        return 0.0

    return mean_excess / std_ann


# ── MAX DRAWDOWN ──────────────────────────────────────────────────────────────

def max_drawdown(log_returns):
    """
    Drawdown = how far you are from your previous peak, in % terms.
    Max drawdown = the worst peak-to-trough decline in the history.

    Steps:
    1. Build equity curve (cumulative returns)
    2. At each point, find the running maximum up to that point (the "peak")
    3. Drawdown at t = (equity[t] - peak[t]) / peak[t]
    4. Max drawdown = min of all drawdowns (most negative value)

    WHY this matters: a strategy with Sharpe 2.0 but max DD of -60%
    is unacceptable — you'd panic-exit long before recovery.
    Calmar = annual return / |max drawdown| — penalises deep drawdowns.
    """
    curve = equity_curve(log_returns)

    # running maximum: at each point, what's the highest value seen so far
    running_max = curve.cummax()

    # drawdown at each point: how far below the peak are we
    drawdown = (curve - running_max) / running_max

    max_dd = drawdown.min()        # most negative value = worst drawdown
    drawdown_series = drawdown     # full series for plotting

    return max_dd, drawdown_series


# ── CALMAR RATIO ──────────────────────────────────────────────────────────────

def calmar_ratio(log_returns, periods_per_year=252):
    """
    Calmar = annualised return / |max drawdown|

    Better than Sharpe for trend-following strategies because it measures
    return per unit of ACTUAL LOSS EXPERIENCED, not just volatility.

    A strategy with Calmar > 1.0 is considered decent.
    Your Strategy B has Calmar 4.02 — that's excellent.
    """
    ann_return = log_returns.mean() * periods_per_year
    max_dd, _ = max_drawdown(log_returns)

    if max_dd == 0:
        return np.inf

    return ann_return / abs(max_dd)


# ── SORTINO RATIO ─────────────────────────────────────────────────────────────

def sortino_ratio(log_returns, risk_free_rate=0.065, periods_per_year=252):
    """
    Sortino is like Sharpe but only penalises DOWNSIDE volatility.

    Sharpe penalises ALL volatility — including upside moves, which is unfair.
    A strategy that has huge positive days looks "risky" under Sharpe.
    Sortino fixes this: only std of NEGATIVE returns goes in the denominator.

    downside_std = std of returns below the target (we use rf rate as target)
    """
    rf_daily = risk_free_rate / periods_per_year
    excess = log_returns - rf_daily

    # Only keep negative excess returns for downside deviation
    downside = excess[excess < 0]
    downside_std = downside.std() * np.sqrt(periods_per_year)

    ann_excess = excess.mean() * periods_per_year

    if downside_std == 0:
        return np.inf

    return ann_excess / downside_std


# ── ANNUALISED RETURN ─────────────────────────────────────────────────────────

def annualised_return(log_returns, periods_per_year=252):
    """
    Compound annual growth rate (CAGR) from log returns.

    Total log return = sum of daily log returns
    Total simple return = exp(total log return) - 1
    CAGR = (1 + total simple return)^(1/years) - 1
    """
    n = len(log_returns)
    total_log_ret = log_returns.sum()
    total_simple_ret = np.exp(total_log_ret) - 1
    years = n / periods_per_year
    cagr = (1 + total_simple_ret) ** (1 / years) - 1
    return cagr


# ── FULL TEARSHEET ────────────────────────────────────────────────────────────

def tearsheet(log_returns, name="Strategy"):
    """
    Prints a clean summary of all key metrics.
    This is what pyfolio does — we're building the same thing from scratch.
    """
    sharpe  = sharpe_ratio(log_returns)
    calmar  = calmar_ratio(log_returns)
    sortino = sortino_ratio(log_returns)
    cagr    = annualised_return(log_returns)
    max_dd, _ = max_drawdown(log_returns)
    ann_vol = log_returns.std() * np.sqrt(252)

    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  CAGR            : {cagr*100:.2f}%")
    print(f"  Annual vol      : {ann_vol*100:.2f}%")
    print(f"  Sharpe ratio    : {sharpe:.2f}")
    print(f"  Sortino ratio   : {sortino:.2f}")
    print(f"  Calmar ratio    : {calmar:.2f}")
    print(f"  Max drawdown    : {max_dd*100:.2f}%")
    print(f"  Days in sample  : {len(log_returns)}")
    print(f"{'='*40}\n")


# ── RUN EVERYTHING ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    log_ret = compute_returns(df)
    signal, _, _ = sma_crossover_signal(df, fast=50, slow=200)
    strat_ret, market_ret, _ = strategy_returns(log_ret, signal)

    tearsheet(strat_ret,  name="SMA 50/200 Strategy")
    tearsheet(market_ret, name="Buy and Hold (Market)")
