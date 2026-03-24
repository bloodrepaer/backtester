import pandas as pd
import numpy as np
from data_loader import load_data
from returns import compute_returns, equity_curve
from metrics import sharpe_ratio, calmar_ratio, sortino_ratio, max_drawdown, annualised_return
from strategy import validate_signal


# ── TRANSACTION COSTS ─────────────────────────────────────────────────────────

def _transaction_costs(signal: pd.Series, slippage_bps: float, commission_bps: float) -> pd.Series:
    """
    Cost is incurred on days the signal CHANGES (a trade happens).
    signal.diff().abs() is 1 on trade days, 0 otherwise.
    Total cost per trade = slippage + commission, in decimal form.
    """
    cost_per_trade = (slippage_bps + commission_bps) / 10_000
    trade_days = signal.diff().abs().fillna(0)
    return trade_days * cost_per_trade


# ── VOLATILITY-TARGETED POSITION SIZING ──────────────────────────────────────

def _vol_size(log_ret: pd.Series, signal: pd.Series,
              target_vol: float, lookback: int, max_leverage: float) -> pd.Series:
    """
    Scales position size so portfolio vol stays near target_vol.
    Uses shift(1) on the vol estimate — no lookahead.
    Binary signal (0/1/-1) is multiplied by the scalar size.
    When signal=0, position=0 regardless of size.
    """
    rolling_vol = log_ret.rolling(lookback).std() * np.sqrt(252)
    rolling_vol = rolling_vol.shift(1).ffill().fillna(0.01)
    size = (target_vol / rolling_vol).clip(upper=max_leverage)
    return signal * size


# ── THE ENGINE ────────────────────────────────────────────────────────────────

def run(
    strategy_fn,
    df: pd.DataFrame = None,
    filepath: str = None,
    slippage_bps: float = 8,
    commission_bps: float = 2,
    vol_target: float = 0.20,
    max_leverage: float = 3.0,
    vol_sizing: bool = True,
) -> dict:
    """
    The single entry point. User passes their strategy function, engine does the rest.

    Parameters
    ----------
    strategy_fn   : callable — user's strategy. Must take df, return pd.Series of {-1,0,1}
    df            : DataFrame — pre-loaded OHLCV data (optional, loads sample if None)
    filepath      : str — path to CSV if user wants to load their own data
    slippage_bps  : float — slippage per trade in basis points (default 8)
    commission_bps: float — commission per trade in basis points (default 2)
    vol_target    : float — annualised vol target for position sizing (default 20%)
    max_leverage  : float — maximum position size multiplier (default 3x)
    vol_sizing    : bool  — whether to apply vol targeting (default True)

    Returns
    -------
    dict with keys:
        net_ret, gross_ret, market_ret, signal, costs, equity_curve, metrics
    """

    # ── 1. LOAD DATA ──────────────────────────────────────────────────────────
    if df is None:
        df = load_data(filepath)

    # ── 2. RUN USER STRATEGY ──────────────────────────────────────────────────
    try:
        raw_signal = strategy_fn(df)
    except Exception as e:
        raise RuntimeError(
            f"Your strategy raised an error:\n  {type(e).__name__}: {e}\n"
            f"Check your strategy function."
        ) from e

    # ── 3. VALIDATE SIGNAL ────────────────────────────────────────────────────
    signal = validate_signal(raw_signal, df)

    # ── 4. SHIFT BY 1 — ENGINE DOES THIS, USER DOES NOT ──────────────────────
    #
    # This is the single most important line in the engine.
    # The user's signal at time t says what position to take based on close[t].
    # But we can only trade AFTER seeing close[t] — i.e. at open[t+1] or later.
    # So we shift every signal forward by 1 day.
    # The user never has to think about this — the engine guarantees it.
    #
    signal = signal.shift(1)

    # ── 5. COMPUTE MARKET RETURNS ─────────────────────────────────────────────
    log_ret = compute_returns(df)

    # ── 6. ALIGN (drops warmup NaNs from rolling windows + the shift) ─────────
    aligned = pd.DataFrame({
        "ret":    log_ret,
        "signal": signal,
    }).dropna()

    ret    = aligned["ret"]
    sig    = aligned["signal"]

    # ── 7. POSITION SIZING ────────────────────────────────────────────────────
    if vol_sizing:
        sized_sig = _vol_size(ret, sig, vol_target, lookback=20, max_leverage=max_leverage)
    else:
        sized_sig = sig

    # ── 8. GROSS RETURNS ──────────────────────────────────────────────────────
    gross_ret = ret * sized_sig

    # ── 9. TRANSACTION COSTS ──────────────────────────────────────────────────
    costs   = _transaction_costs(sig, slippage_bps, commission_bps)
    net_ret = gross_ret - costs

    # ── 10. METRICS ───────────────────────────────────────────────────────────
    max_dd, dd_series = max_drawdown(net_ret)

    metrics = {
        "cagr"        : annualised_return(net_ret),
        "sharpe"      : sharpe_ratio(net_ret),
        "sortino"     : sortino_ratio(net_ret),
        "calmar"      : calmar_ratio(net_ret),
        "max_dd"      : max_dd,
        "annual_vol"  : net_ret.std() * np.sqrt(252),
        "win_rate"    : (net_ret > 0).mean(),
        "n_trades"    : int((sig.diff().abs() > 0).sum()),
        "total_costs" : costs.sum(),
    }

    return {
        "net_ret"     : net_ret,
        "gross_ret"   : gross_ret,
        "market_ret"  : ret,
        "signal"      : sig,
        "sized_signal": sized_sig,
        "costs"       : costs,
        "dd_series"   : dd_series,
        "eq_curve"    : equity_curve(net_ret),
        "mkt_curve"   : equity_curve(ret),
        "metrics"     : metrics,
        "df"          : df,
    }
