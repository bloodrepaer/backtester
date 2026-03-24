import pandas as pd
import numpy as np


# ── THE CONTRACT ──────────────────────────────────────────────────────────────
#
# Every strategy the user writes must follow this contract:
#
#   Input : df — a pandas DataFrame with columns [open, high, low, close, volume]
#                indexed by date (DatetimeIndex), sorted chronologically
#
#   Output: a pandas Series of the same index as df
#           values must be in {-1, 0, 1}
#             +1 = long
#              0 = flat / no position
#             -1 = short
#
# The engine handles everything else:
#   - shift(1) to prevent lookahead bias   ← user does NOT do this
#   - slippage and commission              ← user does NOT do this
#   - position sizing                      ← user does NOT do this
#   - metrics and charts                   ← user does NOT do this
#
# The user only answers one question: "given this data, what is my position?"
# ─────────────────────────────────────────────────────────────────────────────


def validate_signal(signal: pd.Series, df: pd.DataFrame) -> pd.Series:
    """
    Validates that the user's signal output is safe to run through the engine.

    Checks:
    1. Is it a pandas Series?
    2. Does it have the same index as df?
    3. Are all values in {-1, 0, 1}?
    4. Are there any NaNs? (warn but don't fail — warmup NaNs are expected)

    Returns the cleaned signal ready for the engine.
    Raises ValueError with a clear message if something is wrong.
    """
    if not isinstance(signal, pd.Series):
        raise ValueError(
            f"Strategy must return a pandas Series. Got {type(signal).__name__}.\n"
            f"Make sure your strategy ends with: return pd.Series(..., index=df.index)"
        )

    if not signal.index.equals(df.index):
        raise ValueError(
            f"Signal index does not match df index.\n"
            f"Signal has {len(signal)} rows, df has {len(df)} rows.\n"
            f"Make sure your signal uses df.index as its index."
        )

    # Check for values outside {-1, 0, 1}
    unique_vals = set(signal.dropna().unique())
    allowed = {-1, 0, 1, -1.0, 0.0, 1.0}
    invalid = unique_vals - allowed
    if invalid:
        raise ValueError(
            f"Signal contains invalid values: {invalid}.\n"
            f"Only -1 (short), 0 (flat), or 1 (long) are allowed."
        )

    # NaN check — warn but don't fail (rolling windows produce warmup NaNs)
    nan_count = signal.isna().sum()
    if nan_count > 0:
        pct = nan_count / len(signal) * 100
        if pct > 30:
            print(f"[WARNING] Signal has {nan_count} NaN values ({pct:.1f}%).")
            print(f"          This is unusually high. Check your rolling window size.")
        else:
            print(f"[INFO] Signal has {nan_count} NaN warmup rows ({pct:.1f}%) — normal for rolling strategies.")

    # Fill NaN with 0 (flat during warmup period)
    signal = signal.fillna(0).astype(float)

    return signal


# ── EXAMPLE STRATEGIES (templates only, not features) ─────────────────────────
#
# These show users what a valid strategy looks like.
# In the web UI, these appear as starter templates the user can edit.
# ─────────────────────────────────────────────────────────────────────────────

def example_sma_crossover(df: pd.DataFrame) -> pd.Series:
    """
    Template: SMA crossover.
    Go long when fast MA > slow MA, flat otherwise.
    """
    fast = df["close"].rolling(50).mean()
    slow = df["close"].rolling(200).mean()
    signal = (fast > slow).astype(int)
    return signal.rename("signal")


def example_rsi_mean_reversion(df: pd.DataFrame) -> pd.Series:
    """
    Template: RSI mean reversion.
    Go long when RSI < 30 (oversold), short when RSI > 70 (overbought).
    """
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))

    signal = pd.Series(0, index=df.index, dtype=float)
    signal[rsi < 30] =  1   # oversold → long
    signal[rsi > 70] = -1   # overbought → short
    return signal.rename("signal")


def example_breakout(df: pd.DataFrame) -> pd.Series:
    """
    Template: Donchian channel breakout.
    Go long when price breaks above N-day high, short when below N-day low.
    """
    n = 20
    high_n = df["high"].rolling(n).max()
    low_n  = df["low"].rolling(n).min()

    signal = pd.Series(0, index=df.index, dtype=float)
    signal[df["close"] >= high_n] =  1
    signal[df["close"] <= low_n]  = -1
    return signal.rename("signal")
