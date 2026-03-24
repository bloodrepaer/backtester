import pandas as pd
import numpy as np

# ── SIMULATE NIFTY-LIKE DATA ─────────────────────────────────────────────────
# In production, replace this function with:
#   df = pd.read_csv("your_nifty_file.csv", parse_dates=["datetime"])
# Everything below this function stays exactly the same.

def generate_sample_data(n_days=2000, seed=42):
    """
    Generates daily OHLCV data that mimics NIFTY50 price behaviour.
    Uses GBM — geometric brownian motion — the standard model for stock prices.

    GBM formula:  S(t) = S(t-1) * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
    where Z ~ N(0,1), mu = drift, sigma = volatility

    WHY GBM: stock returns are approximately log-normally distributed.
    Taking log of prices makes them normally distributed — GBM models that.
    """
    np.random.seed(seed)

    dates = pd.bdate_range(start="2017-01-01", periods=n_days)  # business days only

    # NIFTY50 historical params: ~12% annual return, ~18% annual vol
    mu    = 0.12 / 252          # daily drift (annualised / 252 trading days)
    sigma = 0.18 / np.sqrt(252) # daily vol   (annualised / sqrt(252) by sqrt-time rule)

    # Generate log returns then cumulative price path
    log_returns = (mu - 0.5 * sigma**2) + sigma * np.random.randn(n_days)
    close = 10000 * np.exp(np.cumsum(log_returns))  # starts at ~10000

    # Build OHLC from close (realistic approximation)
    daily_range = close * sigma * np.abs(np.random.randn(n_days)) * 0.5
    open_  = close * (1 + 0.002 * np.random.randn(n_days))
    high   = np.maximum(open_, close) + daily_range
    low    = np.minimum(open_, close) - daily_range
    volume = np.random.randint(5_000_000, 20_000_000, size=n_days)

    df = pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }, index=dates)

    df.index.name = "date"
    return df


# ── LOAD AND VALIDATE ─────────────────────────────────────────────────────────

def load_data(filepath=None):
    """
    Load OHLCV data. If no filepath given, uses simulated data.

    What we do here and WHY:
    1. parse_dates  — tells pandas the index is timestamps, not plain strings
    2. sort_index   — guarantees chronological order (.shift() depends on this)
    3. dropna       — removes rows where any OHLCV value is missing
    4. astype(float)— ensures all price columns are float64, not object dtype
    """
    if filepath:
        df = pd.read_csv(
            filepath,
            index_col=0,
            parse_dates=True,
        )
    else:
        df = generate_sample_data()

    # Standardise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Keep only what we need
    required = ["open", "high", "low", "close", "volume"]
    df = df[required]

    df = df.sort_index()    # chronological order — critical for .shift()
    df = df.dropna()        # drop any row with a missing value
    df = df.astype(float)   # price columns must be float, never object/string

    print(f"Loaded {len(df)} rows | {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Columns : {list(df.columns)}")
    print(f"Any NaNs: {df.isna().any().any()}")
    print()
    print(df.head())
    return df


if __name__ == "__main__":
    df = load_data()
