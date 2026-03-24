"""
ingestion.py — Smart data loader for Indian market data.

Automatically detects and handles:
  1. NSE Bhavcopy       (SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, TIMESTAMP)
  2. Zerodha Kite       (date with IST timezone offset, ohlcv)
  3. Upstox             (Date, Open, High, Low, Close, Volume, OI)
  4. AngelOne / Shoonya (unix timestamp column named 'time')
  5. Yahoo Finance      (Date, Open, High, Low, Close, Adj Close, Volume)
  6. TradingView        (ISO8601 datetime, ohlcv)
  7. Generic 1-min      (datetime, open, high, low, close, volume)

After detection, all formats are normalised to:
  DatetimeIndex (timezone-naive, IST implied)
  Columns: open, high, low, close, volume  (float64)
  Sorted chronologically, NaNs dropped.

Intraday data is automatically resampled to daily OHLCV.
"""

import pandas as pd
import numpy as np
import io
from pathlib import Path


# ── FORMAT DETECTION ──────────────────────────────────────────────────────────

def _detect_format(df_raw: pd.DataFrame) -> str:
    """
    Inspects column names and index to identify the data source.
    Returns a format string used to route to the correct parser.
    """
    cols = set(df_raw.columns.str.lower().str.strip())

    # NSE Bhavcopy — has SYMBOL and SERIES columns
    if "symbol" in cols and "series" in cols:
        return "nse_bhavcopy"

    # Yahoo Finance — has 'adj close' column
    if "adj close" in cols or "adj_close" in cols:
        return "yahoo"

    # Upstox — has OI column
    if "oi" in cols:
        return "upstox"

    # AngelOne/Shoonya — index or column is unix timestamp (large integer)
    if df_raw.index.dtype in [np.int64, np.float64]:
        return "unix_timestamp"
    if "time" in cols:
        try:
            val = df_raw["time"].iloc[0]
            if isinstance(val, (int, float)) and val > 1e9:
                return "unix_timestamp"
        except Exception:
            pass

    # TradingView — 'time' column with ISO8601 strings
    if "time" in cols:
        return "tradingview"

    # Zerodha — datetime with timezone offset
    idx_str = str(df_raw.index[0]) if len(df_raw) > 0 else ""
    if "+05:30" in idx_str or "IST" in idx_str:
        return "zerodha"

    # Generic fallback — any file with ohlcv columns
    required = {"open", "high", "low", "close"}
    if required.issubset(cols):
        return "generic"

    return "unknown"


# ── FORMAT-SPECIFIC PARSERS ───────────────────────────────────────────────────

def _parse_nse_bhavcopy(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    NSE Bhavcopy format.
    Date column is named TIMESTAMP with format like '28-NOV-2023'.
    Price columns are OPEN, HIGH, LOW, CLOSE.
    Volume column is TOTTRDQTY.
    """
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Parse date
    date_col = next((c for c in df.columns if "timestamp" in c or "date" in c), None)
    if date_col is None:
        raise ValueError("NSE Bhavcopy: cannot find date column.")

    df.index = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df.index.name = "date"

    # Map columns
    col_map = {
        "open"      : "open",
        "high"      : "high",
        "low"       : "low",
        "close"     : "close",
        "tottrdqty" : "volume",
    }
    df = df.rename(columns=col_map)

    if "volume" not in df.columns and "totaltrades" in df.columns:
        df["volume"] = df["totaltrades"]

    return df[["open", "high", "low", "close", "volume"]]


def _parse_zerodha(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Zerodha Kite format.
    Index has timezone-aware timestamps like '2023-11-28 09:15:00+05:30'.
    Strip timezone, keep IST implied.
    """
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Strip timezone info
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    else:
        df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None) \
                   if df.index.dtype == object else df.index

    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def _parse_upstox(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Upstox format. Standard Date column, OI extra column to drop.
    """
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.strip()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def _parse_unix_timestamp(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    AngelOne / Shoonya format. Index or 'time' column is unix epoch seconds.
    """
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.strip()

    if "time" in df.columns:
        df.index = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
        df = df.drop(columns=["time"])
    else:
        df.index = pd.to_datetime(df.index, unit="s", utc=True).dt.tz_localize(None)

    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def _parse_yahoo(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Yahoo Finance format. Has 'Adj Close' which we ignore (using Close).
    """
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.index.name = "date"
    df = df.rename(columns={"adj_close": "_adj_close"})
    return df[["open", "high", "low", "close", "volume"]]


def _parse_tradingview(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    TradingView export. Has a 'time' column with ISO8601 strings.
    """
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.strip()
    df.index = pd.to_datetime(df["time"], errors="coerce").dt.tz_localize(None)
    df = df.drop(columns=["time"])
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def _parse_generic(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Any CSV with ohlcv columns and a parseable date index.
    Best-effort parsing for user's own data.
    """
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Try to parse index as datetime
    try:
        df.index = pd.to_datetime(df.index, errors="coerce")
    except Exception:
        pass

    # If index has tz, strip it
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.index.name = "date"

    # Handle volume column variants
    vol_candidates = ["volume", "vol", "tottrdqty", "qty"]
    vol_col = next((c for c in vol_candidates if c in df.columns), None)
    if vol_col and vol_col != "volume":
        df["volume"] = df[vol_col]
    if "volume" not in df.columns:
        df["volume"] = 0  # volume unknown — fill with 0

    return df[["open", "high", "low", "close", "volume"]]


# ── INTRADAY → DAILY RESAMPLER ────────────────────────────────────────────────

def _is_intraday(df: pd.DataFrame) -> bool:
    """
    Checks if data is intraday (multiple rows per day).
    Looks at the first 10 rows — if any two share the same date, it's intraday.
    """
    if len(df) < 2:
        return False
    dates = df.index[:20].normalize()  # strip time component
    return dates.nunique() < len(dates)


def _resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples intraday OHLCV to daily OHLCV.

    OHLC aggregation rules:
      open   = first bar's open
      high   = max of all highs
      low    = min of all lows
      close  = last bar's close
      volume = sum of all volumes

    Uses 'B' (business day) frequency to align with market calendar.
    """
    daily = df.resample("B").agg({
        "open"  : "first",
        "high"  : "max",
        "low"   : "min",
        "close" : "last",
        "volume": "sum",
    }).dropna(subset=["open", "close"])

    return daily


# ── VALIDATION ────────────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame) -> list:
    """
    Returns a list of warning strings. Empty list = all good.
    """
    warnings = []

    if len(df) < 100:
        warnings.append(f"Only {len(df)} rows — most strategies need 200+ days for meaningful results.")

    if df["high"].lt(df["low"]).any():
        n = df["high"].lt(df["low"]).sum()
        warnings.append(f"{n} rows where high < low — possible data corruption.")

    if df["close"].lt(0).any():
        warnings.append("Negative close prices detected — check your data.")

    # Check for large gaps (weekends excluded — business day gaps > 5 days)
    idx = df.index
    gaps = pd.Series(idx).diff().dt.days.dropna()
    large_gaps = gaps[gaps > 7]
    if len(large_gaps) > 0:
        warnings.append(f"{len(large_gaps)} gaps > 7 days in the data — possible missing periods.")

    # Check for duplicate dates
    dupes = df.index.duplicated().sum()
    if dupes > 0:
        warnings.append(f"{dupes} duplicate dates found and removed.")

    return warnings


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

def smart_load(source, filename: str = "") -> tuple[pd.DataFrame, dict]:
    """
    Main entry point. Accepts a filepath (str) or file-like object (BytesIO).

    Returns
    -------
    df   : clean daily OHLCV DataFrame
    info : dict with detected format, row count, date range, warnings, resampled flag
    """
    # Read raw CSV
    try:
        if isinstance(source, (str, Path)):
            df_raw = pd.read_csv(source, index_col=0, parse_dates=True)
            filename = str(source)
        else:
            # file-like (Streamlit UploadedFile)
            content  = source.read()
            df_raw   = pd.read_csv(io.BytesIO(content), index_col=0, parse_dates=True)
    except Exception as e:
        raise ValueError(f"Could not read file: {e}")

    # Detect format
    fmt = _detect_format(df_raw)

    # Route to parser
    parsers = {
        "nse_bhavcopy"  : _parse_nse_bhavcopy,
        "zerodha"       : _parse_zerodha,
        "upstox"        : _parse_upstox,
        "unix_timestamp": _parse_unix_timestamp,
        "yahoo"         : _parse_yahoo,
        "tradingview"   : _parse_tradingview,
        "generic"       : _parse_generic,
    }

    if fmt == "unknown":
        raise ValueError(
            "Could not detect data format. Make sure your CSV has "
            "open, high, low, close, volume columns and a date index."
        )

    parser = parsers[fmt]
    try:
        df = parser(df_raw)
    except Exception as e:
        raise ValueError(f"Failed to parse {fmt} format: {e}")

    # Drop duplicate index entries
    df = df[~df.index.duplicated(keep="first")]

    # Convert to float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop NaNs in price columns
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Fill missing volume with 0
    df["volume"] = df["volume"].fillna(0)

    # Sort chronologically
    df = df.sort_index()

    # Resample to daily if intraday
    resampled = False
    original_rows = len(df)
    if _is_intraday(df):
        df = _resample_to_daily(df)
        resampled = True

    # Validate
    warnings = _validate(df)

    info = {
        "format"       : fmt,
        "original_rows": original_rows,
        "daily_rows"   : len(df),
        "resampled"    : resampled,
        "date_start"   : df.index[0].date() if len(df) > 0 else None,
        "date_end"     : df.index[-1].date() if len(df) > 0 else None,
        "warnings"     : warnings,
        "filename"     : filename,
    }

    return df, info


def print_load_report(info: dict):
    """Pretty-prints the load report."""
    fmt_labels = {
        "nse_bhavcopy"  : "NSE Bhavcopy",
        "zerodha"       : "Zerodha Kite",
        "upstox"        : "Upstox",
        "unix_timestamp": "AngelOne / Shoonya (unix ts)",
        "yahoo"         : "Yahoo Finance",
        "tradingview"   : "TradingView",
        "generic"       : "Generic OHLCV",
    }
    print(f"\n{'='*50}")
    print(f"  Data loaded successfully")
    print(f"{'='*50}")
    print(f"  Format    : {fmt_labels.get(info['format'], info['format'])}")
    print(f"  File      : {info['filename'] or 'uploaded'}")
    if info["resampled"]:
        print(f"  Rows      : {info['original_rows']} intraday → {info['daily_rows']} daily (resampled)")
    else:
        print(f"  Rows      : {info['daily_rows']} daily bars")
    print(f"  Period    : {info['date_start']} → {info['date_end']}")
    if info["warnings"]:
        print(f"\n  Warnings:")
        for w in info["warnings"]:
            print(f"    ⚠  {w}")
    print(f"{'='*50}\n")


# ── TEST WITH SIMULATED FORMATS ───────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os
    from data_loader import generate_sample_data

    base_df = generate_sample_data(500)

    test_cases = []

    # 1. Generic (our own format)
    buf = io.StringIO()
    base_df.to_csv(buf)
    buf.seek(0)
    test_cases.append(("Generic OHLCV", buf))

    # 2. Yahoo Finance format
    yf = base_df.copy()
    yf["Adj Close"] = yf["close"]
    yf.columns = ["Open","High","Low","Close","Volume","Adj Close"]
    yf.index.name = "Date"
    buf2 = io.StringIO(); yf.to_csv(buf2); buf2.seek(0)
    test_cases.append(("Yahoo Finance", buf2))

    # 3. Upstox format
    up = base_df.copy()
    up["OI"] = 0
    up.columns = ["Open","High","Low","Close","Volume","OI"]
    up.index.name = "Date"
    buf3 = io.StringIO(); up.to_csv(buf3); buf3.seek(0)
    test_cases.append(("Upstox", buf3))

    # 4. Zerodha Kite (timezone-aware index)
    zd = base_df.copy()
    zd.index = pd.DatetimeIndex(
        [str(d) + " 15:30:00+05:30" for d in base_df.index]
    )
    buf4 = io.StringIO(); zd.to_csv(buf4); buf4.seek(0)
    test_cases.append(("Zerodha Kite", buf4))

    # 5. Intraday (should auto-resample)
    intra = base_df.iloc[:50].copy()
    intra_idx = pd.date_range("2023-01-02 09:15", periods=50, freq="15min")
    intra.index = intra_idx
    buf5 = io.StringIO(); intra.to_csv(buf5); buf5.seek(0)
    test_cases.append(("Intraday 15-min (auto-resample)", buf5))

    print("Testing all format parsers...\n")
    all_passed = True
    for name, buf in test_cases:
        try:
            buf_bytes = io.BytesIO(buf.read().encode())
            df_out, info = smart_load(buf_bytes)
            status = "PASS" if len(df_out) > 0 else "FAIL (0 rows)"
        except Exception as e:
            status = f"FAIL — {e}"
            all_passed = False
        print(f"  [{status}] {name} → {info['format']} → {info['daily_rows']} daily rows")

    print(f"\n{'All tests passed.' if all_passed else 'Some tests failed.'}")
