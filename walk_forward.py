import pandas as pd
import numpy as np
from data_loader import load_data
from engine import run
from metrics import sharpe_ratio, calmar_ratio, annualised_return, max_drawdown
from returns import equity_curve


# ── WINDOW GENERATOR ──────────────────────────────────────────────────────────

def generate_windows(df, train_pct=0.70, n_windows=5):
    """
    Splits the full date range into n_windows rolling train/test splits.

    WHY rolling and not a single split:
    A single train/test split gives you one out-of-sample period.
    That period might be a bull market, a crash, or a sideways grind —
    one regime. You can't know if your result is luck or skill.

    Rolling windows give you n_windows out-of-sample periods across
    different market regimes. If the strategy works across all of them,
    that's evidence it generalises.

    Structure of each window:
      total_days = len(df)
      each window covers: total_days / n_windows days
      train = first train_pct of the window
      test  = remaining (1 - train_pct) of the window

    Example with n_windows=5, train_pct=0.70, 2000 days total:
      each window = 400 days
      train = 280 days, test = 120 days

    Parameters
    ----------
    df          : full DataFrame
    train_pct   : fraction of each window used for training (0.70 = 70%)
    n_windows   : number of rolling windows

    Returns
    -------
    list of dicts: [{"train": df_train, "test": df_test, "window": i}, ...]
    """
    n = len(df)
    window_size = n // n_windows
    train_size  = int(window_size * train_pct)
    test_size   = window_size - train_size

    if test_size < 20:
        raise ValueError(
            f"Test window too small ({test_size} days). "
            f"Reduce n_windows or train_pct."
        )

    windows = []
    for i in range(n_windows):
        start = i * window_size
        end   = start + window_size
        end   = min(end, n)           # last window takes remainder

        df_window = df.iloc[start:end]
        split     = int(len(df_window) * train_pct)

        df_train = df_window.iloc[:split]
        df_test  = df_window.iloc[split:]

        windows.append({
            "window" : i + 1,
            "train"  : df_train,
            "test"   : df_test,
            "train_start": df_train.index[0].date(),
            "train_end"  : df_train.index[-1].date(),
            "test_start" : df_test.index[0].date(),
            "test_end"   : df_test.index[-1].date(),
            "train_days" : len(df_train),
            "test_days"  : len(df_test),
        })

    return windows


# ── SINGLE WINDOW RUNNER ──────────────────────────────────────────────────────

def run_window(strategy_fn, window, engine_kwargs):
    """
    Runs the strategy on the TEST portion of a window.

    WHY we pass both train and test:
    Some strategies need to be 'fitted' on train data (e.g. finding optimal
    params via optimisation — that's the next module).
    For now, the strategy_fn sees full df but we only evaluate on test.

    The engine runs on the FULL window (train+test) so that rolling indicators
    have enough warmup data. We then slice the results to test only.
    This prevents the first few test bars from having NaN signals due to
    insufficient warmup history.
    """
    df_full = pd.concat([window["train"], window["test"]])

    result = run(strategy_fn, df=df_full, **engine_kwargs)

    # Slice to test period only for evaluation
    test_start = window["test"].index[0]
    test_end   = window["test"].index[-1]

    net_ret_test = result["net_ret"].loc[test_start:test_end]
    mkt_ret_test = result["market_ret"].loc[test_start:test_end]

    if len(net_ret_test) == 0:
        return None

    max_dd, _ = max_drawdown(net_ret_test)

    return {
        "window"     : window["window"],
        "test_start" : window["test_start"],
        "test_end"   : window["test_end"],
        "test_days"  : len(net_ret_test),
        "net_ret"    : net_ret_test,
        "market_ret" : mkt_ret_test,
        "cagr"       : annualised_return(net_ret_test),
        "sharpe"     : sharpe_ratio(net_ret_test),
        "calmar"     : calmar_ratio(net_ret_test),
        "max_dd"     : max_dd,
        "n_trades"   : int((result["signal"].loc[test_start:test_end].diff().abs() > 0).sum()),
    }


# ── WALK-FORWARD RUNNER ───────────────────────────────────────────────────────

def walk_forward(
    strategy_fn,
    df             = None,
    filepath       = None,
    train_pct      = 0.70,
    n_windows      = 5,
    slippage_bps   = 8,
    commission_bps = 2,
    vol_target     = 0.20,
    max_leverage   = 3.0,
    vol_sizing     = True,
    verbose        = True,
):
    """
    Runs walk-forward validation across n_windows rolling windows.

    Returns
    -------
    dict with:
        window_results : list of per-window metrics
        oos_net_ret    : stitched out-of-sample returns (the honest equity curve)
        oos_mkt_ret    : stitched market returns for the same periods
        summary        : aggregate metrics across all windows
    """
    if df is None:
        df = load_data(filepath)

    engine_kwargs = dict(
        slippage_bps   = slippage_bps,
        commission_bps = commission_bps,
        vol_target     = vol_target,
        max_leverage   = max_leverage,
        vol_sizing     = vol_sizing,
    )

    windows = generate_windows(df, train_pct=train_pct, n_windows=n_windows)

    if verbose:
        print(f"\nWalk-forward validation — {n_windows} windows, "
              f"{int(train_pct*100)}/{int((1-train_pct)*100)} train/test split")
        print("=" * 64)
        print(f"  {'Win':<5} {'Test period':<24} {'Days':<6} "
              f"{'CAGR':>7} {'Sharpe':>8} {'Max DD':>8} {'Trades':>7}")
        print("-" * 64)

    window_results = []
    oos_net_rets   = []
    oos_mkt_rets   = []

    for w in windows:
        res = run_window(strategy_fn, w, engine_kwargs)
        if res is None:
            continue

        window_results.append(res)
        oos_net_rets.append(res["net_ret"])
        oos_mkt_rets.append(res["market_ret"])

        if verbose:
            print(f"  {res['window']:<5} "
                  f"{str(res['test_start'])+' → '+str(res['test_end']):<24} "
                  f"{res['test_days']:<6} "
                  f"{res['cagr']*100:>6.1f}% "
                  f"{res['sharpe']:>8.2f} "
                  f"{res['max_dd']*100:>7.1f}% "
                  f"{res['n_trades']:>7}")

    # Stitch all out-of-sample periods together
    oos_net_ret = pd.concat(oos_net_rets).sort_index()
    oos_mkt_ret = pd.concat(oos_mkt_rets).sort_index()

    # Aggregate metrics
    oos_max_dd, oos_dd_series = max_drawdown(oos_net_ret)
    summary = {
        "oos_cagr"      : annualised_return(oos_net_ret),
        "oos_sharpe"    : sharpe_ratio(oos_net_ret),
        "oos_calmar"    : calmar_ratio(oos_net_ret),
        "oos_max_dd"    : oos_max_dd,
        "oos_win_rate"  : (oos_net_ret > 0).mean(),
        "n_windows"     : len(window_results),
        "pct_profitable": sum(r["cagr"] > 0 for r in window_results) / len(window_results),
        "avg_sharpe"    : np.mean([r["sharpe"] for r in window_results]),
        "std_sharpe"    : np.std([r["sharpe"]  for r in window_results]),
    }

    if verbose:
        print("=" * 64)
        mkt_cagr   = annualised_return(oos_mkt_ret)
        mkt_sharpe = sharpe_ratio(oos_mkt_ret)
        mkt_dd,_   = max_drawdown(oos_mkt_ret)
        print(f"\n  Out-of-sample summary (stitched across all test windows)")
        print(f"  {'Metric':<30} {'Strategy':>10}  {'Market':>8}")
        print(f"  {'-'*52}")
        print(f"  {'CAGR':<30} {summary['oos_cagr']*100:>9.2f}%  {mkt_cagr*100:>7.2f}%")
        print(f"  {'Sharpe':<30} {summary['oos_sharpe']:>10.2f}  {mkt_sharpe:>8.2f}")
        print(f"  {'Calmar':<30} {summary['oos_calmar']:>10.2f}  {'—':>8}")
        print(f"  {'Max drawdown':<30} {summary['oos_max_dd']*100:>9.2f}%  {mkt_dd*100:>7.2f}%")
        print(f"  {'Win rate (daily)':<30} {summary['oos_win_rate']*100:>9.2f}%  {'—':>8}")
        print(f"  {'Windows profitable':<30} {summary['pct_profitable']*100:>9.0f}%  {'—':>8}")
        print(f"  {'Sharpe mean / std':<30} {summary['avg_sharpe']:>6.2f} / {summary['std_sharpe']:.2f}  {'—':>8}")
        print()

        # Consistency check — the most important output
        _consistency_check(summary, window_results)

    return {
        "window_results" : window_results,
        "oos_net_ret"    : oos_net_ret,
        "oos_mkt_ret"    : oos_mkt_ret,
        "oos_dd_series"  : oos_dd_series,
        "summary"        : summary,
    }


# ── CONSISTENCY CHECK ─────────────────────────────────────────────────────────

def _consistency_check(summary, window_results):
    """
    The most important output of walk-forward validation.
    Tells the user whether their strategy is robust or overfit.

    Rules of thumb:
    - Profitable in 4/5+ windows → consistent
    - Sharpe std < Sharpe mean   → consistent (not all over the place)
    - OOS Sharpe > 0.5           → usable
    - OOS Sharpe > in-sample     → red flag (impossible — means IS was underfit)
    """
    pct  = summary["pct_profitable"]
    sh   = summary["avg_sharpe"]
    shsd = summary["std_sharpe"]
    oos  = summary["oos_sharpe"]

    print("  CONSISTENCY VERDICT")
    print("  " + "-" * 40)

    if pct >= 0.8:
        print(f"  Profitable windows : {pct*100:.0f}%  ← consistent")
    elif pct >= 0.6:
        print(f"  Profitable windows : {pct*100:.0f}%  ← acceptable")
    else:
        print(f"  Profitable windows : {pct*100:.0f}%  ← inconsistent, likely overfit")

    if shsd < abs(sh) * 0.5:
        print(f"  Sharpe stability   : {sh:.2f} +/- {shsd:.2f}  ← stable")
    else:
        print(f"  Sharpe stability   : {sh:.2f} +/- {shsd:.2f}  ← highly variable across windows")

    if oos > 1.0:
        print(f"  OOS Sharpe         : {oos:.2f}  ← strong")
    elif oos > 0.5:
        print(f"  OOS Sharpe         : {oos:.2f}  ← acceptable")
    else:
        print(f"  OOS Sharpe         : {oos:.2f}  ← weak, reconsider strategy")

    print()


# ── TEST ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    def my_strategy(df):
        fast = df["close"].rolling(20).mean()
        slow = df["close"].rolling(50).mean()
        signal = (fast > slow).astype(int)
        return signal

    result = walk_forward(my_strategy, df=df, n_windows=5, train_pct=0.70)
