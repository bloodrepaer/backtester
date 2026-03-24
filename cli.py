"""
=====================================================================
  BACKTESTER — cli.py
=====================================================================

USAGE:

  python cli.py [OPTIONS]

OPTIONS:

  --data        PATH     Path to OHLCV CSV file (default: simulated data)
  --strategy    PATH     Path to a .py file containing a strategy function
                         OR a strategy factory + PARAM_GRID for optimised WF
  --mode        INT      1 = simple backtest (default)
                         2 = walk-forward validation
                         3 = optimised walk-forward
  --name        STR      Strategy name for chart title (default: "Strategy")
  --windows     INT      Number of walk-forward windows (default: 5)
  --train-pct   FLOAT    Train fraction per window (default: 0.70)
  --optimise-on STR      Metric to optimise: sharpe|calmar|cagr (default: sharpe)
  --slippage    FLOAT    Slippage in bps (default: 8)
  --commission  FLOAT    Commission in bps (default: 2)
  --vol-target  FLOAT    Annualised vol target (default: 0.20)
  --max-lev     FLOAT    Max leverage multiplier (default: 3.0)
  --no-vol-size          Disable vol-targeted position sizing
  --out         PATH     Output directory for chart (default: current dir)
  --no-plot              Skip chart generation

EXAMPLES:

  # Backtest using simulated data, default SMA strategy
  python cli.py

  # Backtest your strategy on real data
  python cli.py --data nifty.csv --strategy my_strategy.py

  # Walk-forward validation
  python cli.py --data nifty.csv --strategy my_strategy.py --mode 2

  # Optimised walk-forward with custom windows
  python cli.py --data nifty.csv --strategy my_strategy.py --mode 3 --windows 8

  # No vol sizing, custom slippage
  python cli.py --strategy my_strategy.py --slippage 5 --no-vol-size

=====================================================================

STRATEGY FILE FORMAT:

  Your strategy .py file must define a function called `strategy`.
  Optionally define `strategy_factory` and `PARAM_GRID` for mode 3.

  Minimal example (save as my_strategy.py):
  ──────────────────────────────────────────
  import pandas as pd

  def strategy(df):
      fast = df["close"].rolling(20).mean()
      slow = df["close"].rolling(50).mean()
      return (fast > slow).astype(int)
  ──────────────────────────────────────────

  For mode 3 (optimised walk-forward), also define:
  ──────────────────────────────────────────
  def strategy_factory(params):
      fast = params["fast"]
      slow = params["slow"]
      def strategy(df):
          return (df["close"].rolling(fast).mean() >
                  df["close"].rolling(slow).mean()).astype(int)
      return strategy

  PARAM_GRID = {
      "fast": [10, 20, 50],
      "slow": [100, 150, 200],
  }
  ──────────────────────────────────────────

=====================================================================
"""

import argparse
import importlib.util
import os
import sys
import traceback
from pathlib import Path

import pandas as pd
import numpy as np

from data_loader import load_data
from engine import run
from metrics import max_drawdown, sharpe_ratio, calmar_ratio, sortino_ratio, annualised_return
from returns import equity_curve
from walk_forward import walk_forward
from optimiser import optimised_walk_forward


# ── LOAD STRATEGY FROM FILE ───────────────────────────────────────────────────

def load_strategy_file(filepath: str):
    """
    Dynamically imports a user's strategy .py file.
    Extracts: strategy fn, strategy_factory fn (optional), PARAM_GRID (optional).

    WHY dynamic import:
    The user writes their strategy in a separate file — they never edit cli.py.
    importlib.util lets us load any .py file as a module at runtime.
    This is the same mechanism pytest uses to load test files.
    """
    path = Path(filepath).resolve()

    if not path.exists():
        print(f"[ERROR] Strategy file not found: {filepath}")
        sys.exit(1)

    if path.suffix != ".py":
        print(f"[ERROR] Strategy file must be a .py file. Got: {filepath}")
        sys.exit(1)

    # Load the file as a module
    spec   = importlib.util.spec_from_file_location("user_strategy", path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"[ERROR] Failed to load strategy file: {filepath}")
        print(f"        {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract strategy function
    if not hasattr(module, "strategy"):
        print(f"[ERROR] Strategy file must define a function called `strategy`.")
        print(f"        Found: {[x for x in dir(module) if not x.startswith('_')]}")
        sys.exit(1)

    strategy_fn      = module.strategy
    strategy_factory = getattr(module, "strategy_factory", None)
    param_grid       = getattr(module, "PARAM_GRID", None)

    return strategy_fn, strategy_factory, param_grid


# ── DEFAULT STRATEGY (used when no --strategy flag given) ─────────────────────

def _default_strategy(df):
    fast = df["close"].rolling(20).mean()
    slow = df["close"].rolling(50).mean()
    return (fast > slow).astype(int)

def _default_factory(params):
    f, s = params["fast"], params["slow"]
    def strat(df):
        return (df["close"].rolling(f).mean() > df["close"].rolling(s).mean()).astype(int)
    return strat

DEFAULT_PARAM_GRID = {"fast": [10, 20, 50], "slow": [100, 150, 200]}


# ── BENCHMARK COMPARISON ──────────────────────────────────────────────────────

def print_benchmark(net_ret, market_ret, strategy_name):
    rf_daily = pd.Series(0.065 / 252, index=net_ret.index)

    rows = []
    for name, ret in [(strategy_name, net_ret), ("Buy & Hold", market_ret), ("Risk-free", rf_daily)]:
        max_dd, _ = max_drawdown(ret)
        rows.append({
            "name"    : name,
            "cagr"    : annualised_return(ret),
            "vol"     : ret.std() * np.sqrt(252),
            "sharpe"  : sharpe_ratio(ret),
            "sortino" : sortino_ratio(ret),
            "calmar"  : calmar_ratio(ret),
            "max_dd"  : max_dd,
            "win_rate": (ret > 0).mean(),
        })

    col_w = max(14, max(len(r["name"]) + 2 for r in rows))
    width = 24 + len(rows) * (col_w + 2)

    print(f"\n{'='*width}")
    print(f"  BENCHMARK COMPARISON")
    print(f"{'='*width}")
    hdr = f"  {'Metric':<22}"
    for r in rows:
        hdr += f"  {r['name']:>{col_w}}"
    print(hdr)
    print(f"  {'-'*(width-2)}")

    for label, key, unit in [
        ("CAGR",         "cagr",     "%"),
        ("Annual vol",   "vol",      "%"),
        ("Sharpe",       "sharpe",   "x"),
        ("Sortino",      "sortino",  "x"),
        ("Calmar",       "calmar",   "x"),
        ("Max drawdown", "max_dd",   "%"),
        ("Win rate",     "win_rate", "%"),
    ]:
        line = f"  {label:<22}"
        for r in rows:
            v = r[key]
            if unit == "%":
                line += f"  {v*100:>{col_w}.2f}%"[:-1] + "%"
            else:
                line += f"  {'—':>{col_w}}" if (np.isinf(v) or np.isnan(v)) else f"  {v:>{col_w}.2f}"
        print(line)
    print(f"{'='*width}\n")


# ── CHART ─────────────────────────────────────────────────────────────────────

def plot_results(net_ret, market_ret, dd_series, df,
                 strategy_name, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    eq_strat = equity_curve(net_ret)
    eq_mkt   = equity_curve(market_ret)
    rf_curve = equity_curve(pd.Series(0.065/252, index=net_ret.index))

    roll_sharpe = (
        (net_ret - 0.065/252).rolling(126).mean() /
         net_ret.rolling(126).std()
    ) * np.sqrt(252)

    simple_ret  = np.exp(net_ret) - 1
    monthly     = simple_ret.resample("ME").apply(
        lambda x: np.exp(np.sum(np.log(1 + x))) - 1
    )
    monthly_tbl = monthly.groupby(
        [monthly.index.year, monthly.index.month]
    ).first().unstack(level=1)
    monthly_tbl.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"]

    BG, PANEL = "#0f0f0f", "#1a1a1a"
    BLUE="#4da6ff"; GRAY="#888888"; RED="#ff4d4d"
    GREEN="#4dff91"; AMBER="#ffbf40"; TEXT="#cccccc"; GRID="#2a2a2a"

    def style(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=9, pad=6, loc="left")
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.grid(color=GRID, lw=0.5)
        for s in ax.spines.values(): s.set_edgecolor(GRID)

    fig = plt.figure(figsize=(14, 17))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(eq_strat.index, eq_strat.values, color=BLUE,  lw=1.5, label=strategy_name)
    ax1.plot(eq_mkt.index,   eq_mkt.values,   color=GRAY,  lw=1.0, label="Buy & Hold", alpha=0.7)
    ax1.plot(rf_curve.index, rf_curve.values, color=AMBER, lw=0.8, label="Risk-free (6.5%)", ls="--", alpha=0.6)
    ax1.set_ylabel("Portfolio value (Rs.)", color=TEXT, fontsize=8)
    ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style(ax1, "Equity curve")

    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(dd_series.index, dd_series.values*100, 0, color=RED, alpha=0.5)
    ax2.plot(dd_series.index, dd_series.values*100, color=RED, lw=0.8)
    ax2.set_ylabel("Drawdown (%)", color=TEXT, fontsize=8)
    style(ax2, "Drawdown")

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(df.index, df["close"], color=GRAY, lw=0.8, alpha=0.7)
    ax3.set_ylabel("Price", color=TEXT, fontsize=8)
    style(ax3, "Price")

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(roll_sharpe.index, roll_sharpe.values, color=GREEN, lw=1.0)
    ax4.axhline(0, color=GRID, lw=0.8, ls="--")
    ax4.axhline(1, color=BLUE, lw=0.6, ls=":", alpha=0.6)
    ax4.set_ylabel("Sharpe (126d rolling)", color=TEXT, fontsize=8)
    style(ax4, "Rolling Sharpe")

    ax5  = fig.add_subplot(gs[3, :])
    vals = monthly_tbl.values * 100
    good = ~np.isnan(vals)
    vmax = np.nanpercentile(np.abs(vals[good]), 95) if good.any() else 1
    im   = ax5.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax5.set_xticks(range(len(monthly_tbl.columns)))
    ax5.set_xticklabels(monthly_tbl.columns, color=TEXT, fontsize=7)
    ax5.set_yticks(range(len(monthly_tbl.index)))
    ax5.set_yticklabels(monthly_tbl.index,   color=TEXT, fontsize=7)
    ax5.set_facecolor(PANEL)
    ax5.tick_params(colors=TEXT)
    for s in ax5.spines.values(): s.set_edgecolor(GRID)
    for i in range(len(monthly_tbl.index)):
        for j in range(len(monthly_tbl.columns)):
            v = vals[i, j]
            if not np.isnan(v):
                ax5.text(j, i, f"{v:.1f}", ha="center", va="center",
                         fontsize=6, color="black" if abs(v) < vmax*0.6 else "white")
    plt.colorbar(im, ax=ax5, fraction=0.02, pad=0.01).ax.tick_params(colors=TEXT, labelsize=7)
    style(ax5, "Monthly returns (%)")

    fig.suptitle(f"Backtester — {strategy_name}", color=TEXT, fontsize=12, y=0.99)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Chart saved → {save_path}")


# ── ARG PARSER ────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="python cli.py",
        description="Backtester — run strategies, validate, optimise.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data",        type=str,   default=None,     help="Path to OHLCV CSV")
    p.add_argument("--strategy",    type=str,   default=None,     help="Path to strategy .py file")
    p.add_argument("--mode",        type=int,   default=1,        choices=[1,2,3], help="1=backtest 2=WF 3=optWF")
    p.add_argument("--name",        type=str,   default="Strategy", help="Strategy display name")
    p.add_argument("--windows",     type=int,   default=5,        help="Walk-forward windows")
    p.add_argument("--train-pct",   type=float, default=0.70,     help="Train fraction (0-1)")
    p.add_argument("--optimise-on", type=str,   default="sharpe", choices=["sharpe","calmar","cagr"])
    p.add_argument("--slippage",    type=float, default=8,        help="Slippage in bps")
    p.add_argument("--commission",  type=float, default=2,        help="Commission in bps")
    p.add_argument("--vol-target",  type=float, default=0.20,     help="Vol target (annualised)")
    p.add_argument("--max-lev",     type=float, default=3.0,      help="Max leverage")
    p.add_argument("--no-vol-size", action="store_true",          help="Disable vol sizing")
    p.add_argument("--out",         type=str,   default=".",      help="Output directory")
    p.add_argument("--no-plot",     action="store_true",          help="Skip chart")
    return p


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + "="*54)
    print("  BACKTESTER")
    mode_labels = {1: "Simple backtest", 2: "Walk-forward", 3: "Optimised walk-forward"}
    print(f"  Mode     : {args.mode} — {mode_labels[args.mode]}")
    print(f"  Data     : {args.data or 'simulated'}")
    print(f"  Strategy : {args.strategy or 'default SMA 20/50'}")
    print(f"  Slippage : {args.slippage} bps  |  Commission: {args.commission} bps")
    print(f"  Vol size : {'off' if args.no_vol_size else f'on (target={args.vol_target*100:.0f}%, max={args.max_lev}x)'}")
    print("="*54)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_data(filepath=args.data)

    # ── Load strategy ─────────────────────────────────────────────────────────
    if args.strategy:
        strategy_fn, strategy_factory, param_grid = load_strategy_file(args.strategy)
        print(f"\n[OK] Loaded strategy from {args.strategy}")
    else:
        strategy_fn      = _default_strategy
        strategy_factory = _default_factory
        param_grid       = DEFAULT_PARAM_GRID
        print(f"\n[OK] Using default SMA 20/50 strategy")

    engine_kwargs = dict(
        slippage_bps   = args.slippage,
        commission_bps = args.commission,
        vol_target     = args.vol_target,
        max_leverage   = args.max_lev,
        vol_sizing     = not args.no_vol_size,
    )

    chart_name = args.name
    out_dir    = args.out

    # ── Run chosen mode ───────────────────────────────────────────────────────
    if args.mode == 1:
        result  = run(strategy_fn, df=df, **engine_kwargs)
        net_ret = result["net_ret"]
        mkt_ret = result["market_ret"]
        print_benchmark(net_ret, mkt_ret, chart_name)
        _, dd = max_drawdown(net_ret)
        save_path = os.path.join(out_dir, "tearsheet.png")

    elif args.mode == 2:
        wf      = walk_forward(strategy_fn, df=df,
                               n_windows=args.windows,
                               train_pct=args.train_pct,
                               **engine_kwargs)
        net_ret   = wf["oos_net_ret"]
        mkt_ret   = wf["oos_mkt_ret"]
        dd        = wf["oos_dd_series"]
        chart_name = f"{chart_name} (OOS)"
        save_path  = os.path.join(out_dir, "wf_tearsheet.png")

    elif args.mode == 3:
        if strategy_factory is None or param_grid is None:
            print("[ERROR] Mode 3 requires `strategy_factory` and `PARAM_GRID` in your strategy file.")
            print("        See cli.py --help for the required format.")
            sys.exit(1)

        owf     = optimised_walk_forward(
                      strategy_factory, param_grid, df=df,
                      n_windows=args.windows, train_pct=args.train_pct,
                      optimise_on=args.optimise_on, **engine_kwargs)
        net_ret   = owf["oos_net_ret"]
        mkt_ret   = owf["oos_mkt_ret"]
        dd        = owf["oos_dd_series"]
        chart_name = f"{chart_name} Optimised (OOS)"
        save_path  = os.path.join(out_dir, "owf_tearsheet.png")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        if args.mode == 1:
            _, dd = max_drawdown(net_ret)
        plot_results(net_ret, mkt_ret, dd, df,
                     strategy_name=chart_name,
                     save_path=save_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
