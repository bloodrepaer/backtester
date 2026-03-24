"""
app.py — Backtester web interface
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import traceback
import io
import os
import sys
import types

# Add backtester directory to path (if running from a different folder)
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import generate_sample_data
from ingestion import smart_load, print_load_report
from engine import run
from metrics import sharpe_ratio, calmar_ratio, sortino_ratio, annualised_return, max_drawdown
from returns import equity_curve
from walk_forward import walk_forward
from optimiser import optimised_walk_forward


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "Backtester",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Courier New', monospace; font-size: 13px; }
    .metric-box { background: #1a1a1a; border-radius: 8px; padding: 12px 16px; margin: 4px 0; }
    .metric-label { color: #888; font-size: 12px; margin-bottom: 2px; }
    .metric-value { color: #fff; font-size: 20px; font-weight: 600; }
    .metric-delta-pos { color: #4dff91; font-size: 12px; }
    .metric-delta-neg { color: #ff4d4d; font-size: 12px; }
    div[data-testid="stSidebar"] { background: #111; }
    .verdict-good { color: #4dff91; font-weight: 600; }
    .verdict-warn { color: #ffbf40; font-weight: 600; }
    .verdict-bad  { color: #ff4d4d; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── PRESET STRATEGIES ─────────────────────────────────────────────────────────

PRESETS = {
    "SMA crossover": {
        "code": '''\
def strategy(df):
    fast = df["close"].rolling(50).mean()
    slow = df["close"].rolling(200).mean()
    return (fast > slow).astype(int)

def strategy_factory(params):
    f, s = params["fast"], params["slow"]
    def strat(df):
        return (df["close"].rolling(f).mean() >
                df["close"].rolling(s).mean()).astype(int)
    return strat

PARAM_GRID = {
    "fast": [20, 50, 100],
    "slow": [100, 150, 200],
}
''',
        "description": "Go long when fast MA > slow MA. Classic trend-following."
    },
    "RSI mean reversion": {
        "code": '''\
import numpy as np

def strategy(df):
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    sig   = (rsi < 30).astype(int) - (rsi > 70).astype(int)
    return sig.fillna(0)

def strategy_factory(params):
    period = params["period"]
    lo, hi = params["oversold"], params["overbought"]
    def strat(df):
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        sig   = (rsi < lo).astype(int) - (rsi > hi).astype(int)
        return sig.fillna(0)
    return strat

PARAM_GRID = {
    "period"    : [7, 14, 21],
    "oversold"  : [25, 30],
    "overbought": [70, 75],
}
''',
        "description": "Long when oversold (RSI < 30), short when overbought (RSI > 70)."
    },
    "Bollinger band breakout": {
        "code": '''\
import numpy as np

def strategy(df):
    close  = df["close"]
    mid    = close.rolling(20).mean()
    std    = close.rolling(20).std()
    upper  = mid + 2 * std
    lower  = mid - 2 * std
    sig    = (close > upper).astype(int) - (close < lower).astype(int)
    return sig.fillna(0)

def strategy_factory(params):
    window = params["window"]
    nstd   = params["nstd"]
    def strat(df):
        close = df["close"]
        mid   = close.rolling(window).mean()
        std   = close.rolling(window).std()
        sig   = (close > mid + nstd*std).astype(int) - (close < mid - nstd*std).astype(int)
        return sig.fillna(0)
    return strat

PARAM_GRID = {
    "window": [10, 20, 30],
    "nstd"  : [1.5, 2.0, 2.5],
}
''',
        "description": "Long on upper band breakout, short on lower band breakdown."
    },
    "Donchian channel": {
        "code": '''\
def strategy(df):
    n     = 20
    high  = df["high"].rolling(n).max()
    low   = df["low"].rolling(n).min()
    sig   = (df["close"] >= high).astype(int) - (df["close"] <= low).astype(int)
    return sig.fillna(0)

def strategy_factory(params):
    n = params["n"]
    def strat(df):
        high = df["high"].rolling(n).max()
        low  = df["low"].rolling(n).min()
        sig  = (df["close"] >= high).astype(int) - (df["close"] <= low).astype(int)
        return sig.fillna(0)
    return strat

PARAM_GRID = {
    "n": [10, 20, 40, 60],
}
''',
        "description": "Long on N-day high breakout, short on N-day low breakdown."
    },
    "Momentum": {
        "code": '''\
def strategy(df):
    ret = df["close"].pct_change(20)
    sig = (ret > 0.02).astype(int) - (ret < -0.02).astype(int)
    return sig.fillna(0)

def strategy_factory(params):
    lookback  = params["lookback"]
    threshold = params["threshold"]
    def strat(df):
        ret = df["close"].pct_change(lookback)
        sig = (ret > threshold).astype(int) - (ret < -threshold).astype(int)
        return sig.fillna(0)
    return strat

PARAM_GRID = {
    "lookback" : [10, 20, 40],
    "threshold": [0.01, 0.02, 0.05],
}
''',
        "description": "Long if price up >2% over N days, short if down >2%."
    },
}


# ── STRATEGY EXECUTOR ─────────────────────────────────────────────────────────

def execute_strategy_code(code: str):
    """
    Safely executes user strategy code in an isolated namespace.
    Returns (strategy_fn, strategy_factory, param_grid, error_str).

    SECURITY NOTE (for local use):
    exec() runs arbitrary Python — fine for local/trusted use.
    For a hosted product, replace this with a sandboxed subprocess
    or a restricted exec environment (RestrictedPython / gVisor).
    """
    namespace = {}

    # Allow common imports inside the strategy
    import numpy as np_mod
    import pandas as pd_mod
    namespace["np"]      = np_mod
    namespace["numpy"]   = np_mod
    namespace["pd"]      = pd_mod
    namespace["pandas"]  = pd_mod

    try:
        exec(compile(code, "<strategy>", "exec"), namespace)
    except Exception as e:
        return None, None, None, f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"

    strategy_fn      = namespace.get("strategy")
    strategy_factory = namespace.get("strategy_factory")
    param_grid       = namespace.get("PARAM_GRID")

    if strategy_fn is None:
        return None, None, None, "No function named `strategy` found in your code."

    return strategy_fn, strategy_factory, param_grid, None


# ── CHART ─────────────────────────────────────────────────────────────────────

def make_chart(net_ret, market_ret, dd_series, df, strategy_name):
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

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf


# ── METRICS DISPLAY ───────────────────────────────────────────────────────────

def show_metrics(net_ret, market_ret, strategy_name):
    m_cagr    = annualised_return(net_ret)
    m_sharpe  = sharpe_ratio(net_ret)
    m_sortino = sortino_ratio(net_ret)
    m_calmar  = calmar_ratio(net_ret)
    m_vol     = net_ret.std() * np.sqrt(252)
    m_dd, _   = max_drawdown(net_ret)
    m_win     = (net_ret > 0).mean()

    bm_cagr   = annualised_return(market_ret)
    bm_sharpe = sharpe_ratio(market_ret)
    bm_dd, _  = max_drawdown(market_ret)

    st.markdown(f"#### {strategy_name} vs Buy & Hold")

    cols = st.columns(4)
    def metric(col, label, val, ref=None, fmt="%"):
        delta_str = ""
        if ref is not None:
            diff = val - ref
            sign = "+" if diff >= 0 else ""
            col.metric(label,
                       f"{val*100:.2f}%" if fmt=="%" else f"{val:.2f}",
                       f"{sign}{diff*100:.2f}%" if fmt=="%" else f"{sign}{diff:.2f}")
        else:
            col.metric(label,
                       f"{val*100:.2f}%" if fmt=="%" else f"{val:.2f}")

    metric(cols[0], "CAGR",        m_cagr,   bm_cagr)
    metric(cols[1], "Sharpe",      m_sharpe, bm_sharpe, fmt="x")
    metric(cols[2], "Max DD",      m_dd,     bm_dd)
    metric(cols[3], "Annual vol",  m_vol,    None)

    cols2 = st.columns(4)
    metric(cols2[0], "Sortino",    m_sortino, None, fmt="x")
    metric(cols2[1], "Calmar",     m_calmar,  None, fmt="x")
    metric(cols2[2], "Win rate",   m_win,     None)
    cols2[3].metric("Days tested", len(net_ret))


# ── WALK-FORWARD RESULTS TABLE ────────────────────────────────────────────────

def show_wf_table(window_results):
    rows = []
    for r in window_results:
        rows.append({
            "Window" : r["window"],
            "Test start" : str(r["test_start"]),
            "Test end"   : str(r["test_end"]),
            "Days"   : r["test_days"],
            "CAGR"   : f"{r['cagr']*100:.1f}%",
            "Sharpe" : f"{r['sharpe']:.2f}",
            "Max DD" : f"{r['max_dd']*100:.1f}%",
            "Trades" : r["n_trades"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── MAIN APP ──────────────────────────────────────────────────────────────────

def main():
    st.title("📈 Backtester")
    st.caption("Write your strategy. The engine handles everything else.")

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        # Data
        st.subheader("Data")
        data_source = st.radio("Source", ["Simulated (NIFTY-like)", "Upload CSV"], index=0)
        df = None
        if data_source == "Upload CSV":
            uploaded = st.file_uploader(
                "Upload OHLCV CSV",
                type=["csv"],
                help="Supports: Zerodha, NSE Bhavcopy, Upstox, Yahoo Finance, TradingView, AngelOne, or any generic OHLCV CSV. Intraday data is auto-resampled to daily."
            )
            if uploaded:
                try:
                    df, info = smart_load(uploaded, filename=uploaded.name)
                    fmt_labels = {
                        "nse_bhavcopy"  : "NSE Bhavcopy",
                        "zerodha"       : "Zerodha Kite",
                        "upstox"        : "Upstox",
                        "unix_timestamp": "AngelOne / Shoonya",
                        "yahoo"         : "Yahoo Finance",
                        "tradingview"   : "TradingView",
                        "generic"       : "Generic OHLCV",
                    }
                    label = fmt_labels.get(info["format"], info["format"])
                    resample_note = f" (resampled from {info['original_rows']} intraday bars)" if info["resampled"] else ""
                    st.success(f"{label} detected — {info['daily_rows']} daily rows{resample_note}  \n{info['date_start']} → {info['date_end']}")
                    for w in info["warnings"]:
                        st.warning(w)
                except ValueError as e:
                    st.error(f"Could not load file: {e}")
                    st.info(
                        "Supported formats: Zerodha Kite, NSE Bhavcopy, Upstox, "
                        "Yahoo Finance, TradingView, AngelOne/Shoonya, or any CSV "
                        "with open/high/low/close/volume columns and a date index."
                    )
                    df = None
        if df is None:
            df = pd.DataFrame(generate_sample_data())
            if data_source == "Upload CSV":
                st.caption("Using simulated data — upload a valid CSV to use real data.")

        # Mode
        st.subheader("Mode")
        mode = st.radio(
            "Run mode",
            ["1 — Simple backtest", "2 — Walk-forward", "3 — Optimised walk-forward"],
            index=0,
        )
        mode_num = int(mode[0])

        if mode_num in [2, 3]:
            n_windows = st.slider("Windows", 3, 10, 5)
            train_pct = st.slider("Train %", 50, 85, 70) / 100
        if mode_num == 3:
            optimise_on = st.selectbox("Optimise on", ["sharpe", "calmar", "cagr"])

        # Engine params
        st.subheader("Engine")
        slippage    = st.slider("Slippage (bps)",   0, 30, 8)
        commission  = st.slider("Commission (bps)",  0, 10, 2)
        vol_sizing  = st.toggle("Vol-targeted sizing", value=True)
        if vol_sizing:
            vol_target  = st.slider("Vol target %", 5, 50, 20) / 100
            max_lev     = st.slider("Max leverage",  1.0, 5.0, 3.0, step=0.5)
        else:
            vol_target, max_lev = 0.20, 3.0

        engine_kwargs = dict(
            slippage_bps   = slippage,
            commission_bps = commission,
            vol_target     = vol_target,
            max_leverage   = max_lev,
            vol_sizing     = vol_sizing,
        )

    # ── STRATEGY INPUT ────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Strategy")

        tab_preset, tab_custom = st.tabs(["Presets", "Write your own"])

        with tab_preset:
            preset_name = st.selectbox("Choose a preset", list(PRESETS.keys()))
            st.caption(PRESETS[preset_name]["description"])
            strategy_code = PRESETS[preset_name]["code"]
            st.code(strategy_code, language="python")

        with tab_custom:
            st.caption("Write a `strategy(df)` function returning a pd.Series of {-1, 0, 1}.")
            custom_code = st.text_area(
                "Strategy code",
                height=320,
                value='''\
def strategy(df):
    # Your logic here.
    # df has: open, high, low, close, volume
    # Return pd.Series of -1 (short), 0 (flat), 1 (long)
    fast = df["close"].rolling(20).mean()
    slow = df["close"].rolling(50).mean()
    return (fast > slow).astype(int)
''',
                label_visibility="collapsed"
            )
            strategy_code = custom_code

        strategy_name = st.text_input("Strategy name", value=preset_name if tab_preset else "My Strategy")

        run_btn = st.button("▶ Run backtest", type="primary", use_container_width=True)

    # ── RESULTS ───────────────────────────────────────────────────────────────
    with col_right:
        st.subheader("Results")

        if not run_btn:
            st.info("Configure your strategy and click **Run backtest**.")
        else:
            # Parse strategy
            strategy_fn, strategy_factory, param_grid, err = execute_strategy_code(strategy_code)

            if err:
                st.error("Strategy failed to load")
                if "No function named" in err:
                    st.info("Your code must define a function called `strategy(df)`. Check the spelling.")
                else:
                    st.info("Fix the error below and click Run again.")
                with st.expander("Error details", expanded=True):
                    st.code(err, language="python")
            else:
                with st.spinner("Running..."):
                    try:
                        if mode_num == 1:
                            result  = run(strategy_fn, df=df, **engine_kwargs)
                            net_ret = result["net_ret"]
                            mkt_ret = result["market_ret"]
                            _, dd   = max_drawdown(net_ret)

                        elif mode_num == 2:
                            wf_res  = walk_forward(
                                strategy_fn, df=df,
                                n_windows=n_windows, train_pct=train_pct,
                                verbose=False, **engine_kwargs
                            )
                            net_ret = wf_res["oos_net_ret"]
                            mkt_ret = wf_res["oos_mkt_ret"]
                            dd      = wf_res["oos_dd_series"]

                        elif mode_num == 3:
                            if strategy_factory is None or param_grid is None:
                                st.error("Mode 3 needs `strategy_factory` and `PARAM_GRID` defined in your code.")
                                st.stop()
                            owf_res = optimised_walk_forward(
                                strategy_factory, param_grid, df=df,
                                n_windows=n_windows, train_pct=train_pct,
                                optimise_on=optimise_on, verbose=False,
                                **engine_kwargs
                            )
                            net_ret = owf_res["oos_net_ret"]
                            mkt_ret = owf_res["oos_mkt_ret"]
                            dd      = owf_res["oos_dd_series"]

                        # Metrics
                        show_metrics(net_ret, mkt_ret, strategy_name)

                        # Walk-forward table
                        if mode_num == 2:
                            st.markdown("#### Per-window results")
                            show_wf_table(wf_res["window_results"])
                            summary = wf_res["summary"]
                            pct = summary["pct_profitable"]
                            verdict_cls = "verdict-good" if pct >= 0.8 else "verdict-warn" if pct >= 0.6 else "verdict-bad"
                            st.markdown(
                                f'<span class="{verdict_cls}">{"✓ Consistent" if pct >= 0.8 else "⚠ Acceptable" if pct >= 0.6 else "✗ Inconsistent"}</span>'
                                f" — {pct*100:.0f}% of windows profitable, "
                                f"Sharpe {summary['avg_sharpe']:.2f} ± {summary['std_sharpe']:.2f}",
                                unsafe_allow_html=True
                            )

                        if mode_num == 3:
                            st.markdown("#### Per-window optimisation")
                            rows = []
                            for r in owf_res["window_results"]:
                                rows.append({
                                    "Window"      : r["window"],
                                    "Best params" : str(r["best_params"]),
                                    "IS score"    : f"{r['is_score']:.2f}",
                                    "OOS CAGR"    : f"{r['oos_cagr']*100:.1f}%",
                                    "OOS Sharpe"  : f"{r['oos_sharpe']:.2f}",
                                    "OOS Max DD"  : f"{r['oos_max_dd']*100:.1f}%",
                                })
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                            s = owf_res["summary"]
                            decay = s["is_to_oos_decay"]
                            dcls  = "verdict-good" if decay < 0.3 else "verdict-warn" if decay < 0.8 else "verdict-bad"
                            st.markdown(
                                f'<span class="{dcls}">IS→OOS decay: {s["avg_is_score"]:.2f} → {s["avg_oos_sharpe"]:.2f} ({decay:+.2f})</span>',
                                unsafe_allow_html=True
                            )

                        # Chart
                        st.markdown("#### Tearsheet")
                        chart_buf = make_chart(net_ret, mkt_ret, dd, df, strategy_name)
                        st.image(chart_buf, use_container_width=True)

                        # Download button
                        st.download_button(
                            "⬇ Download chart",
                            data=chart_buf,
                            file_name=f"{strategy_name.replace(' ','_')}_tearsheet.png",
                            mime="image/png",
                        )

                    except Exception as e:
                        err_type = type(e).__name__
                        err_msg  = str(e)

                        # Give actionable guidance for common errors
                        if "shift" in err_msg.lower():
                            hint = "Hint: do not call `.shift()` in your strategy — the engine does this automatically."
                        elif "KeyError" in err_type:
                            hint = f"Hint: column {err_msg} not found. Available columns: open, high, low, close, volume."
                        elif "window must be" in err_msg:
                            hint = "Hint: rolling window must be a plain Python int, not a float or numpy type."
                        elif "index" in err_msg.lower():
                            hint = "Hint: make sure your strategy returns a pd.Series with index = df.index."
                        elif "memory" in err_msg.lower():
                            hint = "Hint: dataset may be too large. Try a shorter date range."
                        else:
                            hint = "Check your strategy code and the traceback below."

                        st.error(f"**{err_type}**: {err_msg}")
                        st.info(hint)
                        with st.expander("Full traceback"):
                            st.code(traceback.format_exc(), language="python")


if __name__ == "__main__":
    main()
