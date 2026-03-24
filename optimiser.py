import pandas as pd
import numpy as np
import itertools
from data_loader import load_data
from engine import run
from metrics import sharpe_ratio, calmar_ratio, annualised_return, max_drawdown
from walk_forward import generate_windows, walk_forward


# ── GRID SEARCH ON A SINGLE TRAIN WINDOW ─────────────────────────────────────

def grid_search(strategy_factory, param_grid, df_train,
                optimise_on="sharpe", engine_kwargs=None):
    """
    Exhaustive grid search over all parameter combinations.
    ONLY runs on df_train — never touches test data.

    WHY strategy_factory and not strategy_fn directly:
    The strategy function must be parameterised. We need a way to create
    a new strategy function for each parameter combination.

    strategy_factory is a function that TAKES params and RETURNS a strategy_fn.

    Example:
        def make_sma_strategy(fast, slow):
            def strategy(df):
                return (df["close"].rolling(fast).mean() >
                        df["close"].rolling(slow).mean()).astype(int)
            return strategy

        param_grid = {"fast": [10, 20, 50], "slow": [100, 200]}

    Parameters
    ----------
    strategy_factory : callable(params_dict) → strategy_fn
    param_grid       : dict of {param_name: [list of values]}
    df_train         : training DataFrame only
    optimise_on      : metric to maximise — "sharpe", "calmar", or "cagr"
    engine_kwargs    : dict of engine settings (slippage, vol_sizing, etc.)

    Returns
    -------
    best_params  : dict of the winning parameter combination
    best_score   : score of the winning combination
    all_results  : full results table sorted by score
    """
    if engine_kwargs is None:
        engine_kwargs = {}

    # Generate all combinations
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    results = []

    for combo in combos:
        params = dict(zip(keys, combo))

        try:
            strategy_fn = strategy_factory(params)
            result      = run(strategy_fn, df=df_train, **engine_kwargs)
            net_ret     = result["net_ret"]

            if len(net_ret) < 20:
                continue

            score = {
                "sharpe" : sharpe_ratio(net_ret),
                "calmar" : calmar_ratio(net_ret),
                "cagr"   : annualised_return(net_ret),
            }.get(optimise_on, sharpe_ratio(net_ret))

            max_dd, _ = max_drawdown(net_ret)

            results.append({
                **params,
                "sharpe"  : sharpe_ratio(net_ret),
                "calmar"  : calmar_ratio(net_ret),
                "cagr"    : annualised_return(net_ret),
                "max_dd"  : max_dd,
                "score"   : score,
            })

        except Exception:
            continue

    if not results:
        raise RuntimeError("All parameter combinations failed. Check your strategy factory.")

    results_df   = pd.DataFrame(results).sort_values("score", ascending=False)
    best_row     = results_df.iloc[0]
    best_params  = {k: type(param_grid[k][0])(best_row[k]) for k in keys}
    best_score   = best_row["score"]

    return best_params, best_score, results_df


# ── OPTIMISED WALK-FORWARD ────────────────────────────────────────────────────

def optimised_walk_forward(
    strategy_factory,
    param_grid,
    df             = None,
    filepath       = None,
    train_pct      = 0.70,
    n_windows      = 5,
    optimise_on    = "sharpe",
    slippage_bps   = 8,
    commission_bps = 2,
    vol_target     = 0.20,
    max_leverage   = 3.0,
    vol_sizing     = True,
    verbose        = True,
):
    """
    Walk-forward validation WITH parameter optimisation.

    For each window:
      1. Grid search on TRAIN portion → find best params
      2. Run strategy with those params on TEST portion
      3. Record test performance

    This is the correct way to evaluate a parameterised strategy.
    Optimising on the full dataset and then testing on the same data
    is a form of lookahead — here the engine prevents it structurally.

    The test set is NEVER touched during optimisation.
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

    n_combos = len(list(itertools.product(*param_grid.values())))

    if verbose:
        print(f"\nOptimised walk-forward — {n_windows} windows, "
              f"{n_combos} parameter combinations, optimising on {optimise_on}")
        print("=" * 78)
        print(f"  {'Win':<5} {'Best params':<30} {'IS '+optimise_on:>10} "
              f"{'OOS CAGR':>9} {'OOS Sharpe':>11} {'OOS Max DD':>11}")
        print("-" * 78)

    window_results = []
    oos_net_rets   = []
    oos_mkt_rets   = []

    for w in windows:
        df_full  = pd.concat([w["train"], w["test"]])
        df_train = w["train"]
        df_test  = w["test"]

        # Step 1: optimise on train only
        best_params, best_is_score, _ = grid_search(
            strategy_factory, param_grid, df_train,
            optimise_on=optimise_on, engine_kwargs=engine_kwargs
        )

        # Step 2: build best strategy and run on full window (for warmup)
        strategy_fn = strategy_factory(best_params)
        result_full = run(strategy_fn, df=df_full, **engine_kwargs)

        # Step 3: slice to test only
        test_start = df_test.index[0]
        test_end   = df_test.index[-1]

        net_ret_test = result_full["net_ret"].loc[test_start:test_end]
        mkt_ret_test = result_full["market_ret"].loc[test_start:test_end]

        if len(net_ret_test) < 5:
            continue

        oos_cagr   = annualised_return(net_ret_test)
        oos_sharpe = sharpe_ratio(net_ret_test)
        oos_dd, _  = max_drawdown(net_ret_test)

        params_str = ", ".join(f"{k}={v}" for k, v in best_params.items())

        if verbose:
            print(f"  {w['window']:<5} {params_str:<30} {best_is_score:>10.2f} "
                  f"{oos_cagr*100:>8.1f}% {oos_sharpe:>11.2f} {oos_dd*100:>10.1f}%")

        window_results.append({
            "window"       : w["window"],
            "best_params"  : best_params,
            "is_score"     : best_is_score,
            "oos_cagr"     : oos_cagr,
            "oos_sharpe"   : oos_sharpe,
            "oos_max_dd"   : oos_dd,
            "net_ret"      : net_ret_test,
            "market_ret"   : mkt_ret_test,
        })

        oos_net_rets.append(net_ret_test)
        oos_mkt_rets.append(mkt_ret_test)

    oos_net_ret = pd.concat(oos_net_rets).sort_index()
    oos_mkt_ret = pd.concat(oos_mkt_rets).sort_index()

    oos_max_dd, oos_dd_series = max_drawdown(oos_net_ret)
    mkt_cagr   = annualised_return(oos_mkt_ret)
    mkt_sharpe = sharpe_ratio(oos_mkt_ret)
    mkt_dd, _  = max_drawdown(oos_mkt_ret)

    summary = {
        "oos_cagr"       : annualised_return(oos_net_ret),
        "oos_sharpe"     : sharpe_ratio(oos_net_ret),
        "oos_calmar"     : calmar_ratio(oos_net_ret),
        "oos_max_dd"     : oos_max_dd,
        "pct_profitable" : sum(r["oos_cagr"] > 0 for r in window_results) / len(window_results),
        "avg_is_score"   : np.mean([r["is_score"]   for r in window_results]),
        "avg_oos_sharpe" : np.mean([r["oos_sharpe"] for r in window_results]),
        "is_to_oos_decay": np.mean([r["is_score"]   for r in window_results]) -
                           np.mean([r["oos_sharpe"] for r in window_results]),
    }

    if verbose:
        print("=" * 78)
        print(f"\n  Out-of-sample aggregate")
        print(f"  {'Metric':<32} {'Strategy':>10}  {'Market':>8}")
        print(f"  {'-'*54}")
        print(f"  {'CAGR':<32} {summary['oos_cagr']*100:>9.2f}%  {mkt_cagr*100:>7.2f}%")
        print(f"  {'Sharpe':<32} {summary['oos_sharpe']:>10.2f}  {mkt_sharpe:>8.2f}")
        print(f"  {'Max drawdown':<32} {summary['oos_max_dd']*100:>9.2f}%  {mkt_dd*100:>7.2f}%")
        print(f"  {'Profitable windows':<32} {summary['pct_profitable']*100:>9.0f}%")
        print()

        # IS-to-OOS decay: how much does performance drop from train to test?
        # High decay = overfitting. Low decay = robust.
        decay = summary["is_to_oos_decay"]
        print(f"  IS→OOS Sharpe decay: {summary['avg_is_score']:.2f} → "
              f"{summary['avg_oos_sharpe']:.2f} (decay = {decay:.2f})")
        if decay < 0.3:
            print(f"  Verdict: low decay — strategy is robust to parameter choice")
        elif decay < 0.8:
            print(f"  Verdict: moderate decay — acceptable, watch for overfitting")
        else:
            print(f"  Verdict: high decay — likely overfit, simplify the strategy")
        print()

    return {
        "window_results" : window_results,
        "oos_net_ret"    : oos_net_ret,
        "oos_mkt_ret"    : oos_mkt_ret,
        "oos_dd_series"  : oos_dd_series,
        "summary"        : summary,
    }


# ── TEST ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    # Strategy factory: takes params dict, returns a strategy function
    def make_sma_strategy(params):
        fast = params["fast"]
        slow = params["slow"]
        def strategy(df):
            return (df["close"].rolling(fast).mean() >
                    df["close"].rolling(slow).mean()).astype(int)
        return strategy

    # Parameter grid to search
    param_grid = {
        "fast": [10, 20, 50],
        "slow": [100, 150, 200],
    }

    result = optimised_walk_forward(
        strategy_factory = make_sma_strategy,
        param_grid       = param_grid,
        df               = df,
        n_windows        = 5,
        train_pct        = 0.70,
        optimise_on      = "sharpe",
    )
