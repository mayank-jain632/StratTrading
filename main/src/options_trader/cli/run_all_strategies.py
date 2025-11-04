from __future__ import annotations
import argparse
import copy
import json
import os
import time
from typing import Any, Dict, List
from scipy.optimize import minimize

import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ..strategies import STRATEGY_CATALOG
from ..metrics.performance import compute_performance
from .run_backtest import build_from_config


def _optimize_weights(returns_df: pd.DataFrame, l2: float = 0.0) -> List[float]:
    """Compute long-only weights that maximize annualized Sharpe of portfolio returns.

    Parameters
    - returns_df: dataframe of period returns (columns=strategies)
    - l2: L2 regularization strength (adds l2 * ||w||^2 penalty to objective)
    """
    # returns_df: columns = strategies, index = dates, values = period returns (e.g., daily)
    # daily mean and cov
    rets = returns_df.dropna()
    if rets.shape[1] == 0:
        return []
    mu = rets.mean()
    cov = rets.cov()
    n = len(mu)
    init = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0) for _ in range(n)]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    def neg_sharpe(w):
        # portfolio daily return and vol
        port_mean = float(np.dot(mu.values, w))
        port_vol = float(np.sqrt(w @ cov.values @ w))
        if port_vol <= 0:
            return 1e6
        # annualize (assume 252 trading days)
        ann_sharpe = (port_mean / port_vol) * np.sqrt(252)
        # add L2 penalty to discourage extreme weights (minimize negative Sharpe + penalty)
        penalty = float(l2) * float(np.dot(w, w)) if l2 and l2 > 0.0 else 0.0
        return -ann_sharpe + penalty

    res = minimize(neg_sharpe, init, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 200})
    if not res.success:
        return init.tolist()
    w = res.x
    # numerical cleanup
    w[w < 1e-8] = 0.0
    w = w / (w.sum() + 1e-12)
    return w.tolist()


def bootstrap_optimize_weights(returns_df: pd.DataFrame, rounds: int = 100, l2: float = 0.0, random_state: int | None = None) -> List[float]:
    """Compute robust weights by bootstrapping the in-sample returns.

    Procedure:
    - Resample rows (with replacement) from returns_df `rounds` times
    - Call _optimize_weights on each resample (with same l2)
    - Return the element-wise median weights across rounds (stabilizes against sampling noise)
    """
    rets = returns_df.dropna()
    if rets.shape[1] == 0:
        return []
    rng = np.random.RandomState(random_state) if random_state is not None else np.random
    ws = []
    n = rets.shape[0]
    for i in range(max(1, int(rounds))):
        idx = rng.randint(0, n, size=n)
        sample = rets.iloc[idx].reset_index(drop=True)
        w = _optimize_weights(sample, l2=l2)
        ws.append(w)
    W = np.array(ws, dtype=float)
    # take median across bootstrap runs for robustness
    med = np.median(W, axis=0)
    # cleanup and renormalize
    med[med < 1e-8] = 0.0
    med = med / (med.sum() + 1e-12)
    return med.tolist()


def run_all(base_config_path: str, strategies: List[str] | None = None, start: str | None = None, end: str | None = None, optimize: bool = False) -> str:
    with open(base_config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    if not strategies or strategies == ["ALL"]:
        strategies = list(STRATEGY_CATALOG.keys())

    # Optional date overrides
    if start or end:
        base_cfg.setdefault("data", {})
        if start:
            base_cfg["data"]["start"] = start
        if end:
            base_cfg["data"]["end"] = end

    # Output folder for this batch (store all subruns here)
    ts = time.strftime("%Y%m%d-%H%M%S")
    base_runs = base_cfg.get("output_dir", "runs")
    out_dir = os.path.join(base_runs, f"multi-strategy-{ts}")
    os.makedirs(out_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    subruns: List[str] = []
    equity_curves: List[pd.DataFrame] = []  # collect for aggregate plot/metrics
    trade_counts: List[int] = []
    strategy_names: List[str] = []

    for strat_type in strategies:
        if strat_type not in STRATEGY_CATALOG:
            print(f"Skipping unknown strategy: {strat_type}")
            continue
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("strategy", {})
        cfg["strategy"]["type"] = strat_type
        # Let strategy default params apply; preserve symbols from data provider
        cfg["strategy"]["symbols"] = cfg.get("data", {}).get("symbols", [])
        # Ensure this strategy's outputs go inside the batch folder
        cfg["output_dir"] = out_dir
        # Run
        bt = build_from_config(cfg)
        # annotate meta
        try:
            bt.config.meta = {
                "source_config_path": os.path.abspath(base_config_path),
                "config": cfg,
                "batch_folder": out_dir,
            }
        except Exception:
            pass
        result = bt.run()
        eq = result["equity_curve"]
        trades = result.get("trades", [])
        # determine number of trades for gating
        try:
            if hasattr(trades, "shape"):
                tcount = int(trades.shape[0])
            else:
                tcount = int(len(trades))
        except Exception:
            tcount = 0
        trade_counts.append(tcount)
        summ = compute_performance(eq, trades)
        run_dir = result["run_dir"]
        subruns.append(run_dir)
        equity_curves.append(eq.copy())
        strategy_names.append(strat_type)
        row = {"strategy": strat_type, "run_dir": run_dir, **summ.to_dict()}
        results.append(row)

    # Save aggregate CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "results.csv")
    df.to_csv(csv_path, index=False)

    # Compute equal-weight portfolio aggregate metrics
    aggregate_metrics = {}
    composite_eq = None
    if equity_curves:
        # Align all equity curves to common index (intersection)
        aligned = []
        common_idx = None
        for eq in equity_curves:
            eq_copy = eq.copy()
            if not isinstance(eq_copy.index, pd.DatetimeIndex):
                eq_copy.index = pd.to_datetime(eq_copy.index)
            if common_idx is None:
                common_idx = eq_copy.index
            else:
                common_idx = common_idx.intersection(eq_copy.index)
        for eq in equity_curves:
            eq_copy = eq.copy()
            if not isinstance(eq_copy.index, pd.DatetimeIndex):
                eq_copy.index = pd.to_datetime(eq_copy.index)
            aligned.append(eq_copy.reindex(common_idx))
        # Equal-weight composite: average returns
        rets = [eq_al["equity"].pct_change().fillna(0.0) for eq_al in aligned]
        composite_rets = pd.concat(rets, axis=1).mean(axis=1)
        composite_eq_series = (1 + composite_rets).cumprod()
        # Scale to starting cash
        start_cash = base_cfg.get("starting_cash", 100000.0)
        composite_eq_series = composite_eq_series * start_cash
        composite_eq = pd.DataFrame({"equity": composite_eq_series}, index=common_idx)
        # Compute metrics for composite
        comp_summ = compute_performance(composite_eq, trades=[])
        aggregate_metrics = comp_summ.to_dict()
        # Save composite equity
        composite_eq.to_csv(os.path.join(out_dir, "composite_equity_curve.csv"))

    # Optionally compute optimized weights across strategies using returns
    optimized_weights = None
    optimized_metrics = None
    optimized_eq = None
    if optimize and equity_curves:
        # Build aligned returns dataframe (columns = strategy)
        aligned = []
        common_idx = None
        for eq in equity_curves:
            eq_copy = eq.copy()
            if not isinstance(eq_copy.index, pd.DatetimeIndex):
                eq_copy.index = pd.to_datetime(eq_copy.index)
            if common_idx is None:
                common_idx = eq_copy.index
            else:
                common_idx = common_idx.intersection(eq_copy.index)
        rets_df = pd.DataFrame(index=common_idx)
        for eq, name in zip(equity_curves, strategy_names):
            eqc = eq.copy()
            if not isinstance(eqc.index, pd.DatetimeIndex):
                eqc.index = pd.to_datetime(eqc.index)
            eqc = eqc.reindex(common_idx)
            rets_df[name] = eqc['equity'].pct_change().fillna(0.0)

        weights = _optimize_weights(rets_df, l2=float(base_cfg.get('optimizer_l2', 0.0)))
        # Optional volatility-scaling: scale weights by inverse of in-sample vol
        if bool(base_cfg.get('volatility_scaling', False)):
            vols = rets_df.std().replace(0.0, np.nan).fillna(np.inf)
            inv_vol = 1.0 / vols.values
            w = np.array(weights, dtype=float)
            # scale and renormalize
            w = w * inv_vol
            if w.sum() <= 1e-12:
                w = np.repeat(1.0 / len(w), len(w))
            else:
                w = w / (w.sum() + 1e-12)
            weights = w.tolist()
            print(f"[run_all_strategies] Applied volatility scaling to optimizer weights")
        # Apply minimum-trades gating: don't allow strategies with too few in-sample trades
        min_trades = int(base_cfg.get("min_trades_for_weight", 5))
        if min_trades > 0:
            tc = np.array(trade_counts)
            # tc aligns to strategy_names order which aligns to rets_df columns
            low_mask = tc < min_trades
            if low_mask.any():
                w = np.array(weights, dtype=float)
                w[low_mask] = 0.0
                if w.sum() <= 1e-12:
                    # If all weights zeroed (no strategy meets min trades), fall back to equal weights among all
                    w = np.repeat(1.0 / len(w), len(w))
                else:
                    w = w / (w.sum() + 1e-12)
                weights = w.tolist()
                print(f"[run_all_strategies] Applied min_trades_for_weight={min_trades}; zeroed strategies: {list(np.array(strategy_names)[low_mask])}")
        # Regularize / cap optimizer weights to avoid single-strategy dominance which
        # often indicates overfitting to the training window. Allow users to override
        # caps via base config keys: `max_strategy_weight` and `min_strategy_weight`.
        max_cap = float(base_cfg.get("max_strategy_weight", 0.6))
        min_floor = float(base_cfg.get("min_strategy_weight", 0.0))
        weights = np.array(weights, dtype=float)
        if len(weights) > 1 and weights.max() > max_cap:
            # Cap the weights but redistribute the leftover mass to uncapped strategies.
            orig = weights.copy()
            capped_mask = orig > max_cap
            weights[capped_mask] = max_cap
            remaining = 1.0 - weights.sum()
            uncapped_idx = np.where(~capped_mask)[0]
            if len(uncapped_idx) > 0:
                # distribute remaining proportionally to the original uncapped weights if possible,
                # otherwise distribute equally among uncapped strategies
                uncapped_orig_sum = orig[uncapped_idx].sum()
                if uncapped_orig_sum > 0:
                    weights[uncapped_idx] += remaining * (orig[uncapped_idx] / uncapped_orig_sum)
                else:
                    weights[uncapped_idx] += remaining / len(uncapped_idx)
            else:
                # all strategies were capped (rare); fall back to equal weights
                weights = np.repeat(1.0 / len(weights), len(weights))
            # enforce floor then renormalize defensively
            if min_floor > 0.0:
                weights = np.maximum(weights, min_floor)
            weights = weights / (weights.sum() + 1e-12)
            print(f"[run_all_strategies] Optimizer weights capped to max={max_cap} and redistributed: {weights}")

        optimized_weights = {name: float(w) for name, w in zip(rets_df.columns, weights)}
        # compute optimized composite returns and equity
        comp_rets = rets_df.fillna(0.0).dot(np.array(weights))
        comp_eq = (1 + comp_rets).cumprod() * base_cfg.get('starting_cash', 100000.0)
        optimized_eq = pd.DataFrame({'equity': comp_eq}, index=comp_rets.index)
        optimized_eq.to_csv(os.path.join(out_dir, 'optimized_composite_equity_curve.csv'))
        optimized_metrics = compute_performance(optimized_eq, trades=[])

    # Save batch summary.json with aggregate metrics
    batch_summary = {
        "base_config_path": os.path.abspath(base_config_path),
        "strategies": strategies,
        "subruns": subruns,
        "results_count": len(results),
        "aggregate_portfolio_metrics": aggregate_metrics,
        "optimized": {
            "enabled": bool(optimize),
            "weights": optimized_weights,
            "optimized_portfolio_metrics": optimized_metrics.to_dict() if optimized_metrics is not None else None,
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(batch_summary, f, indent=2)

    # Generate aggregate plot: overlay all strategies + composite
    if equity_curves:
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (eq, strat) in enumerate(zip(equity_curves, strategies)):
            eq_copy = eq.copy()
            if not isinstance(eq_copy.index, pd.DatetimeIndex):
                eq_copy.index = pd.to_datetime(eq_copy.index)
            eq_aligned = eq_copy.reindex(common_idx)
            ax.plot(eq_aligned.index, eq_aligned["equity"], label=strat, alpha=0.6, linewidth=1.2)
        # Composite overlay
        if composite_eq is not None:
            ax.plot(composite_eq.index, composite_eq["equity"], label="Equal-Weight Composite", color="black", linewidth=2.0, linestyle="--")
        # Optimized composite overlay (if optimization was performed)
        if optimized_eq is not None:
            # Use a distinct color and style so it stands out from individual strategies
            ax.plot(optimized_eq.index, optimized_eq["equity"], label="Optimized Composite", color="#d62728", linewidth=2.0, linestyle="-")
        ax.set_title("Multi-Strategy Batch: All Strategies + Composite", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Equity ($)", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        plot_path = os.path.join(out_dir, "aggregate_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Aggregate plot saved to: {plot_path}")

    # Save an index.html for quick browsing
    def fmt_pct(x):
        try:
            return f"{float(x)*100:.2f}%"
        except Exception:
            return "-"
    def fmt_float(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return "-"
    rows = []
    for r in results:
        rows.append(
            (
                r.get("strategy"),
                os.path.relpath(r.get("run_dir"), out_dir),
                fmt_pct(r.get("total_return")),
                fmt_pct(r.get("cagr")),
                fmt_float(r.get("sharpe")),
                fmt_pct(r.get("max_drawdown")),
            )
        )
    table_rows = "\n".join([
        f"<tr><td>{s}</td><td><a href='{p}'>open</a></td><td>{tr}</td><td>{cagr}</td><td>{sh}</td><td>{mdd}</td></tr>"
        for s, p, tr, cagr, sh, mdd in rows
    ])

    # Aggregate metrics table
    agg_rows_html = ""
    if aggregate_metrics:
        agg_fields = [
            ("Total Return", fmt_pct(aggregate_metrics.get("total_return"))),
            ("CAGR", fmt_pct(aggregate_metrics.get("cagr"))),
            ("Sharpe", fmt_float(aggregate_metrics.get("sharpe"))),
            ("Sortino", fmt_float(aggregate_metrics.get("sortino"))),
            ("Calmar", fmt_float(aggregate_metrics.get("calmar"))),
            ("Volatility (ann)", fmt_float(aggregate_metrics.get("volatility"))),
            ("Max Drawdown", fmt_pct(aggregate_metrics.get("max_drawdown"))),
            ("Max DD Duration", str(aggregate_metrics.get("max_drawdown_duration"))),
        ]
        agg_rows_html = "\n".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in agg_fields])

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Multi-Strategy Batch</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 20px; }}
        .sub {{ color: #666; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin-top: 12px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 16px; }}
    </style>
    </head>
<body>
    <h1>Multi-Strategy Batch</h1>
    <div class=\"sub\">Batch dir: {out_dir}</div>
    <div class=\"sub\">Config: {os.path.basename(base_config_path)}</div>
    <div style=\"margin: 12px 0;\">
        <a href=\"results.csv\">results.csv</a>
        <a style=\"margin-left:12px\" href=\"summary.json\">summary.json</a>
        <a style=\"margin-left:12px\" href=\"composite_equity_curve.csv\">composite_equity_curve.csv</a>
        <a style=\"margin-left:12px\" href=\"optimized_composite_equity_curve.csv\">optimized_composite_equity_curve.csv</a>
    </div>
    <div class=\"grid\">
        <div>
            <h3>Equal-Weight Composite Portfolio</h3>
            <table><tbody>{agg_rows_html}</tbody></table>
        </div>
        <div>
            <h3>Individual Strategies</h3>
            <table>
                <thead>
                    <tr><th>Strategy</th><th>Run</th><th>Total Return</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
    </div>
    <h3>Aggregate Equity Plot</h3>
    <img src=\"aggregate_plot.png\" alt=\"Aggregate Plot\" />
</body>
</html>
"""
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html)

    print("Batch results saved to:", csv_path)
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run all or selected strategies on a base config and aggregate results")
    parser.add_argument("--config", required=True, help="Base YAML config (e.g., configs/large_caps.yaml)")
    parser.add_argument("--strategies", nargs="*", default=["ALL"], help="Strategy IDs from STRATEGY_CATALOG; default ALL")
    parser.add_argument("--start", help="Override data.start in base config (YYYY-MM-DD)")
    parser.add_argument("--end", help="Override data.end in base config (YYYY-MM-DD)")
    parser.add_argument("--optimize", action="store_true", help="Enable long-only Sharpe optimizer for strategy weights")
    args = parser.parse_args(argv)

    run_all(args.config, strategies=args.strategies, start=args.start, end=args.end, optimize=args.optimize)


if __name__ == "__main__":
    main()
