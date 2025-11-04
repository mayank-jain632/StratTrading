from __future__ import annotations
import argparse
import json
import os
import pandas as pd
import yaml

from .run_backtest import build_from_config
from ..metrics.performance import compute_performance, save_summary, plot_results


def run_multi(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    portfolio_cfg = cfg["portfolio"]
    strategies = portfolio_cfg["strategies"]
    output_dir = cfg.get("output_dir", "runs")
    os.makedirs(output_dir, exist_ok=True)

    # Run each strategy and collect equity curves
    series = []
    weights = []
    subruns = []
    base_dir = os.path.dirname(os.path.abspath(config_path))
    project_root = os.path.dirname(base_dir)
    for s in strategies:
        sub_cfg_path = s["config"]
        # Resolve relative paths against the portfolio config directory
        if not os.path.isabs(sub_cfg_path):
            # Primary: relative to the portfolio config directory
            candidate = os.path.normpath(os.path.join(base_dir, sub_cfg_path))
            if os.path.exists(candidate):
                sub_cfg_path = candidate
            else:
                # Fallback: relative to the project root (one level above configs)
                candidate2 = os.path.normpath(os.path.join(project_root, sub_cfg_path))
                sub_cfg_path = candidate2
        w = float(s.get("weight", 1.0))
        with open(sub_cfg_path, "r") as f:
            sub_cfg = yaml.safe_load(f)
        bt = build_from_config(sub_cfg)
        # Attach per-strategy meta for traceability
        try:
            bt.config.meta = {
                "source_config_path": os.path.abspath(sub_cfg_path),
                "config": sub_cfg,
                "parent_portfolio_config": os.path.abspath(config_path),
                "weight": float(s.get("weight", 1.0)),
            }
        except Exception:
            pass
        res = bt.run()
        eq = res["equity_curve"]["equity"].astype(float)
        eq.index = pd.to_datetime(res["equity_curve"].index)
        series.append(eq.pct_change().fillna(0.0))
        weights.append(w)
        subruns.append(res["run_dir"])

    # Combine returns by weights
    rets = pd.concat(series, axis=1).fillna(0.0)
    weights = pd.Series(weights) / sum(weights)
    combined = (rets * weights.values).sum(axis=1)

    # Build composite equity curve
    eq = (1 + combined).cumprod()
    start_capital = float(cfg.get("starting_cash", 100000.0))
    equity_curve = pd.DataFrame({"equity": eq * start_capital}, index=combined.index)

    # Save outputs
    run_dir = os.path.join(output_dir, f"portfolio-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    equity_curve.to_csv(os.path.join(run_dir, "equity_curve.csv"))

    # Metrics and plots
    summary = compute_performance(equity_curve)
    save_summary(summary, os.path.join(run_dir, "summary.json"))
    plot_results(equity_curve, run_dir, title="Multi-Strategy Portfolio")

    # Save meta
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "portfolio_config_path": os.path.abspath(config_path),
            "subruns": subruns,
            "weights": weights.tolist(),
        }, f, indent=2)

    # Simple HTML dashboard for portfolio results
    table_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in [
            ("Total Return", f"{summary.total_return*100:.2f}%"),
            ("CAGR", f"{summary.cagr*100:.2f}%"),
            ("Sharpe", f"{summary.sharpe:.4f}"),
            ("Sortino", f"{summary.sortino:.4f}"),
            ("Calmar", f"{summary.calmar:.4f}"),
            ("Volatility (ann)", f"{summary.volatility:.4f}"),
            ("Max Drawdown", f"{summary.max_drawdown*100:.2f}%"),
            ("Max DD Duration (bars)", f"{summary.max_drawdown_dur}"),
            ("Num Trades", f"{summary.num_trades}"),
            ("Win Rate", f"{((summary.win_rate or 0)*100):.2f}%"),
            ("Profit Factor", f"{summary.profit_factor:.4f}"),
        ]
    )
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Portfolio Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 20px; }}
        .sub {{ color: #666; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    </style>
    </head>
<body>
    <h1>Multi-Strategy Portfolio</h1>
    <div class=\"sub\">Run dir: {run_dir}</div>
    <div class=\"sub\">Config: {os.path.basename(config_path)}</div>
    <div style=\"margin: 12px 0;\">
        <a href=\"summary.json\">summary.json</a>
        <a style=\"margin-left:12px\" href=\"equity_curve.csv\">equity_curve.csv</a>
        <a style=\"margin-left:12px\" href=\"meta.json\">meta.json</a>
    </div>
    <div style=\"display:grid;grid-template-columns:1fr 1fr; gap:24px\"> 
        <div>
            <h3>Metrics</h3>
            <table><tbody>{table_rows}</tbody></table>
        </div>
        <div>
            <h3>Equity Curve</h3>
            <img src=\"backtest_plots.png\" alt=\"Equity Plot\" />
        </div>
    </div>
</body>
</html>
"""
    with open(os.path.join(run_dir, "index.html"), "w") as f: f.write(html)

    print("Portfolio run directory:", run_dir)
    print("Summary:", summary.to_dict())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run multi-strategy weighted portfolio")
    parser.add_argument("--config", required=True, help="Portfolio YAML config")
    args = parser.parse_args(argv)
    run_multi(args.config)


if __name__ == "__main__":
    main()
