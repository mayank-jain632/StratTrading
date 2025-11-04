from __future__ import annotations
import argparse
import json
import os
from typing import List

import pandas as pd

from ..metrics.performance import compute_performance, save_summary, plot_results


def repair_run_dir(run_dir: str) -> List[str]:
    """Ensure a run directory has equity_curve.csv, trades.csv, summary.json, plot image, and index.html.
    Returns list of actions taken.
    """
    actions: List[str] = []
    eq_path = os.path.join(run_dir, "equity_curve.csv")
    if not os.path.exists(eq_path):
        return ["skip: no equity_curve.csv"]

    # Load equity
    eq_df = pd.read_csv(eq_path)
    if "equity" not in eq_df.columns:
        # Try to handle index-based CSVs
        if eq_df.shape[1] >= 1:
            eq_df.columns = ["dt", "equity"][: eq_df.shape[1]]
        if "equity" not in eq_df.columns:
            return ["skip: equity column missing"]
    # Set datetime index if possible
    if "dt" in eq_df.columns:
        try:
            eq_df["dt"] = pd.to_datetime(eq_df["dt"])  # type: ignore
            eq_df = eq_df.set_index("dt")
        except Exception:
            pass

    # Trades
    trades_path = os.path.join(run_dir, "trades.csv")
    trades = []
    if os.path.exists(trades_path):
        try:
            trades = pd.read_csv(trades_path).to_dict(orient="records")
        except Exception:
            trades = []
    else:
        # create empty with headers
        pd.DataFrame(columns=[
            "symbol", "entry_dt", "exit_dt", "entry_price", "exit_price", "qty", "pnl", "bars_held"
        ]).to_csv(trades_path, index=False)
        actions.append("created trades.csv (empty)")

    # Summary
    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(summary_path):
        summ = compute_performance(eq_df, trades=trades)
        save_summary(summ, summary_path)
        actions.append("created summary.json")

    # Plot
    plot_path = os.path.join(run_dir, "backtest_plots.png")
    if not os.path.exists(plot_path):
        plot_results(eq_df, run_dir, title=f"Backtest: {os.path.basename(run_dir)}")
        actions.append("created backtest_plots.png")

    # Dashboard
    index_path = os.path.join(run_dir, "index.html")
    if not os.path.exists(index_path):
        links = [
            '<a href="summary.json">summary.json</a>',
            '<a style="margin-left:12px" href="equity_curve.csv">equity_curve.csv</a>',
            '<a style="margin-left:12px" href="trades.csv">trades.csv</a>',
        ]
        try:
            with open(summary_path, "r") as f:
                sd = json.load(f)
        except Exception:
            sd = {}
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
        rows = [
            ("Total Return", fmt_pct(sd.get("total_return"))),
            ("CAGR", fmt_pct(sd.get("cagr"))),
            ("Sharpe", fmt_float(sd.get("sharpe"))),
            ("Sortino", fmt_float(sd.get("sortino"))),
            ("Calmar", fmt_float(sd.get("calmar"))),
            ("Volatility (ann)", fmt_float(sd.get("volatility"))),
            ("Max Drawdown", fmt_pct(sd.get("max_drawdown"))),
            ("Max DD Duration (bars)", str(sd.get("max_drawdown_duration"))),
            ("Num Trades", str(sd.get("num_trades"))),
            ("Win Rate", fmt_pct(sd.get("win_rate"))),
            ("Profit Factor", fmt_float(sd.get("profit_factor"))),
        ]
        table_rows = "\n".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows])
        html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Backtest Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 20px; }}
        .sub {{ color: #666; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    </style>
    </head>
<body>
    <h1>Backtest Dashboard</h1>
    <div class=\"sub\">Run dir: {run_dir}</div>
    <div style=\"margin: 12px 0;\"> {' '.join(links)} </div>
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
        with open(index_path, "w") as f:
            f.write(html)
        actions.append("created index.html")

    return actions


def main(argv=None):
    parser = argparse.ArgumentParser(description="Backfill/repair run artifacts in a runs folder")
    parser.add_argument("--root", default="runs", help="Path to runs folder containing timestamped subfolders")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print("Not a directory:", root)
        return
    print("Repairing runs under:", root)
    for name in sorted(os.listdir(root)):
        rd = os.path.join(root, name)
        if not os.path.isdir(rd):
            continue
        actions = repair_run_dir(rd)
        if actions and not (len(actions) == 1 and actions[0].startswith("skip")):
            print(f" - {name}: " + ", ".join(actions))


if __name__ == "__main__":
    main()
