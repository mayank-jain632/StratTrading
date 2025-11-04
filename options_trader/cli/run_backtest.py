from __future__ import annotations
import argparse
import os
from typing import Any, Dict

import yaml

from ..core.backtester import Backtester, BacktestConfig
from ..core.broker import PaperBroker, CommissionModel, ExecutionCostModel
from ..core.portfolio import BasicPortfolio
from ..core.risk import FixedFractionSizer
from ..data.synthetic import SyntheticDataProvider, SyntheticConfig
from ..data.yfinance import YFinanceDataProvider, YFConfig
from ..strategies import STRATEGY_CATALOG
from ..metrics.performance import compute_performance, save_summary, plot_results
import pandas as pd
import numpy as np
try:
    import yfinance as yf
except Exception:
    yf = None


def build_from_config(cfg: Dict[str, Any]) -> Backtester:
    output_dir = cfg.get("output_dir", "runs")

    data_cfg = cfg["data"]
    if data_cfg["type"] == "synthetic":
        sc = SyntheticConfig(
            symbols=data_cfg["symbols"],
            start=data_cfg["start"],
            end=data_cfg["end"],
            freq=data_cfg.get("freq", "1D"),
            seed=data_cfg.get("seed", 42),
            start_price=float(data_cfg.get("start_price", 100.0)),
            drift=float(data_cfg.get("drift", 0.0002)),
            vol=float(data_cfg.get("vol", 0.01)),
        )
        data = SyntheticDataProvider(sc)
    elif data_cfg["type"] == "yfinance":
        yc = YFConfig(
            symbols=data_cfg["symbols"],
            start=data_cfg["start"],
            end=data_cfg["end"],
            interval=data_cfg.get("interval", "1d"),
            auto_adjust=bool(data_cfg.get("auto_adjust", True)),
            cache_dir=data_cfg.get("cache_dir"),
            use_cache=bool(data_cfg.get("use_cache", True)),
        )
        data = YFinanceDataProvider(yc)
    else:
        raise ValueError(f"Unsupported data type: {data_cfg['type']}")

    # Dynamic strategy loading from catalog
    strat_cfg = cfg["strategy"]
    strat_type = strat_cfg["type"]
    
    if strat_type not in STRATEGY_CATALOG:
        available = ", ".join(STRATEGY_CATALOG.keys())
        raise ValueError(f"Unknown strategy '{strat_type}'. Available: {available}")
    
    strategy_info = STRATEGY_CATALOG[strat_type]
    StrategyClass = strategy_info["class"]
    ConfigClass = strategy_info["config"]
    
    # Build config from YAML parameters
    # If no symbols provided for the strategy, default to data provider symbols
    strat_symbols = strat_cfg.get("symbols")
    if not strat_symbols:
        try:
            strat_symbols = list(data.symbols())
        except Exception:
            strat_symbols = []
    config_params = {"symbols": strat_symbols}
    config_params.update(strat_cfg.get("params", {}))
    strategy = StrategyClass(ConfigClass(**config_params))


    risk_cfg = cfg.get("risk", {})
    risk = FixedFractionSizer(
        fraction=float(risk_cfg.get("fraction", 0.1)),
        max_positions=int(risk_cfg.get("max_positions", 5)),
    )

    broker_cfg = cfg.get("broker", {})
    commission = CommissionModel(
        per_trade=float(broker_cfg.get("per_trade", 0.0)),
        per_share=float(broker_cfg.get("per_share", 0.0)),
    )
    execution = ExecutionCostModel(
        slippage_bps=float(broker_cfg.get("slippage_bps", 0.0)),
        spread_bps=float(broker_cfg.get("spread_bps", 0.0)),
    )

    portfolio = BasicPortfolio(starting_cash=float(cfg.get("starting_cash", 100000.0)))

    bt = Backtester(
        data=data,
        strategy=strategy,
        broker=PaperBroker(event_queue=None, commission=commission, execution=execution),  # Backtester will set event_queue
        portfolio=portfolio,
        risk=risk,
        config=BacktestConfig(output_dir=output_dir, run_prefix=strat_type),
    )
    return bt


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run a backtest from YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--start", help="Override data.start (YYYY-MM-DD)")
    parser.add_argument("--end", help="Override data.end (YYYY-MM-DD)")
    args = parser.parse_args(argv)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Optional date overrides
    if args.start or args.end:
        cfg.setdefault("data", {})
        if args.start:
            cfg["data"]["start"] = args.start
        if args.end:
            cfg["data"]["end"] = args.end

    bt = build_from_config(cfg)
    # Attach meta so core backtester persists it
    try:
        bt.config.meta = {"source_config_path": os.path.abspath(args.config), "config": cfg}
    except Exception:
        pass
    result = bt.run()

    eq = result["equity_curve"]
    trades = result.get("trades", [])
    
    # Optional benchmark download if using yfinance and benchmark provided
    benchmark_series = None
    bmk_cfg = cfg.get("benchmark")
    if bmk_cfg and bmk_cfg.get("symbol") and isinstance(eq.index, pd.DatetimeIndex) and yf is not None:
        sym = bmk_cfg["symbol"]
        start = eq.index.min().strftime("%Y-%m-%d")
        end = eq.index.max().strftime("%Y-%m-%d")
        try:
            bmk_df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
            if not bmk_df.empty:
                benchmark_series = bmk_df["Close"]
        except Exception:
            pass

    summary = compute_performance(eq, trades=trades)
    # Build merged summary including benchmark comparison if available
    summ_dict = summary.to_dict()
    if benchmark_series is not None and len(benchmark_series) > 1:
        # Align benchmark to equity index
        bmk = benchmark_series.copy()
        if not isinstance(bmk.index, pd.DatetimeIndex):
            bmk.index = pd.to_datetime(bmk.index)
        bmk = bmk.reindex(eq.index, method="pad").dropna()
        # Daily returns
        strat_rets = eq.pct_change().fillna(0.0)
        bmk_rets = bmk.pct_change().fillna(0.0)
        # Use daily ann factor
        ann_factor = 252
        # Totals
        b_total = float(bmk.iloc[-1] / bmk.iloc[0] - 1.0)
        s_total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        # CAGR approximation using period length
        n = max(len(strat_rets), 1)
        b_cagr = (1 + b_total) ** (ann_factor / n) - 1.0
        s_cagr = (1 + s_total) ** (ann_factor / n) - 1.0
        # Regression stats
        cov = float(np.cov(strat_rets, bmk_rets)[0, 1])
        var_b = float(np.var(bmk_rets)) + 1e-12
        beta = cov / var_b
        corr = float(np.corrcoef(strat_rets, bmk_rets)[0, 1])
        alpha_daily = float(strat_rets.mean() - beta * bmk_rets.mean())
        alpha_annual = alpha_daily * ann_factor
        # Tracking error and information ratio
        diff = strat_rets - bmk_rets
        te = float(diff.std() * np.sqrt(ann_factor))
        ir = float(((strat_rets.mean() - bmk_rets.mean()) / (diff.std() + 1e-12)) * np.sqrt(ann_factor))
        # Benchmark Sharpe for reference
        b_sharpe = float((bmk_rets.mean() / (bmk_rets.std() + 1e-12)) * np.sqrt(ann_factor))
        summ_dict.update({
            "benchmark_symbol": bmk_cfg.get("symbol"),
            "benchmark_total_return": b_total,
            "benchmark_cagr": b_cagr,
            "excess_total_return": s_total - b_total,
            "excess_cagr": s_cagr - b_cagr,
            "beta": beta,
            "alpha": alpha_annual,
            "correlation": corr,
            "tracking_error": te,
            "information_ratio": ir,
            "benchmark_sharpe": b_sharpe,
        })
    # Write merged summary.json
    with open(os.path.join(result["run_dir"], "summary.json"), "w") as f:
        import json
        json.dump(summ_dict, f, indent=2)
    
    # Generate plots
    plot_results(eq, result["run_dir"], title=f"Backtest: {os.path.basename(args.config)}", benchmark=benchmark_series)

    # Create a simple HTML dashboard
    dash_path = os.path.join(result["run_dir"], "index.html")
    def fmt_pct(x):
        try:
            return f"{x*100:.2f}%"
        except Exception:
            return "-"
    def fmt_float(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return "-"
    rows = []
    # Core metrics
    core_fields = [
        ("Total Return", fmt_pct(summ_dict.get("total_return"))),
        ("CAGR", fmt_pct(summ_dict.get("cagr"))),
        ("Sharpe", fmt_float(summ_dict.get("sharpe"))),
        ("Sortino", fmt_float(summ_dict.get("sortino"))),
        ("Calmar", fmt_float(summ_dict.get("calmar"))),
        ("Volatility (ann)", fmt_float(summ_dict.get("volatility"))),
        ("Max Drawdown", fmt_pct(summ_dict.get("max_drawdown"))),
        ("Max DD Duration (bars)", str(summ_dict.get("max_drawdown_duration"))),
        ("Num Trades", str(summ_dict.get("num_trades"))),
        ("Win Rate", fmt_pct(summ_dict.get("win_rate"))),
        ("Avg Win ($)", fmt_float(summ_dict.get("avg_win"))),
        ("Avg Loss ($)", fmt_float(summ_dict.get("avg_loss"))),
        ("Profit Factor", fmt_float(summ_dict.get("profit_factor"))),
        ("Exposure (bars/total)", fmt_float(summ_dict.get("exposure_time"))),
    ]
    # Benchmark metrics (if present)
    if summ_dict.get("benchmark_symbol"):
        bench_fields = [
            ("Benchmark", summ_dict.get("benchmark_symbol")),
            ("Benchmark Total Return", fmt_pct(summ_dict.get("benchmark_total_return"))),
            ("Benchmark CAGR", fmt_pct(summ_dict.get("benchmark_cagr"))),
            ("Excess Total Return", fmt_pct(summ_dict.get("excess_total_return"))),
            ("Excess CAGR", fmt_pct(summ_dict.get("excess_cagr"))),
            ("Beta", fmt_float(summ_dict.get("beta"))),
            ("Alpha (ann)", fmt_float(summ_dict.get("alpha"))),
            ("Correlation", fmt_float(summ_dict.get("correlation"))),
            ("Tracking Error", fmt_float(summ_dict.get("tracking_error"))),
            ("Information Ratio", fmt_float(summ_dict.get("information_ratio"))),
            ("Benchmark Sharpe", fmt_float(summ_dict.get("benchmark_sharpe"))),
        ]
    else:
        bench_fields = []
    rows = core_fields + bench_fields
    table_rows = "\n".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows])
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Backtest Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; margin: 20px; }}
        h1 {{ margin-bottom: 0; }}
        .sub {{ color: #666; margin-top: 4px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 16px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; }}
        .links a {{ margin-right: 16px; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        .panel {{ padding: 12px; border: 1px solid #eee; border-radius: 8px; }}
    </style>
    </head>
<body>
    <h1>Backtest Dashboard</h1>
    <div class=\"sub\">Run dir: {result["run_dir"]}</div>
    <div class=\"sub\">Config: {os.path.basename(args.config)}</div>
    <div class=\"links\" style=\"margin: 12px 0;\">
        <a href=\"summary.json\">summary.json</a>
        <a href=\"equity_curve.csv\">equity_curve.csv</a>
        <a href=\"trades.csv\">trades.csv</a>
    </div>
    <div class=\"grid\">
        <div class=\"panel\">
            <h3>Metrics</h3>
            <table>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        <div class=\"panel\">
            <h3>Equity & Benchmark</h3>
            <img src=\"backtest_plots.png\" alt=\"Equity Plot\" />
        </div>
    </div>
</body>
</html>
"""
    with open(dash_path, "w") as f: f.write(html)

    print("Run directory:", result["run_dir"])
    print("Summary:", summary.to_dict())


if __name__ == "__main__":
    main()
