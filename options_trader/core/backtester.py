from __future__ import annotations
import os
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Deque, List, Optional

import pandas as pd

from .events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from .interfaces import DataProvider, Strategy, Broker, Portfolio, RiskManager
from ..metrics.performance import compute_performance, save_summary, plot_results


@dataclass
class BacktestConfig:
    output_dir: str
    run_prefix: str = ""
    # Control artifact generation
    save_equity_csv: bool = True
    save_trades_csv: bool = True
    save_summary: bool = True
    save_plots: bool = True
    create_dashboard: bool = True
    # Optional meta to persist with the run (e.g., original YAML config)
    meta: Optional[Dict[str, Any]] = None


class Backtester:
    def __init__(
        self,
        data: DataProvider,
        strategy: Strategy,
        broker: Broker,
        portfolio: Portfolio,
        risk: RiskManager,
        config: BacktestConfig,
    ) -> None:
        self.data = data
        self.strategy = strategy
        self.broker = broker
        self.portfolio = portfolio
        self.risk = risk
        self.config = config

        self.event_queue: Deque[Any] = deque()
        self._setup_output()
        # Connect broker to event queue if it expects it
        if hasattr(self.broker, "event_queue"):
            self.broker.event_queue = self.event_queue
        
        # Track trades for metrics
        self.trades: List[Dict[str, Any]] = []
        self._open_positions: Dict[str, Dict[str, Any]] = {}  # symbol -> {entry_dt, entry_price, qty, bars_held}
        self._bar_count = 0

    def _setup_output(self) -> None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        prefix = (self.config.run_prefix.strip().replace(" ", "_") + "-") if self.config.run_prefix else ""
        self.run_dir = os.path.join(self.config.output_dir, f"{prefix}{ts}")
        os.makedirs(self.run_dir, exist_ok=True)

    def _emit_market_events(self) -> None:
        for sym in self.data.symbols():
            bar = self.data.get_bar(sym)
            me = MarketEvent(symbol=sym, dt=bar["dt"], price=bar["close"])
            self.event_queue.append(me)
            # Update broker with latest price for fills
            if hasattr(self.broker, "update_price"):
                self.broker.update_price(sym, bar["close"])

    def _context(self) -> Dict[str, Any]:
        syms = list(self.data.symbols())
        prices = {s: self.data.get_bar(s)["close"] for s in syms}
        dt = self.data.get_bar(syms[0])["dt"] if syms else None
        return {
            "dt": dt,
            "prices": prices,
            "portfolio": self.portfolio,
            "event_queue": self.event_queue,
            "risk": self.risk,
        }

    def run(self) -> Dict[str, Any]:
        equity_curve = []
        while self.data.next():
            self._bar_count += 1
            self._emit_market_events()

            # Strategy reacts per bar using context
            ctx = self._context()
            self.strategy.on_bar(ctx)

            # Process all events for this bar
            while self.event_queue:
                ev = self.event_queue.popleft()
                if isinstance(ev, SignalEvent):
                    qty = self.risk.size_order(ev.symbol, ev.direction, ctx)
                    if qty != 0:
                        order = OrderEvent(
                            symbol=ev.symbol,
                            dt=ev.dt,
                            order_type="MKT",
                            quantity=abs(qty),
                            direction="BUY" if qty > 0 else "SELL",
                        )
                        self.broker.send_order(order)
                elif isinstance(ev, FillEvent):
                    self.portfolio.update_from_fill(ev)
                    self._track_trade(ev, ctx)
                # MarketEvents are informational

            # Next-bar fills: process any pending orders now that this bar is closed
            if hasattr(self.broker, "process_pending_fills"):
                self.broker.process_pending_fills(ctx["dt"])  # fills use current close with slippage/spread
            # Process fills emitted by broker
            while self.event_queue:
                ev = self.event_queue.popleft()
                if isinstance(ev, FillEvent):
                    self.portfolio.update_from_fill(ev)
                    self._track_trade(ev, ctx)

            # Increment bars held for open positions
            for sym in self._open_positions:
                self._open_positions[sym]["bars_held"] += 1

            # Mark-to-market at close
            dt = ctx["dt"]
            prices = ctx["prices"]
            self.portfolio.on_bar_close(dt, prices)
            equity_curve.append({"dt": dt, "equity": self.portfolio.total_equity})

        eq_df = pd.DataFrame(equity_curve).set_index("dt")
        # Save equity curve
        if self.config.save_equity_csv:
            eq_path = os.path.join(self.run_dir, "equity_curve.csv")
            eq_df.to_csv(eq_path)
        
        # Close any remaining open positions at final prices for trade tracking
        if self._open_positions:
            final_dt = ctx["dt"]
            final_prices = ctx["prices"]
            for sym, pos in list(self._open_positions.items()):
                final_price = final_prices.get(sym)
                if final_price:
                    # Assume zero commission for end-of-backtest close
                    pnl = (final_price - pos["entry_price"]) * pos["qty"]
                    self.trades.append({
                        "symbol": sym,
                        "entry_dt": pos["entry_dt"],
                        "exit_dt": final_dt,
                        "entry_price": pos["entry_price"],
                        "exit_price": final_price,
                        "qty": pos["qty"],
                        "pnl": pnl,
                        "bars_held": pos["bars_held"],
                    })
            self._open_positions.clear()
        
        # Save trades log (always create the file for consistency)
        if self.config.save_trades_csv:
            trades_path = os.path.join(self.run_dir, "trades.csv")
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv(trades_path, index=False)
            else:
                # Write empty CSV with headers
                pd.DataFrame(columns=[
                    "symbol", "entry_dt", "exit_dt", "entry_price", "exit_price", "qty", "pnl", "bars_held"
                ]).to_csv(trades_path, index=False)

        # Compute and save summary/plots/dashboard
        if self.config.save_summary or self.config.save_plots or self.config.create_dashboard:
            try:
                summary = compute_performance(eq_df, trades=self.trades)
            except Exception:
                summary = None
            if self.config.save_summary and summary is not None:
                save_summary(summary, os.path.join(self.run_dir, "summary.json"))
            if self.config.save_plots:
                title = f"Backtest: {self.config.run_prefix or 'Strategy'}"
                # Backtester doesn't handle benchmark overlay; portfolio/CLI can overwrite with richer plots
                plot_results(eq_df, self.run_dir, title=title)
            if self.config.create_dashboard:
                # Minimal index.html; richer dashboard may overwrite later
                dash_path = os.path.join(self.run_dir, "index.html")
                # Read summary if created
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
                rows_html = ""
                if summary is not None:
                    sd = summary.to_dict()
                    core_fields = [
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
                        ("Avg Win ($)", fmt_float(sd.get("avg_win"))),
                        ("Avg Loss ($)", fmt_float(sd.get("avg_loss"))),
                        ("Profit Factor", fmt_float(sd.get("profit_factor"))),
                        ("Exposure (bars/total)", fmt_float(sd.get("exposure_time"))),
                    ]
                    rows_html = "\n".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in core_fields])
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
    <div class=\"sub\">Run dir: {self.run_dir}</div>
    <div class=\"sub\">Strategy: {self.config.run_prefix or '-'}
    </div>
    <div style=\"margin: 12px 0;\"> 
        <a href=\"summary.json\">summary.json</a>
        <a style=\"margin-left:12px\" href=\"equity_curve.csv\">equity_curve.csv</a>
        <a style=\"margin-left:12px\" href=\"trades.csv\">trades.csv</a>
    </div>
    <div style=\"display:grid;grid-template-columns:1fr 1fr; gap:24px\"> 
        <div>
            <h3>Metrics</h3>
            <table><tbody>{rows_html}</tbody></table>
        </div>
        <div>
            <h3>Equity Curve</h3>
            <img src=\"backtest_plots.png\" alt=\"Equity Plot\" />
        </div>
    </div>
</body>
</html>
"""
                with open(dash_path, "w") as f:
                    f.write(html)

        # Persist meta if provided
        if self.config.meta:
            import json as _json
            with open(os.path.join(self.run_dir, "meta.json"), "w") as f:
                _json.dump(self.config.meta, f, indent=2)

        return {"run_dir": self.run_dir, "equity_curve": eq_df, "trades": self.trades}
    
    def _track_trade(self, fill: FillEvent, ctx: Dict[str, Any]) -> None:
        """Track round-trip trades for metrics."""
        sym = fill.symbol
        if fill.direction == "BUY":
            # Open or add to position
            if sym in self._open_positions:
                # Average in (not implemented for simplicity, just replace)
                pass
            self._open_positions[sym] = {
                "entry_dt": fill.dt,
                "entry_price": fill.fill_price,
                "qty": fill.quantity,
                "bars_held": 0,
            }
        elif fill.direction == "SELL":
            # Close position
            if sym in self._open_positions:
                pos = self._open_positions.pop(sym)
                pnl = (fill.fill_price - pos["entry_price"]) * pos["qty"] - fill.commission
                self.trades.append({
                    "symbol": sym,
                    "entry_dt": pos["entry_dt"],
                    "exit_dt": fill.dt,
                    "entry_price": pos["entry_price"],
                    "exit_price": fill.fill_price,
                    "qty": pos["qty"],
                    "pnl": pnl,
                    "bars_held": pos["bars_held"],
                })

