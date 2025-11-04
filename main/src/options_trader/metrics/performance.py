from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


@dataclass
class PerfSummary:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_dur: int
    volatility: float
    num_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    exposure_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_dur,
            "volatility": self.volatility,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "exposure_time": self.exposure_time,
        }


def compute_performance(equity_curve: pd.DataFrame, trades: Optional[List[Dict[str, Any]]] = None, freq: str = "D") -> PerfSummary:
    """Compute comprehensive performance metrics from equity curve and optional trade log."""
    eq = equity_curve["equity"].astype(float)
    rets = eq.pct_change().fillna(0.0)
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1.0 if len(eq) > 1 else 0.0
    ann_factor = 252 if freq.upper().startswith("D") else 52 if freq.upper().startswith("W") else 12
    
    # CAGR
    cagr = (1 + total_return) ** (ann_factor / max(len(rets), 1)) - 1.0 if len(rets) > 0 else 0.0
    
    # Volatility (annualized)
    volatility = rets.std() * np.sqrt(ann_factor) if len(rets) > 1 else 0.0
    
    # Sharpe
    sharpe = np.sqrt(ann_factor) * (rets.mean() / (rets.std() + 1e-12)) if len(rets) > 1 else 0.0
    
    # Sortino (downside deviation only)
    downside_rets = rets[rets < 0]
    downside_std = downside_rets.std() if len(downside_rets) > 0 else rets.std()
    sortino = np.sqrt(ann_factor) * (rets.mean() / (downside_std + 1e-12)) if len(rets) > 1 else 0.0

    # Drawdown
    cum_max = eq.cummax()
    dd = (eq / cum_max) - 1.0
    max_drawdown = dd.min() if len(dd) else 0.0
    
    # Drawdown duration
    dd_dur = 0
    max_dd_dur = 0
    for v in dd:
        if v < 0:
            dd_dur += 1
            max_dd_dur = max(max_dd_dur, dd_dur)
        else:
            dd_dur = 0
    
    # Calmar ratio (CAGR / abs(max_dd))
    calmar = cagr / (abs(max_drawdown) + 1e-12) if max_drawdown != 0 else 0.0

    # Trade-based metrics
    num_trades = 0
    win_rate = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    profit_factor = 0.0
    exposure_time = 0.0

    if trades:
        num_trades = len(trades)
        pnls = [t.get("pnl", 0.0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        win_rate = len(wins) / num_trades if num_trades > 0 else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / (gross_loss + 1e-12) if gross_loss > 0 else 0.0
        # Exposure: sum of bars in market / total bars
        total_bars = len(eq)
        bars_in_market = sum(t.get("bars_held", 0) for t in trades)
        exposure_time = bars_in_market / total_bars if total_bars > 0 else 0.0

    return PerfSummary(
        total_return=float(total_return),
        cagr=float(cagr),
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(calmar),
        max_drawdown=float(max_drawdown),
        max_drawdown_dur=int(max_dd_dur),
        volatility=float(volatility),
        num_trades=num_trades,
        win_rate=float(win_rate),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        profit_factor=float(profit_factor),
        exposure_time=float(exposure_time),
    )


def save_summary(summary: PerfSummary, path: str) -> None:
    with open(path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)


def plot_results(equity_curve: pd.DataFrame, output_dir: str, title: str = "Backtest Results", benchmark: Optional[pd.Series] = None) -> None:
    """Generate and save comprehensive backtest plots."""
    eq = equity_curve["equity"].astype(float)
    idx = equity_curve.index
    
    # Convert index to datetime if not already
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    
    rets = eq.pct_change().fillna(0.0)
    cum_max = eq.cummax()
    dd = (eq / cum_max) - 1.0

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1) Equity curve
    axes[0].plot(idx, eq, label="Equity", linewidth=1.5, color="#2E86AB")
    # Optional benchmark overlay (scaled to starting equity)
    if benchmark is not None and not benchmark.empty:
        bmk = benchmark.copy()
        if not isinstance(bmk.index, pd.DatetimeIndex):
            bmk.index = pd.to_datetime(bmk.index)
        bmk = bmk.reindex(idx, method="pad").dropna()
        if len(bmk) > 0:
            bmk_norm = (bmk / bmk.iloc[0]) * eq.iloc[0]
            axes[0].plot(bmk_norm.index, bmk_norm.values, label="Benchmark (B&H)", linewidth=1.2, color="#888888", linestyle="--")
    axes[0].fill_between(idx, eq.iloc[0], eq, alpha=0.2, color="#2E86AB")
    axes[0].axhline(eq.iloc[0], color="gray", linestyle="--", linewidth=0.8, label="Starting Capital")
    axes[0].set_ylabel("Equity ($)", fontsize=10)
    axes[0].set_title("Equity Curve", fontsize=11, loc="left")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # 2) Drawdown
    axes[1].fill_between(idx, 0, dd * 100, color="#A23B72", alpha=0.6, label="Drawdown %")
    axes[1].set_ylabel("Drawdown (%)", fontsize=10)
    axes[1].set_title("Drawdown", fontsize=11, loc="left")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    # 3) Monthly returns (if enough data)
    if len(rets) > 20:
        try:
            monthly_rets = rets.groupby(pd.Grouper(freq="ME")).apply(lambda x: (1 + x).prod() - 1) * 100
            # Bar chart with color based on positive/negative
            colors = ["#06A77D" if r >= 0 else "#D72638" for r in monthly_rets]
            axes[2].bar(monthly_rets.index, monthly_rets, color=colors, alpha=0.7, width=20)
            axes[2].axhline(0, color="black", linewidth=0.8)
            axes[2].set_ylabel("Return (%)", fontsize=10)
            axes[2].set_title("Monthly Returns", fontsize=11, loc="left")
            axes[2].grid(True, alpha=0.3, axis="y")
        except Exception:
            # Fallback to daily returns histogram
            axes[2].hist(rets * 100, bins=50, color="#06A77D", alpha=0.7, edgecolor="black")
            axes[2].axvline(0, color="red", linewidth=1.0, linestyle="--")
            axes[2].set_ylabel("Frequency", fontsize=10)
            axes[2].set_xlabel("Daily Return (%)", fontsize=10)
            axes[2].set_title("Return Distribution", fontsize=11, loc="left")
            axes[2].grid(True, alpha=0.3, axis="y")
    else:
        # Daily returns histogram for short backtests
        axes[2].hist(rets * 100, bins=30, color="#06A77D", alpha=0.7, edgecolor="black")
        axes[2].axvline(0, color="red", linewidth=1.0, linestyle="--")
        axes[2].set_ylabel("Frequency", fontsize=10)
        axes[2].set_xlabel("Daily Return (%)", fontsize=10)
        axes[2].set_title("Return Distribution", fontsize=11, loc="left")
        axes[2].grid(True, alpha=0.3, axis="y")

    # Format x-axis dates
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "backtest_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to: {plot_path}")
