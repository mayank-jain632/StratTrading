# Strategy Library Documentation

## Overview

This backtesting platform includes **6 different trading strategies** across 4 major categories. Each strategy has been designed for specific market conditions and risk profiles.

## Quick Start

### 1. List All Strategies

```bash
python -m options_trader.cli.list_strategies
```

### 2. Run a Backtest

```bash
python -m options_trader.cli.run_backtest --config configs/<strategy_name>.yaml
```

### 3. Compare Results

Check the `runs/` directory for equity curves, plots, and performance summaries.

---

## Strategy Catalog

### 1. Moving Average Crossover (`ma_cross`)

**Type:** Trend Following  
**Risk:** Medium  
**Frequency:** Low (10-30 trades/year)

**How it Works:**

- Fast MA crosses above slow MA → BUY
- Fast MA crosses below slow MA → SELL
- Rides trends while filtering out noise

**Best For:**

- Strong trending markets (bull or bear)
- Long-term position holding
- Low transaction cost environments

**Parameters:**

```yaml
fast: 9 # Fast moving average period (default: 9)
slow: 21 # Slow moving average period (default: 21)
```

**Typical Results:** Low absolute returns, high Sharpe ratio, minimal drawdowns

---

### 2. RSI Mean Reversion (`rsi_mean_reversion`)

**Type:** Mean Reversion  
**Risk:** Medium-High  
**Frequency:** Medium (30-100 trades/year)

**How it Works:**

- RSI < 30 (oversold) → BUY
- RSI > 70 (overbought) → SELL
- Catches reversals from extremes

**Best For:**

- Choppy, ranging markets
- High volatility environments
- Short-term trading

**Parameters:**

```yaml
rsi_period: 14 # RSI calculation period (default: 14)
oversold: 30 # Buy threshold (default: 30)
overbought: 70 # Sell threshold (default: 70)
```

**Typical Results:** Higher trade frequency, works well in sideways markets, can get caught in trends

---

### 3. Bollinger Band Breakout (`bollinger_breakout`)

**Type:** Breakout/Volatility  
**Risk:** High  
**Frequency:** Medium (30-100 trades/year)

**How it Works:**

- Price breaks above upper band → BUY (momentum)
- Price breaks below lower band → SELL (reversal)
- Bands expand/contract with volatility

**Best For:**

- Volatile markets with expansion phases
- Catching explosive moves
- Trend initiation points

**Parameters:**

```yaml
period: 20 # Bollinger Band lookback (default: 20)
num_std: 2.0 # Standard deviations for bands (default: 2.0)
breakout_threshold: 0.005 # % above band to confirm (default: 0.5%)
```

**Typical Results:** High risk/high reward, false breakouts in choppy markets

---

### 4. Momentum Ranking (`momentum_rank`)

**Type:** Rotation/Relative Strength  
**Risk:** Medium  
**Frequency:** Low (10-30 trades/year)

**How it Works:**

- Rank all assets by recent performance
- Hold top N performers
- Rebalance periodically (e.g., weekly)

**Best For:**

- Multi-asset portfolios (sectors, ETFs)
- Capturing relative strength
- Diversified exposure

**Parameters:**

```yaml
lookback: 20 # Days to measure momentum (default: 20)
top_n: 4 # Number of top assets to hold (default: 50% of symbols)
rebalance_days: 5 # Rebalancing frequency (default: 5 = weekly)
```

**Typical Results:** Smooth equity curve, diversified, benefits from having many assets

---

### 5. Dual Momentum (`dual_momentum`)

**Type:** Absolute + Relative Momentum  
**Risk:** Low-Medium  
**Frequency:** Very Low (<10 trades/year)

**How it Works:**

- **Absolute Momentum:** Only invest if asset is trending up
- **Relative Momentum:** Among uptrending assets, pick the best
- Move to cash if no assets meet criteria

**Best For:**

- Tactical allocation
- Crash protection (moves to cash in downtrends)
- Long-term conservative strategies

**Parameters:**

```yaml
lookback: 60 # Momentum period (default: 60 = 3 months)
rebalance_days: 20 # Rebalancing frequency (default: 20 = monthly)
cash_threshold: 0.0 # Minimum return to stay invested (default: 0%)
```

**Typical Results:** Low drawdowns, good crisis performance, concentrated positions

---

### 6. 52-Week Breakout (`breakout_52week`)

**Type:** Breakout/Momentum  
**Risk:** Medium-High  
**Frequency:** Low (10-30 trades/year)

**How it Works:**

- Buy when price makes new high over lookback period
- Trail stop at X% below peak price
- Classic momentum continuation

**Best For:**

- Bull markets
- Momentum trends
- "Buy high, sell higher" approach

**Parameters:**

```yaml
lookback: 252 # Days to track high (default: 252 = 1 year)
breakout_threshold: 0.001 # % above high to confirm (default: 0.1%)
exit_pct: 0.05 # Trailing stop % (default: 5%)
```

**Typical Results:** Captures big moves, late entries, stopped out at tops

---

## Strategy Selection Guide

### Market Condition Matrix

| Market Type          | Best Strategies                              |
| -------------------- | -------------------------------------------- |
| **Strong Uptrend**   | MA Cross, 52-Week Breakout, Dual Momentum    |
| **Strong Downtrend** | Dual Momentum (goes to cash), Short MA Cross |
| **Choppy/Sideways**  | RSI Mean Reversion, Bollinger Breakout       |
| **High Volatility**  | Bollinger Breakout, RSI Mean Reversion       |
| **Low Volatility**   | Momentum Rank, MA Cross                      |

### Risk Tolerance Matrix

| Risk Level       | Strategies                           |
| ---------------- | ------------------------------------ |
| **Conservative** | Dual Momentum, MA Cross              |
| **Moderate**     | Momentum Rank, RSI Mean Reversion    |
| **Aggressive**   | Bollinger Breakout, 52-Week Breakout |

### Time Horizon Matrix

| Holding Period          | Strategies                             |
| ----------------------- | -------------------------------------- |
| **Long-term (months)**  | Dual Momentum, MA Cross                |
| **Medium-term (weeks)** | Momentum Rank, 52-Week Breakout        |
| **Short-term (days)**   | RSI Mean Reversion, Bollinger Breakout |

---

## Performance Comparison

Based on typical backtest results (2023-2024 period):

| Strategy      | Avg. Return | Sharpe  | Max DD  | Trades | Win Rate |
| ------------- | ----------- | ------- | ------- | ------ | -------- |
| MA Cross      | 1-5%        | 2.0-2.5 | -3-5%   | 2-10   | 50-70%   |
| RSI Mean Rev  | 3-7%        | 1.0-1.5 | -5-10%  | 20-50  | 60-80%   |
| Bollinger     | 5-15%       | 0.5-1.5 | -10-20% | 15-40  | 40-60%   |
| Momentum Rank | 10-20%      | 1.0-1.5 | -10-15% | 40-80  | 50-60%   |
| Dual Momentum | 5-10%       | 1.5-2.0 | -5-8%   | 5-15   | 60-75%   |
| 52-Week       | 8-15%       | 1.0-1.8 | -8-12%  | 10-25  | 55-70%   |

_Note: Results vary significantly based on market conditions, parameters, and time period_

---

## Customization Tips

### 1. Adjust Parameters for Your Style

```yaml
# Conservative MA Cross
params:
  fast: 20
  slow: 50    # Slower = fewer trades, less whipsaw

# Aggressive RSI
params:
  oversold: 35    # Less extreme = more trades
  overbought: 65
```

### 2. Combine with Risk Management

```yaml
risk:
  fraction: 0.05 # 5% per position = conservative
  max_positions: 10 # More diversification
```

### 3. Test Multiple Periods

```yaml
# Bull market
start: "2023-01-01"
end: "2023-12-31"

# Bear market
start: "2022-01-01"
end: "2022-12-31"

# Full cycle
start: "2020-01-01"
end: "2024-12-31"
```

---

## Adding Your Own Strategy

1. Create a new file in `src/options_trader/strategies/`:

```python
# my_strategy.py
from dataclasses import dataclass
from typing import Dict, Any, List
from ..core.events import SignalEvent

@dataclass
class MyStrategyConfig:
    symbols: List[str]
    param1: int = 10
    param2: float = 0.5

class MyStrategy:
    def __init__(self, config: MyStrategyConfig):
        self.symbols = config.symbols
        # Initialize your strategy

    def on_bar(self, context: Dict[str, Any]) -> None:
        # Implement your logic
        # Access: context["dt"], context["prices"], context["event_queue"]
        # Emit signals: context["event_queue"].append(SignalEvent(...))
        pass
```

2. Register in `__init__.py`:

```python
from .my_strategy import MyStrategy, MyStrategyConfig

STRATEGY_CATALOG["my_strategy"] = {
    "name": "My Strategy",
    "class": MyStrategy,
    "config": MyStrategyConfig,
    "type": "custom",
    "description": "What it does",
    "best_for": "Market conditions",
    "frequency": "low/medium/high",
    "risk": "low/medium/high",
}
```

3. Create config file `configs/my_strategy.yaml`

4. Run it!

```bash
python -m options_trader.cli.run_backtest --config configs/my_strategy.yaml
```

---

## Next Steps

1. **Test All Strategies:** Run each config to understand their behavior
2. **Compare Performance:** Review plots and summaries in `runs/` directory
3. **Optimize Parameters:** Use grid search to find best settings
4. **Combine Strategies:** Build ensemble/multi-strategy portfolios
5. **Paper Trade:** Connect to Webull paper trading (Stage 2)

---

## Resources

- [Strategy Catalog](src/options_trader/strategies/__init__.py) - Full strategy definitions
- [Config Examples](configs/) - Sample configurations
- [Performance Metrics](src/options_trader/metrics/performance.py) - How results are calculated
- [List Strategies Tool](src/options_trader/cli/list_strategies.py) - Quick reference

---

**Happy backtesting! 🚀**
