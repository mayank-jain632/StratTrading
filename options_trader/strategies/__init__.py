"""
Trading Strategies Library

STRATEGY TYPES:
1. Trend Following: Follow momentum, ride trends (MA Cross, Dual Momentum, 52-Week Breakout)
2. Mean Reversion: Buy dips, sell rips (RSI Mean Reversion, Bollinger Bands)
3. Momentum/Rotation: Rank assets, hold winners (Momentum Rank, Dual Momentum)

USAGE:
    from options_trader.strategies import STRATEGY_CATALOG
    
    # List all available strategies
    for name, info in STRATEGY_CATALOG.items():
        print(f"{name}: {info['description']}")
    
    # Get strategy class and config
    strategy_info = STRATEGY_CATALOG['ma_cross']
    Strategy = strategy_info['class']
    Config = strategy_info['config']
"""
from .ma_cross import MovingAverageCrossStrategy, MACrossConfig
from .rsi_mean_reversion import RSIMeanReversionStrategy, RSIMeanReversionConfig
from .bollinger_breakout import BollingerBreakoutStrategy, BollingerBreakoutConfig
from .momentum_rank import MomentumRankStrategy, MomentumRankConfig
from .dual_momentum import DualMomentumStrategy, DualMomentumConfig
from .breakout_52week import Breakout52WeekStrategy, Breakout52WeekConfig
from .macd_crossover import MACDCrossoverStrategy, MACDConfig
from .donchian_breakout import DonchianBreakoutStrategy, DonchianConfig
from .zscore_mean_reversion import ZScoreMeanReversionStrategy, ZScoreMeanReversionConfig
from .vol_momentum import VolatilityFilteredMomentum, VolMomentumConfig
from .atr_breakout import ATRBreakoutStrategy, ATRBreakoutConfig


# Strategy Catalog for easy discovery and testing
STRATEGY_CATALOG = {
    "ma_cross": {
        "name": "Moving Average Crossover",
        "class": MovingAverageCrossStrategy,
        "config": MACrossConfig,
        "type": "trend_following",
        "description": "Classic trend-following using fast/slow MA crossovers",
        "best_for": "Strong trending markets",
        "frequency": "low",
        "risk": "medium",
    },
    "macd_crossover": {
        "name": "MACD Crossover",
        "class": MACDCrossoverStrategy,
        "config": MACDConfig,
        "type": "trend_following",
        "description": "MACD line crosses signal line to enter/exit",
        "best_for": "Medium-term trends",
        "frequency": "low",
        "risk": "medium",
    },
    "donchian_breakout": {
        "name": "Donchian Channel Breakout",
        "class": DonchianBreakoutStrategy,
        "config": DonchianConfig,
        "type": "breakout",
        "description": "Breakouts of highest high/lowest low over N days",
        "best_for": "Trend initiation, momentum continuation",
        "frequency": "low",
        "risk": "medium-high",
    },
    "rsi_mean_reversion": {
        "name": "RSI Mean Reversion",
        "class": RSIMeanReversionStrategy,
        "config": RSIMeanReversionConfig,
        "type": "mean_reversion",
        "description": "Buy oversold (RSI<30), sell overbought (RSI>70)",
        "best_for": "Choppy, ranging markets",
        "frequency": "medium",
        "risk": "medium-high",
    },
    "bollinger_breakout": {
        "name": "Bollinger Band Breakout",
        "class": BollingerBreakoutStrategy,
        "config": BollingerBreakoutConfig,
        "type": "breakout",
        "description": "Trade volatility breakouts from Bollinger Bands",
        "best_for": "Volatile, expanding ranges",
        "frequency": "medium",
        "risk": "high",
    },
    "momentum_rank": {
        "name": "Momentum Ranking/Rotation",
        "class": MomentumRankStrategy,
        "config": MomentumRankConfig,
        "type": "rotation",
        "description": "Hold top N performers by momentum, rebalance periodically",
        "best_for": "Multi-asset portfolios, sector rotation",
        "frequency": "low",
        "risk": "medium",
    },
    "dual_momentum": {
        "name": "Dual Momentum (Absolute + Relative)",
        "class": DualMomentumStrategy,
        "config": DualMomentumConfig,
        "type": "momentum",
        "description": "Hold best performer only if trending up, else cash",
        "best_for": "Tactical allocation, crash protection",
        "frequency": "very_low",
        "risk": "low-medium",
    },
    "breakout_52week": {
        "name": "52-Week Breakout",
        "class": Breakout52WeekStrategy,
        "config": Breakout52WeekConfig,
        "type": "breakout",
        "description": "Buy new highs with trailing stop protection",
        "best_for": "Bull markets, momentum continuation",
        "frequency": "low",
        "risk": "medium-high",
    },
    "zscore_reversion": {
        "name": "Z-Score Mean Reversion",
        "class": ZScoreMeanReversionStrategy,
        "config": ZScoreMeanReversionConfig,
        "type": "mean_reversion",
        "description": "Mean reversion using z-score of recent returns",
        "best_for": "Choppy markets",
        "frequency": "medium",
        "risk": "medium",
    },
    "vol_momentum": {
        "name": "Volatility-Filtered Momentum",
        "class": VolatilityFilteredMomentum,
        "config": VolMomentumConfig,
        "type": "momentum",
        "description": "Momentum that prefers low-volatility winners",
        "best_for": "Momentum with lower drawdown",
        "frequency": "medium",
        "risk": "medium",
    },
    "atr_breakout": {
        "name": "ATR Breakout",
        "class": ATRBreakoutStrategy,
        "config": ATRBreakoutConfig,
        "type": "breakout",
        "description": "Breakout with ATR-style volatility filter",
        "best_for": "Volatile breakout environments",
        "frequency": "low",
        "risk": "medium-high",
    },
}


__all__ = [
    "MovingAverageCrossStrategy",
    "MACrossConfig",
    "RSIMeanReversionStrategy",
    "RSIMeanReversionConfig",
    "BollingerBreakoutStrategy",
    "BollingerBreakoutConfig",
    "MomentumRankStrategy",
    "MomentumRankConfig",
    "DualMomentumStrategy",
    "DualMomentumConfig",
    "Breakout52WeekStrategy",
    "Breakout52WeekConfig",
    "MACDCrossoverStrategy",
    "MACDConfig",
    "DonchianBreakoutStrategy",
    "DonchianConfig",
    "STRATEGY_CATALOG",
    "ZScoreMeanReversionStrategy",
    "ZScoreMeanReversionConfig",
    "VolatilityFilteredMomentum",
    "VolMomentumConfig",
    "ATRBreakoutStrategy",
    "ATRBreakoutConfig",
]
