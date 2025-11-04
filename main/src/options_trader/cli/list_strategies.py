"""
Strategy Comparison Tool

List all available strategies with their characteristics and best use cases.
Helps you choose the right strategy for your market conditions.
"""
from options_trader.strategies import STRATEGY_CATALOG


def print_strategy_catalog():
    """Print a formatted table of all available strategies."""
    
    print("\n" + "=" * 100)
    print("AVAILABLE TRADING STRATEGIES")
    print("=" * 100)
    
    for strategy_id, info in STRATEGY_CATALOG.items():
        print(f"\n📊 {info['name'].upper()}")
        print(f"   ID: {strategy_id}")
        print(f"   Type: {info['type']}")
        print(f"   Description: {info['description']}")
        print(f"   Best For: {info['best_for']}")
        print(f"   Trade Frequency: {info['frequency']}")
        print(f"   Risk Level: {info['risk']}")
        print(f"   Config: configs/{strategy_id}.yaml")
    
    print("\n" + "=" * 100)
    print("\nSTRATEGY TYPE GUIDE:")
    print("-" * 100)
    print("  • TREND FOLLOWING: Follow momentum, ride trends (best in trending markets)")
    print("  • MEAN REVERSION: Buy dips, sell rips (best in ranging/choppy markets)")
    print("  • MOMENTUM/ROTATION: Rank assets, hold winners (best for multi-asset portfolios)")
    print("  • BREAKOUT: Trade volatility expansions (best in volatile markets)")
    print("\n" + "=" * 100)
    print("\nUSAGE:")
    print("  python -m options_trader.cli.run_backtest --config configs/<strategy_id>.yaml")
    print("=" * 100 + "\n")


def compare_strategies():
    """Compare strategies by type and characteristics."""
    
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON BY TYPE")
    print("=" * 100)
    
    # Group by type
    by_type = {}
    for strategy_id, info in STRATEGY_CATALOG.items():
        strat_type = info['type']
        if strat_type not in by_type:
            by_type[strat_type] = []
        by_type[strat_type].append((strategy_id, info))
    
    for strat_type, strategies in sorted(by_type.items()):
        print(f"\n{strat_type.upper().replace('_', ' ')}:")
        print("-" * 100)
        for strategy_id, info in strategies:
            print(f"  • {info['name']:30s} | Risk: {info['risk']:12s} | Freq: {info['frequency']:10s}")
    
    print("\n" + "=" * 100)
    print("\nFREQUENCY GUIDE:")
    print("  • very_low: Monthly rebalancing, <10 trades/year")
    print("  • low: Weekly/bi-weekly, 10-30 trades/year")
    print("  • medium: Few times per week, 30-100 trades/year")
    print("  • high: Daily signals, >100 trades/year")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    print_strategy_catalog()
    compare_strategies()
