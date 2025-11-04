"""
Momentum Ranking Strategy

CONCEPT: Buy the strongest performers, avoid the weakest
- Rank all symbols by recent performance (e.g., 20-day return)
- Hold top N performers (e.g., top 50%)
- Rebalance periodically (e.g., weekly)

CHARACTERISTICS:
- Works best in: Strong trending markets, multi-asset portfolios
- Risk: Concentration in recent winners (chasing performance)
- Frequency: Low (rebalances on schedule, not every bar)
- Best for: Sector rotation, relative strength investing

PARAMETERS:
- lookback: Days to measure momentum (default 20)
- top_n: How many top symbols to hold (default: 50% of symbols)
- rebalance_days: How often to rebalance (default 5 = weekly)
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, List

from ..core.events import SignalEvent


@dataclass
class MomentumRankConfig:
    symbols: List[str]
    lookback: int = 20
    top_n: int = None  # Will default to len(symbols) // 2
    rebalance_days: int = 5


class MomentumRankStrategy:
    """Rotational strategy based on momentum ranking."""
    
    def __init__(self, config: MomentumRankConfig):
        self.symbols = config.symbols
        self.lookback = config.lookback
        self.top_n = config.top_n or len(self.symbols) // 2
        self.rebalance_days = config.rebalance_days
        
        self.price_buffers: Dict[str, deque] = {
            s: deque(maxlen=self.lookback + 1) for s in self.symbols
        }
        self.current_holdings: set = set()
        self.days_since_rebalance = 0
    
    def _calculate_momentum(self, prices: deque) -> float:
        """Calculate momentum as % return over lookback period."""
        if len(prices) < self.lookback + 1:
            return 0.0
        
        price_list = list(prices)
        old_price = price_list[0]
        new_price = price_list[-1]
        
        if old_price == 0:
            return 0.0
        
        return (new_price - old_price) / old_price
    
    def on_bar(self, context: Dict[str, Any]) -> None:
        dt = context["dt"]
        prices = context["prices"]
        q = context["event_queue"]
        
        # Update price buffers
        for s in self.symbols:
            self.price_buffers[s].append(prices[s])
        
        # Check if it's time to rebalance
        self.days_since_rebalance += 1
        if self.days_since_rebalance < self.rebalance_days:
            return
        
        # Calculate momentum for all symbols
        momentum_scores = {}
        for s in self.symbols:
            buf = self.price_buffers[s]
            if len(buf) >= self.lookback + 1:
                momentum_scores[s] = self._calculate_momentum(buf)
        
        if not momentum_scores:
            return
        
        # Rank symbols by momentum
        ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_symbols = set([s for s, _ in ranked[:self.top_n]])
        
        # Exit positions not in top N
        for s in self.current_holdings - top_symbols:
            q.append(SignalEvent(symbol=s, dt=dt, direction="EXIT"))
        
        # Enter positions in top N
        for s in top_symbols - self.current_holdings:
            q.append(SignalEvent(symbol=s, dt=dt, direction="LONG"))
        
        self.current_holdings = top_symbols
        self.days_since_rebalance = 0
