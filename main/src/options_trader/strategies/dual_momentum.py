"""
Dual Momentum Strategy (Absolute + Relative)

CONCEPT: Combine trend-following with relative strength
- Absolute Momentum: Only buy if asset is in uptrend (vs itself)
- Relative Momentum: Among uptrending assets, buy the strongest
- Cash filter: Move to cash if no assets are trending up

CHARACTERISTICS:
- Works best in: Trending markets with clear winners/losers
- Risk: Whipsaw in sideways markets, late entries
- Frequency: Low (monthly rebalancing typical)
- Best for: Long-term tactical allocation, crash protection

PARAMETERS:
- lookback: Momentum measurement period (default 60 = 3 months)
- rebalance_days: Rebalancing frequency (default 20 = monthly)
- cash_threshold: Min return to stay invested (default 0%)
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from ..core.events import SignalEvent


@dataclass
class DualMomentumConfig:
    symbols: List[str]
    lookback: int = 60
    rebalance_days: int = 20
    cash_threshold: float = 0.0  # Stay invested only if return > this


class DualMomentumStrategy:
    """Dual momentum strategy combining absolute and relative momentum."""
    
    def __init__(self, config: DualMomentumConfig):
        self.symbols = config.symbols
        self.lookback = config.lookback
        self.rebalance_days = config.rebalance_days
        self.cash_threshold = config.cash_threshold
        
        self.price_buffers: Dict[str, deque] = {
            s: deque(maxlen=self.lookback + 1) for s in self.symbols
        }
        self.current_position: Optional[str] = None
        self.days_since_rebalance = 0
    
    def _calculate_return(self, prices: deque) -> float:
        """Calculate return over lookback period."""
        if len(prices) < self.lookback + 1:
            return -999.0  # Flag as invalid
        
        price_list = list(prices)
        old_price = price_list[0]
        new_price = price_list[-1]
        
        if old_price == 0:
            return -999.0
        
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
        
        # Calculate returns for all symbols
        returns = {}
        for s in self.symbols:
            ret = self._calculate_return(self.price_buffers[s])
            if ret > -999.0:  # Valid return
                returns[s] = ret
        
        if not returns:
            return
        
        # Find best performing asset
        best_symbol = max(returns.items(), key=lambda x: x[1])[0]
        best_return = returns[best_symbol]
        
        # ABSOLUTE MOMENTUM: Only invest if best return > threshold
        if best_return < self.cash_threshold:
            # Move to cash (exit all)
            if self.current_position:
                q.append(SignalEvent(symbol=self.current_position, dt=dt, direction="EXIT"))
                self.current_position = None
        else:
            # RELATIVE MOMENTUM: Invest in best performer
            if self.current_position != best_symbol:
                # Exit current position
                if self.current_position:
                    q.append(SignalEvent(symbol=self.current_position, dt=dt, direction="EXIT"))
                # Enter new position
                q.append(SignalEvent(symbol=best_symbol, dt=dt, direction="LONG"))
                self.current_position = best_symbol
        
        self.days_since_rebalance = 0
