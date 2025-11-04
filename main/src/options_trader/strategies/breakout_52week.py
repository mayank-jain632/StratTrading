"""
52-Week Breakout Strategy

CONCEPT: Buy when price makes new highs
- Track highest price over lookback period (e.g., 252 days = 1 year)
- Buy when price breaks above recent high (strong momentum)
- Exit when price falls below trailing stop or support level

CHARACTERISTICS:
- Works best in: Strong bull markets, momentum trends
- Risk: Buy high, whipsaw at tops
- Frequency: Low (only on genuine breakouts)
- Best for: Capturing major trend continuations

PARAMETERS:
- lookback: Days to track high (default 252 = 1 year)
- breakout_threshold: % above high to confirm (default 0.1%)
- exit_pct: Trailing stop % (default 5%)
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, List

from ..core.events import SignalEvent


@dataclass
class Breakout52WeekConfig:
    symbols: List[str]
    lookback: int = 252  # ~1 year of trading days
    breakout_threshold: float = 0.001  # 0.1% above high
    exit_pct: float = 0.05  # 5% trailing stop


class Breakout52WeekStrategy:
    """Breakout strategy based on new highs."""
    
    def __init__(self, config: Breakout52WeekConfig):
        self.symbols = config.symbols
        self.lookback = config.lookback
        self.breakout_threshold = config.breakout_threshold
        self.exit_pct = config.exit_pct
        
        self.price_buffers: Dict[str, deque] = {
            s: deque(maxlen=self.lookback) for s in self.symbols
        }
        self.in_market: Dict[str, bool] = {s: False for s in self.symbols}
        self.entry_prices: Dict[str, float] = {}
        self.peak_prices: Dict[str, float] = {}
    
    def on_bar(self, context: Dict[str, Any]) -> None:
        dt = context["dt"]
        prices = context["prices"]
        q = context["event_queue"]
        
        for s in self.symbols:
            price = prices[s]
            buf = self.price_buffers[s]
            buf.append(price)
            
            if len(buf) < 20:  # Need some history
                continue
            
            # Calculate recent high
            recent_high = max(buf)
            breakout_level = recent_high * (1 + self.breakout_threshold)
            
            # BUY on breakout to new high
            if not self.in_market[s] and price >= breakout_level:
                q.append(SignalEvent(symbol=s, dt=dt, direction="LONG"))
                self.in_market[s] = True
                self.entry_prices[s] = price
                self.peak_prices[s] = price
            
            # EXIT on trailing stop
            elif self.in_market[s]:
                # Update peak price
                if price > self.peak_prices[s]:
                    self.peak_prices[s] = price
                
                # Check trailing stop
                stop_level = self.peak_prices[s] * (1 - self.exit_pct)
                if price < stop_level:
                    q.append(SignalEvent(symbol=s, dt=dt, direction="EXIT"))
                    self.in_market[s] = False
                    if s in self.entry_prices:
                        del self.entry_prices[s]
                    if s in self.peak_prices:
                        del self.peak_prices[s]
