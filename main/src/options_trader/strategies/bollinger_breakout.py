"""
Bollinger Band Breakout Strategy

CONCEPT: Trade breakouts from volatility bands
- Price breaks above upper band = STRONG uptrend (BUY)
- Price breaks below lower band = STRONG downtrend (EXIT/SHORT)
- Bands expand/contract based on volatility

CHARACTERISTICS:
- Works best in: Trending markets with volatility expansion
- Risk: False breakouts in choppy markets
- Frequency: Moderate (only on genuine volatility breakouts)
- Best for: Catching explosive moves, momentum trading

PARAMETERS:
- period: Lookback for moving average (default 20)
- num_std: Number of standard deviations for bands (default 2.0)
- breakout_threshold: % above/below band to confirm breakout (default 0.5%)
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, List
import math

from ..core.events import SignalEvent


@dataclass
class BollingerBreakoutConfig:
    symbols: List[str]
    period: int = 20
    num_std: float = 2.0
    breakout_threshold: float = 0.005  # 0.5% beyond band


class BollingerBreakoutStrategy:
    """Breakout strategy using Bollinger Bands."""
    
    def __init__(self, config: BollingerBreakoutConfig):
        self.symbols = config.symbols
        self.period = config.period
        self.num_std = config.num_std
        self.breakout_threshold = config.breakout_threshold
        
        self.price_buffers: Dict[str, deque] = {
            s: deque(maxlen=self.period) for s in self.symbols
        }
        self.in_market: Dict[str, bool] = {s: False for s in self.symbols}
    
    def _calculate_bands(self, prices: deque) -> tuple:
        """Calculate Bollinger Bands (middle, upper, lower)."""
        if len(prices) < self.period:
            return None, None, None
        
        price_list = list(prices)
        
        # Middle band (SMA)
        middle = sum(price_list) / len(price_list)
        
        # Standard deviation
        variance = sum((p - middle) ** 2 for p in price_list) / len(price_list)
        std = math.sqrt(variance)
        
        # Upper and lower bands
        upper = middle + (self.num_std * std)
        lower = middle - (self.num_std * std)
        
        return middle, upper, lower
    
    def on_bar(self, context: Dict[str, Any]) -> None:
        dt = context["dt"]
        prices = context["prices"]
        q = context["event_queue"]
        
        for s in self.symbols:
            price = prices[s]
            buf = self.price_buffers[s]
            buf.append(price)
            
            if len(buf) < self.period:
                continue
            
            middle, upper, lower = self._calculate_bands(buf)
            
            if middle is None:
                continue
            
            # Calculate breakout thresholds
            upper_breakout = upper * (1 + self.breakout_threshold)
            lower_breakout = lower * (1 - self.breakout_threshold)
            
            # Buy on upper band breakout (momentum)
            if not self.in_market[s] and price > upper_breakout:
                q.append(SignalEvent(symbol=s, dt=dt, direction="LONG"))
                self.in_market[s] = True
            
            # Exit on lower band breach (reversal/stop loss)
            elif self.in_market[s] and price < lower_breakout:
                q.append(SignalEvent(symbol=s, dt=dt, direction="EXIT"))
                self.in_market[s] = False
