"""
RSI Mean Reversion Strategy

CONCEPT: Buy when oversold, sell when overbought
- Uses Relative Strength Index (RSI) to identify extreme conditions
- RSI > 70 = overbought (SELL/EXIT)
- RSI < 30 = oversold (BUY)

CHARACTERISTICS:
- Works best in: Ranging/choppy markets, high volatility
- Risk: Can get caught in strong trends (catching falling knife)
- Frequency: More trades than MA crossover
- Best for: Counter-trend trading, short-term positions

PARAMETERS:
- rsi_period: Lookback for RSI calculation (default 14)
- oversold: RSI level to trigger buy (default 30)
- overbought: RSI level to trigger sell (default 70)
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, List

from ..core.events import SignalEvent


@dataclass
class RSIMeanReversionConfig:
    symbols: List[str]
    rsi_period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0


class RSIMeanReversionStrategy:
    """Mean reversion strategy using RSI indicator."""
    
    def __init__(self, config: RSIMeanReversionConfig):
        self.symbols = config.symbols
        self.rsi_period = config.rsi_period
        self.oversold = config.oversold
        self.overbought = config.overbought
        
        # Price buffers for RSI calculation
        self.price_buffers: Dict[str, deque] = {
            s: deque(maxlen=self.rsi_period + 1) for s in self.symbols
        }
        self.in_market: Dict[str, bool] = {s: False for s in self.symbols}
    
    def _calculate_rsi(self, prices: deque) -> float:
        """Calculate RSI from price buffer."""
        if len(prices) < self.rsi_period + 1:
            return 50.0  # Neutral RSI
        
        # Calculate price changes
        changes = []
        price_list = list(prices)
        for i in range(1, len(price_list)):
            changes.append(price_list[i] - price_list[i-1])
        
        # Separate gains and losses
        gains = [c if c > 0 else 0 for c in changes]
        losses = [-c if c < 0 else 0 for c in changes]
        
        # Calculate average gains and losses
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # Avoid division by zero
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def on_bar(self, context: Dict[str, Any]) -> None:
        dt = context["dt"]
        prices = context["prices"]
        q = context["event_queue"]
        
        for s in self.symbols:
            price = prices[s]
            buf = self.price_buffers[s]
            buf.append(price)
            
            if len(buf) < self.rsi_period + 1:
                continue
            
            rsi = self._calculate_rsi(buf)
            
            # Buy when oversold
            if not self.in_market[s] and rsi < self.oversold:
                q.append(SignalEvent(symbol=s, dt=dt, direction="LONG"))
                self.in_market[s] = True
            
            # Sell when overbought
            elif self.in_market[s] and rsi > self.overbought:
                q.append(SignalEvent(symbol=s, dt=dt, direction="EXIT"))
                self.in_market[s] = False
