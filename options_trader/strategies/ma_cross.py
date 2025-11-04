from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List

from ..core.events import SignalEvent


@dataclass
class MACrossConfig:
    symbols: List[str]
    fast: int = 9
    slow: int = 21


class MovingAverageCrossStrategy:
    def __init__(self, config: MACrossConfig):
        self.symbols = config.symbols
        self.fast = config.fast
        self.slow = config.slow
        self.buffers: Dict[str, deque] = {s: deque(maxlen=self.slow) for s in self.symbols}
        self.in_market: Dict[str, bool] = {s: False for s in self.symbols}

    def on_bar(self, context: Dict[str, Any]) -> None:
        dt = context["dt"]
        prices = context["prices"]
        q = context["event_queue"]

        for s in self.symbols:
            price = prices[s]
            buf = self.buffers[s]
            buf.append(price)
            if len(buf) < self.slow:
                continue
            fast_ma = sum(list(buf)[-self.fast:]) / self.fast
            slow_ma = sum(buf) / self.slow

            if not self.in_market[s] and fast_ma > slow_ma:
                q.append(SignalEvent(symbol=s, dt=dt, direction="LONG"))
                self.in_market[s] = True
            elif self.in_market[s] and fast_ma < slow_ma:
                q.append(SignalEvent(symbol=s, dt=dt, direction="EXIT"))
                self.in_market[s] = False

