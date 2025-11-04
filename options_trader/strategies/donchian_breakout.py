from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
from collections import deque

from ..core.events import SignalEvent


@dataclass
class DonchianConfig:
    symbols: List[str]
    period: int = 20


class DonchianBreakoutStrategy:
    def __init__(self, config: DonchianConfig):
        self.symbols = config.symbols
        self.period = config.period
        self.buffers: Dict[str, deque] = {s: deque(maxlen=self.period) for s in self.symbols}
        self.in_market: Dict[str, bool] = {s: False for s in self.symbols}

    def on_bar(self, context: Dict[str, Any]) -> None:
        dt = context["dt"]
        prices = context["prices"]
        q = context["event_queue"]

        for s in self.symbols:
            price = prices[s]
            buf = self.buffers[s]
            buf.append(price)
            if len(buf) < self.period:
                continue
            hi = max(buf)
            lo = min(buf)
            if not self.in_market[s] and price >= hi:
                q.append(SignalEvent(symbol=s, dt=dt, direction="LONG"))
                self.in_market[s] = True
            elif self.in_market[s] and price <= lo:
                q.append(SignalEvent(symbol=s, dt=dt, direction="EXIT"))
                self.in_market[s] = False
