from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Dict, Any, List

from ..core.events import SignalEvent


@dataclass
class ATRBreakoutConfig:
    symbols: List[str]
    lookback: int = 20
    atr_mult: float = 1.5


class ATRBreakoutStrategy:
    """Simple breakout using rolling max plus an ATR-like volatility filter computed from pct returns.

    Entry: price > rolling_max
    Exit: price < rolling_max - atr_mult * recent_vol
    """
    def __init__(self, config: ATRBreakoutConfig):
        self.symbols = config.symbols
        self.lookback = config.lookback
        self.atr_mult = config.atr_mult
        self.price_bufs: Dict[str, deque] = {s: deque(maxlen=self.lookback) for s in self.symbols}
        self.in_market: Dict[str, bool] = {s: False for s in self.symbols}

    def on_bar(self, context: Dict[str, Any]) -> None:
        dt = context["dt"]
        prices = context["prices"]
        q = context["event_queue"]

        for s in self.symbols:
            price = prices[s]
            buf = self.price_bufs[s]
            buf.append(price)
            if len(buf) < self.lookback:
                continue
            it = list(buf)
            rolling_max = max(it[:-1]) if len(it) > 1 else it[-1]
            # compute recent vol as std of pct returns
            rets = []
            for i in range(1, len(it)):
                p0 = it[i - 1]
                p1 = it[i]
                rets.append((p1 - p0) / p0 if p0 != 0 else 0.0)
            import math
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / len(rets)
            vol = math.sqrt(var)

            if not self.in_market[s] and price > rolling_max:
                q.append(SignalEvent(symbol=s, dt=dt, direction="LONG"))
                self.in_market[s] = True
            elif self.in_market[s] and price < (rolling_max - self.atr_mult * vol * price):
                q.append(SignalEvent(symbol=s, dt=dt, direction="EXIT"))
                self.in_market[s] = False
