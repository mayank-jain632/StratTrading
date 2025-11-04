from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Dict, Any, List

from ..core.events import SignalEvent


@dataclass
class VolMomentumConfig:
    symbols: List[str]
    mom_lookback: int = 63  # quarter approx
    vol_lookback: int = 21
    vol_threshold: float = 0.03  # enter only if recent vol < threshold


class VolatilityFilteredMomentum:
    """Momentum signal that prefers low-volatility winners.

    Entry: momentum (price change over mom_lookback) > 0 AND recent vol < vol_threshold
    Exit: momentum <= 0 or vol > vol_threshold
    """
    def __init__(self, config: VolMomentumConfig):
        self.symbols = config.symbols
        self.mom_lookback = config.mom_lookback
        self.vol_lookback = config.vol_lookback
        self.vol_threshold = config.vol_threshold
        self.price_bufs: Dict[str, deque] = {s: deque(maxlen=max(self.mom_lookback, self.vol_lookback)) for s in self.symbols}
        self.in_market: Dict[str, bool] = {s: False for s in self.symbols}

    def on_bar(self, context: Dict[str, Any]) -> None:
        dt = context["dt"]
        prices = context["prices"]
        q = context["event_queue"]

        for s in self.symbols:
            price = prices[s]
            buf = self.price_bufs[s]
            buf.append(price)
            if len(buf) < max(self.mom_lookback, self.vol_lookback):
                continue
            it = list(buf)
            mom = (it[-1] - it[-self.mom_lookback]) / it[-self.mom_lookback] if it[-self.mom_lookback] != 0 else 0.0
            # compute recent vol as std of pct returns over vol_lookback
            rets = []
            for i in range(-self.vol_lookback, 0):
                p0 = it[i - 1]
                p1 = it[i]
                rets.append((p1 - p0) / p0 if p0 != 0 else 0.0)
            import math
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / len(rets)
            vol = math.sqrt(var)

            if not self.in_market[s] and mom > 0 and vol < self.vol_threshold:
                q.append(SignalEvent(symbol=s, dt=dt, direction="LONG"))
                self.in_market[s] = True
            elif self.in_market[s] and (mom <= 0 or vol > self.vol_threshold):
                q.append(SignalEvent(symbol=s, dt=dt, direction="EXIT"))
                self.in_market[s] = False
