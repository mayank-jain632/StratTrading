from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Dict, Any, List

from ..core.events import SignalEvent


@dataclass
class ZScoreMeanReversionConfig:
    symbols: List[str]
    lookback: int = 30
    entry_z: float = 2.0
    exit_z: float = 0.5


class ZScoreMeanReversionStrategy:
    """Mean-reversion using z-score of recent returns.

    Entry: when latest return's z-score < -entry_z
    Exit: when z-score > -exit_z (i.e., mean reversion done)
    """
    def __init__(self, config: ZScoreMeanReversionConfig):
        self.symbols = config.symbols
        self.lookback = config.lookback
        self.entry_z = config.entry_z
        self.exit_z = config.exit_z
        self.buffers: Dict[str, deque] = {s: deque(maxlen=self.lookback) for s in self.symbols}
        self.in_market: Dict[str, bool] = {s: False for s in self.symbols}

    def on_bar(self, context: Dict[str, Any]) -> None:
        dt = context["dt"]
        prices = context["prices"]
        q = context["event_queue"]

        for s in self.symbols:
            price = prices[s]
            buf = self.buffers[s]
            if len(buf) > 0:
                prev = buf[-1]
                ret = (price - prev) / prev if prev != 0 else 0.0
            else:
                ret = 0.0
            buf.append(price)
            # Need at least 2 points to compute returns; wait until buffer filled
            if len(buf) < self.lookback:
                continue
            # compute return series from prices
            rets = []
            it = list(buf)
            for i in range(1, len(it)):
                p0 = it[i - 1]
                p1 = it[i]
                rets.append((p1 - p0) / p0 if p0 != 0 else 0.0)
            # compute z-score of latest return
            import math
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / len(rets)
            std = math.sqrt(var)
            latest_ret = rets[-1]
            z = (latest_ret - mean) / (std + 1e-12)

            if not self.in_market[s] and z < -self.entry_z:
                q.append(SignalEvent(symbol=s, dt=dt, direction="LONG"))
                self.in_market[s] = True
            elif self.in_market[s] and z > -self.exit_z:
                q.append(SignalEvent(symbol=s, dt=dt, direction="EXIT"))
                self.in_market[s] = False
