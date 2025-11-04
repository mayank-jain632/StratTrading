from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

from ..core.events import SignalEvent


@dataclass
class MACDConfig:
    symbols: List[str]
    fast: int = 12
    slow: int = 26
    signal: int = 9


class MACDCrossoverStrategy:
    """MACD line crossing signal line."""

    def __init__(self, config: MACDConfig):
        self.symbols = config.symbols
        self.fast = config.fast
        self.slow = config.slow
        self.signal = config.signal
        # EMA state per symbol
        self.ema_fast: Dict[str, float] = {s: None for s in self.symbols}
        self.ema_slow: Dict[str, float] = {s: None for s in self.symbols}
        self.ema_signal: Dict[str, float] = {s: None for s in self.symbols}
        self.in_market: Dict[str, bool] = {s: False for s in self.symbols}

    def _ema(self, prev: float, price: float, period: int) -> float:
        alpha = 2 / (period + 1)
        return price if prev is None else (alpha * price + (1 - alpha) * prev)

    def on_bar(self, context: Dict[str, Any]) -> None:
        dt = context["dt"]
        prices = context["prices"]
        q = context["event_queue"]

        for s in self.symbols:
            price = prices[s]
            self.ema_fast[s] = self._ema(self.ema_fast[s], price, self.fast)
            self.ema_slow[s] = self._ema(self.ema_slow[s], price, self.slow)
            if self.ema_fast[s] is None or self.ema_slow[s] is None:
                continue
            macd = self.ema_fast[s] - self.ema_slow[s]
            self.ema_signal[s] = self._ema(self.ema_signal[s], macd, self.signal)
            if self.ema_signal[s] is None:
                continue
            hist = macd - self.ema_signal[s]
            # Cross above zero -> long; cross below -> exit
            if not self.in_market[s] and hist > 0:
                q.append(SignalEvent(symbol=s, dt=dt, direction="LONG"))
                self.in_market[s] = True
            elif self.in_market[s] and hist < 0:
                q.append(SignalEvent(symbol=s, dt=dt, direction="EXIT"))
                self.in_market[s] = False
