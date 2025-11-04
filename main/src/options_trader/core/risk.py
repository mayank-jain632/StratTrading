from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class FixedFractionSizer:
    fraction: float = 0.1  # 10% of equity per trade
    max_positions: int = 5

    def size_order(self, symbol: str, signal_direction: str, context: Dict[str, Any]) -> int:
        prices = context["prices"]
        price = prices[symbol]
        port = context["portfolio"]
        # Simple equal-weight sizing by fraction of equity
        budget = port.total_equity * self.fraction
        qty = int(budget // price)
        if qty <= 0:
            return 0
        if signal_direction == "LONG":
            return qty
        elif signal_direction == "EXIT":
            return -port.position_size(symbol)
        return 0
