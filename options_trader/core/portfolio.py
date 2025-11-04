from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

from .events import FillEvent


@dataclass
class Position:
    quantity: int = 0
    avg_price: float = 0.0


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    total_equity: float = 0.0


class BasicPortfolio:
    def __init__(self, starting_cash: float = 100000.0):
        self.cash = starting_cash
        self.positions: Dict[str, Position] = {}
        self.total_equity = starting_cash

    def update_from_fill(self, fill: FillEvent) -> None:
        pos = self.positions.get(fill.symbol, Position())
        qty = fill.quantity if fill.direction == "BUY" else -fill.quantity
        cost = qty * fill.fill_price
        # Update cash (include commission)
        self.cash -= cost
        self.cash -= fill.commission

        # Update position
        new_qty = pos.quantity + qty
        if new_qty == 0:
            pos = Position(0, 0.0)
        elif qty > 0:
            # Recalculate average price on buys
            pos.avg_price = (pos.avg_price * pos.quantity + qty * fill.fill_price) / (pos.quantity + qty)
        # On sells keep avg_price
        pos.quantity = new_qty
        self.positions[fill.symbol] = pos

    def on_bar_close(self, dt, prices: Dict[str, float]) -> None:
        mtm = 0.0
        for sym, pos in self.positions.items():
            mtm += pos.quantity * prices.get(sym, pos.avg_price)
        self.total_equity = self.cash + mtm

    def position_size(self, symbol: str) -> int:
        return self.positions.get(symbol, Position()).quantity
