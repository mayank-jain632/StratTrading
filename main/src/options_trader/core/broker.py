from __future__ import annotations
from dataclasses import dataclass
from typing import Deque, Any, Optional, Dict, List

from .events import OrderEvent, FillEvent


@dataclass
class CommissionModel:
    per_trade: float = 0.0
    per_share: float = 0.0


@dataclass
class ExecutionCostModel:
    """Slippage and spread in basis points (bps).

    Example: slippage_bps=5, spread_bps=10 means:
    - BUY price = close * (1 + (5 + 10/2) bps)
    - SELL price = close * (1 - (5 + 10/2) bps)
    """
    slippage_bps: float = 0.0
    spread_bps: float = 0.0

    def adjust_price(self, side: str, base_price: float) -> float:
        half_spread = self.spread_bps / 2.0
        total_bps = self.slippage_bps + half_spread
        if side.upper() == "BUY":
            return base_price * (1.0 + total_bps / 10000.0)
        else:
            return base_price * (1.0 - total_bps / 10000.0)


class PaperBroker:
    """Broker that fills on the next bar at close, with optional commission and execution costs."""

    def __init__(
        self,
        event_queue: Optional[Deque[Any]],
        commission: Optional[CommissionModel] = None,
        execution: Optional[ExecutionCostModel] = None,
    ):
        self.event_queue = event_queue
        self.commission = commission or CommissionModel()
        self.execution = execution or ExecutionCostModel()
        self.last_prices: Dict[str, float] = {}
        # Orders submitted this bar will be filled on the next bar when process_pending_fills is called.
        self._pending: List[OrderEvent] = []

    def update_price(self, symbol: str, price: float) -> None:
        self.last_prices[symbol] = price

    def send_order(self, order_event: OrderEvent) -> None:
        # Queue for next-bar fill
        self._pending.append(order_event)

    def process_pending_fills(self, current_dt) -> None:
        if self.event_queue is None:
            return
        remaining: List[OrderEvent] = []
        for order in self._pending:
            # Next-bar logic: only fill orders whose dt != current_dt
            if order.dt == current_dt:
                remaining.append(order)
                continue
            base_price = self.last_prices.get(order.symbol)
            if base_price is None:
                # Can't fill without a price yet; keep pending
                remaining.append(order)
                continue
            side = order.direction
            exec_price = self.execution.adjust_price(side, base_price)
            qty = order.quantity if side == "BUY" else -order.quantity
            commission = self.commission.per_trade + self.commission.per_share * abs(qty)
            fill = FillEvent(
                symbol=order.symbol,
                dt=current_dt,
                quantity=abs(qty),
                direction="BUY" if qty > 0 else "SELL",
                fill_price=float(exec_price),
                commission=float(commission),
            )
            self.event_queue.append(fill)
        self._pending = remaining
