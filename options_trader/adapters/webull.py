"""
Stage 2 placeholder for Webull paper trading adapter.

Design intent:
- Provide an adapter that implements the Broker interface and routes orders to Webull paper trading.
- Manage authentication and order placement/cancel, and account/position sync back to Portfolio.

For now, this is a stub so Stage 1 remains broker-agnostic.
"""
from __future__ import annotations
from typing import Deque, Any, Optional

from ..core.interfaces import Broker
from ..core.events import OrderEvent


class WebullPaperBroker(Broker):
    def __init__(self, event_queue: Optional[Deque[Any]] = None, credentials: Optional[dict] = None):
        self.event_queue = event_queue
        self.credentials = credentials or {}
        # TODO: Implement auth/session and order routing to Webull in Stage 2

    def send_order(self, order_event: OrderEvent) -> None:
        raise NotImplementedError("Webull integration will be implemented in Stage 2")
