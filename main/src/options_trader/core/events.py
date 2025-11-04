from dataclasses import dataclass
from typing import Optional
from datetime import datetime


class Event:
    pass


@dataclass
class MarketEvent(Event):
    symbol: str
    dt: datetime
    price: float


@dataclass
class SignalEvent(Event):
    symbol: str
    dt: datetime
    direction: str  # 'LONG' or 'EXIT'
    strength: float = 1.0


@dataclass
class OrderEvent(Event):
    symbol: str
    dt: datetime
    order_type: str  # 'MKT' or 'LMT'
    quantity: int
    direction: str  # 'BUY' or 'SELL'
    limit_price: Optional[float] = None


@dataclass
class FillEvent(Event):
    symbol: str
    dt: datetime
    quantity: int
    direction: str  # 'BUY' or 'SELL'
    fill_price: float
    commission: float = 0.0
