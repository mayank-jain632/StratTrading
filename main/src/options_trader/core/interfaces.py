from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any


class DataProvider(ABC):
    @abstractmethod
    def symbols(self) -> Iterable[str]:
        ...

    @abstractmethod
    def next(self) -> bool:
        """Advance to next bar across all symbols; return False when finished."""
        ...

    @abstractmethod
    def get_bar(self, symbol: str) -> Dict[str, Any]:
        """Return dict with keys: dt, open, high, low, close, volume."""
        ...


class Strategy(ABC):
    @abstractmethod
    def on_bar(self, context: Dict[str, Any]) -> None:
        ...


class Broker(ABC):
    @abstractmethod
    def send_order(self, order_event: Any) -> None:
        ...


class RiskManager(ABC):
    @abstractmethod
    def size_order(self, symbol: str, signal_direction: str, context: Dict[str, Any]) -> int:
        ...


class Portfolio(ABC):
    @abstractmethod
    def update_from_fill(self, fill_event: Any) -> None:
        ...

    @abstractmethod
    def on_bar_close(self, dt, prices: Dict[str, float]) -> None:
        ...
