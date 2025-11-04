from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Iterable, List
from datetime import datetime

import numpy as np
import pandas as pd

from ..core.interfaces import DataProvider


@dataclass
class SyntheticConfig:
    symbols: List[str]
    start: str  # ISO date
    end: str
    freq: str = "1D"
    seed: int = 42
    start_price: float = 100.0
    drift: float = 0.0002
    vol: float = 0.01


class SyntheticDataProvider(DataProvider):
    def __init__(self, config: SyntheticConfig):
        self._symbols = list(config.symbols)
        self._rng = np.random.default_rng(config.seed)
        self._index = pd.date_range(config.start, config.end, freq=config.freq)
        n = len(self._index)
        self._bars: Dict[str, pd.DataFrame] = {}
        for s in self._symbols:
            rets = config.drift + config.vol * self._rng.standard_normal(n)
            prices = config.start_price * np.exp(np.cumsum(rets))
            df = pd.DataFrame(index=self._index)
            df["open"] = prices
            df["high"] = df["open"] * (1 + np.abs(self._rng.normal(0, config.vol/2, n)))
            df["low"] = df["open"] * (1 - np.abs(self._rng.normal(0, config.vol/2, n)))
            df["close"] = (df["open"] + df["high"] + df["low"]) / 3.0
            df["volume"] = self._rng.integers(1e5, 5e5, n)
            df.reset_index(inplace=True)
            df.rename(columns={"index": "dt"}, inplace=True)
            self._bars[s] = df
        self._ptr = -1

    @property
    def current_dt(self) -> datetime:
        if self._ptr < 0:
            return self._index[0].to_pydatetime()
        return self._index[self._ptr].to_pydatetime()

    def symbols(self) -> Iterable[str]:
        return self._symbols

    def next(self) -> bool:
        self._ptr += 1
        return self._ptr < len(self._index)

    def get_bar(self, symbol: str) -> Dict[str, Any]:
        row = self._bars[symbol].iloc[self._ptr]
        return {
            "dt": row["dt"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
        }
