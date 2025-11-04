from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Iterable, List, Optional
from datetime import datetime
import os

import pandas as pd
import yfinance as yf

from ..core.interfaces import DataProvider


@dataclass
class YFConfig:
    symbols: List[str]
    start: str  # 'YYYY-MM-DD'
    end: str
    interval: str = "1d"
    auto_adjust: bool = True
    cache_dir: Optional[str] = None
    use_cache: bool = True


class YFinanceDataProvider(DataProvider):
    def __init__(self, config: YFConfig):
        self._symbols = list(dict.fromkeys(config.symbols))
        self._interval = config.interval
        self._auto_adjust = config.auto_adjust
        self._cache_dir = config.cache_dir
        self._use_cache = bool(config.use_cache)
        if self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)

        self._bars: Dict[str, pd.DataFrame] = {}
        for s in self._symbols:
            self._bars[s] = self._load_symbol(s, config.start, config.end)

        # Align index intersection to ensure all symbols have data for each step
        idx = None
        for df in self._bars.values():
            ix = pd.DatetimeIndex(df["dt"]).tz_localize(None)
            idx = ix if idx is None else idx.intersection(ix)
        for s, df in self._bars.items():
            df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize(None)
            self._bars[s] = df[df["dt"].isin(idx)].reset_index(drop=True)

        self._index = sorted(idx) if idx is not None else []
        self._ptr = -1

    def _cache_path(self, symbol: str, start: str, end: str) -> Optional[str]:
        if not self._cache_dir:
            return None
        fname = f"{symbol}_{start}_{end}_{self._interval}_{'adj' if self._auto_adjust else 'raw'}.csv"
        return os.path.join(self._cache_dir, fname)

    def _load_symbol(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        cache = self._cache_path(symbol, start, end)
        if cache and self._use_cache and os.path.exists(cache):
            df = pd.read_csv(cache, parse_dates=["dt"])
            # Validate numeric columns; if invalid, force re-download
            bad = False
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    bad = True
                    break
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[col].isnull().any():
                    bad = True
            if not bad:
                return df
            # Fall through to re-download
        # Download from yfinance
        df = yf.download(symbol, start=start, end=end, interval=self._interval, auto_adjust=self._auto_adjust, progress=False)
        if df.empty:
            raise ValueError(f"No data for {symbol} from {start} to {end}")

        # If multi-index columns (e.g., multi-ticker structure), slice to this symbol
        if isinstance(df.columns, pd.MultiIndex):
            # Try common layouts: (field, ticker) or (ticker, field)
            lvls = df.columns.names
            try:
                if symbol in df.columns.get_level_values(1):
                    df = df.xs(symbol, axis=1, level=1)
                elif symbol in df.columns.get_level_values(0):
                    df = df.xs(symbol, axis=1, level=0)
            except Exception:
                # Fallback: collapse to first level if unique
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        # Now normalize column names
        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        df = df.rename(columns=rename_map).reset_index().rename(columns={"Date": "dt"})
        # Some versions may use lowercase already; ensure we have needed cols
        cols = set(df.columns)
        needed = {"dt", "open", "high", "low", "close", "volume"}
        if not needed.issubset(cols):
            # Try lowercase mapping
            df = df.rename(columns={k.lower(): v for k, v in rename_map.items()})
        # Select columns in order
        df = df[["dt", "open", "high", "low", "close", "volume"]]
        # Ensure numeric dtypes
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if cache:
            df.to_csv(cache, index=False)
        return df

    def symbols(self) -> Iterable[str]:
        return self._symbols

    def next(self) -> bool:
        self._ptr += 1
        return self._ptr < len(self._index)

    def get_bar(self, symbol: str) -> Dict[str, Any]:
        df = self._bars[symbol]
        row = df.iloc[self._ptr]

        def _sc(v):
            # Convert single-element Series to scalar
            if hasattr(v, "iloc"):
                try:
                    return v.iloc[0]
                except Exception:
                    pass
            return v

        dt_val = _sc(row["dt"])
        ts = pd.to_datetime(dt_val)
        if isinstance(ts, pd.Timestamp):
            try:
                ts = ts.tz_localize(None)
            except Exception:
                # Already tz-naive
                pass
        dt_py = ts.to_pydatetime()

        return {
            "dt": dt_py,
            "open": float(_sc(row["open"])),
            "high": float(_sc(row["high"])),
            "low": float(_sc(row["low"])),
            "close": float(_sc(row["close"])),
            "volume": int(_sc(row["volume"])),
        }
