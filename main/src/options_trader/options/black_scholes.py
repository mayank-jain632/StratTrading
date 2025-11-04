from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple

from scipy.stats import norm


@dataclass
class BSResult:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def _d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * math.sqrt(T)


def black_scholes_call(S: float, K: float, r: float, sigma: float, T: float) -> BSResult:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return BSResult(price=max(S - K, 0.0), delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
    d1 = _d1(S, K, r, sigma, T)
    d2 = _d2(d1, sigma, T)
    price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100.0
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0
    rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100.0
    return BSResult(price, delta, gamma, vega, theta, rho)


def black_scholes_put(S: float, K: float, r: float, sigma: float, T: float) -> BSResult:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return BSResult(price=max(K - S, 0.0), delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
    d1 = _d1(S, K, r, sigma, T)
    d2 = _d2(d1, sigma, T)
    price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    delta = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100.0
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365.0
    rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100.0
    return BSResult(price, delta, gamma, vega, theta, rho)
