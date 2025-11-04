# Automated Stock Trading Machine (Backtesting & Paper Trading)

Modular platform for building, backtesting, and paper-trading automated stock strategies. Designed for iterative development:

This project focuses on:

- Data providers (synthetic for offline tests; yfinance optional)
- Strategy development (example: Moving Average Crossover)
- Event-driven backtester, paper broker, and portfolio tracking
- Metrics and reports
- CLI runner with YAML configs

## Quick start

1. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run a sample backtest

```bash
python -m options_trader.cli.run_backtest --config configs/sample_synthetic.yaml
```

Outputs will be saved under `runs/<timestamp>/`.

## Structure

- `src/options_trader/core/` — events, backtester, broker, portfolio, base interfaces
- `src/options_trader/data/` — data providers (synthetic now; extendable)
- `src/options_trader/strategies/` — strategy implementations
- `src/options_trader/metrics/` — performance metrics and reporting
- `src/options_trader/cli/` — command-line runner
- `configs/` — YAML config examples
- `tests/` — unit and smoke tests

## Notes

- For determinism in tests, the synthetic data provider accepts a random seed.
- The broker currently fills orders at the same-bar close price (optimistic). You can switch to next-bar fill easily.
- Stage 2 will add a Webull paper trading adapter.

## License

MIT
