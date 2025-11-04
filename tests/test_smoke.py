import os
import yaml

from options_trader.cli.run_backtest import main as run_main


def test_smoke(tmp_path, monkeypatch):
    cfg = {
        "output_dir": str(tmp_path),
        "starting_cash": 100000,
        "data": {
            "type": "synthetic",
            "symbols": ["SPY"],
            "start": "2021-01-01",
            "end": "2021-03-31",
            "freq": "1D",
            "seed": 1,
            "start_price": 100.0,
            "drift": 0.0002,
            "vol": 0.01,
        },
        "strategy": {
            "type": "ma_cross",
            "symbols": ["SPY"],
            "fast": 5,
            "slow": 15,
        },
        "risk": {
            "fraction": 0.1,
            "max_positions": 3,
        },
        "broker": {
            "per_trade": 0.0,
            "per_share": 0.0,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # Run the CLI with our temp config
    run_main(["--config", str(cfg_path)])

    # Check outputs exist - find the timestamped run directory
    runs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name != "__pycache__"]
    assert runs, "No run directory created"
    run_dir = runs[0]
    assert (run_dir / "equity_curve.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "backtest_plots.png").exists()
