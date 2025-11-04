from __future__ import annotations
import argparse
import itertools
import json
import os
import time
from typing import Any, Dict, List

import yaml
import pandas as pd

from ..metrics.performance import compute_performance
from .run_backtest import build_from_config


def _expand_params(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a dict of params where list-valued items are treated as a grid."""
    keys = list(params.keys())
    values_product = []
    for k in keys:
        v = params[k]
        if isinstance(v, list):
            values_product.append(v)
        else:
            values_product.append([v])
    combos = []
    for combo in itertools.product(*values_product):
        combos.append({k: combo[i] for i, k in enumerate(keys)})
    return combos


def run_grid_search(cfg_path: str, topn: int = 5) -> str:
    with open(cfg_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    strat_cfg = base_cfg.get("strategy", {})
    params = strat_cfg.get("params", {})
    combos = _expand_params(params)
    if not combos:
        combos = [params]

    # Output directory for grid search
    ts = time.strftime("%Y%m%d-%H%M%S")
    strategy_id = strat_cfg.get("type", "strategy")
    out_dir = os.path.join(base_cfg.get("output_dir", "runs"), f"{strategy_id}-grid-{ts}")
    os.makedirs(out_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for i, combo in enumerate(combos, start=1):
        cfg = json.loads(json.dumps(base_cfg))  # deep copy via json
        cfg.setdefault("strategy", {}).setdefault("params", {})
        cfg["strategy"]["params"].update(combo)

        bt = build_from_config(cfg)
        result = bt.run()
        eq = result["equity_curve"]
        trades = result.get("trades", [])
        summ = compute_performance(eq, trades)

        row = {
            "run_dir": result["run_dir"],
            **{f"param_{k}": v for k, v in combo.items()},
            **summ.to_dict(),
        }
        results.append(row)

    # Save results CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "results.csv")
    df.to_csv(csv_path, index=False)

    # Save top-N configs
    if topn and topn > 0:
        # Rank by Sharpe then total_return
        df_sorted = df.sort_values(["sharpe", "total_return"], ascending=[False, False]).head(topn)
        top_dir = os.path.join(out_dir, "top_configs")
        os.makedirs(top_dir, exist_ok=True)
        for idx, row in df_sorted.iterrows():
            cfg = json.loads(json.dumps(base_cfg))
            cfg.setdefault("strategy", {}).setdefault("params", {})
            for k, v in row.items():
                if k.startswith("param_"):
                    cfg["strategy"]["params"][k.replace("param_", "")] = v
            with open(os.path.join(top_dir, f"config_{idx}.yaml"), "w") as f:
                yaml.safe_dump(cfg, f)

    print(f"Grid search complete. Results saved to: {csv_path}")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Grid search over strategy parameters")
    parser.add_argument("--config", required=True, help="Base YAML config with strategy.params ranges")
    parser.add_argument("--topn", type=int, default=5, help="Save top-N configs by Sharpe")
    args = parser.parse_args(argv)

    run_grid_search(args.config, topn=args.topn)


if __name__ == "__main__":
    main()
