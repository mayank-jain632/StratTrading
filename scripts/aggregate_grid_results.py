#!/usr/bin/env python3
"""Aggregate rolling_oos_summary.json files from the grid runs and produce a CSV summary."""
from pathlib import Path
import json
import pandas as pd


def read_summary(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def main():
    base = Path('runs/rolling_oos_grid')
    runs = ['relaxcap_only', 'relaxcap_nomin', 'aggressive']
    rows = []
    for r in runs:
        summ = read_summary(base / r / 'rolling_oos_summary.json')
        for entry in summ:
            row = {
                'run': r,
                'scenario': entry.get('scenario'),
                'train_start': entry.get('train_start'),
                'train_end': entry.get('train_end'),
                'test_end': entry.get('test_end'),
            }
            metrics = entry.get('metrics', {})
            for k in ['total_return','cagr','sharpe','max_drawdown','num_trades','win_rate','profit_factor']:
                row[k] = metrics.get(k)
            rows.append(row)

    df = pd.DataFrame(rows)
    out = base / 'summary.csv'
    df.to_csv(out, index=False)
    print('Wrote', out)


if __name__ == '__main__':
    main()
