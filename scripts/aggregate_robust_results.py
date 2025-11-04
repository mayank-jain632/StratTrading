#!/usr/bin/env python3
"""Aggregate selected rolling OOS summaries into a single CSV for easy comparison."""
from pathlib import Path
import json
import pandas as pd


def read_summary(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def main():
    base = Path('runs/rolling_oos_robust')
    runs = ['aggressive_boot1000', 'aggressive_highcosts', 'aggressive_l2']
    rows = []
    for r in runs:
        summ_path = base / r / 'rolling_oos_summary.json'
        if not summ_path.exists():
            print('Missing', summ_path)
            continue
        summ = read_summary(summ_path)
        for entry in summ:
            metrics = entry.get('metrics', {})
            rows.append({
                'run': r,
                'scenario': entry.get('scenario'),
                'train_start': entry.get('train_start'),
                'train_end': entry.get('train_end'),
                'test_end': entry.get('test_end'),
                'total_return': metrics.get('total_return'),
                'cagr': metrics.get('cagr'),
                'sharpe': metrics.get('sharpe'),
                'max_drawdown': metrics.get('max_drawdown'),
                'num_trades': metrics.get('num_trades'),
                'win_rate': metrics.get('win_rate'),
            })

    df = pd.DataFrame(rows)
    out = base / 'summary.csv'
    df.to_csv(out, index=False)
    print('Wrote', out)


if __name__ == '__main__':
    main()
