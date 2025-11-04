"""Aggregate rolling_oos_summary.json into a CSV of per-split metrics.

Saves: <out_dir>/compare_splits.csv
"""
import os
import json
import sys
import pandas as pd

if len(sys.argv) < 2:
    print('Usage: python scripts/aggregate_rolling_summary.py <rolling_oos_run_dir>')
    sys.exit(1)

run_dir = sys.argv[1]
summary_path = os.path.join(run_dir, 'rolling_oos_summary.json')
if not os.path.exists(summary_path):
    print('No summary at', summary_path)
    sys.exit(1)

with open(summary_path, 'r') as f:
    data = json.load(f)

rows = []
for rec in data:
    metrics = rec.get('metrics', {}) or {}
    rows.append({
        'scenario': rec.get('scenario'),
        'train_start': rec.get('train_start'),
        'train_end': rec.get('train_end'),
        'test_end': rec.get('test_end'),
        'total_return': metrics.get('total_return'),
        'cagr': metrics.get('cagr'),
        'sharpe': metrics.get('sharpe'),
        'max_drawdown': metrics.get('max_drawdown'),
    })

out_csv = os.path.join(run_dir, 'compare_splits.csv')
pd.DataFrame(rows).to_csv(out_csv, index=False)
print('Wrote', out_csv)
