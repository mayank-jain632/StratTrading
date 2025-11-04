"""Collect and compare experiments from runs/rolling_oos_maxcap95 and save CSV + combined plot.
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = "runs/rolling_oos_maxcap95"
folders = sorted([p for p in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, p))])
rows = []
traces = []
labels = []
for f in folders:
    summ = os.path.join(BASE, f, 'rolling_oos_summary.json')
    comp = os.path.join(BASE, f, 'scenario_0', 'composite_20240101_20241226.csv')
    if not os.path.exists(summ):
        continue
    with open(summ, 'r') as fh:
        data = json.load(fh)
    rec = data[0]
    metrics = rec.get('metrics', {})
    rows.append({
        'experiment': f,
        'train_start': rec.get('train_start'),
        'train_end': rec.get('train_end'),
        'test_end': rec.get('test_end'),
        'total_return': metrics.get('total_return'),
        'cagr': metrics.get('cagr'),
        'sharpe': metrics.get('sharpe'),
        'max_drawdown': metrics.get('max_drawdown'),
    })
    if os.path.exists(comp):
        df = pd.read_csv(comp, parse_dates=['dt']).set_index('dt')
        first = df['equity'].iloc[0]
        traces.append(df['equity'] / first)
        labels.append(f)

# write CSV
out_csv = os.path.join(BASE, 'compare_l2_max95.csv')
pd.DataFrame(rows).to_csv(out_csv, index=False)
print('Wrote', out_csv)

# plot combined
if traces:
    plt.figure(figsize=(12,6))
    for s, lab in zip(traces, labels):
        plt.plot(s.index, s.values, label=lab)
    plt.title('Rolling OOS (max cap 0.95): normalized composite equity')
    plt.xlabel('Date')
    plt.ylabel('Normalized equity')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out = os.path.join(BASE, 'combined_l2_max95.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print('Wrote', out)
else:
    print('No composite traces found')
