"""Plot and compare composite equity curves across multiple rolling-OOS experiments.

Saves:
 - runs/rolling_oos_compare_l2_vol/combined_all_experiments.png
 - runs/rolling_oos_compare_l2_vol/compare_all_experiments.csv  (if not already created)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = "runs/rolling_oos_compare_l2_vol"
EXPS = [
    ("baseline", "baseline"),
    ("l2_0.005", "l2_grid_0.005"),
    ("l2_0.01", "l2_grid_0.01"),
    ("l2_0.02", "l2_grid_0.02"),
    ("l2_0.05", "l2_grid_0.05"),
    ("l2_only_0.02", "l2_only"),
    ("vol_only", "vol_only"),
    ("l2_plus_vol_0.02", "l2_plus_vol"),
]

traces = []
labels = []
for label, folder in EXPS:
    comp_path = os.path.join(BASE, folder, "scenario_0", "composite_20240101_20241226.csv")
    if not os.path.exists(comp_path):
        print(f"Missing composite for {label}: {comp_path}")
        continue
    df = pd.read_csv(comp_path, parse_dates=["dt"]).set_index("dt")
    # normalize to 1.0 at start for fair comparison (relative growth)
    first = df["equity"].iloc[0]
    df["norm"] = df["equity"] / first
    traces.append(df["norm"])
    labels.append(label)

if not traces:
    raise SystemExit("No composites found; nothing to plot")

plt.figure(figsize=(12, 6))
for s, lab in zip(traces, labels):
    plt.plot(s.index, s.values, label=lab)
plt.title("Rolling OOS: Composite Equity - All Experiments (normalized)")
plt.xlabel("Date")
plt.ylabel("Normalized Equity (start=1.0)")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
out = os.path.join(BASE, "combined_all_experiments.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
print("Saved combined plot to:", out)

# Additionally, generate a simple CSV summary by reading rolling_oos_summary.json from each experiment
import json
rows = []
for label, folder in EXPS:
    summ_path = os.path.join(BASE, folder, "rolling_oos_summary.json")
    if not os.path.exists(summ_path):
        continue
    with open(summ_path, 'r') as f:
        data = json.load(f)
    # take first record (single-split runs)
    rec = data[0]
    metrics = rec.get('metrics', {})
    rows.append({
        'experiment': label,
        'folder': folder,
        'train_start': rec.get('train_start'),
        'train_end': rec.get('train_end'),
        'test_end': rec.get('test_end'),
        'total_return': metrics.get('total_return'),
        'cagr': metrics.get('cagr'),
        'sharpe': metrics.get('sharpe'),
        'max_drawdown': metrics.get('max_drawdown'),
    })

csv_out = os.path.join(BASE, 'compare_all_experiments.csv')
df_sum = pd.DataFrame(rows)
df_sum.to_csv(csv_out, index=False)
print('Wrote comparison CSV to:', csv_out)
