#!/usr/bin/env python3
"""Reconstruct composite equity from per-strategy weighted trade PnLs for a single split
and compare to the composite CSV produced by `rolling_oos.py`.

Saves reconciliation CSV under the run folder and prints diagnostics.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys


def main():
    base = Path('runs/rolling_oos_new_full_weighted')
    summary_path = base / 'rolling_oos_summary.json'
    if not summary_path.exists():
        print('Summary not found at', summary_path)
        sys.exit(1)

    summary = json.loads(summary_path.read_text())
    # pick the first scenario_0 entry (train 2024-01-01 -> test 2024-09-27)
    entry = None
    for e in summary:
        if e.get('scenario') == 'scenario_0' and e.get('train_start') == '2024-01-01' and e.get('test_end') == '2024-09-27':
            entry = e
            break
    if entry is None:
        entry = summary[0]
        print('Requested split not found; falling back to first entry:', entry.get('train_start'), entry.get('test_end'))

    weights = entry.get('weights', {})
    # locate composite CSV
    scenario = entry.get('scenario')
    stamp = f"runs_{entry.get('train_start').replace('-', '')}_{pd.to_datetime(entry.get('train_end')).strftime('%Y%m%d')}_{pd.to_datetime(entry.get('test_end')).strftime('%Y%m%d')}"
    runs_dir = base / scenario / stamp
    if not runs_dir.exists():
        # fallback: search for the runs_* folder under scenario
        cand = list((base / scenario).glob('runs_*'))
        if not cand:
            print('No runs_* folder found under', base / scenario)
            sys.exit(1)
        runs_dir = cand[0]

    composite_csv = base / scenario / f"composite_{entry.get('train_start').replace('-', '')}_{entry.get('test_end').replace('-', '')}.csv"
    if not composite_csv.exists():
        # try alternative name
        composite_csv = base / scenario / f"composite_{pd.to_datetime(entry.get('train_start')).strftime('%Y%m%d')}_{pd.to_datetime(entry.get('test_end')).strftime('%Y%m%d')}.csv"
    if not composite_csv.exists():
        print('Composite CSV not found for split at', composite_csv)
        sys.exit(1)

    comp_df = pd.read_csv(composite_csv, parse_dates=['dt'])
    comp_df = comp_df.set_index('dt').sort_index()

    # Aggregate weighted trade PnLs by exit date
    per_day = {}
    for strat, w in weights.items():
        strat_dir = runs_dir / strat
        if not strat_dir.exists():
            # try nested
            possible = list(runs_dir.glob(f"{strat}*"))
            if possible:
                strat_dir = possible[0]
        # find trades.csv under strat_dir (recursively)
        trades_files = list(strat_dir.rglob('trades.csv'))
        if not trades_files:
            # no trades for this strategy
            continue
        # pick the most recent trades.csv
        trades_files = sorted(trades_files, key=lambda p: p.stat().st_mtime, reverse=True)
        trades = pd.read_csv(trades_files[0], parse_dates=['exit_dt'])
        if trades.empty:
            continue
        # ensure pnl column
        pnl_col = 'pnl' if 'pnl' in trades.columns else ('profit' if 'profit' in trades.columns else None)
        if pnl_col is None:
            continue
        trades['exit_date'] = pd.to_datetime(trades['exit_dt']).dt.normalize()
        # scale pnl by weight (weights are floats summing to 1)
        trades['weighted_pnl'] = trades[pnl_col].astype(float) * float(w)
        for dt, grp in trades.groupby('exit_date'):
            per_day.setdefault(dt, 0.0)
            per_day[dt] += grp['weighted_pnl'].sum()

    # Build DataFrame of per-day weighted pnl (only test period days)
    pd_series = pd.Series(per_day).sort_index()
    pd_series = pd_series.reindex(comp_df.index, fill_value=0.0)

    starting_cash = 100000.0
    per_day_return = pd_series / starting_cash
    recon_eq = (1 + per_day_return).cumprod() * starting_cash

    recon_df = pd.DataFrame({
        'composite_equity': comp_df['equity'],
        'reconstructed_equity': recon_eq,
        'per_day_weighted_pnl': pd_series,
    })
    recon_df['diff'] = recon_df['composite_equity'] - recon_df['reconstructed_equity']

    out_csv = base / 'reconciliation_split1.csv'
    recon_df.to_csv(out_csv, index_label='dt')

    # Quick diagnostics
    final_comp = recon_df['composite_equity'].iloc[-1]
    final_recon = recon_df['reconstructed_equity'].iloc[-1]
    mse = np.mean((recon_df['composite_equity'] - recon_df['reconstructed_equity']) ** 2)
    max_abs = np.max(np.abs(recon_df['diff']))
    corr = np.corrcoef((recon_df['composite_equity'].pct_change().fillna(0.0), recon_df['reconstructed_equity'].pct_change().fillna(0.0)))[0,1]

    print('Reconciliation saved to:', out_csv)
    print('Final composite equity:', final_comp)
    print('Final reconstructed equity:', final_recon)
    print('Final diff (abs):', abs(final_comp - final_recon))
    print('MSE:', mse)
    print('Max abs diff:', max_abs)
    print('Return corr (daily pct):', corr)


if __name__ == '__main__':
    main()
