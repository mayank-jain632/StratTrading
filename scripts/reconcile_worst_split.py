#!/usr/bin/env python3
"""Reconcile composite mark-to-market equity vs reconstructed equity from weighted exit PnLs for a split.
Usage: python reconcile_worst_split.py --split-dir /absolute/path/to/split_folder

Saves reconciliation CSV and PNG into the split folder.
"""
import argparse
import os
import sys
import json
import re
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_run_root(start_dir):
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.exists(os.path.join(cur, 'per_split_summary.csv')) or os.path.exists(os.path.join(cur, 'rolling_oos_summary.json')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def parse_dates_from_split_dir(split_dir):
    # Expect name like runs_YYYYMMDD_YYYYMMDD_YYYYMMDD
    base = os.path.basename(os.path.normpath(split_dir))
    m = re.search(r'runs_(\d{8})_(\d{8})_(\d{8})', base)
    if not m:
        return None
    fmt = '%Y%m%d'
    return {"train_start": datetime.strptime(m.group(1), fmt).date().isoformat(),
            "train_end": datetime.strptime(m.group(2), fmt).date().isoformat(),
            "test_end": datetime.strptime(m.group(3), fmt).date().isoformat()}


def load_weights_from_per_split(per_split_csv, dates):
    df = pd.read_csv(per_split_csv)
    # Normalize date formats possibly like 2021-03-16
    mask = (
        df['train_start'].astype(str).str.contains(dates['train_start']) &
        df['train_end'].astype(str).str.contains(dates['train_end']) &
        df['test_end'].astype(str).str.contains(dates['test_end'])
    )
    rows = df[mask]
    if len(rows) == 0:
        # Try matching using startswith (some formats)
        mask2 = (
            df['train_start'].astype(str).str.startswith(dates['train_start']) &
            df['train_end'].astype(str).str.startswith(dates['train_end']) &
            df['test_end'].astype(str).str.startswith(dates['test_end'])
        )
        rows = df[mask2]
    if len(rows) == 0:
        # fallback: try to match by train_end only
        rows = df[df['train_end'].astype(str).str.contains(dates['train_end'])]
    if len(rows) == 0:
        raise RuntimeError('Could not find matching row in per_split_summary.csv for dates: %s' % (dates,))
    row = rows.iloc[0]
    weights = {c.replace('w_', ''): float(row[c]) for c in df.columns if c.startswith('w_')}
    return weights


def find_trades_and_equities(split_dir):
    trades_files = []
    equity_files = []
    for root, dirs, files in os.walk(split_dir):
        for f in files:
            if f == 'trades.csv':
                trades_files.append(os.path.join(root, f))
            if f == 'equity_curve.csv':
                equity_files.append(os.path.join(root, f))
    return trades_files, equity_files


def strategy_name_from_trades_path(trades_path, split_dir):
    # trades path looks like .../<strategy>/<strategy-<timestamp>)/trades.csv
    p = os.path.abspath(trades_path)
    # parent is timestamp folder; parent of that is strategy folder
    strategy = os.path.basename(os.path.dirname(os.path.dirname(p)))
    # handle if structure different
    if strategy == os.path.basename(split_dir) or strategy == '':
        # fallback: one level up
        strategy = os.path.basename(os.path.dirname(p))
    return strategy


def read_trades_grouped_by_exit(trades_path):
    df = pd.read_csv(trades_path, parse_dates=['entry_dt','exit_dt'])
    if 'exit_dt' not in df.columns or 'pnl' not in df.columns:
        return pd.DataFrame(columns=['exit_date','pnl'])
    df['exit_date'] = df['exit_dt'].dt.date
    grouped = df.groupby('exit_date', as_index=False)['pnl'].sum()
    grouped['exit_date'] = pd.to_datetime(grouped['exit_date'])
    return grouped.set_index('exit_date')['pnl']


def read_equity_series(equity_path):
    df = pd.read_csv(equity_path, parse_dates=['dt'])
    if 'equity' not in df.columns:
        raise RuntimeError('equity_curve.csv missing equity column: %s' % equity_path)
    s = df.set_index(pd.to_datetime(df['dt']))['equity'].sort_index()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-dir', required=True, help='Absolute path to the split folder (runs_YYYYMMDD_YYYYMMDD_YYYYMMDD)')
    ap.add_argument('--initial-capital', type=float, default=None, help='Initial capital override (defaults to first-strategy equity first value)')
    args = ap.parse_args()

    split_dir = os.path.abspath(args.split_dir)
    if not os.path.isdir(split_dir):
        print('split-dir not found:', split_dir)
        sys.exit(2)

    run_root = find_run_root(split_dir)
    if run_root is None:
        print('Could not find run root containing per_split_summary.csv or rolling_oos_summary.json above split-dir')
        sys.exit(2)

    dates = parse_dates_from_split_dir(split_dir)
    if not dates:
        print('Could not parse dates from split dir name:', split_dir)
        sys.exit(2)

    per_split_csv = os.path.join(run_root, 'per_split_summary.csv')
    if not os.path.exists(per_split_csv):
        # fallback to rolling_oos_summary.json
        print('per_split_summary.csv not found in run root, trying rolling_oos_summary.json')
        ro_json = os.path.join(run_root, 'rolling_oos_summary.json')
        if not os.path.exists(ro_json):
            print('rolling_oos_summary.json not found either in', run_root)
            sys.exit(2)
        with open(ro_json,'r') as f:
            ro = json.load(f)
        # find matching by dates
        match = None
        for r in ro:
            if r.get('train_start','').startswith(dates['train_start']) and r.get('train_end','').startswith(dates['train_end']) and r.get('test_end','').startswith(dates['test_end']):
                match = r
                break
        if not match:
            # try by train_end
            for r in ro:
                if r.get('train_end','').startswith(dates['train_end']):
                    match = r
                    break
        if not match:
            print('Could not find matching entry in rolling_oos_summary.json for split dates', dates)
            sys.exit(2)
        weights = match.get('weights', {})
    else:
        weights = load_weights_from_per_split(per_split_csv, dates)

    print('Using weights for split (from per_split_summary):')
    for k,v in weights.items():
        print('  %s: %.4f' % (k, v))

    trades_files, equity_files = find_trades_and_equities(split_dir)
    if len(trades_files) == 0:
        print('No trades.csv files found under split dir:', split_dir)
        sys.exit(1)

    # Aggregate weighted exit PnLs
    pnl_dfs = []
    for tf in trades_files:
        strategy = strategy_name_from_trades_path(tf, split_dir)
        s = read_trades_grouped_by_exit(tf)
        w = float(weights.get(strategy, 0.0))
        if w == 0.0:
            print('Warning: weight for strategy %s is 0.0 (or missing). Found trades file: %s' % (strategy, tf))
        s = s.rename(strategy)
        s = s * w
        pnl_dfs.append(s)

    if len(pnl_dfs) == 0:
        print('No pnl series collected')
        sys.exit(1)

    combined_pnl = pd.concat(pnl_dfs, axis=1).fillna(0.0)
    combined_pnl['weighted_exit_pnl'] = combined_pnl.sum(axis=1)
    weighted_exit_pnl = combined_pnl['weighted_exit_pnl'].sort_index()

    # Reconstructed equity from exit-pnls
    # Determine initial capital
    initial_capital = args.initial_capital
    equity_series = {}
    for ef in equity_files:
        try:
            s = read_equity_series(ef)
            # strategy name
            strategy = os.path.basename(os.path.dirname(os.path.dirname(ef)))
            equity_series[strategy] = s
        except Exception as e:
            print('Skipping equity file', ef, 'error:', e)

    if initial_capital is None:
        if len(equity_series) > 0:
            # pick minimum first value
            initial_capital = min(s.iloc[0] for s in equity_series.values())
        else:
            initial_capital = 100000.0
    print('Using initial capital =', initial_capital)

    recon = weighted_exit_pnl.sort_index().cumsum() + initial_capital
    recon = recon.rename('reconstructed_equity')

    # Build composite mtm from per-strategy equity curves using weights
    # Align all strategy equities, fill forward
    strategy_equities_weighted = []
    for strat, s in equity_series.items():
        w = float(weights.get(strat, 0.0))
        # if weight zero skip
        if w == 0.0:
            continue
        # reindex to daily business days covering range
        s2 = s.copy()
        strategy_equities_weighted.append((w, s2))

    if len(strategy_equities_weighted) == 0:
        composite_mtm = None
    else:
        # union dates
        all_dates = pd.Index(sorted(set().union(*[s.index for _,s in strategy_equities_weighted])))
        df_join = pd.DataFrame(index=all_dates)
        for w,s in strategy_equities_weighted:
            # reindex and forward-fill
            s2 = s.reindex(all_dates).ffill().fillna(method='bfill')
            df_join = df_join.join((s2 * w).rename('tmp'), how='left')
            df_join = df_join.rename(columns={'tmp': f'strat_{w}'})
        # sum across columns
        composite_mtm = df_join.sum(axis=1)
        composite_mtm.name = 'composite_mtm'

    # Create reconciliation dataframe
    dfs = []
    dfs.append(recon)
    if composite_mtm is not None:
        dfs.append(composite_mtm)
    recon_df = pd.concat(dfs, axis=1)
    recon_df = recon_df.sort_index()
    # fill missing reconstructed equity forward/back
    recon_df['reconstructed_equity'] = recon_df['reconstructed_equity'].ffill().bfill()
    if 'composite_mtm' in recon_df.columns:
        recon_df['diff'] = recon_df['composite_mtm'] - recon_df['reconstructed_equity']
    else:
        recon_df['diff'] = np.nan

    out_csv = os.path.join(split_dir, 'reconciliation_worst_split.csv')
    recon_df.to_csv(out_csv, index_label='date')
    print('WROTE', out_csv)

    # Plot
    plt.figure(figsize=(10,6))
    if 'composite_mtm' in recon_df.columns:
        plt.plot(recon_df.index, recon_df['composite_mtm'], label='composite_mtm')
    plt.plot(recon_df.index, recon_df['reconstructed_equity'], label='reconstructed_from_weighted_exits')
    plt.legend()
    plt.title('Reconciliation: composite MTM vs reconstructed (split: %s)' % os.path.basename(split_dir))
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True)
    out_png = os.path.join(split_dir, 'reconciliation_worst_split.png')
    plt.savefig(out_png, bbox_inches='tight')
    print('WROTE', out_png)

    # Print top contributing negative weighted trades
    # Use combined_pnl to get per-strategy exit pnls; now create flattened weighted trade list by reading trades and multiplying pnl by weight
    weighted_trades = []
    for tf in trades_files:
        strat = strategy_name_from_trades_path(tf, split_dir)
        w = float(weights.get(strat, 0.0))
        try:
            df = pd.read_csv(tf, parse_dates=['entry_dt','exit_dt'])
        except Exception as e:
            continue
        if 'pnl' in df.columns and not df.empty:
            df['weighted_pnl'] = df['pnl'] * w
            for _,row in df.iterrows():
                weighted_trades.append({'strategy': strat, 'exit_dt': row.get('exit_dt'), 'symbol': row.get('symbol'), 'pnl': row.get('pnl'), 'weighted_pnl': row.get('weighted_pnl')})
    wtr_df = pd.DataFrame(weighted_trades)
    if not wtr_df.empty:
        worst = wtr_df.sort_values('weighted_pnl').head(10)
        worst_csv = os.path.join(split_dir, 'worst_weighted_trades.csv')
        worst.to_csv(worst_csv, index=False)
        print('WROTE', worst_csv)
        print('Top negative weighted trades:')
        print(worst.to_string(index=False))
    else:
        print('No trades to report in worst split')

if __name__ == '__main__':
    main()
