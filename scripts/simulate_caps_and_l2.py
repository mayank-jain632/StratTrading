#!/usr/bin/env python3
"""Simulate applying a max_strategy_weight cap and optimizer_l2 blending to per-split weights.
Produces a per_split_summary_simulated.csv with new weights and composite metrics built from per-strategy equity_curve.csv files.

Usage: python simulate_caps_and_l2.py --run-dir /path/to/run_folder --cap 0.5 --l2 0.1
"""
import argparse
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime


def herfindahl_from_weights(wdict):
    arr = np.array(list(wdict.values()))
    if arr.sum() == 0:
        return 0.0
    w = arr / arr.sum()
    return float((w**2).sum())


def compute_metrics_from_equity(series):
    # series: pd.Series indexed by datetime, values equity
    s = series.sort_index()
    if s.empty:
        return {'total_return': np.nan, 'cagr': np.nan, 'max_drawdown': np.nan, 'sharpe': np.nan}
    initial = s.iloc[0]
    final = s.iloc[-1]
    total_return = (final / initial) - 1.0
    days = (s.index[-1] - s.index[0]).days or 1
    years = days / 365.25
    if years > 0:
        cagr = (final / initial) ** (1.0 / years) - 1.0
    else:
        cagr = np.nan
    # max drawdown
    roll_max = s.cummax()
    drawdown = (s - roll_max) / roll_max
    max_dd = drawdown.min()
    # daily returns
    daily = s.pct_change().dropna()
    if len(daily) >= 2:
        ann_mean = daily.mean() * 252
        ann_std = daily.std() * np.sqrt(252)
        sharpe = ann_mean / ann_std if ann_std > 0 else np.nan
    else:
        sharpe = np.nan
    return {'total_return': total_return, 'cagr': cagr, 'max_drawdown': max_dd, 'sharpe': sharpe}


def find_split_folders(run_dir):
    return sorted([p for p in glob.glob(os.path.join(run_dir, 'scenario_*','runs_*')) if os.path.isdir(p)])


def load_strategy_equities_for_split(split_dir):
    equities = {}
    for ef in glob.glob(os.path.join(split_dir, '**','equity_curve.csv'), recursive=True):
        # infer strategy
        # parent of parent is strategy folder
        try:
            parts = os.path.normpath(ef).split(os.sep)
            # find index of split_dir
            # fallback: take two levels up
            strategy = os.path.basename(os.path.dirname(os.path.dirname(ef)))
        except Exception:
            strategy = os.path.basename(os.path.dirname(ef))
        try:
            df = pd.read_csv(ef, parse_dates=['dt'])
            if 'equity' in df.columns:
                s = df.set_index(pd.to_datetime(df['dt']))['equity'].sort_index()
                equities[strategy] = s
        except Exception:
            continue
    return equities


def apply_cap_and_l2(orig_weights, cap=0.5, l2=0.1):
    # orig_weights: dict strategy->weight
    # cap: maximum allowed weight
    # l2: blending toward uniform (0-1)
    w = {k: max(0.0, float(v)) for k,v in orig_weights.items()}
    # apply cap
    w_capped = {k: min(v, cap) for k,v in w.items()}
    total = sum(w_capped.values())
    n = max(1, len(w_capped))
    if total == 0:
        # fallback: uniform
        w_norm = {k: 1.0/n for k in w_capped}
    else:
        w_norm = {k: v/total for k,v in w_capped.items()}
    # blend toward uniform
    uni = {k: 1.0/n for k in w_norm}
    w_blend = {k: (1.0 - l2) * w_norm[k] + l2 * uni[k] for k in w_norm}
    # final normalize
    s = sum(w_blend.values()) or 1.0
    w_final = {k: v/s for k,v in w_blend.items()}
    return w_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--cap', type=float, default=0.5)
    ap.add_argument('--l2', type=float, default=0.1)
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    per_split_csv = os.path.join(run_dir, 'per_split_summary.csv')
    if not os.path.exists(per_split_csv):
        print('per_split_summary.csv not found in', run_dir)
        return
    df = pd.read_csv(per_split_csv)
    split_folders = find_split_folders(run_dir)
    rows = []
    for split in split_folders:
        base = os.path.basename(split)
        parts = base.split('_')
        if len(parts) < 4:
            continue
        train_start, train_end, test_end = parts[1], parts[2], parts[3]
        # find row
        mask = (df['train_start'].astype(str).str.contains(train_start)) & (df['train_end'].astype(str).str.contains(train_end)) & (df['test_end'].astype(str).str.contains(test_end))
        row = df[mask]
        if row.empty:
            row = df[df['train_end'].astype(str).str.contains(train_end)]
        if row.empty:
            # skip
            continue
        row = row.iloc[0]
        # extract orig weights
        wcols = [c for c in df.columns if c.startswith('w_')]
        orig_weights = {c.replace('w_',''): float(row[c]) for c in wcols}
        # apply cap and l2
        new_weights = apply_cap_and_l2(orig_weights, cap=args.cap, l2=args.l2)
        # load equities
        equities = load_strategy_equities_for_split(split)
        # build composite mtm
        # determine date union
        if equities:
            all_dates = sorted(set().union(*[s.index for s in equities.values()]))
            index = pd.DatetimeIndex(all_dates)
            comp = pd.Series(0.0, index=index)
            initial_candidates = []
            for strat,s in equities.items():
                weight = new_weights.get(strat, 0.0)
                s2 = s.reindex(index).ffill().bfill()
                comp = comp.add(s2 * weight, fill_value=0.0)
                initial_candidates.append(s.iloc[0])
            if initial_candidates:
                initial_cap = min(initial_candidates)
            else:
                initial_cap = 100000.0
            metrics = compute_metrics_from_equity(comp)
            herf = herfindahl_from_weights(new_weights)
            rows.append({'split_folder': split, 'train_start': row['train_start'], 'train_end': row['train_end'], 'test_end': row['test_end'], 'herfindahl': herf, 'max_weight': max(new_weights.values() if new_weights else [0.0]), 'total_return': metrics['total_return'], 'cagr': metrics['cagr'], 'max_drawdown': metrics['max_drawdown'], 'sharpe': metrics['sharpe']})
        else:
            continue
    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(run_dir, f'per_split_summary_simulated_cap{args.cap}_l2{args.l2}.csv')
    out_df.to_csv(out_csv, index=False)
    print('WROTE', out_csv)
    # summary
    if not out_df.empty:
        summary = out_df.agg({'total_return':['mean','median'],'max_drawdown':['mean','median'],'sharpe':['mean','median']}).T
        summary_csv = os.path.join(run_dir, f'simulated_summary_cap{args.cap}_l2{args.l2}.csv')
        summary.to_csv(summary_csv)
        print('WROTE', summary_csv)

if __name__ == '__main__':
    main()
