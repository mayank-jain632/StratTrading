#!/usr/bin/env python3
"""Compare rolling OOS runs and produce aggregated diagnostics.
Usage: python compare_diagnostics.py --runs <path1> <path2> [--labels label1 label2]
Saves comparison CSV and plots into the first run folder (as report outputs).
"""
import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_summary(run_dir):
    f = os.path.join(run_dir, 'rolling_oos_summary.json')
    if not os.path.exists(f):
        raise FileNotFoundError(f)
    with open(f,'r') as fh:
        data = json.load(fh)
    rows = []
    for r in data:
        if r.get('error'):
            continue
        metrics = r.get('metrics') or {}
        weights = r.get('weights') or {}
        # flatten metrics
        row = {'run': os.path.basename(run_dir), 'train_start': r.get('train_start'), 'train_end': r.get('train_end'), 'test_end': r.get('test_end')}
        for k,v in metrics.items():
            row[k] = v
        # weights as columns
        for s,w in weights.items():
            row[f'w_{s}'] = w
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def agg_stats(df):
    # metrics of interest
    cols = ['total_return','cagr','sharpe','max_drawdown','volatility','num_trades','exposure_time']
    stats = {}
    for c in cols:
        if c in df.columns:
            stats[c+'_mean'] = float(df[c].mean())
            stats[c+'_median'] = float(df[c].median())
            stats[c+'_std'] = float(df[c].std())
            stats[c+'_count'] = int(df[c].count())
        else:
            stats[c+'_mean'] = np.nan
    return stats


def herfindahl_from_row(row):
    wcols = [c for c in row.index if c.startswith('w_')]
    ws = np.array([row[c] if pd.notna(row[c]) else 0.0 for c in wcols], dtype=float)
    if ws.sum() == 0:
        return np.nan
    w = ws/ (ws.sum() + 1e-12)
    return float((w**2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--labels', nargs='*', help='Optional labels matching runs')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    run_dirs = [os.path.abspath(p) for p in args.runs]
    labels = args.labels or [os.path.basename(p) for p in run_dirs]
    # load each
    dfs = []
    for rd,lab in zip(run_dirs, labels):
        try:
            df = load_summary(rd)
        except Exception as e:
            print('Skipping', rd, 'error:', e)
            continue
        df['run_label'] = lab
        # compute herfindahl
        df['herfindahl'] = df.apply(herfindahl_from_row, axis=1)
        dfs.append((lab, rd, df))
    if not dfs:
        print('No run data found')
        return
    # produce per-run aggregate CSV
    agg_list = []
    for lab,rd,df in dfs:
        stats = agg_stats(df)
        stats['label'] = lab
        stats['run_dir'] = rd
        agg_list.append(stats)
        # write per-run worst splits
        out_dir = rd
        df_sorted = df.sort_values('total_return')
        df_sorted[['train_start','train_end','test_end','total_return','max_drawdown'] + [c for c in df.columns if c.startswith('w_')]].head(5).to_csv(os.path.join(out_dir,'worst_splits_by_return.csv'), index=False)
        df_sorted_dd = df.sort_values('max_drawdown')
        df_sorted_dd[['train_start','train_end','test_end','total_return','max_drawdown'] + [c for c in df.columns if c.startswith('w_')]].head(5).to_csv(os.path.join(out_dir,'worst_splits_by_drawdown.csv'), index=False)
        # histogram plots
        plt.figure(figsize=(6,4))
        if 'total_return' in df.columns:
            plt.hist(df['total_return'].dropna(), bins=20)
            plt.title(f'Total return distribution: {lab}')
            plt.xlabel('Total return')
            plt.savefig(os.path.join(out_dir,'hist_total_return.png'), bbox_inches='tight')
            plt.close()
        plt.figure(figsize=(6,4))
        if 'max_drawdown' in df.columns:
            plt.hist(df['max_drawdown'].dropna(), bins=20)
            plt.title(f'Max drawdown distribution: {lab}')
            plt.xlabel('Max drawdown')
            plt.savefig(os.path.join(out_dir,'hist_max_drawdown.png'), bbox_inches='tight')
            plt.close()
        # herfindahl
        plt.figure(figsize=(5,3))
        plt.hist(df['herfindahl'].dropna(), bins=20)
        plt.title(f'Herfindahl (concentration): {lab}')
        plt.savefig(os.path.join(out_dir,'hist_herfindahl.png'), bbox_inches='tight')
        plt.close()
    agg_df = pd.DataFrame(agg_list)
    # save comparison into first run dir
    master_out = args.out or dfs[0][1]
    agg_df.to_csv(os.path.join(master_out,'comparison_summary.csv'), index=False)
    print('WROTE', os.path.join(master_out,'comparison_summary.csv'))
    # combined plot of total_return means
    try:
        labels = agg_df['label'].tolist()
        means = agg_df['total_return_mean'].tolist()
        medians = agg_df['total_return_median'].tolist()
        x = np.arange(len(labels))
        plt.figure(figsize=(8,4))
        plt.bar(x-0.15, means, width=0.3, label='mean')
        plt.bar(x+0.15, medians, width=0.3, label='median')
        plt.xticks(x, labels)
        plt.ylabel('Total return')
        plt.title('Run comparison: total return (mean & median)')
        plt.legend()
        plt.savefig(os.path.join(master_out,'comparison_total_return.png'), bbox_inches='tight')
        plt.close()
        print('WROTE', os.path.join(master_out,'comparison_total_return.png'))
    except Exception as e:
        print('Failed to write combined plot:', e)

if __name__ == '__main__':
    main()
