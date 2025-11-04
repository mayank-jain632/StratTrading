#!/usr/bin/env python3
"""Analyze per-split weight concentration and recurring bad tickers.
Reads per_split_summary.csv in a run folder and aggregates weighted negative trades across splits.
Usage: python analyze_weights_and_bad_tickers.py --run-dir /absolute/path/to/run_folder
Saves: weight_concentration.csv, weight_concentration.png, top_bad_tickers.csv in run folder
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import glob


def herfindahl(weights):
    arr = np.array(list(weights))
    if arr.sum() == 0:
        return 0.0
    w = arr / arr.sum()
    return float((w**2).sum())


def find_split_folders(run_dir):
    # folders named runs_YYYYMMDD_YYYYMMDD_YYYYMMDD
    return [p for p in glob.glob(os.path.join(run_dir, 'scenario_*','runs_*')) if os.path.isdir(p)]


def load_weights(per_split_csv):
    df = pd.read_csv(per_split_csv)
    return df


def aggregate_bad_tickers(split_folders, weights_df):
    # weights_df has train_start, train_end, test_end and w_<strategy> columns
    ticker_counter = Counter()
    ticker_unweighted = Counter()
    missing = []
    for split in split_folders:
        base = os.path.basename(split)
        # parse dates
        # attempt to build keys to find row
        parts = base.split('_')
        # runs_YYYYMMDD_train_end_test_end
        if len(parts) < 4:
            continue
        train_start, train_end, test_end = parts[1], parts[2], parts[3]
        # match row
        mask = (weights_df['train_start'].astype(str).str.contains(train_start)) & (weights_df['train_end'].astype(str).str.contains(train_end)) & (weights_df['test_end'].astype(str).str.contains(test_end))
        row = weights_df[mask]
        if row.empty:
            # try match by train_end
            row = weights_df[weights_df['train_end'].astype(str).str.contains(train_end)]
        if row.empty:
            missing.append(split)
            continue
        row = row.iloc[0]
        # collect weights by strategy
        weights = {c.replace('w_',''): float(row[c]) for c in weights_df.columns if c.startswith('w_')}
        # read trades for split
        for tfile in glob.glob(os.path.join(split, '**','trades.csv'), recursive=True):
            # strategy name
            strat = os.path.basename(os.path.dirname(os.path.dirname(tfile)))
            w = weights.get(strat, 0.0)
            try:
                tdf = pd.read_csv(tfile)
            except Exception:
                continue
            if 'pnl' not in tdf.columns:
                continue
            # for negative trades, add weighted contribution
            negs = tdf[tdf['pnl'] < 0]
            for _,r in negs.iterrows():
                sym = r.get('symbol') or r.get('ticker') or 'UNKNOWN'
                pnl = float(r['pnl'])
                ticker_counter[sym] += (-pnl) * w  # weight contribution (positive magnitude)
                ticker_unweighted[sym] += -pnl
    return ticker_counter, ticker_unweighted, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    per_split_csv = os.path.join(run_dir, 'per_split_summary.csv')
    if not os.path.exists(per_split_csv):
        print('per_split_summary.csv not found in', run_dir)
        return
    weights_df = load_weights(per_split_csv)
    split_folders = find_split_folders(run_dir)
    hc = []
    for _,row in weights_df.iterrows():
        wcols = [c for c in weights_df.columns if c.startswith('w_')]
        ws = [float(row[c]) for c in wcols]
        hc.append({'train_start': row['train_start'], 'train_end': row['train_end'], 'test_end': row['test_end'], 'herfindahl': herfindahl(ws), 'max_weight': max(ws)})
    hc_df = pd.DataFrame(hc)
    out_conc = os.path.join(run_dir, 'weight_concentration.csv')
    hc_df.to_csv(out_conc, index=False)
    print('WROTE', out_conc)
    # plot
    plt.figure(figsize=(6,4))
    plt.hist(hc_df['herfindahl'].dropna(), bins=20)
    plt.xlabel('Herfindahl (sum w^2)')
    plt.title('Weight concentration across splits')
    out_png = os.path.join(run_dir, 'weight_concentration.png')
    plt.savefig(out_png, bbox_inches='tight')
    print('WROTE', out_png)

    ticker_counter, ticker_unweighted, missing = aggregate_bad_tickers(split_folders, weights_df)
    # write top bad tickers
    top = ticker_counter.most_common(50)
    df_top = pd.DataFrame(top, columns=['symbol','weighted_loss_contribution'])
    df_top['unweighted_loss'] = df_top['symbol'].map(lambda s: ticker_unweighted.get(s,0.0))
    out_top = os.path.join(run_dir, 'top_bad_tickers.csv')
    df_top.to_csv(out_top, index=False)
    print('WROTE', out_top)
    if missing:
        print('Warning: could not match rows for %d splits (see list)' % len(missing))

if __name__ == '__main__':
    main()
