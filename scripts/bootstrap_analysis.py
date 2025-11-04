#!/usr/bin/env python3
"""Bootstrap analysis for equity curve metrics.

Usage: python3 scripts/bootstrap_analysis.py --equity-file <path> [--n-iter 2000] [--block-size 5]

This script reads a composite equity curve CSV with columns (dt,equity), computes daily returns,
and performs a circular block bootstrap on returns to generate distributions for CAGR, Sharpe, and
max drawdown. Outputs CSVs and simple histograms to an output folder.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def calc_metrics(equity_series, periods_per_year=252):
    # equity_series: pd.Series indexed by datetime
    returns = equity_series.pct_change().dropna()
    if len(returns) == 0:
        return {
            'cagr': np.nan,
            'ann_vol': np.nan,
            'sharpe': np.nan,
            'max_dd': np.nan,
        }

    # determine total_years: if index is datetime-like, use actual span, otherwise infer from periods_per_year
    try:
        idx0 = equity_series.index[0]
        idx1 = equity_series.index[-1]
        if hasattr(idx0, 'year') and hasattr(idx1, 'year'):
            total_years = (idx1 - idx0).days / 365.25
        else:
            total_years = len(equity_series) / periods_per_year
    except Exception:
        total_years = len(equity_series) / periods_per_year

    start = float(equity_series.iloc[0])
    end = float(equity_series.iloc[-1])
    cagr = (end / start) ** (1 / total_years) - 1 if total_years > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    ann_ret = returns.mean() * periods_per_year
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    # max drawdown
    cum = equity_series / equity_series.iloc[0]
    highwater = cum.cummax()
    drawdown = cum / highwater - 1
    max_dd = drawdown.min()
    return {
        'cagr': cagr,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
    }


def circular_block_bootstrap(returns, n_samples, block_size):
    # returns: 1D numpy array of returns
    n = len(returns)
    samples = np.empty((n_samples, n))
    for i in range(n_samples):
        idx = 0
        res = []
        while idx < n:
            start = np.random.randint(0, n)
            block = []
            for b in range(block_size):
                block_idx = (start + b) % n
                block.append(returns[block_idx])
            take = min(block_size, n - idx)
            res.extend(block[:take])
            idx += take
        samples[i, :] = np.array(res)
    return samples


def sample_to_equity(starting_equity, returns_sample):
    # returns_sample: 1D array
    eq = starting_equity * np.cumprod(1 + returns_sample)
    idx = pd.RangeIndex(len(eq))
    return pd.Series(eq, index=idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--equity-file', required=True)
    p.add_argument('--n-iter', type=int, default=2000)
    p.add_argument('--block-size', type=int, default=5)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)

    df = pd.read_csv(args.equity_file, parse_dates=['dt'])
    if 'equity' not in df.columns:
        raise SystemExit('equity column not found in CSV')
    df = df.sort_values('dt').reset_index(drop=True)
    eq_series = pd.Series(df['equity'].values, index=pd.to_datetime(df['dt']))
    starting_equity = float(eq_series.iloc[0])

    metrics_obs = calc_metrics(eq_series)

    returns = eq_series.pct_change().dropna().values
    n = len(returns)
    print(f'Found {n} return periods; performing {args.n_iter} bootstrap iterations (block_size={args.block_size})')

    samples = circular_block_bootstrap(returns, args.n_iter, args.block_size)

    # compute metrics for each sample
    cagr_list = []
    sharpe_list = []
    maxdd_list = []
    for i in range(args.n_iter):
        sample_returns = samples[i]
        # convert to an equity series with same index dates (use numeric index)
        sample_eq = sample_to_equity(starting_equity, sample_returns)
        metrics = calc_metrics(sample_eq)
        cagr_list.append(metrics['cagr'])
        sharpe_list.append(metrics['sharpe'])
        maxdd_list.append(metrics['max_dd'])

    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.equity_file).parent / 'bootstrap_analysis')
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame({
        'cagr': cagr_list,
        'sharpe': sharpe_list,
        'max_dd': maxdd_list,
    })
    metrics_df.to_csv(out_dir / 'bootstrap_metrics.csv', index=False)

    # Summaries and p-values (one-sided: how often bootstrap >= observed for cagr/sharpe; for max_dd, <= observed since it's negative)
    def p_value_one_sided(obs, sims, greater=True):
        if np.isnan(obs):
            return np.nan
        if greater:
            return np.mean(np.array(sims) >= obs)
        else:
            return np.mean(np.array(sims) <= obs)

    summary = {
        'obs_cagr': metrics_obs['cagr'],
        'obs_sharpe': metrics_obs['sharpe'],
        'obs_max_dd': metrics_obs['max_dd'],
        'cagr_mean': np.nanmean(cagr_list),
        'cagr_5pct': np.nanpercentile(cagr_list, 5),
        'cagr_95pct': np.nanpercentile(cagr_list, 95),
        'sharpe_mean': np.nanmean(sharpe_list),
        'sharpe_5pct': np.nanpercentile(sharpe_list, 5),
        'sharpe_95pct': np.nanpercentile(sharpe_list, 95),
        'maxdd_mean': np.nanmean(maxdd_list),
        'maxdd_5pct': np.nanpercentile(maxdd_list, 5),
        'maxdd_95pct': np.nanpercentile(maxdd_list, 95),
        'p_cagr_ge_obs': p_value_one_sided(metrics_obs['cagr'], cagr_list, greater=True),
        'p_sharpe_ge_obs': p_value_one_sided(metrics_obs['sharpe'], sharpe_list, greater=True),
        'p_maxdd_le_obs': p_value_one_sided(metrics_obs['max_dd'], maxdd_list, greater=False),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(out_dir / 'bootstrap_summary.csv', index=False)

    # simple histograms
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(cagr_list, bins=50, alpha=0.7)
        plt.axvline(metrics_obs['cagr'], color='red', linestyle='--', label='observed')
        plt.title('Bootstrap CAGR')
        plt.legend()
        plt.savefig(out_dir / 'cagr_hist.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(sharpe_list, bins=50, alpha=0.7)
        plt.axvline(metrics_obs['sharpe'], color='red', linestyle='--', label='observed')
        plt.title('Bootstrap Sharpe')
        plt.legend()
        plt.savefig(out_dir / 'sharpe_hist.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(maxdd_list, bins=50, alpha=0.7)
        plt.axvline(metrics_obs['max_dd'], color='red', linestyle='--', label='observed')
        plt.title('Bootstrap Max Drawdown')
        plt.legend()
        plt.savefig(out_dir / 'maxdd_hist.png')
        plt.close()
    except Exception:
        print('matplotlib plotting failed; results still saved as CSVs.')

    print('Bootstrap analysis complete. Results saved to:', out_dir)


if __name__ == '__main__':
    main()
