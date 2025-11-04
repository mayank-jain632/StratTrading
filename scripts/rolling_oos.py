#!/usr/bin/env python3
"""Walk-forward (rolling-window) out-of-sample evaluation with transaction-cost sensitivity.

This script performs the following for each rolling split:
 - Runs each strategy over the full window (train+test) using the backtester
 - Computes optimized long-only strategy weights on the TRAIN period
 - Applies those weights to strategy returns in the TEST period to produce an out-of-sample composite
 - Supports multiple broker cost scenarios (per_trade, per_share, slippage_bps, spread_bps)
 - Saves per-split and per-scenario metrics and composite equity curves

Usage example:
 ./scripts/rolling_oos.py --config webull_paper/configs/large_caps.yaml --train-days 180 --test-days 90 --step-days 90 --out-dir runs/rolling_oos_test

Note: this will invoke the project's backtester multiple times and can be slow. Use small windows for a quick smoke test.
"""
from __future__ import annotations
import argparse
import yaml
import pandas as pd
import numpy as np
import sys
import copy
import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Ensure project src path is importable (so scripts can import package modules)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'webull_paper' / 'src'))

from options_trader.cli.run_backtest import build_from_config
from options_trader.cli.run_all_strategies import _optimize_weights, bootstrap_optimize_weights
from options_trader.metrics.performance import compute_performance


def daterange(start: datetime, end: datetime, step: timedelta):
    cur = start
    while cur < end:
        yield cur
        cur = cur + step


def load_base_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def run_split(base_cfg: Dict[str, Any], strategies: List[str], train_start: datetime, train_end: datetime, test_end: datetime, broker_override: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    # run each strategy over full window [train_start, test_end]
    equity_curves = {}
    strategy_rets = {}
    strategy_trade_counts: List[int] = []
    strategy_trades: Dict[str, Any] = {}
    for strat in strategies:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault('strategy', {})
        cfg['strategy']['type'] = strat
        cfg['strategy']['symbols'] = cfg.get('data', {}).get('symbols', [])
        cfg['data'] = cfg.get('data', {})
        cfg['data']['start'] = train_start.strftime('%Y-%m-%d')
        cfg['data']['end'] = test_end.strftime('%Y-%m-%d')
        # apply broker overrides
        cfg.setdefault('broker', {})
        cfg['broker'].update(broker_override)
        # ensure outputs go to a split-specific folder to avoid collisions
        stamp = f"{train_start.strftime('%Y%m%d')}_{train_end.strftime('%Y%m%d')}_{test_end.strftime('%Y%m%d')}"
        cfg['output_dir'] = str(out_dir / f"runs_{stamp}" / strat)
        os.makedirs(cfg['output_dir'], exist_ok=True)
        bt = build_from_config(cfg)
        result = bt.run()
        eq = result['equity_curve'].copy()
        trades = result.get('trades', [])
        # normalize trades into a DataFrame when possible and store per-strategy
        try:
            if hasattr(trades, 'shape'):
                trades_df = trades.copy()
                tcount = int(trades_df.shape[0])
            else:
                trades_df = pd.DataFrame(trades)
                tcount = int(len(trades_df))
        except Exception:
            trades_df = pd.DataFrame()
            tcount = 0
        strategy_trade_counts.append(tcount)
        strategy_trades[strat] = trades_df
        # ensure datetime index
        if not isinstance(eq.index, pd.DatetimeIndex):
            eq.index = pd.to_datetime(eq.index)
        equity_curves[strat] = eq
        strategy_rets[strat] = eq['equity'].pct_change().fillna(0.0)

    # align returns to common index
    common_idx = None
    for rets in strategy_rets.values():
        if common_idx is None:
            common_idx = rets.index
        else:
            common_idx = common_idx.intersection(rets.index)
    rets_df = pd.DataFrame({s: r.reindex(common_idx).fillna(0.0) for s, r in strategy_rets.items()})

    # training mask and test mask
    train_mask = (rets_df.index >= pd.to_datetime(train_start)) & (rets_df.index <= pd.to_datetime(train_end))
    test_mask = (rets_df.index > pd.to_datetime(train_end)) & (rets_df.index <= pd.to_datetime(test_end))

    if rets_df[train_mask].shape[0] == 0 or rets_df[test_mask].shape[0] == 0:
        return {'error': 'No overlapping data for this split'}

    # compute optimized weights on train (with optional L2 regularization)
    opt_l2 = float(base_cfg.get('optimizer_l2', 0.0))
    bootstrap_rounds = int(base_cfg.get('optimizer_bootstrap_rounds', 0))
    if bootstrap_rounds > 0:
        weights = bootstrap_optimize_weights(rets_df[train_mask], rounds=bootstrap_rounds, l2=opt_l2)
        print(f"[rolling_oos] Computed bootstrap-aggregated weights using {bootstrap_rounds} rounds")
    else:
        weights = _optimize_weights(rets_df[train_mask], l2=opt_l2)
    # Optional volatility-scaling before gating/capping
    if bool(base_cfg.get('volatility_scaling', False)):
        vols = rets_df[train_mask].std().replace(0.0, np.nan).fillna(np.inf)
        inv_vol = 1.0 / vols.values
        w = np.array(weights, dtype=float)
        w = w * inv_vol
        if w.sum() <= 1e-12:
            w = np.repeat(1.0 / len(w), len(w))
        else:
            w = w / (w.sum() + 1e-12)
        weights = w
        print(f"[rolling_oos] Applied volatility scaling to optimizer weights")
    # Apply minimum-trades gating: strategies with too few trades in-sample get zero weight
    min_trades = int(base_cfg.get('min_trades_for_weight', 5))
    if min_trades > 0:
        tc = np.array(strategy_trade_counts)
        low_mask = tc < min_trades
        if low_mask.any():
            w = np.array(weights, dtype=float)
            w[low_mask] = 0.0
            if w.sum() <= 1e-12:
                # no strategy meets min trades -> fall back to equal weights
                w = np.repeat(1.0 / len(w), len(w))
            else:
                w = w / (w.sum() + 1e-12)
            weights = w
            print(f"[rolling_oos] Applied min_trades_for_weight={min_trades}; zeroed strategies: {list(np.array(list(rets_df.columns))[low_mask])}")
    # Apply cap/floor from base config to avoid single-strategy dominance (same logic as run_all_strategies)
    max_cap = float(base_cfg.get('max_strategy_weight', 0.6))
    min_floor = float(base_cfg.get('min_strategy_weight', 0.0))
    weights = np.array(weights, dtype=float)
    if len(weights) > 1 and weights.max() > max_cap:
        orig = weights.copy()
        capped_mask = orig > max_cap
        weights[capped_mask] = max_cap
        remaining = 1.0 - weights.sum()
        uncapped_idx = np.where(~capped_mask)[0]
        if len(uncapped_idx) > 0:
            uncapped_orig_sum = orig[uncapped_idx].sum()
            if uncapped_orig_sum > 0:
                weights[uncapped_idx] += remaining * (orig[uncapped_idx] / uncapped_orig_sum)
            else:
                weights[uncapped_idx] += remaining / len(uncapped_idx)
        else:
            weights = np.repeat(1.0 / len(weights), len(weights))
        if min_floor > 0.0:
            weights = np.maximum(weights, min_floor)
        weights = weights / (weights.sum() + 1e-12)
        print(f"[rolling_oos] Optimizer weights capped to max={max_cap} and redistributed: {weights}")
    weights_map = {name: float(w) for name, w in zip(rets_df.columns, weights)}

    # apply weights to test returns
    test_rets = rets_df[test_mask].fillna(0.0)
    comp_rets = test_rets.dot(np.array(weights))
    comp_eq = (1 + comp_rets).cumprod() * base_cfg.get('starting_cash', 100000.0)
    comp_eq_df = pd.DataFrame({'equity': comp_eq}, index=test_rets.index)
    # Build composite trade list: include trades from individual strategies that closed in TEST period
    composite_trades: List[Dict[str, Any]] = []
    try:
        test_start = pd.to_datetime(train_end)  # exclusive
        test_end_dt = pd.to_datetime(test_end)
        for sname, tdf in strategy_trades.items():
            if tdf is None or tdf.empty:
                continue
            # ensure columns exist and parse exit_dt
            if 'exit_dt' in tdf.columns:
                try:
                    exit_dates = pd.to_datetime(tdf['exit_dt'], errors='coerce')
                except Exception:
                    exit_dates = pd.to_datetime(tdf['exit_dt'].astype(str), errors='coerce')
            elif 'exit_date' in tdf.columns:
                exit_dates = pd.to_datetime(tdf['exit_date'], errors='coerce')
            else:
                # no exit date column, skip
                continue
            mask = (exit_dates > test_start) & (exit_dates <= test_end_dt)
            if mask.any():
                selected = tdf.loc[mask]
                for _, row in selected.iterrows():
                    # construct minimal trade dict expected by compute_performance
                    trade = {}
                    raw_pnl = float(row.get('pnl', 0.0)) if 'pnl' in row.index else float(row.get('profit', 0.0) if 'profit' in row.index else 0.0)
                    raw_bars = float(row.get('bars_held', 0)) if 'bars_held' in row.index else float(row.get('bars', 0) if 'bars' in row.index else 0)
                    # scale pnl and exposure by the optimizer weight for this strategy so composite trade stats reflect weighted composite PnL
                    w = float(weights_map.get(sname, 1.0)) if 'weights_map' in locals() else float(weights.get(sname, 1.0)) if 'weights' in locals() else 1.0
                    trade['pnl'] = raw_pnl * w
                    trade['bars_held'] = raw_bars * w
                    trade['symbol'] = row.get('symbol', None) if 'symbol' in row.index else None
                    trade['entry_dt'] = row.get('entry_dt', None) if 'entry_dt' in row.index else None
                    trade['exit_dt'] = row.get('exit_dt', None) if 'exit_dt' in row.index else None
                    trade['strategy'] = sname
                    composite_trades.append(trade)
    except Exception:
        composite_trades = []

    metrics = compute_performance(comp_eq_df, trades=composite_trades)

    return {
        'train_start': train_start.strftime('%Y-%m-%d'),
        'train_end': train_end.strftime('%Y-%m-%d'),
        'test_end': test_end.strftime('%Y-%m-%d'),
        'weights': weights_map,
        'metrics': metrics.to_dict(),
        'composite_equity': comp_eq_df,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--strategies', nargs='*', help='Strategies to include; default from config')
    p.add_argument('--train-days', type=int, default=180)
    p.add_argument('--test-days', type=int, default=90)
    p.add_argument('--step-days', type=int, default=90)
    p.add_argument('--out-dir', default='runs/rolling_oos')
    p.add_argument('--optimizer-l2', type=float, default=None, help='L2 regularization strength for optimizer')
    p.add_argument('--optimizer-bootstrap-rounds', type=int, default=None, help='If set, compute bootstrap-aggregated weights over this many resamples')
    p.add_argument('--volatility-scaling', action='store_true', help='Enable volatility scaling (inverse-vol) on optimized weights')
    p.add_argument('--costs', nargs='*', help='Cost scenario JSON strings, e.g. "{\"per_trade\":0,\"slippage_bps\":0.001}"', default=None)
    p.add_argument('--robinhood', action='store_true', help='Include a Robinhood-like preset cost scenario (commission-free, small slippage)')
    args = p.parse_args()

    base_cfg = load_base_config(args.config)
    # allow CLI override for bootstrap rounds
    if args.optimizer_bootstrap_rounds is not None:
        base_cfg['optimizer_bootstrap_rounds'] = int(args.optimizer_bootstrap_rounds)
    # Resolve strategies: CLI > base_cfg['strategies'] > base_cfg['strategy']['type']
    if args.strategies:
        strategies = args.strategies
    else:
        strategies = base_cfg.get('strategies')
        if not strategies:
            st = base_cfg.get('strategy', {}).get('type')
            if isinstance(st, str):
                strategies = [st]
            elif isinstance(st, list):
                strategies = st
            else:
                strategies = []
    if not strategies:
        raise SystemExit('No strategies specified and none found in config. Pass --strategies')

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # Build cost scenarios
    scenarios: List[Dict[str, Any]] = []
    if args.costs:
        for s in args.costs:
            scenarios.append(json.loads(s))
    else:
        # default: no cost
        scenarios.append({'per_trade': 0.0, 'per_share': 0.0, 'slippage_bps': 0.0, 'spread_bps': 0.0})

    if args.robinhood:
        scenarios.append({'per_trade': 0.0, 'per_share': 0.0, 'slippage_bps': 0.001, 'spread_bps': 0.0005, 'name': 'robinhood'})

    # derive global date range from base config
    data_cfg = base_cfg.get('data', {})
    start = pd.to_datetime(data_cfg.get('start'))
    end = pd.to_datetime(data_cfg.get('end'))
    train_delta = timedelta(days=args.train_days)
    test_delta = timedelta(days=args.test_days)
    step_delta = timedelta(days=args.step_days)

    all_results = []
    for scenario_idx, scenario in enumerate(scenarios):
        scen_name = scenario.get('name', f'scenario_{scenario_idx}')
        scen_out = out_base / scen_name
        scen_out.mkdir(parents=True, exist_ok=True)
        cur_start = start
        while cur_start + train_delta + test_delta <= end:
            train_start = cur_start
            train_end = cur_start + train_delta
            test_end = train_end + test_delta
            print(f'Running split: train {train_start.date()} -> {train_end.date()}, test until {test_end.date()}, scenario {scen_name}')
            try:
                res = run_split(base_cfg, strategies, train_start, train_end, test_end, scenario, scen_out)
            except Exception as e:
                res = {'error': str(e), 'train_start': train_start.strftime('%Y-%m-%d'), 'train_end': train_end.strftime('%Y-%m-%d')}
            # save composite equity if present
            if 'composite_equity' in res:
                fname = scen_out / f'composite_{train_start.strftime("%Y%m%d")}_{test_end.strftime("%Y%m%d")}.csv'
                res['composite_equity'].to_csv(fname)
            all_results.append({'scenario': scen_name, 'train_start': res.get('train_start'), 'train_end': res.get('train_end'), 'test_end': res.get('test_end'), 'weights': res.get('weights'), 'metrics': res.get('metrics'), 'error': res.get('error')})
            cur_start = cur_start + step_delta

    # Save master summary
    pd.DataFrame(all_results).to_json(out_base / 'rolling_oos_summary.json', orient='records', lines=False)
    print('Rolling OOS complete. Results saved to', out_base)

    # Combined overlay plot: read per-scenario composite CSVs and overlay them for quick visual comparison
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        for scen_dir in sorted(out_base.iterdir()):
            if not scen_dir.is_dir():
                continue
            # find composite files
            for csvf in sorted(scen_dir.glob('composite_*.csv')):
                try:
                    df = pd.read_csv(csvf, parse_dates=['dt'])
                    if 'equity' not in df.columns:
                        continue
                    # normalize index
                    ax.plot(df['dt'], df['equity'], label=f"{scen_dir.name}: {csvf.name}", alpha=0.8)
                except Exception:
                    continue
        ax.set_title('Rolling OOS: Per-split Composite Equity Overlays')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_plot = out_base / 'combined_optimized_overlay.png'
        plt.savefig(out_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Combined overlay plot saved to: {out_plot}')
    except Exception as e:
        print('Failed to create combined overlay plot:', e)


if __name__ == '__main__':
    main()
