#!/usr/bin/env python3
"""Generate an execution-order plan from trades.csv files.
Usage: python generate_execution_orders.py --split-dir /path/to/split_folder --out execution_plan.csv
Generates entry and exit orders with estimated price, slippage, and fees.
"""
import argparse
import os
import glob
import pandas as pd
from datetime import datetime

DEFAULT_SLIPPAGE_PCT = 0.001  # 0.1% per side
FEE_PER_TRADE = 0.0  # flat fee per trade


def strategy_name_from_path(path, split_dir):
    p = os.path.abspath(path)
    strategy = os.path.basename(os.path.dirname(os.path.dirname(p)))
    if strategy == os.path.basename(split_dir) or strategy == '':
        strategy = os.path.basename(os.path.dirname(p))
    return strategy


def create_orders_for_trade(row, strategy, slippage_pct, fee_per_trade):
    # Create two orders: entry and exit.
    # We estimate entry_price and exit_price using avg_entry_price/exit_price if available, else set NaN.
    entry_dt = row.get('entry_dt')
    exit_dt = row.get('exit_dt')
    symbol = row.get('symbol') or row.get('ticker') or row.get('underlying')
    qty = row.get('qty') if 'qty' in row and not pd.isna(row.get('qty')) else row.get('quantity') if 'quantity' in row else None
    pnl = row.get('pnl') if 'pnl' in row else None
    # If price columns exist, prefer them
    entry_px = row.get('entry_price') or row.get('entry_px') or None
    exit_px = row.get('exit_price') or row.get('exit_px') or None
    orders = []
    # entry order
    orders.append({
        'strategy': strategy,
        'symbol': symbol,
        'side': 'BUY' if qty is None or (qty is not None and float(qty) > 0) else 'SELL',
        'qty': qty,
        'scheduled_dt': entry_dt,
        'price_est': entry_px,
        'slippage_pct': slippage_pct,
        'fee': fee_per_trade,
        'type': 'entry'
    })
    # exit order
    orders.append({
        'strategy': strategy,
        'symbol': symbol,
        'side': 'SELL' if qty is None or (qty is not None and float(qty) > 0) else 'BUY',
        'qty': qty,
        'scheduled_dt': exit_dt,
        'price_est': exit_px,
        'slippage_pct': slippage_pct,
        'fee': fee_per_trade,
        'type': 'exit'
    })
    return orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-dir', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--slippage-pct', type=float, default=DEFAULT_SLIPPAGE_PCT)
    ap.add_argument('--fee', type=float, default=FEE_PER_TRADE)
    args = ap.parse_args()
    split_dir = os.path.abspath(args.split_dir)
    if not os.path.isdir(split_dir):
        print('split-dir not found', split_dir)
        return
    orders = []
    for tfile in glob.glob(os.path.join(split_dir, '**','trades.csv'), recursive=True):
        strategy = strategy_name_from_path(tfile, split_dir)
        try:
            df = pd.read_csv(tfile)
        except Exception:
            continue
        if df.empty:
            continue
        for _,r in df.iterrows():
            ods = create_orders_for_trade(r, strategy, args.slippage_pct, args.fee)
            orders.extend(ods)
    if not orders:
        print('No trades found under', split_dir)
        return
    out_df = pd.DataFrame(orders)
    out_csv = args.out or os.path.join(split_dir, 'execution_plan.csv')
    out_df.to_csv(out_csv, index=False)
    print('WROTE', out_csv)

if __name__ == '__main__':
    main()
