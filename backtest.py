#!/usr/bin/env python3
"""TradeBot v1: educational, long-only, next-open daily-bar backtest.
CSV columns: Date,Open,High,Low,Close,Volume. No live orders.
"""
import argparse
import json
import math
from pathlib import Path
import pandas as pd


def backtest(df, initial=1000.0, risk_pct=.005, max_alloc=.25, fee=0.0, slip_bps=10):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    needed = {'date','open','high','low','close','volume'}
    if not needed.issubset(df.columns):
        raise ValueError('Missing CSV columns: '+str(sorted(needed-set(df.columns))))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='raise')
    if (df[['open','high','low','close']] <= 0).any().any():
        raise ValueError('Prices must be positive')
    if (df['high'] < df[['open','close','low']].max(axis=1)).any() or (df['low'] > df[['open','close','high']].min(axis=1)).any():
        raise ValueError('Invalid OHLC bars')
    df['ma20'] = df.close.rolling(20).mean()
    df['ma200'] = df.close.rolling(200).mean()
    prior_close = df.close.shift(1)
    tr = pd.concat([df.high-df.low, (df.high-prior_close).abs(), (df.low-prior_close).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    # Signal formed after today's close; entry is next session open.
    df['signal'] = (df.close > df.ma200) & (df.low <= df.ma20*1.01) & (df.close > df.ma20) & (df.close > df.open)
    cash = float(initial)
    shares = 0
    entry = stop = entry_cost = 0.0
    entry_date = None
    trades = []
    equity = []
    halted = False
    peak = initial
    for i in range(200, len(df)):
        bar = df.iloc[i]
        date = str(bar.date.date())
        if shares:
            # Stop triggered at open or intraday. Stop fill may gap through stop.
            if bar.open <= stop or bar.low <= stop:
                raw_exit = min(float(bar.open), stop)
                exit_px = raw_exit*(1-slip_bps/10000)
                proceeds = shares*exit_px-fee
                cash += proceeds
                trades.append({'entry_date':entry_date,'exit_date':date,'entry':round(entry,4),'exit':round(exit_px,4),'shares':shares,'pnl':round(proceeds-entry_cost,2),'exit_reason':'stop_or_gap'})
                shares = 0
            elif i > 0 and df.iloc[i-1].close < df.iloc[i-1].ma20:
                exit_px = float(bar.open)*(1-slip_bps/10000)
                proceeds = shares*exit_px-fee
                cash += proceeds
                trades.append({'entry_date':entry_date,'exit_date':date,'entry':round(entry,4),'exit':round(exit_px,4),'shares':shares,'pnl':round(proceeds-entry_cost,2),'exit_reason':'trend_exit'})
                shares = 0
        mark = cash + shares*float(bar.close)
        peak = max(peak, mark)
        if mark <= peak*.92:
            halted = True
        if not shares and not halted and i > 0:
            prev = df.iloc[i-1]
            if bool(prev.signal) and pd.notna(prev.atr) and prev.atr > 0:
                buy_px = float(bar.open)*(1+slip_bps/10000)
                # Stop anchored to the entry, not to today's unknowable low.
                proposed_stop = buy_px - 2*float(prev.atr)
                per_share_risk = buy_px-proposed_stop
                budget = min(initial*risk_pct, mark*risk_pct)
                quantity = min(math.floor(budget/per_share_risk), math.floor((mark*max_alloc-fee)/buy_px), math.floor((cash-fee)/buy_px))
                if quantity >= 1 and proposed_stop > 0:
                    shares = quantity
                    entry = buy_px
                    stop = proposed_stop
                    entry_date = date
                    entry_cost = shares*entry+fee
                    cash -= entry_cost
        equity.append({'date':date,'equity':round(cash+shares*float(bar.close),2),'cash':round(cash,2),'halted':halted})
    if shares:
        last = df.iloc[-1]
        px = float(last.close)*(1-slip_bps/10000)
        proceeds = shares*px-fee
        cash += proceeds
        trades.append({'entry_date':entry_date,'exit_date':str(last.date.date()),'entry':round(entry,4),'exit':round(px,4),'shares':shares,'pnl':round(proceeds-entry_cost,2),'exit_reason':'end_of_data'})
        if equity:
            equity[-1]['equity'] = round(cash,2)
            equity[-1]['cash'] = round(cash,2)
    curve = pd.DataFrame(equity)
    if curve.empty:
        raise ValueError('At least 201 daily bars required')
    dd = curve.equity / curve.equity.cummax() - 1
    pnl = [t['pnl'] for t in trades]
    metrics = {'initial_equity':initial,'ending_equity':round(float(curve.equity.iloc[-1]),2), 'total_return_pct':round((float(curve.equity.iloc[-1])/initial-1)*100,2),'max_drawdown_pct':round(float(dd.min())*100,2),'trades':len(trades),'win_rate_pct':round(100*sum(x>0 for x in pnl)/len(pnl),2) if pnl else None,'avg_pnl_per_trade':round(sum(pnl)/len(pnl),2) if pnl else None,'halted':halted}
    return metrics, pd.DataFrame(trades), curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv', help='Historical daily OHLCV CSV')
    p.add_argument('--output', default='results')
    p.add_argument('--fee', type=float, default=0)
    p.add_argument('--slippage-bps', type=float, default=10)
    args = p.parse_args()
    if args.fee < 0 or args.slippage_bps < 0:
        p.error('Costs must be nonnegative')
    metrics, trades, curve = backtest(pd.read_csv(args.csv), fee=args.fee, slip_bps=args.slippage_bps)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    trades.to_csv(out/'trades.csv',index=False)
    curve.to_csv(out/'equity_curve.csv',index=False)
    print(json.dumps(metrics,indent=2))

if __name__ == '__main__':
    main()
