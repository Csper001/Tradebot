"""TradeBot cloud research dashboard. Never sends brokerage orders."""
import io
import numpy as np
import pandas as pd
import streamlit as st
from backtest import backtest

st.set_page_config(page_title='TradeBot | Research Dashboard',page_icon='📈',layout='wide')
st.title('📈 TradeBot — Research Dashboard')
st.caption('Historical backtesting only • No brokerage connection • No live trading')
st.warning('Educational research prototype. Backtests are not forecasts. A simulated stop does not cap real losses.')

with st.sidebar:
    st.header('Strategy controls')
    initial=st.number_input('Starting account ($)',min_value=100.0,max_value=100000.0,value=1000.0,step=100.0)
    risk=st.slider('Risk budget per trade (%)',0.1,1.0,0.5,0.1)
    allocation=st.slider('Max allocation per position (%)',5,50,25,5)
    fee=st.number_input('Commission per order ($)',min_value=0.0,max_value=20.0,value=0.0,step=0.5)
    slippage=st.slider('Slippage per side (basis points)',0,100,10,5)
    st.caption('Fixed research strategy: MA200 trend + MA20 pullback, next-open entry, 2×ATR stop, MA20 exit, 8% entry halt. Whole shares only.')

st.subheader('1. Supply historical daily prices')
st.write('Upload a CSV with **Date, Open, High, Low, Close, Volume** columns. At least 201 trading days required; 5+ years recommended. Use consistent, split-adjusted OHLC prices.')
upload=st.file_uploader('Historical OHLCV CSV',type=['csv'])
if upload is None:
    st.info('Upload a historical-price CSV to run your first backtest. No live market data or fabricated sample returns are used.')
    st.download_button('Download blank CSV template','Date,Open,High,Low,Close,Volume\n','tradebot_data_template.csv','text/csv')
    st.stop()
try:
    data=pd.read_csv(upload)
    cols={c.lower().strip():c for c in data.columns}
    if 'date' not in cols or 'close' not in cols:
        raise ValueError('CSV needs Date and Close columns, plus Open, High, Low, Volume.')
    data[cols['date']]=pd.to_datetime(data[cols['date']],errors='raise')
    if data[cols['date']].duplicated().any():
        st.warning('Duplicate dates detected. Backtester retains one record per date.')
    st.write(f'**Loaded:** {len(data):,} rows · {data[cols["date"]].min().date()} through {data[cols["date"]].max().date()}')
    metrics,trades,curve=backtest(data,initial=initial,risk_pct=risk/100,max_alloc=allocation/100,fee=fee,slip_bps=slippage)
    dates=pd.to_datetime(curve['date'])
    close_series=data.rename(columns={v:k for k,v in cols.items()}).set_index('date')['close'].sort_index()
    # Benchmark is buy-and-hold starting on the first equity-curve day, fractional units,
    # without fees/dividends. This is intentionally an illustrative comparison only.
    prices=close_series.reindex(dates).astype(float)
    if prices.isna().any():
        raise ValueError('Could not align benchmark prices to the equity curve.')
    benchmark=initial*prices.to_numpy()/float(prices.iloc[0])
    bench_return=(benchmark[-1]/initial-1)*100
    bench_dd=(pd.Series(benchmark)/pd.Series(benchmark).cummax()-1).min()*100
    st.subheader('2. Results')
    a,b,c,d=st.columns(4)
    a.metric('Ending equity',f'${metrics["ending_equity"]:,.2f}',f'{metrics["total_return_pct"]:+.2f}%')
    b.metric('Max drawdown',f'{metrics["max_drawdown_pct"]:.2f}%')
    c.metric('Closed trades',str(metrics['trades']))
    d.metric('Avg P/L per trade','N/A' if metrics['avg_pnl_per_trade'] is None else f'${metrics["avg_pnl_per_trade"]:,.2f}')
    if metrics['halted']:
        st.error('The 8% drawdown threshold was breached: the strategy stopped opening new positions. Existing positions may have remained open.')
    if metrics['trades']<30:
        st.warning('Fewer than 30 completed trades. Results are too sparse to support a reliable conclusion.')
    chart=pd.DataFrame({'TradeBot equity':curve['equity'].to_numpy(),'Buy-and-hold benchmark':benchmark},index=dates)
    st.line_chart(chart)
    st.caption(f'Benchmark: same asset held from first simulated equity date, fractional shares, no trading costs or dividends. Return {bench_return:+.2f}%, max drawdown {bench_dd:.2f}%. Not a broad-market index benchmark.')
    st.subheader('3. Trades and exports')
    st.dataframe(trades,use_container_width=True,hide_index=True)
    x,y=st.columns(2)
    x.download_button('Download trade log',trades.to_csv(index=False).encode(),'tradebot_trades.csv','text/csv')
    y.download_button('Download equity curve',chart.rename_axis('Date').to_csv().encode(),'tradebot_equity.csv','text/csv')
    st.subheader('4. Interpretation checklist')
    st.markdown('''- Did it outperform holding the same asset after realistic costs?\n- Is the sample large enough and spread across market conditions?\n- Does performance survive doubling slippage and fees?\n- Does it work on unseen dates and other preselected symbols?\n- Were delistings, dividends, splits and survivorship bias handled?''')
    st.info('This version tests one symbol at a time. It is **not** a multi-stock portfolio simulation, live scanner, or automated execution engine.')
except Exception as exc:
    st.error(f'Could not run backtest: {exc}')
    st.caption('Check CSV format, date parsing, valid OHLC prices and number of trading days.')
