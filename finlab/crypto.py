# IMPORTS
import pandas as pd
import math
import os.path
import time
from bitmex import bitmex
from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser
from tqdm import tqdm_notebook #(Optional, used for progress-bars)
import config
import backtrader as bt



### ORIGINAL SOURCE CODE
# crawler source code is revised from the following blog post
# https://medium.com/better-programming/easiest-way-to-use-the-bitmex-api-with-python-fbf66dc38633

### CONSTANTS
binsizes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h":120, "4h": 240, "6h": 360, "8h": 480, "1d": 1440}
batch_size = 750
bitmex_client = bitmex(test=False, api_key=config.bitmex_api_key, api_secret=config.bitmex_api_secret)
binance_client = Client(api_key=config.binance_api_key, api_secret=config.binance_api_secret)


### FUNCTIONS
def minutes_of_new_data(symbol, kline_size, data, source):
    if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1])
    elif source == "binance": old = datetime.strptime('1 Jan 2017', '%d %b %Y')
    elif source == "bitmex": old = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=False).result()[0][0]['timestamp']
    if source == "binance": new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    if source == "bitmex": new = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=True).result()[0][0]['timestamp']
    return old, new

def get_all_binance(symbol, kline_size, save=True, update=True):
    
    
    if not os.path.exists('history'):
        os.mkdir('history')
        
    if not os.path.exists(os.path.join('history', 'crypto')):
        os.mkdir(os.path.join('history', 'crypto'))
    
    # create dataframe from file
    filename = os.path.join('history', 'crypto', '%s-%s-data.csv' % (symbol, kline_size))
    if os.path.isfile(filename):
        data_df = pd.read_csv(filename, index_col='Timestamp', parse_dates=True)
    else:
        data_df = pd.DataFrame()
    
    if update == False:
        data_df.columns = data_df.columns.str.capitalize()
        data_df.index = pd.to_datetime(data_df.index)
        return data_df

        
    # find time period
    #oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, source = "binance")
    if not data_df.empty:
        oldest_point = data_df.index[-1].to_pydatetime()
    else:
        oldest_point = datetime.strptime('1 Jan 2017', '%d %b %Y')
    newest_point = datetime.now()
    print(oldest_point, newest_point)
    
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])
    
    # print some info
    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): print('Downloading all available %s data for %s. Be patient..!' % (kline_size, symbol))
    else: print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))
        
    # download kbars
    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"))
    
    
    # processing
    data = pd.DataFrame(klines, columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_av', 'Trades', 'Tb_base_av', 'Tb_quote_av', 'Ignore' ])

    data.Timestamp = pd.to_datetime(data.Timestamp, unit='ms')
    data.set_index('Timestamp', inplace=True)
    data = data[~data.index.duplicated(keep='last')]
    data = data.astype(float)
    
    # combine dataframe
    if len(data_df) > 0:
        data_df = data_df.append(data)
    else:
        data_df = data
        
    data_df = data_df[~data_df.index.duplicated(keep='last')]
    
    assert data_df.index.duplicated().sum() == 0
    # save data   
    if save: data_df.to_csv(filename)
    print('All caught up..!')
    return data_df

"""
def get_all_bitmex(symbol, kline_size, save = False):
    filename = '%s-%s-data.csv' % (symbol, kline_size)
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, source = "bitmex")
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])
    rounds = math.ceil(available_data / batch_size)
    if rounds > 0:
        print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data in %d rounds.' % (delta_min, symbol, available_data, kline_size, rounds))
        for round_num in tqdm_notebook(range(rounds)):
            time.sleep(1)
            new_time = (oldest_point + timedelta(minutes = round_num * batch_size * binsizes[kline_size]))
            data = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=batch_size, startTime = new_time).result()[0]
            temp_df = pd.DataFrame(data)
            data_df = data_df.append(temp_df)
    data_df.set_index('timestamp', inplace=True)
    if save and rounds > 0: data_df.to_csv(filename)
    print('All caught up..!')
    return data_df
"""


from collections import OrderedDict

OHLCV_AGG = OrderedDict((
    ('Open', 'first'),
    ('High', 'max'),
    ('Low', 'min'),
    ('Close', 'last'),
    ('Volume', 'sum'),
))

def resample(price_df, period):
    return price_df.resample(period, label='right').agg(OHLCV_AGG)

from backtesting.backtesting import _Data, _Indicator, _Broker
import numpy as np
from backtesting._util import _as_str, _Indicator, _Data, _data_period
from backtesting import Strategy

def compute_stats(data, broker: _Broker, strategy: Strategy) -> pd.Series:
    
    def _drawdown_duration_peaks(dd, index):
        # XXX: possible to vectorize any of this?
        durations = [np.nan] * len(dd)
        peaks = [np.nan] * len(dd)
        i = 0
        for j in range(1, len(dd)):
            if dd[j] == 0:
                if dd[j - 1] != 0:
                    durations[j - 1] = index[j] - index[i]
                    peaks[j - 1] = dd[i:j].max()
                i = j
        return pd.Series(durations), pd.Series(peaks)

    df = pd.DataFrame()
    df['Equity'] = pd.Series(broker.log.equity).bfill().fillna(broker._cash)
    equity = df.Equity.values
    df['Exit Entry'] = broker.log.exit_entry
    exits = df['Exit Entry']
    df['Exit Position'] = broker.log.exit_position
    df['Entry Price'] = broker.log.entry_price
    df['Exit Price'] = broker.log.exit_price
    df['P/L'] = broker.log.pl
    pl = df['P/L']
    df['Returns'] = returns = pl.dropna() / equity[exits.dropna().values.astype(int)]
    df['Drawdown'] = dd = 1 - equity / np.maximum.accumulate(equity)
    dd_dur, dd_peaks = _drawdown_duration_peaks(dd, data.index)
    df['Drawdown Duration'] = dd_dur
    dd_dur = df['Drawdown Duration']

    df.index = data.index

    def _round_timedelta(value, _period=_data_period(df)):
        if not isinstance(value, pd.Timedelta):
            return value
        resolution = getattr(_period, 'resolution_string', None) or _period.resolution
        return value.ceil(resolution)

    s = pd.Series()
    s.loc['Start'] = df.index[0]
    s.loc['End'] = df.index[-1]
    s.loc['Duration'] = s.End - s.Start
    exits = df['Exit Entry']  # After reindexed
    durations = (exits.dropna().index - df.index[exits.dropna().values.astype(int)]).to_series()
    s.loc['Exposure [%]'] = np.nan_to_num(durations.sum() / (s.loc['Duration'] or np.nan) * 100)
    s.loc['Equity Final [$]'] = equity[-1]
    s.loc['Equity Peak [$]'] = equity.max()
    s.loc['Return [%]'] = (equity[-1] - equity[0]) / equity[0] * 100
    c = data.Close
    s.loc['Buy & Hold Return [%]'] = abs(c[-1] - c[0]) / c[0] * 100  # long OR short
    s.loc['Max. Drawdown [%]'] = max_dd = -np.nan_to_num(dd.max()) * 100
    s.loc['Avg. Drawdown [%]'] = -dd_peaks.mean() * 100
    s.loc['Max. Drawdown Duration'] = _round_timedelta(dd_dur.max())
    s.loc['Avg. Drawdown Duration'] = _round_timedelta(dd_dur.mean())
    s.loc['# Trades'] = n_trades = pl.count()
    s.loc['Win Rate [%]'] = win_rate = np.nan if not n_trades else (pl > 0).sum() / n_trades * 100  # noqa: E501
    s.loc['Best Trade [%]'] = returns.max() * 100
    s.loc['Worst Trade [%]'] = returns.min() * 100
    mean_return = returns.mean()
    s.loc['Avg. Trade [%]'] = mean_return * 100
    s.loc['Max. Trade Duration'] = _round_timedelta(durations.max())
    s.loc['Avg. Trade Duration'] = _round_timedelta(durations.mean())
    s.loc['Expectancy [%]'] = ((returns[returns > 0].mean() * win_rate -
                                returns[returns < 0].mean() * (100 - win_rate)))
    pl = pl.dropna()
    s.loc['SQN'] = np.sqrt(n_trades) * pl.mean() / pl.std()
    s.loc['Sharpe Ratio'] = mean_return / (returns.std() or np.nan)
    s.loc['Sortino Ratio'] = mean_return / (returns[returns < 0].std() or np.nan)
    s.loc['Calmar Ratio'] = mean_return / ((-max_dd / 100) or np.nan)

    s.loc['_strategy'] = strategy
    s._trade_data = df  # Private API
    return s


def live_bt(Strategy, data, **kwargs):
    """
    Run the backtest. Returns `pd.Series` with results and statistics.
    Keyword arguments are interpreted as strategy parameters.
    """
    data = _Data(data.ffill())    
    broker = _Broker(data=data, cash=1, commission=0.001, margin=1,
        trade_on_close=False, length=len(data))
    strategy = Strategy(broker, data)  # type: Strategy

    strategy._set_params(**kwargs)

    strategy.init()
    indicator_attrs = {attr: indicator
                       for attr, indicator in strategy.__dict__.items()
                       if isinstance(indicator, _Indicator)}.items()

    # Skip first few candles where indicators are still "warming up"
    # +1 to have at least two entries available
    start = 1 + max((np.isnan(indicator.astype(float)).argmin()
                     for _, indicator in indicator_attrs), default=0)

    # Disable "invalid value encountered in ..." warnings. Comparison
    # np.nan >= 3 is not invalid; it's False.
    with np.errstate(invalid='ignore'):

        for i in range(start, len(data)):
            # Prepare data and indicators for `next` call
            data._set_length(i + 1)
            for attr, indicator in indicator_attrs:
                # Slice indicator on the last dimension (case of 2d indicator)
                setattr(strategy, attr, indicator[..., :i + 1])

            # Handle orders processing and broker stuff
            try:
                broker.next()
            except _OutOfMoneyError:
                break

            # Next tick, a moment before bar close
            strategy.next()
            #if strategy.orders.entry:
            #    print(data.index[i], broker.last_close, strategy.orders.entry, strategy.orders.is_long, strategy)

    
    return strategy, compute_stats(data, broker, strategy)


class BinanceAccount():
    
    def __init__(self):
        self.update()
        
    def update(self):
        
        # tickers: {'BTCUSDT': 9000, ....}
        tickers = binance_client.get_all_tickers()
        self.tickers = {t['symbol']: float(t['price']) for t in tickers}
        
        # get account info
        self.account_info = binance_client.get_account()
        
        # get exchange info
        self.exchange_info = binance_client.get_exchange_info()
        self.symbol_info = {s['symbol']: s for s in self.exchange_info['symbols']}
    
        
    def account_balance(self):
        self.account_info = binance_client.get_account()
        return self.account_info['balances']
    
    def altcoin_value_to_btc(self, symbol, amount):
        if amount == 0:
            return 0
        
        if symbol == 'BTC':
            return amount
        if symbol + 'BTC' in self.tickers:
            p = self.tickers[symbol + 'BTC']
            return p * amount
        elif 'BTC' + symbol in self.tickers:
            p = 1/self.tickers['BTC' + symbol]
            return p * amount
        return 0

    def balance_sum_btc(self, locked=False):
    
        account_balance = self.account_info['balances']

        ret = 0
        for a in account_balance:
            amount = float(a['free'])
            if locked:
                amount += float(a['locked'])
                
            ret += self.altcoin_value_to_btc(a['asset'], amount)
        return ret

def crossover(s1, s2):
    prev_s1 = s1 if isinstance(s1, float) else s1.shift()
    prev_s2 = s2 if isinstance(s2, float) else s2.shift()
    return (prev_s1 < prev_s2) & (s1 > s2)

def crossunder(s1, s2):
    prev_s1 = s1 if isinstance(s1, float) else s1.shift()
    prev_s2 = s2 if isinstance(s2, float) else s2.shift()
    return (prev_s1 > prev_s2) & (s1 < s2)

def cross(s1, s2):
    prev_s1 = s1 if isinstance(s1, float) else s1.shift()
    prev_s2 = s2 if isinstance(s2, float) else s2.shift()
    return ((prev_s1 > prev_s2) & (s1 < s2)) | ((prev_s1 < prev_s2) & (s1 > s2))
    
class SmartExit(bt.Strategy):
    
    tp = None
    sl = None
    tp_trailing = None
    sl_trailing = None
    max_profit = 0
    memory = [None] * 6
    
    def set_tp(self, tp: float, trailing: float=None, on_close=False):
        self.tp = tp
        self.tp_trailing = trailing
        self.tp_on_close = on_close
        self.memory = [tp, trailing, on_close] + self.memory[3:]
        
    def set_sl(self, sl: float=None, trailing: float=None, on_close=False):
        self.sl = sl
        self.sl_trailing = trailing
        self.sl_on_close = on_close
        self.memory = self.memory[:3] + [sl, trailing, on_close]
    
    def next(self):
        super().next()
        
        if self.position.size == 0:
            self.max_profit = 0
            [self.tp, self.tp_trailing, self.tp_on_close, self.sl, self.sl_trailing, self.sl_on_close] = self.memory

        if self.position and not self.orders._close:
            sl = self.sl
            tp = self.tp
            
            if self.position.is_short:
                pl_pct_without_commission = - self.data.Close[-1] / self.position.open_price + 1
                self.max_profit = max(self.max_profit, - self.data.Low[-1] / self.position.open_price + 1)
            else:
                pl_pct_without_commission = self.data.Close[-1] / self.position.open_price - 1
                self.max_profit = max(self.max_profit, self.data.High[-1] / self.position.open_price - 1)
                
            
            if self.tp_trailing:
                if pl_pct_without_commission > tp:
                    self.sl = None
                    self.sl_trailing = self.tp_trailing
                    self.tp = None
                    self.tp_trailing = None
            elif tp:
                if pl_pct_without_commission > tp:
                    self.position.close()
                    return
                if not self.tp_on_close:
                    if self.position.is_long:
                        self.orders.set_tp(self.position.open_price*(1+tp))
                    else:
                        self.orders.set_tp(self.position.open_price*(1-tp))
            
            #print(self.sl_trailing, self.memory)
            if self.sl_trailing is not None:
                
                #print('price now', self.data.Close[-1])
                #print('sl1', self.data.Low[-1] / self.high - 1)
                
                if pl_pct_without_commission - self.max_profit < self.sl_trailing:
                    self.position.close()
                    return
                    
                if not self.sl_on_close:
                    if self.position.is_long:
                        self.orders.set_sl(self.position.open_price * (1 + self.max_profit + self.sl_trailing))
                    else:
                        self.orders.set_sl(self.position.open_price * (1 - self.max_profit - self.sl_trailing))
                    
                #print(self.data.Low[-1] / self.high -1, self.sl_trailing)
    
            elif sl:
                if pl_pct_without_commission < sl:
                    self.position.close()
                    return
                if not self.sl_on_close:
                    if self.position.is_long:
                        self.orders.set_sl(self.position.open_price*(1+sl))
                    else:
                        self.orders.set_sl(self.position.open_price*(1-sl))