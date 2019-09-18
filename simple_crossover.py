# imports
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import pandas as pd
import datetime  # For datetime objects
import pprint
import itertools
import backtrader as bt
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo

from binance.client import Client
from klychi import api_key, api_secret

client = Client(api_key, api_secret)

from matplotlib import style

style.use('ggplot')


def organise_binance_klines(df, save_csv=False, csv_name=''):
    col_names = ['open_time_unix', 'open', 'high', 'low', 'close', 'volume',
                 'close_time_unix', 'quote_asset_volume', 'number_of_trades',
                 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

    df_new = df.copy()

    # change unix time to datetime; binance show time in ms
    df_new['candle-start'] = pd.to_datetime(df_new[0], unit='ms')
    df_new['candle-end'] = pd.to_datetime(df_new[6], unit='ms')

    # rename columns
    df_new.columns = col_names + ['candle-start', 'candle-end']

    # set index and drop extra columns
    df_new.set_index(['candle-end'], inplace=True)

    # change dtypes of data from object to float/int
    df_new = df_new.astype({'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64',
                            'volume': 'float64', 'quote_asset_volume': 'float64', 'number_of_trades': 'int64',
                            'taker_buy_base_asset_volume': 'float64',
                            'taker_buy_quote_asset_volume': 'float64'})

    if save_csv:
        df_new.to_csv(path_or_buf=str(csv_name) + '.csv')

    return df_new


def printTradeAnalysis(analyzer):
    '''
    Function to print the Technical Analysis results in a nice format.
    '''
    # Get the results we are interested in
    total_open = analyzer.total.open
    total_closed = analyzer.total.closed
    total_won = analyzer.won.total
    total_lost = analyzer.lost.total
    win_streak = analyzer.streak.won.longest
    lose_streak = analyzer.streak.lost.longest
    pnl_net = round(analyzer.pnl.net.total, 2)
    strike_rate = round(number=((total_won / total_closed) * 100), ndigits=2)
    # Designate the rows
    h1 = ['Total Open', 'Total Closed', 'Total Won', 'Total Lost']
    h2 = ['Strike Rate %', 'Win Streak', 'Losing Streak', 'PnL Net']
    r1 = [total_open, total_closed, total_won, total_lost]
    r2 = [strike_rate, win_streak, lose_streak, pnl_net]
    # Check which set of headers is the longest.
    if len(h1) > len(h2):
        header_length = len(h1)
    else:
        header_length = len(h2)
    # Print the rows
    print_list = [h1, r1, h2, r2]
    row_format = "{:<15}" * (header_length + 1)
    print("Trade Analysis Results:")
    for row in print_list:
        print(row_format.format('', *row))


def printDrawDown(analyzer):
    '''
    Function to print the DrawDown results in a nice format.
    '''
    ddpct = round(analyzer.drawdown, 2)
    moneydd = round(analyzer.moneydown, 2)
    ddlen = analyzer.len
    maxdd = round(analyzer.max.drawdown, 2)
    maxmoneydd = round(analyzer.max.moneydown, 2)
    maxddlen = analyzer.max.len

    # Designate the rows
    h1 = ['DrawDown %', 'Money Down $', 'DrawDown Length']
    h2 = ['Max DrawDown %', 'Max MoneyDown $ ', 'Max DrawDown Length']
    r1 = [ddpct, moneydd, ddlen]
    r2 = [maxdd, maxmoneydd, maxddlen]

    # Check which set of headers is the longest.
    if len(h1) > len(h2):
        header_length = len(h1)
    else:
        header_length = len(h2)
    # Print the rows
    print_list = [h1, r1, h2, r2]
    row_format = "{:<15}" * (header_length + 1)
    print("DrawDown Results:")
    for row in print_list:
        print(row_format.format('', *row))


# Create a Stratey
class TestStrategy(bt.Strategy):
    #TODO make this work with argparse and make translation between hours and minutes
    params = (
        ('short_ema', 40),
        ('long_ema', 100),
        ('mult', 10),
        ('lever', 10),
        ('printlog', True),
        ('trailstoppct', 0.05),  # trailstop order trail 0.05 = 5%
    )

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.data.datetime[0]
            dt = bt.num2date(dt)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add exponential moving averages
        ema_short = bt.ind.EMA(period=self.params.short_ema)
        ema_long = bt.ind.EMA(period=self.params.long_ema)

        self.crossover = bt.ind.CrossOver(ema_short, ema_long)

    def next(self):

        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.dataclose[0])

        if self.order:
            return  # if an order is active, no new orders are allowed

        if self.crossover > 0:  # cross upwards
            if self.position:
                self.log('CLOSE SHORT , %.2f' % self.dataclose[0])
                self.close()

            self.log('BUY CREATE , %.2f' % self.dataclose[0])
            # to make orders at 5% of portfolio value, no more than 500$
            self.buy(size=round(min((cerebro.broker.getvalue() * 0.05), 500) / self.dataclose[0], 2))
            # exectype=bt.Order.StopTrail, trailpercent=self.params.trailstoppct)

        elif self.crossover < 0:  # cross downwards
            if self.position:
                self.log('CLOSE LONG , %.2f' % self.dataclose[0])
                self.close()

            self.log('SELL CREATE , %.2f' % self.dataclose[0])
            # to make orders at 5% of portfolio value, no more than 500$
            self.sell(size=round(min((cerebro.broker.getvalue() * 0.05), 500) / self.dataclose[0], 2))
            # exectype=bt.Order.StopTrail, trailpercent=self.params.trailstoppct)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:

            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm)
                         )

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm

            elif order.issell():
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm)
                         )

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected', doprint=True)

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f, Acc Balance: %.2f' %
                     (trade.pnl, trade.pnlcomm, cerebro.broker.getvalue()))

        elif trade.justopened:
            self.log('TRADE OPENED, SIZE: %2d , VAL: %.2f' % (trade.size, trade.value))


def parse_args(pargs=None):
    #TODO fix arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Sample for Signal concepts')

    parser.add_argument('--data', required=False,
                        default='btc_usdt1h.csv',
                        help='Specific data to be read in')

    parser.add_argument('--fromdate', required=False, default=None,
                        help='Starting date in YYYY-MM-DD format')

    parser.add_argument('--todate', required=False, default=None,
                        help='Ending date in YYYY-MM-DD format')

    parser.add_argument('--cash', required=False, action='store',
                        type=float, default=10000,
                        help='Cash to start with')

    parser.add_argument('--emaperiod', required=False, action='store',
                        type=int, default=30,
                        help='Period for the moving average')

    parser.add_argument('--mult', required=False, action='store',
                        type=int, default=10,
                        help='multiplier to be applied to PnL (margin trading)')

    parser.add_argument('--lever', required=False, action='store',
                        type=int, default=10,
                        help='leverage level to be used')

    if pargs is not None:
        return parser.parse_args(pargs)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    dfraw = pd.read_csv(args.data, index_col='candle-end')

    dfraw.index = pd.to_datetime(dfraw.index)
    dfraw['candle-start'] = pd.to_datetime(dfraw['candle-start'])

    # timeframe and compression tells the system we have hourly data
    data = bt.feeds.PandasData(dataname=dfraw,
                               timeframe=bt.TimeFrame.Minutes,
                               compression=30)  #TODO make compression updatable depending on data passed

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(args.cash)  # 10,000

    # Set the commission
    cerebro.broker.setcommission(commission=0.001,  # 0.1% binance commission
                                 mult=args.mult,
                                 interest=0.01,  # long/short interest 0.01 -> 1%
                                 leverage=args.lever,
                                 interest_long=True
                                 )

    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    # Create Analyzers
    # RF = 1%,
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='mysharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='mydrawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='mytradeanal')
    cerebro.addanalyzer(bt.analyzers.BasicTradeStats, _name='tradestats')

    # Create Observers
    cerebro.addobserver(bt.observers.DrawDown)

    # Run over everything
    thestrats = cerebro.run()
    thestrat = thestrats[0]

    # Print Analyzers
    print()
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
    print()
    print('Annualised SharpeR:', thestrat.analyzers.mysharpe.get_analysis()['sharperatio'])
    print()
    # CAGR = (end bal / beg bal) ^ 1/n of years - 1
    print('CAGR % ', ((cerebro.broker.getvalue() / 10000) ** 0.5 - 1) * 100)
    #TODO substitute 0.5 with proper calculation of year
    print()
    printTradeAnalysis(thestrat.analyzers.mytradeanal.get_analysis())
    print()
    printDrawDown(thestrat.analyzers.mydrawdown.get_analysis())
    print()
    pprint.pprint(dict(thestrat.analyzers.tradestats.get_analysis()))

    b = Bokeh(style='line', plot_mode='single', scheme=Tradimo())
    cerebro.plot(b)

    # to plot part of the data
    # cerebro.plot(start=datetime.date(2017, 12, 1), end=datetime.date(2018, 2, 1))

    # cerebro.plot()  # plot backtest
